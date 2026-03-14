from __future__ import annotations

import json
import os
import selectors
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class MCPError(Exception):
    pass


class MCPConfigError(MCPError):
    pass


class MCPProtocolError(MCPError):
    pass


class MCPToolCallError(MCPError):
    pass


_STDIO_TRANSPORTS = {"stdio"}
_HTTP_TRANSPORTS = {"http", "streamable_http", "streamable-http"}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    command: tuple[str, ...] = ()
    url: str | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 10000

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MCPServerConfig":
        name = str(payload.get("name") or "").strip()
        if not name:
            raise MCPConfigError("MCP server config requires a non-empty name.")

        transport_value = str(payload.get("transport") or "stdio").strip().lower()
        if transport_value in _STDIO_TRANSPORTS:
            transport = "stdio"
        elif transport_value in _HTTP_TRANSPORTS:
            transport = "http"
        else:
            raise MCPConfigError(
                f"MCP server {name!r} uses unsupported transport {transport_value!r}."
            )

        raw_env = payload.get("env") or {}
        if not isinstance(raw_env, dict):
            raise MCPConfigError(f"MCP server {name!r} env must be an object.")
        env = {str(key): str(value) for key, value in raw_env.items()}

        raw_headers = payload.get("headers") or {}
        if not isinstance(raw_headers, dict):
            raise MCPConfigError(f"MCP server {name!r} headers must be an object.")
        headers = {str(key): str(value) for key, value in raw_headers.items()}

        cwd = payload.get("cwd")
        cwd_text = str(cwd).strip() if cwd is not None else None

        timeout_ms_raw = payload.get("timeout_ms", 10000)
        try:
            timeout_ms = int(timeout_ms_raw)
        except (TypeError, ValueError) as exc:
            raise MCPConfigError(
                f"MCP server {name!r} timeout_ms must be an integer."
            ) from exc
        if timeout_ms < 1:
            raise MCPConfigError(f"MCP server {name!r} timeout_ms must be >= 1.")

        if transport == "stdio":
            raw_command = payload.get("command")
            if not isinstance(raw_command, list) or not raw_command:
                raise MCPConfigError(
                    f"MCP server {name!r} requires a non-empty command array."
                )
            command = tuple(str(part) for part in raw_command if str(part).strip())
            if not command:
                raise MCPConfigError(
                    f"MCP server {name!r} requires at least one non-empty command element."
                )
            return cls(
                name=name,
                transport=transport,
                command=command,
                cwd=cwd_text or None,
                env=env,
                headers=headers,
                timeout_ms=timeout_ms,
            )

        url = str(payload.get("url") or "").strip()
        if not url:
            raise MCPConfigError(f"MCP server {name!r} requires a non-empty url.")
        return cls(
            name=name,
            transport=transport,
            url=url,
            cwd=cwd_text or None,
            env=env,
            headers=headers,
            timeout_ms=timeout_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport,
            "timeout_ms": self.timeout_ms,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.url:
            payload["url"] = self.url
        if self.cwd:
            payload["cwd"] = self.cwd
        if self.env:
            payload["env"] = dict(self.env)
        if self.headers:
            payload["headers"] = dict(self.headers)
        return payload


@dataclass(frozen=True)
class MCPToolDefinition:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    transport: str
    connected: bool
    error: str | None = None
    tool_names: tuple[str, ...] = ()
    server_info: dict[str, Any] | None = None
    protocol_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport,
            "connected": self.connected,
            "tool_names": list(self.tool_names),
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.server_info is not None:
            payload["server_info"] = self.server_info
        if self.protocol_version is not None:
            payload["protocol_version"] = self.protocol_version
        return payload


class MCPSession(Protocol):
    server_info: dict[str, Any] | None
    protocol_version: str | None

    def start(self) -> None: ...

    def close(self) -> None: ...

    def list_tools(self) -> list[MCPToolDefinition]: ...

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class _BaseMCPSession:
    _CLIENT_PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._request_id = 0
        self.server_info: dict[str, Any] | None = None
        self.protocol_version: str | None = None

    def start(self) -> None:
        response = self._request(
            "initialize",
            {
                "protocolVersion": self._CLIENT_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "leo", "version": "0.01"},
            },
        )
        if not isinstance(response, dict):
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} returned a non-object initialize result."
            )

        self.server_info = (
            response["serverInfo"] if isinstance(response.get("serverInfo"), dict) else None
        )
        protocol_version = response.get("protocolVersion")
        if isinstance(protocol_version, str) and protocol_version.strip():
            self.protocol_version = protocol_version.strip()

        self._notify("notifications/initialized", {})

    def list_tools(self) -> list[MCPToolDefinition]:
        result = self._request("tools/list", {})
        tools = self._extract_tools(result)
        next_cursor = result.get("nextCursor") if isinstance(result, dict) else None
        while isinstance(next_cursor, str) and next_cursor.strip():
            result = self._request("tools/list", {"cursor": next_cursor})
            tools.extend(self._extract_tools(result))
            next_cursor = result.get("nextCursor") if isinstance(result, dict) else None
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        if not isinstance(result, dict):
            raise MCPProtocolError(
                f"MCP tool {tool_name!r} returned a non-object result."
            )
        if result.get("isError") is True:
            raise MCPToolCallError(self._summarize_tool_error(tool_name, result))
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send_notification(payload)

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._request_id += 1
        request_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        message = self._send_request(payload)
        if "error" in message:
            error = message["error"]
            if isinstance(error, dict):
                code = error.get("code")
                detail = error.get("message") or error
                raise MCPProtocolError(
                    f"MCP server {self._config.name!r} returned error for {method}: {code} {detail}"
                )
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} returned error for {method}: {error}"
            )
        return message.get("result")

    def _extract_tools(self, result: Any) -> list[MCPToolDefinition]:
        if not isinstance(result, dict):
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} returned a non-object tools/list result."
            )
        raw_tools = result.get("tools")
        if not isinstance(raw_tools, list):
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} returned tools/list without a tools array."
            )

        tools: list[MCPToolDefinition] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            description = str(item.get("description") or "").strip()
            schema = item.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}, "additionalProperties": True}
            tools.append(
                MCPToolDefinition(
                    server_name=self._config.name,
                    name=name,
                    description=description or f"MCP tool from {self._config.name}.",
                    input_schema=schema,
                )
            )
        return tools

    @staticmethod
    def _summarize_tool_error(tool_name: str, result: dict[str, Any]) -> str:
        content = result.get("content")
        if isinstance(content, list):
            snippets: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str) and item["text"].strip():
                    snippets.append(item["text"].strip())
                elif item.get("type") == "image":
                    snippets.append("[image]")
            if snippets:
                return f"MCP tool {tool_name!r} failed: {' | '.join(snippets)}"
        return f"MCP tool {tool_name!r} reported isError=true."

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _send_notification(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MCPStdioSession(_BaseMCPSession):
    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None:
            return

        cwd = Path(self._config.cwd).resolve() if self._config.cwd else None
        if cwd is not None and not cwd.exists():
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} cwd does not exist: {cwd}"
            )

        env = os.environ.copy()
        env.update(self._config.env)
        try:
            self._process = subprocess.Popen(
                list(self._config.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise MCPProtocolError(
                f"Failed to start MCP server {self._config.name!r}: {exc}"
            ) from exc

        super().start()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._write_message(payload)

        while True:
            message = self._read_message()
            if "id" not in message:
                continue
            if message.get("id") != payload["id"]:
                continue
            return message

    def _send_notification(self, payload: dict[str, Any]) -> None:
        self._write_message(payload)

    def _write_message(self, payload: dict[str, Any]) -> None:
        self._ensure_process()
        assert self._process is not None
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        self._ensure_process()
        assert self._process is not None
        assert self._process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        events = selector.select(self._config.timeout_ms / 1000)
        selector.close()
        if not events:
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} did not respond within {self._config.timeout_ms}ms."
            )
        line = self._process.stdout.readline()
        if not line:
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} closed stdout unexpectedly."
            )
        return _decode_jsonrpc_message(line, server_name=self._config.name)

    def _ensure_process(self) -> None:
        process = self._process
        if process is None:
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} session has not been started."
            )
        returncode = process.poll()
        if returncode is None:
            return
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        detail = f" exit_code={returncode}"
        if stderr:
            detail += f" stderr={stderr}"
        raise MCPProtocolError(
            f"MCP server {self._config.name!r} exited unexpectedly.{detail}"
        )


class MCPHTTPSession(_BaseMCPSession):
    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._session_id: str | None = None

    def close(self) -> None:
        if self._session_id is None or not self._config.url:
            return
        request = urllib.request.Request(
            self._config.url,
            method="DELETE",
            headers=self._build_headers(include_content_type=False),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_ms / 1000,
            ):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 405, 501}:
                raise MCPProtocolError(
                    f"MCP server {self._config.name!r} returned HTTP {exc.code} on session close."
                ) from exc
        except urllib.error.URLError:
            return

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body, headers = self._post_json(payload)
        return self._parse_http_response(body, headers, request_id=payload["id"])

    def _send_notification(self, payload: dict[str, Any]) -> None:
        self._post_json(payload, expect_response=False)

    def _post_json(
        self,
        payload: dict[str, Any],
        *,
        expect_response: bool = True,
    ) -> tuple[bytes, dict[str, str]]:
        if not self._config.url:
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} has no HTTP url configured."
            )

        request = urllib.request.Request(
            self._config.url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._build_headers(include_content_type=True),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_ms / 1000,
            ) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
                session_id = headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                if not expect_response or response.status == 202:
                    response.read()
                    return b"", headers
                return response.read(), headers
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = _summarize_http_error(body)
            raise MCPProtocolError(
                f"MCP server {self._config.name!r} returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise MCPProtocolError(
                f"Failed to reach MCP server {self._config.name!r}: {exc.reason}"
            ) from exc

    def _parse_http_response(
        self,
        body: bytes,
        headers: dict[str, str],
        *,
        request_id: int,
    ) -> dict[str, Any]:
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type == "text/event-stream":
            return self._parse_sse_response(body, request_id=request_id)
        if content_type in {"", "application/json"}:
            if not body:
                raise MCPProtocolError(
                    f"MCP server {self._config.name!r} returned an empty HTTP body."
                )
            return _decode_jsonrpc_message(body.decode("utf-8"), server_name=self._config.name)
        raise MCPProtocolError(
            f"MCP server {self._config.name!r} returned unsupported content type {content_type!r}."
        )

    def _parse_sse_response(self, body: bytes, *, request_id: int) -> dict[str, Any]:
        data_lines: list[str] = []
        for raw_line in body.decode("utf-8").splitlines():
            line = raw_line.rstrip("\r")
            if not line:
                payload = self._decode_sse_event(data_lines, request_id=request_id)
                if payload is not None:
                    return payload
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        payload = self._decode_sse_event(data_lines, request_id=request_id)
        if payload is not None:
            return payload
        raise MCPProtocolError(
            f"MCP server {self._config.name!r} returned SSE without a matching response for request {request_id}."
        )

    def _decode_sse_event(
        self,
        data_lines: list[str],
        *,
        request_id: int,
    ) -> dict[str, Any] | None:
        if not data_lines:
            return None
        payload = _decode_jsonrpc_message(
            "\n".join(data_lines),
            server_name=self._config.name,
        )
        if payload.get("id") != request_id:
            return None
        return payload

    def _build_headers(self, *, include_content_type: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        if self._session_id:
            headers["MCP-Session-Id"] = self._session_id
        headers.update(self._config.headers)
        return headers


class MCPToolRuntime:
    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = list(configs)
        self._sessions: dict[str, MCPSession] = {}
        self._tools_by_server: dict[str, tuple[MCPToolDefinition, ...]] = {}
        self._statuses: dict[str, MCPServerStatus] = {}
        self._initialize()

    @classmethod
    def from_sources(
        cls,
        *,
        configs: list[MCPServerConfig] | None = None,
        config_path: str | Path | None = None,
    ) -> "MCPToolRuntime":
        return cls(load_mcp_server_configs(configs=configs, config_path=config_path))

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()

    def list_server_statuses(self) -> list[dict[str, Any]]:
        return [self._statuses[name].to_dict() for name in sorted(self._statuses)]

    def list_tool_definitions(self) -> list[MCPToolDefinition]:
        tools: list[MCPToolDefinition] = []
        for server_name in sorted(self._tools_by_server):
            tools.extend(self._tools_by_server[server_name])
        return tools

    def invoke_tool(self, server_name: str, tool_name: str, **tool_args: Any) -> dict[str, Any]:
        session = self._sessions.get(server_name)
        if session is None:
            raise MCPToolCallError(f"MCP server {server_name!r} is not connected.")
        return session.call_tool(tool_name, tool_args)

    def _initialize(self) -> None:
        for config in self._configs:
            session = _build_session(config)
            try:
                session.start()
                tools = tuple(session.list_tools())
            except MCPError as exc:
                session.close()
                self._statuses[config.name] = MCPServerStatus(
                    name=config.name,
                    transport=config.transport,
                    connected=False,
                    error=str(exc),
                )
                continue

            self._sessions[config.name] = session
            self._tools_by_server[config.name] = tools
            self._statuses[config.name] = MCPServerStatus(
                name=config.name,
                transport=config.transport,
                connected=True,
                tool_names=tuple(tool.name for tool in tools),
                server_info=session.server_info,
                protocol_version=session.protocol_version,
            )


def load_mcp_server_configs(
    *,
    configs: list[MCPServerConfig] | None = None,
    config_path: str | Path | None = None,
) -> list[MCPServerConfig]:
    if configs is not None:
        return list(configs)

    if config_path is not None:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return _decode_mcp_config_payload(payload, source=str(config_path))

    env_path = os.getenv("LEO_MCP_CONFIG")
    if env_path:
        payload = json.loads(Path(env_path).read_text(encoding="utf-8"))
        return _decode_mcp_config_payload(payload, source=env_path)

    env_inline = os.getenv("LEO_MCP_SERVERS")
    if env_inline:
        payload = json.loads(env_inline)
        return _decode_mcp_config_payload(payload, source="LEO_MCP_SERVERS")

    return []


def _decode_mcp_config_payload(payload: Any, *, source: str) -> list[MCPServerConfig]:
    if not isinstance(payload, list):
        raise MCPConfigError(f"MCP config from {source} must be a JSON array.")
    configs: list[MCPServerConfig] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise MCPConfigError(
                f"MCP config entry {index} from {source} must be an object."
            )
        configs.append(MCPServerConfig.from_dict(item))
    return configs


def _build_session(config: MCPServerConfig) -> MCPSession:
    if config.transport == "stdio":
        return MCPStdioSession(config)
    if config.transport == "http":
        return MCPHTTPSession(config)
    raise MCPConfigError(
        f"MCP server {config.name!r} uses unsupported transport {config.transport!r}."
    )


def _decode_jsonrpc_message(raw: str, *, server_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPProtocolError(
            f"MCP server {server_name!r} emitted invalid JSON: {raw.strip()}"
        ) from exc
    if not isinstance(payload, dict):
        raise MCPProtocolError(
            f"MCP server {server_name!r} emitted a non-object JSON message."
        )
    return payload


def _summarize_http_error(body: bytes) -> str:
    if not body:
        return "empty response body"
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return "empty response body"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message") or error
            return f"{code} {message}"
    return text

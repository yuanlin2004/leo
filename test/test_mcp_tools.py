from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import urllib.error

import pytest

from leo.tools.mcp import MCPServerConfig
from leo.tools.registry import ToolsRegistry


_RUN_LIVE_DEEPWIKI_TESTS = os.getenv("LEO_RUN_LIVE_MCP_TESTS") == "1"


def _write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        """from __future__ import annotations
import json
import os
import sys

MODE = os.getenv("FAKE_MCP_MODE", "ok")
TOOL_NAME = os.getenv("FAKE_MCP_TOOL", "mcp_echo")

def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()

if MODE == "badjson":
    sys.stdout.write("not-json\\n")
    sys.stdout.flush()
    raise SystemExit(0)

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "fake-mcp", "version": "1.0"},
                    "capabilities": {"tools": {}},
                },
            }
        )
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "tools": [
                        {
                            "name": TOOL_NAME,
                            "description": "Echo a message.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        if MODE == "tool_error":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "content": [{"type": "text", "text": "simulated failure"}],
                        "isError": True,
                    },
                }
            )
            continue
        msg = message.get("params", {}).get("arguments", {}).get("message", "")
        send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "content": [{"type": "text", "text": f"echo:{msg}"}],
                    "structuredContent": {"echo": msg},
                    "isError": False,
                },
            }
        )
""",
        encoding="utf-8",
    )


def _stdio_config(server_script: Path, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "fake-server",
        "transport": "stdio",
        "command": ["python3", str(server_script)],
    }
    payload.update(extra)
    return payload

class _FakeHTTPResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeHTTPErrorBody:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


def _http_urlopen_factory(
    *,
    response_mode: str = "json",
    tool_error: bool = False,
):
    session_id = "fake-http-session"

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        if request.get_method() == "DELETE":
            return _FakeHTTPResponse(b"", status=204)

        payload = json.loads(request.data.decode("utf-8"))
        method = payload.get("method")
        headers = {key.lower(): value for key, value in request.header_items()}

        if method == "initialize":
            return _FakeHTTPResponse(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "fake-http-mcp", "version": "1.0"},
                            "capabilities": {"tools": {}},
                        },
                    }
                ).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "mcp-session-id": session_id,
                },
            )

        if headers.get("mcp-session-id") != session_id:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "missing session id",
                hdrs=None,
                fp=_FakeHTTPErrorBody(b"missing session id"),
            )

        if method == "notifications/initialized":
            return _FakeHTTPResponse(b"", status=202, headers={"content-type": "application/json"})

        if method == "tools/list":
            body = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "tools": [
                        {
                            "name": "http_echo",
                            "description": "Echo a message over HTTP.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                        }
                    ]
                },
            }
            return _fake_http_payload(body, response_mode=response_mode)

        if method == "tools/call":
            message = payload.get("params", {}).get("arguments", {}).get("message", "")
            body = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "content": [{"type": "text", "text": f"http:{message}"}],
                    "structuredContent": {"echo": message},
                    "isError": False,
                },
            }
            if tool_error:
                body["result"] = {
                    "content": [{"type": "text", "text": "http simulated failure"}],
                    "isError": True,
                }
            return _fake_http_payload(body, response_mode=response_mode)

        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "unknown method",
            hdrs=None,
            fp=_FakeHTTPErrorBody(b"unknown method"),
        )

    return fake_urlopen


def _fake_http_payload(
    payload: dict[str, object],
    *,
    response_mode: str,
) -> _FakeHTTPResponse:
    if response_mode == "sse":
        return _FakeHTTPResponse(
            f"data: {json.dumps(payload)}\n\n".encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )
    return _FakeHTTPResponse(
        json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_registry_registers_and_executes_mcp_tools(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_script)

    empty_registry = ToolsRegistry(workspace_root=tmp_path, mcp_servers=[])
    assert empty_registry.list_mcp_servers() == []
    assert "list_mcp_servers" in empty_registry.get_all_tools()
    assert empty_registry.get_tool_provenance("list_mcp_servers") == "runtime:core"
    assert "mcp_echo" not in empty_registry.get_all_tools()

    registry = ToolsRegistry(
        workspace_root=tmp_path,
        mcp_servers=[MCPServerConfig.from_dict(_stdio_config(server_script))],
    )

    assert registry.get_tool_provenance("mcp_echo") == "mcp:fake-server"
    statuses = registry.list_mcp_servers()
    assert statuses == [
        {
            "name": "fake-server",
            "transport": "stdio",
            "connected": True,
            "tool_names": ["mcp_echo"],
            "server_info": {"name": "fake-mcp", "version": "1.0"},
            "protocol_version": "2024-11-05",
        }
    ]

    result = registry.execute("mcp_echo", message="hello")

    assert result["content"] == [{"type": "text", "text": "echo:hello"}]
    assert result["structuredContent"] == {"echo": "hello"}


def test_registry_loads_mcp_servers_from_environment(tmp_path: Path, monkeypatch) -> None:
    server_script = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_script)
    monkeypatch.setenv(
        "LEO_MCP_SERVERS",
        json.dumps([_stdio_config(server_script, name="env-server", env={"FAKE_MCP_TOOL": "env_echo"})]),
    )

    registry = ToolsRegistry(workspace_root=tmp_path)

    assert registry.get_tool_provenance("env_echo") == "mcp:env-server"
    assert registry.list_mcp_servers()[0]["name"] == "env-server"


def test_registry_reports_mcp_connection_failures(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_script)

    registry = ToolsRegistry(
        workspace_root=tmp_path,
        mcp_servers=[
            MCPServerConfig.from_dict(
                _stdio_config(server_script, env={"FAKE_MCP_MODE": "badjson"})
            )
        ],
    )

    statuses = registry.list_mcp_servers()
    assert statuses[0]["name"] == "fake-server"
    assert statuses[0]["connected"] is False
    assert "invalid JSON" in statuses[0]["error"]


def test_registry_surfaces_mcp_tool_errors(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_mcp_server.py"
    _write_fake_mcp_server(server_script)

    registry = ToolsRegistry(
        workspace_root=tmp_path,
        mcp_servers=[
            MCPServerConfig.from_dict(
                _stdio_config(server_script, env={"FAKE_MCP_MODE": "tool_error"})
            )
        ],
    )

    with pytest.raises(Exception, match="simulated failure"):
        registry.execute("mcp_echo", message="hello")


def test_registry_registers_and_executes_http_mcp_tools(tmp_path: Path) -> None:
    with patch("urllib.request.urlopen", _http_urlopen_factory()):
        registry = ToolsRegistry(
            workspace_root=tmp_path,
            mcp_servers=[
                MCPServerConfig.from_dict(
                    {
                        "name": "http-server",
                        "transport": "http",
                        "url": "http://fake.test/mcp",
                    }
                )
            ],
        )

        assert registry.get_tool_provenance("http_echo") == "mcp:http-server"
        assert registry.list_mcp_servers() == [
            {
                "name": "http-server",
                "transport": "http",
                "connected": True,
                "tool_names": ["http_echo"],
                "server_info": {"name": "fake-http-mcp", "version": "1.0"},
                "protocol_version": "2024-11-05",
            }
        ]

        result = registry.execute("http_echo", message="hello")
        assert result["content"] == [{"type": "text", "text": "http:hello"}]
        assert result["structuredContent"] == {"echo": "hello"}


def test_registry_supports_http_sse_mcp_responses(tmp_path: Path) -> None:
    with patch("urllib.request.urlopen", _http_urlopen_factory(response_mode="sse")):
        registry = ToolsRegistry(
            workspace_root=tmp_path,
            mcp_servers=[
                MCPServerConfig.from_dict(
                    {
                        "name": "http-sse-server",
                        "transport": "streamable_http",
                        "url": "http://fake.test/mcp",
                    }
                )
            ],
        )

        result = registry.execute("http_echo", message="sse")
        assert result["structuredContent"] == {"echo": "sse"}


def test_registry_reports_http_mcp_connection_failures(tmp_path: Path) -> None:
    def fail_urlopen(request, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", fail_urlopen):
        registry = ToolsRegistry(
            workspace_root=tmp_path,
            mcp_servers=[
                MCPServerConfig.from_dict(
                    {
                        "name": "bad-http-server",
                        "transport": "http",
                        "url": "http://fake.test/mcp",
                        "timeout_ms": 200,
                    }
                )
            ],
        )

    statuses = registry.list_mcp_servers()
    assert statuses[0]["name"] == "bad-http-server"
    assert statuses[0]["connected"] is False
    assert "Failed to reach" in statuses[0]["error"]


def test_registry_surfaces_http_mcp_tool_errors(tmp_path: Path) -> None:
    with patch("urllib.request.urlopen", _http_urlopen_factory(tool_error=True)):
        registry = ToolsRegistry(
            workspace_root=tmp_path,
            mcp_servers=[
                MCPServerConfig.from_dict(
                    {
                        "name": "http-error-server",
                        "transport": "http",
                        "url": "http://fake.test/mcp",
                    }
                )
            ],
        )

        with pytest.raises(Exception, match="http simulated failure"):
            registry.execute("http_echo", message="hello")


@pytest.mark.skipif(
    not _RUN_LIVE_DEEPWIKI_TESTS,
    reason="set LEO_RUN_LIVE_MCP_TESTS=1 to run live DeepWiki MCP integration tests",
)
def test_live_deepwiki_mcp_server_supports_http_transport(tmp_path: Path) -> None:
    registry = ToolsRegistry(
        workspace_root=tmp_path,
        mcp_servers=[
            MCPServerConfig.from_dict(
                {
                    "name": "deepwiki",
                    "transport": "http",
                    "url": "https://mcp.deepwiki.com/mcp",
                    "timeout_ms": 30000,
                }
            )
        ],
    )

    statuses = registry.list_mcp_servers()
    assert len(statuses) == 1
    assert statuses[0]["name"] == "deepwiki"
    assert statuses[0]["transport"] == "http"
    assert statuses[0]["connected"] is True
    assert "read_wiki_structure" in statuses[0]["tool_names"]
    assert "ask_question" in statuses[0]["tool_names"]

    result = registry.execute("read_wiki_structure", repoName="badlogic/pi-mono")

    assert result["isError"] is False
    assert isinstance(result.get("structuredContent"), dict)
    summary = result["structuredContent"]["result"]
    assert "Available pages for badlogic/pi-mono" in summary
    assert "Extension System" in summary

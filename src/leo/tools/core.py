from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from leo.skills.runtime import probe_tmux_runtime


class CoreToolError(Exception):
    pass


class CoreToolRuntime:
    _DEFAULT_READ_MAX_CHARS = 20000
    _DEFAULT_PROCESS_OUTPUT_MAX_CHARS = 12000
    _SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = (
            Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        )
        self._tmux_sessions: set[str] = set()

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def reset_state(self) -> None:
        for session_name in list(self._tmux_sessions):
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                check=False,
            )
            self._tmux_sessions.discard(session_name)

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = _DEFAULT_READ_MAX_CHARS,
    ) -> dict[str, object]:
        file_path = self._resolve_existing_path(path)
        if not file_path.is_file():
            raise CoreToolError(f"Path is not a file: {path}")

        if start_line < 1:
            raise CoreToolError("start_line must be >= 1.")
        if end_line is not None and end_line < start_line:
            raise CoreToolError("end_line must be >= start_line.")
        if max_chars < 1:
            raise CoreToolError("max_chars must be >= 1.")

        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        start_index = min(start_line - 1, total_lines)
        if end_line is None:
            selected = lines[start_index:]
            actual_end_line = total_lines
        else:
            selected = lines[start_index:end_line]
            actual_end_line = min(end_line, total_lines)

        content = "".join(selected)
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        return {
            "path": str(file_path),
            "content": content,
            "start_line": start_line,
            "end_line": actual_end_line,
            "total_lines": total_lines,
            "truncated": truncated,
        }

    def write_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, object]:
        file_path = self._resolve_path_for_write(path)
        existed = file_path.exists()
        if existed and not overwrite:
            raise CoreToolError(
                f"Refusing to overwrite existing file without overwrite=true: {path}"
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {
            "path": str(file_path),
            "created": not existed,
            "bytes_written": len(content.encode("utf-8")),
        }

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        replace_all: bool = False,
    ) -> dict[str, object]:
        file_path = self._resolve_existing_path(path)
        if not file_path.is_file():
            raise CoreToolError(f"Path is not a file: {path}")
        if not old_text:
            raise CoreToolError("old_text must be non-empty.")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise CoreToolError("old_text was not found in the target file.")
        if occurrences > 1 and not replace_all:
            raise CoreToolError(
                "old_text appears multiple times; rerun with replace_all=true or use a more specific match."
            )

        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        file_path.write_text(updated, encoding="utf-8")
        return {
            "path": str(file_path),
            "replacements": occurrences if replace_all else 1,
        }

    def run_shell(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_ms: int = 30000,
        max_output_chars: int = _DEFAULT_PROCESS_OUTPUT_MAX_CHARS,
    ) -> dict[str, object]:
        if timeout_ms < 1:
            raise CoreToolError("timeout_ms must be >= 1.")
        if max_output_chars < 1:
            raise CoreToolError("max_output_chars must be >= 1.")

        run_cwd = self._resolve_directory(cwd) if cwd else self._workspace_root
        shell = os.environ.get("SHELL") or "/bin/sh"
        try:
            completed = subprocess.run(
                [shell, "-lc", command],
                cwd=run_cwd,
                capture_output=True,
                text=True,
                timeout=timeout_ms / 1000,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CoreToolError(f"Shell executable not found: {shell}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CoreToolError(f"Command timed out after {timeout_ms}ms.") from exc

        stdout, stdout_truncated = self._truncate_text(
            completed.stdout,
            max_output_chars,
        )
        stderr, stderr_truncated = self._truncate_text(
            completed.stderr,
            max_output_chars,
        )
        return {
            "command": command,
            "cwd": str(run_cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def tmux_start_session(
        self,
        *,
        session_name: str | None = None,
        cwd: str | None = None,
        command: str | None = None,
    ) -> dict[str, object]:
        self._ensure_tmux_available()
        run_cwd = self._resolve_directory(cwd) if cwd else self._workspace_root
        name = self._normalize_session_name(session_name)
        if name in self._tmux_sessions:
            raise CoreToolError(f"tmux session is already tracked: {name}")

        shell = os.environ.get("SHELL") or "/bin/sh"
        argv = ["tmux", "new-session", "-d", "-s", name, "-c", str(run_cwd)]
        if command:
            argv.extend([shell, "-lc", command])

        try:
            subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["tmux", "set-option", "-t", name, "remain-on-exit", "on"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown tmux error"
            raise CoreToolError(f"Failed to start tmux session: {stderr}") from exc

        self._tmux_sessions.add(name)
        return {
            "session_name": name,
            "cwd": str(run_cwd),
            "command": command or "",
        }

    def tmux_send_keys(
        self,
        session_name: str,
        keys: str,
        *,
        enter: bool = True,
    ) -> dict[str, object]:
        self._require_known_session(session_name)
        argv = ["tmux", "send-keys", "-t", session_name, keys]
        if enter:
            argv.append("Enter")
        self._run_tmux_command(argv, "send keys to")
        return {
            "session_name": session_name,
            "sent_keys": keys,
            "enter": enter,
        }

    def tmux_capture_pane(
        self,
        session_name: str,
        *,
        max_chars: int = _DEFAULT_PROCESS_OUTPUT_MAX_CHARS,
    ) -> dict[str, object]:
        self._require_known_session(session_name)
        if max_chars < 1:
            raise CoreToolError("max_chars must be >= 1.")

        try:
            completed = subprocess.run(
                ["tmux", "capture-pane", "-p", "-t", session_name],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown tmux error"
            raise CoreToolError(f"Failed to capture tmux pane: {stderr}") from exc

        content, truncated = self._truncate_text(completed.stdout, max_chars)
        return {
            "session_name": session_name,
            "content": content,
            "truncated": truncated,
        }

    def tmux_kill_session(self, session_name: str) -> dict[str, object]:
        self._require_known_session(session_name)
        self._run_tmux_command(
            ["tmux", "kill-session", "-t", session_name],
            "kill",
        )
        self._tmux_sessions.discard(session_name)
        return {"session_name": session_name, "killed": True}

    def _resolve_existing_path(self, raw_path: str) -> Path:
        path = self._resolve_path(raw_path)
        if not path.exists():
            raise CoreToolError(f"Path does not exist: {raw_path}")
        return path

    def _resolve_path_for_write(self, raw_path: str) -> Path:
        path = self._resolve_path(raw_path)
        if not path.parent.exists():
            parent = path.parent
            while not parent.exists() and parent != self._workspace_root:
                parent = parent.parent
            if not parent.exists():
                raise CoreToolError(f"Parent path is outside the workspace: {raw_path}")
        return path

    def _resolve_directory(self, raw_path: str) -> Path:
        path = self._resolve_existing_path(raw_path)
        if not path.is_dir():
            raise CoreToolError(f"Path is not a directory: {raw_path}")
        return path

    def _resolve_path(self, raw_path: str) -> Path:
        text = (raw_path or "").strip()
        if not text:
            raise CoreToolError("path must be a non-empty string.")

        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate

        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self._workspace_root):
            raise CoreToolError(
                f"Path escapes the workspace root {self._workspace_root}: {raw_path}"
            )
        return resolved

    def _ensure_tmux_available(self) -> None:
        available, error = probe_tmux_runtime()
        if not available:
            raise CoreToolError(error or "tmux is unavailable.")

    def _normalize_session_name(self, session_name: str | None) -> str:
        if session_name is None or not session_name.strip():
            return f"leo-core-{int(time.time() * 1000)}"

        value = session_name.strip()
        if not self._SESSION_NAME_RE.match(value):
            raise CoreToolError(
                "session_name may only contain letters, numbers, underscores, and hyphens."
            )
        return value

    def _require_known_session(self, session_name: str) -> None:
        self._ensure_tmux_available()
        if session_name not in self._tmux_sessions:
            raise CoreToolError(f"Unknown managed tmux session: {session_name}")

    def _run_tmux_command(self, argv: list[str], action: str) -> None:
        try:
            subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown tmux error"
            raise CoreToolError(f"Failed to {action} tmux session: {stderr}") from exc

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
        if len(text) <= max_chars:
            return text, False
        return text[:max_chars], True


def build_core_tool_specs(
    runtime: CoreToolRuntime,
) -> list[tuple[str, str, dict[str, object], object]]:
    return [
        (
            "read_file",
            "Read a UTF-8 text file from the workspace with optional line ranges.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer"},
                    "max_chars": {
                        "type": "integer",
                        "default": CoreToolRuntime._DEFAULT_READ_MAX_CHARS,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            runtime.read_file,
        ),
        (
            "write_file",
            "Write a UTF-8 text file inside the workspace. By default this creates new files and refuses to overwrite existing ones.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            runtime.write_file,
        ),
        (
            "edit_file",
            "Replace one matching text block in a UTF-8 workspace file. If the target text appears multiple times, the call must set replace_all=true or use a more specific match.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            runtime.edit_file,
        ),
        (
            "run_shell",
            "Run a shell command inside the workspace and return its exit code, stdout, and stderr.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_ms": {"type": "integer", "default": 30000},
                    "max_output_chars": {
                        "type": "integer",
                        "default": CoreToolRuntime._DEFAULT_PROCESS_OUTPUT_MAX_CHARS,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            runtime.run_shell,
        ),
        (
            "tmux_start_session",
            "Start a managed tmux session inside the workspace.",
            {
                "type": "object",
                "properties": {
                    "session_name": {"type": "string"},
                    "cwd": {"type": "string"},
                    "command": {"type": "string"},
                },
                "additionalProperties": False,
            },
            runtime.tmux_start_session,
        ),
        (
            "tmux_send_keys",
            "Send keys to a managed tmux session.",
            {
                "type": "object",
                "properties": {
                    "session_name": {"type": "string"},
                    "keys": {"type": "string"},
                    "enter": {"type": "boolean", "default": True},
                },
                "required": ["session_name", "keys"],
                "additionalProperties": False,
            },
            runtime.tmux_send_keys,
        ),
        (
            "tmux_capture_pane",
            "Capture text from a managed tmux session pane.",
            {
                "type": "object",
                "properties": {
                    "session_name": {"type": "string"},
                    "max_chars": {
                        "type": "integer",
                        "default": CoreToolRuntime._DEFAULT_PROCESS_OUTPUT_MAX_CHARS,
                    },
                },
                "required": ["session_name"],
                "additionalProperties": False,
            },
            runtime.tmux_capture_pane,
        ),
        (
            "tmux_kill_session",
            "Kill a managed tmux session created through Leo.",
            {
                "type": "object",
                "properties": {
                    "session_name": {"type": "string"},
                },
                "required": ["session_name"],
                "additionalProperties": False,
            },
            runtime.tmux_kill_session,
        ),
    ]

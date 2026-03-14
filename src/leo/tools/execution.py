from __future__ import annotations

import builtins
import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any


class ExecutionContextError(Exception):
    pass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    error: dict[str, str] | None
    summary: str
    globals_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "error": self.error,
            "summary": self.summary,
            "globals_count": self.globals_count,
        }


class ExecutionContext:
    _DEFAULT_MAX_OUTPUT_CHARS = 12000
    _DEFAULT_MAX_TRACEBACK_CHARS = 4000

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._globals: dict[str, Any] = {
            "__builtins__": builtins,
            "__name__": "__leo_exec__",
        }

    def execute_python(
        self,
        code: str,
        *,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
        max_traceback_chars: int = _DEFAULT_MAX_TRACEBACK_CHARS,
    ) -> dict[str, Any]:
        if not isinstance(code, str) or not code.strip():
            raise ExecutionContextError("code must be a non-empty string.")
        if max_output_chars < 1:
            raise ExecutionContextError("max_output_chars must be >= 1.")
        if max_traceback_chars < 1:
            raise ExecutionContextError("max_traceback_chars must be >= 1.")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        error_payload: dict[str, str] | None = None

        try:
            compiled = compile(code, "<leo-exec>", "exec")
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                exec(compiled, self._globals, self._globals)
            summary = "Execution completed successfully."
        except Exception as exc:  # noqa: BLE001
            tb_text = "".join(traceback.format_exception(exc)).strip()
            if len(tb_text) > max_traceback_chars:
                tb_text = tb_text[:max_traceback_chars]
            error_payload = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb_text,
            }
            summary = f"Execution failed with {type(exc).__name__}: {exc}"

        stdout_text, stdout_truncated = self._truncate_text(
            stdout_buffer.getvalue(),
            max_output_chars,
        )
        stderr_text, stderr_truncated = self._truncate_text(
            stderr_buffer.getvalue(),
            max_output_chars,
        )

        result = ExecutionResult(
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            error=error_payload,
            summary=summary,
            globals_count=len([key for key in self._globals if not key.startswith("__")]),
        )
        return result.to_dict()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        return text[:limit], True

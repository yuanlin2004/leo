from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal


class SkillRuntimeError(Exception):
    pass


_TMUX_EXIT_CODE_RE = re.compile(r"__LEO_EXIT_CODE__:(\d+)")


@lru_cache(maxsize=1)
def probe_tmux_runtime() -> tuple[bool, str | None]:
    session_name = f"leo-probe-{int(time.time() * 1000)}"
    try:
        created = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "tmux is required for this skill command but is not installed."

    stderr = created.stderr.strip()
    if created.returncode != 0 or stderr:
        message = stderr or "unknown tmux error"
        return (
            False,
            "tmux is required for this skill command but is not available in the current environment."
            if "Operation not permitted" in message or "failed to connect" in message
            else f"Failed to start tmux session: {message}",
        )

    exists = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode != 0:
        message = exists.stderr.strip() or "tmux session could not be verified"
        return False, f"Failed to start tmux session: {message}"

    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return True, None


@dataclass(frozen=True)
class SkillRequirement:
    kind: Literal["mcp", "env_var", "binary", "platform", "auth", "compatibility"]
    name: str
    value: str
    required: bool = True
    source: str = "skill"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "value": self.value,
            "required": self.required,
            "source": self.source,
        }


@dataclass(frozen=True)
class SkillCommand:
    name: str
    command_path: str
    execution_mode: Literal["direct", "tmux"]
    executable: str
    source: str = "skill"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "command_path": self.command_path,
            "execution_mode": self.execution_mode,
            "executable": self.executable,
            "source": self.source,
        }


@dataclass(frozen=True)
class SkillCommandResult:
    skill_id: str
    command_name: str
    execution_mode: Literal["direct", "tmux"]
    exit_code: int
    stdout: str
    stderr: str
    command_path: str
    session_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "skill_id": self.skill_id,
            "command_name": self.command_name,
            "execution_mode": self.execution_mode,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command_path": self.command_path,
        }
        if self.session_name is not None:
            payload["session_name"] = self.session_name
        return payload


def _run_direct(
    *,
    argv: list[str],
    cwd: Path,
    timeout_ms: int,
    command_path: str,
    skill_id: str,
    command_name: str,
) -> SkillCommandResult:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_ms) / 1000,
            check=False,
        )
    except FileNotFoundError as exc:
        missing = argv[0] if argv else "<unknown>"
        raise SkillRuntimeError(f"Required executable not found: {missing}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SkillRuntimeError(
            f"Skill command timed out after {timeout_ms}ms: {command_name}"
        ) from exc

    return SkillCommandResult(
        skill_id=skill_id,
        command_name=command_name,
        execution_mode="direct",
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command_path=command_path,
    )


def _run_tmux(
    *,
    argv: list[str],
    cwd: Path,
    timeout_ms: int,
    command_path: str,
    skill_id: str,
    command_name: str,
) -> SkillCommandResult:
    tmux_available, tmux_error = probe_tmux_runtime()
    if not tmux_available:
        raise SkillRuntimeError(tmux_error or "tmux is unavailable.")

    marker = "__LEO_EXIT_CODE__:"
    session_name = (
        f"leo-skill-{skill_id}-{command_name}-{int(time.time() * 1000)}"
        .replace("/", "-")
        .replace("_", "-")
    )
    quoted_command = " ".join(shlex.quote(part) for part in argv)

    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "-c", str(cwd)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "tmux",
                "send-keys",
                "-t",
                session_name,
                f"{quoted_command}; printf '\\n{marker}%s\\n' $? ",
                "Enter",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SkillRuntimeError(
            "tmux is required for this skill command but is not installed."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown tmux error"
        if "Operation not permitted" in stderr or "failed to connect" in stderr:
            raise SkillRuntimeError(
                "tmux is required for this skill command but is not available in the current environment."
            ) from exc
        raise SkillRuntimeError(f"Failed to start tmux session: {stderr}") from exc

    deadline = time.monotonic() + max(1, timeout_ms) / 1000
    output = ""
    exit_code = None

    try:
        while time.monotonic() < deadline:
            capture = subprocess.run(
                ["tmux", "capture-pane", "-p", "-t", session_name],
                capture_output=True,
                text=True,
                check=False,
            )
            output = capture.stdout
            matches = _TMUX_EXIT_CODE_RE.findall(output)
            if matches:
                exit_code = int(matches[-1])
                break
            time.sleep(0.1)

        if exit_code is None:
            raise SkillRuntimeError(
                f"Skill command timed out after {timeout_ms}ms in tmux: {command_name}"
            )
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
            text=True,
            check=False,
        )

    cleaned_output = _TMUX_EXIT_CODE_RE.sub("", output).rstrip()
    return SkillCommandResult(
        skill_id=skill_id,
        command_name=command_name,
        execution_mode="tmux",
        exit_code=exit_code,
        stdout=cleaned_output,
        stderr="",
        command_path=command_path,
        session_name=session_name,
    )


def run_skill_command(
    *,
    skill_id: str,
    command: SkillCommand,
    skill_root: Path,
    args: list[str] | None = None,
    timeout_ms: int = 30000,
) -> SkillCommandResult:
    resolved_script = (skill_root / command.command_path).resolve()
    command_args = [str(arg) for arg in (args or [])]

    if command.execution_mode == "direct":
        if command.executable == "python3":
            argv = ["python3", str(resolved_script), *command_args]
        elif command.executable == "node":
            argv = ["node", str(resolved_script), *command_args]
        else:
            argv = [str(resolved_script), *command_args]
        return _run_direct(
            argv=argv,
            cwd=skill_root,
            timeout_ms=timeout_ms,
            command_path=command.command_path,
            skill_id=skill_id,
            command_name=command.name,
        )

    argv = [str(resolved_script), *command_args]
    return _run_tmux(
        argv=argv,
        cwd=skill_root,
        timeout_ms=timeout_ms,
        command_path=command.command_path,
        skill_id=skill_id,
        command_name=command.name,
    )

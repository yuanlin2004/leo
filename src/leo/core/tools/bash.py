from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

OUTPUT_CAP = 8 * 1024
DEFAULT_TIMEOUT = 30

_SCRATCH_DIR: Path | None = None


def _scratch_dir() -> Path:
    global _SCRATCH_DIR
    if _SCRATCH_DIR is None:
        _SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="leo-bash-"))
    return _SCRATCH_DIR


def _bwrap_argv(
    workspace: Path,
    net_on: bool,
    command: str,
    skill_dirs: list[Path] | None = None,
) -> list[str]:
    argv = ["bwrap", "--die-with-parent", "--unshare-all"]
    if net_on:
        argv.append("--share-net")
    scratch = _scratch_dir()
    user_local_bin = Path.home() / ".local" / "bin"
    user_config = Path.home() / ".config"
    gogcli_config = user_config / "gogcli"
    argv += [
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/sbin", "/sbin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve",
        "--ro-bind-try", str(user_local_bin), str(user_local_bin),
        "--ro-bind-try", str(user_config), str(user_config),
        # gogcli rewrites its keyring file on token refresh; overlay as rw.
        "--bind-try", str(gogcli_config), str(gogcli_config),
        "--bind", str(workspace), str(workspace),
        "--chdir", str(workspace),
        "--proc", "/proc",
        "--dev", "/dev",
        "--bind", str(scratch), "/tmp",
        "--setenv", "HOME", "/tmp",
        "--setenv", "XDG_CONFIG_HOME", str(user_config),
        "--setenv", "PATH", f"{user_local_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
    for var in ("GOG_KEYRING_BACKEND", "GOG_KEYRING_PASSWORD", "GOG_ACCOUNT"):
        val = os.environ.get(var)
        if val is not None:
            argv += ["--setenv", var, val]
    # Symlinks in ~/.local/bin (e.g. gog) point outside mounted paths; expose targets.
    seen: set[Path] = set()
    if user_local_bin.is_dir():
        for entry in user_local_bin.iterdir():
            if not entry.is_symlink():
                continue
            target_dir = entry.resolve().parent
            if target_dir in seen:
                continue
            seen.add(target_dir)
            argv += ["--ro-bind-try", str(target_dir), str(target_dir)]
    for sdir in skill_dirs or []:
        argv += ["--ro-bind-try", str(sdir), str(sdir)]
    argv += ["bash", "-c", command]
    return argv


def _truncate(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    omitted = len(text) - OUTPUT_CAP
    return text[:OUTPUT_CAP] + f"\n...(truncated, {omitted} bytes omitted)"


def _format_result(exit_code: int, stdout: str, stderr: str) -> str:
    parts = [f"exit: {exit_code}"]
    if stdout:
        parts.append("---stdout---")
        parts.append(_truncate(stdout))
    if stderr:
        parts.append("---stderr---")
        parts.append(_truncate(stderr))
    return "\n".join(parts)


def bash(ctx, command: str, timeout_seconds: int = DEFAULT_TIMEOUT) -> str:
    skill_dirs = sorted({s.path.parent.resolve() for s in ctx.skills.values()})
    argv = _bwrap_argv(ctx.workspace, ctx.net_on, command, skill_dirs)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout_seconds}s"
    return _format_result(proc.returncode, proc.stdout, proc.stderr)


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command inside a bubblewrap sandbox. "
                "Read-only system dirs; read-write workspace (the current working "
                "directory). Files written to the workspace and /tmp persist across "
                "calls within a session; env vars, cwd changes, and background jobs "
                "do NOT persist (each call is a fresh shell). Common tools available: "
                "curl, grep, sed, awk, jq, python3, pdftotext (poppler-utils). "
                "Network availability is controlled by the user. Returns exit code, "
                "stdout, and stderr (each capped at 8 KB)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run (passed to 'bash -c').",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": f"Max runtime in seconds (default {DEFAULT_TIMEOUT}).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

FUNCTIONS = {"bash": bash}

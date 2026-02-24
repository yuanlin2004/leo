from __future__ import annotations

import os
from pathlib import Path


def _find_dotenv(start_path: str | Path | None = None) -> Path | None:
    start = Path(start_path) if start_path is not None else Path.cwd()
    if start.is_file():
        start = start.parent
    start = start.resolve()

    for base in [start, *start.parents]:
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_with_builtin_parser(dotenv_path: Path, override: bool) -> None:
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue

        if not override and key in os.environ:
            continue

        value = _strip_wrapping_quotes(raw_value.strip())
        os.environ[key] = value


def load_project_env(start_path: str | Path | None = None, override: bool = False) -> Path | None:
    dotenv_path = _find_dotenv(start_path=start_path)
    if dotenv_path is None:
        return None

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        _load_with_builtin_parser(dotenv_path=dotenv_path, override=override)
        return dotenv_path

    load_dotenv(dotenv_path=dotenv_path, override=override)
    return dotenv_path

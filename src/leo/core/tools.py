from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable


def _tool_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _tool_time() -> str:
    return datetime.now().astimezone().strftime("%H:%M:%S %Z")


def _tool_read_file(path: str) -> str:
    return Path(path).expanduser().read_text()


def _tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "date": _tool_date,
    "time": _tool_time,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "date",
            "description": "Return the current local date as YYYY-MM-DD.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Return the current local time as HH:MM:SS with timezone.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, overwriting if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def dispatch(name: str, arguments_json: str) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return f"error: invalid arguments JSON: {e}"
    try:
        return fn(**kwargs)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

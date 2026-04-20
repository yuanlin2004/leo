from __future__ import annotations

from pathlib import Path

READ_MAX_OUTPUT = 64 * 1024
READ_MAX_LINE_LEN = 2000
READ_DEFAULT_LIMIT = 2000


def _resolve(ctx, path: str) -> Path | str:
    p = Path(path)
    if not p.is_absolute():
        p = ctx.workspace / p
    p = p.resolve()
    try:
        p.relative_to(ctx.workspace)
    except ValueError:
        return f"error: path {p} is outside workspace {ctx.workspace}"
    return p


def read(ctx, path: str, offset: int = 0, limit: int = READ_DEFAULT_LIMIT) -> str:
    resolved = _resolve(ctx, path)
    if isinstance(resolved, str):
        return resolved
    p = resolved
    if not p.is_file():
        return f"error: {p} is not a file"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: {p} is not a utf-8 text file"
    lines = text.splitlines()
    total = len(lines)
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = READ_DEFAULT_LIMIT
    end = min(offset + limit, total)
    out = []
    for i in range(offset, end):
        line = lines[i]
        if len(line) > READ_MAX_LINE_LEN:
            line = line[:READ_MAX_LINE_LEN] + f"... [{len(line) - READ_MAX_LINE_LEN} chars truncated]"
        out.append(f"{i + 1:6}\t{line}")
    result = "\n".join(out)
    if len(result) > READ_MAX_OUTPUT:
        result = result[:READ_MAX_OUTPUT] + "\n... [output truncated]"
    if end < total:
        result += f"\n... ({total - end} more lines; pass offset={end} to continue)"
    if total == 0:
        return f"(empty file: {p})"
    return result


def edit(ctx, path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
    resolved = _resolve(ctx, path)
    if isinstance(resolved, str):
        return resolved
    p = resolved
    if not p.is_file():
        return f"error: {p} is not a file"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: {p} is not a utf-8 text file"
    count = text.count(old_str)
    if count == 0:
        return f"error: old_str not found in {p}"
    if count > 1 and not replace_all:
        return (
            f"error: old_str appears {count} times in {p}; add more surrounding "
            f"context to make it unique, or pass replace_all=true"
        )
    new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
    p.write_text(new_text, encoding="utf-8")
    return f"edited: {p} ({count} replacement{'s' if count > 1 else ''})"


def write(ctx, path: str, content: str) -> str:
    resolved = _resolve(ctx, path)
    if isinstance(resolved, str):
        return resolved
    p = resolved
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    verb = "overwrote" if existed else "wrote"
    return f"{verb}: {p} ({len(content)} bytes)"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a utf-8 text file, returning its contents with 1-based line "
                "numbers (line<tab>text). Use 'offset' and 'limit' to page through "
                "large files. Path may be absolute or relative to the workspace; "
                "reads outside the workspace are rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute, or relative to workspace).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based line index to start reading from. Default 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max lines to return. Default {READ_DEFAULT_LIMIT}.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Replace an exact string in an existing text file. 'old_str' must "
                "match byte-for-byte (including whitespace and indentation) and "
                "must appear exactly once in the file unless 'replace_all' is set. "
                "Path may be absolute or relative to the workspace; edits outside "
                "the workspace are rejected. Use 'read' first to see exact content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute, or relative to workspace).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact substring to replace. Must be unique unless replace_all is true.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string. May be empty to delete old_str.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace every occurrence. Default false.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write a text file, creating parent directories as needed. "
                "Overwrites existing content entirely — use 'edit' for surgical "
                "changes. Path must be inside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (absolute, or relative to workspace).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]

FUNCTIONS = {"read": read, "edit": edit, "write": write}

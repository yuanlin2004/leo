from __future__ import annotations

import re

# Pattern set adapted from hermes-agent/tools/memory_tool.py.
# Lesson bodies are injected into the system prompt; reject inputs that
# look like prompt injection or credential exfiltration before they land
# on disk or in context.

_THREAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), label)
    for p, label in [
        # Prompt injection
        (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
        (r"you\s+are\s+now\s+", "role_hijack"),
        (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
        (r"system\s+prompt\s+override", "sys_prompt_override"),
        (
            r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)",
            "disregard_rules",
        ),
        # Exfiltration
        (
            r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
            "exfil_curl",
        ),
        (
            r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
            "exfil_wget",
        ),
        (
            r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)",
            "read_secrets",
        ),
        # Persistence
        (r"authorized_keys", "ssh_backdoor"),
        (r"\$HOME/\.ssh|~/\.ssh", "ssh_access"),
    ]
)

_INVISIBLE_CHARS = frozenset(
    [
        "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    ]
)


def scan(content: str) -> str | None:
    """Return a reason string if content is unsafe, None if it passes."""
    for ch in _INVISIBLE_CHARS:
        if ch in content:
            return f"contains invisible unicode U+{ord(ch):04X}"
    for pattern, label in _THREAT_PATTERNS:
        if pattern.search(content):
            return f"matches threat pattern {label!r}"
    return None

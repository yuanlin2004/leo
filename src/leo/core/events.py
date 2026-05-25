"""Structured events emitted during an agent run.

Events are the observability stream. They are persisted to
`<session>/events.jsonl` (one JSON object per line) and, in the
forthcoming web layer, also broadcast to live SSE subscribers. The
LLM message format on disk (`messages.jsonl`) is unaffected.

Each event carries a monotonic `seq` per session, a wall-clock `ts`,
a `type` tag, and a small `payload`. Heavy data (full assistant
content, full tool output) stays in `messages.jsonl`; the event
payload only summarizes — enough to render the observability view
without bloating the stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Event:
    seq: int
    ts: str
    type: str
    payload: dict

    def to_dict(self) -> dict:
        return asdict(self)

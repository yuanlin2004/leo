from __future__ import annotations

from pathlib import Path
from typing import Any

from .trace import RunTraceRecorder


def replay_trace(trace_path: str | Path) -> dict[str, Any]:
    recorder = RunTraceRecorder(trace_path)
    events = recorder.read_events()
    event_types: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type") or "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
    return {
        "trace_path": str(Path(trace_path).resolve()),
        "event_count": len(events),
        "event_types": event_types,
        "events": events,
    }

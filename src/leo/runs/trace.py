from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    timestamp: str
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class RunTraceRecorder:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = TraceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            payload=_normalize_trace_payload(payload),
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        return events


def _normalize_trace_payload(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_trace_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_trace_payload(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalize_trace_payload(value.model_dump())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_trace_payload(value.to_dict())
    return repr(value)

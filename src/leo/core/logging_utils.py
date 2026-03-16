from __future__ import annotations

import logging
from typing import Any

TRACE_LEVEL = 5
CONCISE_LEVEL = 15


class LeoOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "leo" or record.name.startswith("leo.")


def ensure_trace_logging() -> None:
    if not hasattr(logging, "TRACE"):
        setattr(logging, "TRACE", TRACE_LEVEL)
        logging.addLevelName(TRACE_LEVEL, "TRACE")
    if not hasattr(logging, "CONCISE"):
        setattr(logging, "CONCISE", CONCISE_LEVEL)
        logging.addLevelName(CONCISE_LEVEL, "CONCISE")

    if not hasattr(logging.Logger, "trace"):
        def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace  # type: ignore[attr-defined]


def resolve_log_level(level_name: str | None) -> int:
    normalized = (level_name or "INFO").strip().upper()
    level_map = {
        "TRACE": TRACE_LEVEL,
        "DEBUG": logging.DEBUG,
        "CONCISE": CONCISE_LEVEL,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(normalized, logging.INFO)


def configure_leo_logging(level_name: str = "INFO", leo_only: bool = True) -> int:
    ensure_trace_logging()
    resolved_level = resolve_log_level(level_name)
    logging.basicConfig(level=resolved_level, format="%(message)s", force=True)

    root = logging.getLogger()
    if leo_only:
        for handler in root.handlers:
            if not any(isinstance(existing, LeoOnlyFilter) for existing in handler.filters):
                handler.addFilter(LeoOnlyFilter())

    logging.getLogger("leo").setLevel(resolved_level)
    return resolved_level

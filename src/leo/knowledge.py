"""Knowledge base with TF-IDF retrieval for per-turn context injection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


@dataclass
class KnowledgeBase:
    """Parsed action/code dictionary with TF-IDF retrieval.

    Supports files with JSON wrapped in ``<Examples>...</Examples>`` tags,
    or plain JSON files. The JSON must be a dict mapping action descriptions
    to lists of code strings.
    """

    entries: list[tuple[str, str]]  # (action_text, code_text)
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _entry_tokens: list[set[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.entries)
        if n == 0:
            return
        self._entry_tokens = [set(_tokenize(action)) for action, _ in self.entries]
        df: dict[str, int] = {}
        for tokens in self._entry_tokens:
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(n / cnt) for t, cnt in df.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "KnowledgeBase":
        text = Path(path).read_text(encoding="utf-8")
        m = re.search(r"<Examples>\s*(.*?)\s*</Examples>", text, re.DOTALL)
        json_text = m.group(1) if m else text.strip()
        # Strip trailing commas (relaxed JSON used by some hand-authored files).
        json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)
        data: dict[str, Any] = json.loads(json_text)
        entries = [
            (action, "\n".join(codes) if isinstance(codes, list) else str(codes))
            for action, codes in data.items()
        ]
        return cls(entries=entries)

    def retrieve(self, query: str, top_k: int = 15) -> list[tuple[str, str]]:
        """Return the top-K (action, code) pairs most relevant to *query*."""
        if not self.entries:
            return []
        query_tokens = set(_tokenize(query))
        scores = [
            (sum(self._idf.get(t, 0.0) for t in query_tokens & tokens), i)
            for i, tokens in enumerate(self._entry_tokens)
        ]
        scores.sort(reverse=True)
        return [self.entries[i] for score, i in scores[:top_k] if score > 0]

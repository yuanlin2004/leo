"""Session persistence — one task's record inside a workspace.

A session lives at `<workspace>/.leo/sessions/<id>/`:
  meta.json       — id, title, started_at, last_active, model, toggles,
                    reflection_idx, injected_ids, loaded_skills
  messages.jsonl  — one JSON message object per line, append-only
  events.jsonl    — one JSON event envelope per line, append-only
                    (observability stream — see core/events.py)
  artifacts/      — reserved for per-session scratch (unused in v1)
"""

from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from leo.core.events import Event

SESSIONS_SUBDIR = "sessions"
_ID_CHARS = "abcdefghijkmnopqrstuvwxyz23456789"  # no 0/1/l confusables


def _sessions_root(workspace: Path) -> Path:
    return workspace / ".leo" / SESSIONS_SUBDIR


def _new_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    suffix = "".join(secrets.choice(_ID_CHARS) for _ in range(4))
    return f"{stamp}-{suffix}"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _title_from_prompt(prompt: str, limit: int = 60) -> str:
    line = prompt.strip().splitlines()[0] if prompt.strip() else "(empty)"
    return line[:limit]


@dataclass
class Session:
    id: str
    workspace: Path
    dir: Path
    title: str
    model: str | None
    started_at: str
    last_active: str
    reflection_idx: int = 1
    toggles: dict = field(default_factory=dict)
    injected_ids: list[str] = field(default_factory=list)
    loaded_skills: list[str] = field(default_factory=list)
    # Agent type this session was created with. Legacy sessions (created
    # before agents existed) have agent_id="leo" virtualized at load.
    agent_id: str = "leo"
    # In-memory only — not persisted to meta.json. Source of truth for the
    # value is events.jsonl itself; recomputed on session load.
    next_event_seq: int = 1

    # ---- on-disk paths --------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def messages_path(self) -> Path:
        return self.dir / "messages.jsonl"

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"

    # ---- serialization --------------------------------------------------
    def to_meta_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "started_at": self.started_at,
            "last_active": self.last_active,
            "reflection_idx": self.reflection_idx,
            "toggles": self.toggles,
            "injected_ids": self.injected_ids,
            "loaded_skills": self.loaded_skills,
            "agent_id": self.agent_id,
        }

    def write_meta(self) -> None:
        self.last_active = _now_iso()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_meta_dict(), indent=2))
        tmp.replace(self.meta_path)


def new_session(
    workspace: Path,
    *,
    model: str | None,
    toggles: dict,
    title: str = "(untitled)",
    agent_id: str = "leo",
) -> Session:
    sid = _new_id()
    sdir = _sessions_root(workspace) / sid
    sdir.mkdir(parents=True, exist_ok=False)
    now = _now_iso()
    session = Session(
        id=sid,
        workspace=workspace,
        dir=sdir,
        title=title,
        model=model,
        started_at=now,
        last_active=now,
        toggles=dict(toggles),
        agent_id=agent_id,
    )
    # Initialize files so an empty session is still well-formed.
    session.messages_path.touch()
    session.events_path.touch()
    session.write_meta()
    return session


def load_session(workspace: Path, sid: str) -> tuple[Session, list[dict]]:
    sdir = _sessions_root(workspace) / sid
    if not sdir.is_dir():
        raise FileNotFoundError(f"no session {sid!r} in {workspace}")
    meta = json.loads((sdir / "meta.json").read_text())
    session = Session(
        id=meta["id"],
        workspace=workspace,
        dir=sdir,
        title=meta.get("title", "(untitled)"),
        model=meta.get("model"),
        started_at=meta.get("started_at", _now_iso()),
        last_active=meta.get("last_active", _now_iso()),
        reflection_idx=meta.get("reflection_idx", 1),
        toggles=meta.get("toggles", {}),
        injected_ids=list(meta.get("injected_ids", [])),
        loaded_skills=list(meta.get("loaded_skills", [])),
        # Legacy sessions (pre-agent feature) virtualize to "leo".
        agent_id=meta.get("agent_id", "leo"),
    )
    messages: list[dict] = []
    mp = session.messages_path
    if mp.is_file():
        for line in mp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            messages.append(json.loads(line))
    session.next_event_seq = _scan_next_event_seq(session.events_path)
    return session, messages


def append_messages(session: Session, new_msgs: list[dict]) -> None:
    if not new_msgs:
        return
    with session.messages_path.open("a") as f:
        for m in new_msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    session.write_meta()


def _scan_next_event_seq(path: Path) -> int:
    """Return the seq to assign to the next event.

    Reads the file line-by-line; the last well-formed line's seq + 1
    becomes the next seq. An empty or missing file resets to 1. A
    line that fails to parse is skipped (best-effort — we never want
    persistence layer corruption to refuse a resume).
    """
    if not path.is_file():
        return 1
    last_seq = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            seq = int(obj.get("seq", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if seq > last_seq:
            last_seq = seq
    return last_seq + 1


def append_event(session: Session, type: str, payload: dict) -> Event:
    """Persist one event to <session>/events.jsonl and return it.

    Assigns `seq` from the session's in-memory counter and `ts` from
    the wall clock. Does not write meta — events are high-frequency
    and meta.write on every event would be wasteful. `last_active`
    is refreshed via `append_messages` on turn boundaries.
    """
    ev = Event(seq=session.next_event_seq, ts=_now_iso(), type=type, payload=payload)
    session.next_event_seq += 1
    with session.events_path.open("a") as f:
        f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
    return ev


def read_events(session: Session, *, since: int = 0) -> list[Event]:
    """Read events with seq > since from disk.

    Used by the (forthcoming) web SSE endpoint to backfill before
    switching to live tailing.
    """
    out: list[Event] = []
    if not session.events_path.is_file():
        return out
    for line in session.events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            seq = int(obj.get("seq", 0))
            if seq <= since:
                continue
            out.append(Event(
                seq=seq,
                ts=obj.get("ts", ""),
                type=obj.get("type", ""),
                payload=obj.get("payload") or {},
            ))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return out


def list_sessions(workspace: Path) -> list[Session]:
    root = _sessions_root(workspace)
    if not root.is_dir():
        return []
    out: list[Session] = []
    for sdir in sorted(root.iterdir()):
        meta_path = sdir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        out.append(Session(
            id=meta.get("id", sdir.name),
            workspace=workspace,
            dir=sdir,
            title=meta.get("title", "(untitled)"),
            model=meta.get("model"),
            started_at=meta.get("started_at", ""),
            last_active=meta.get("last_active", ""),
            reflection_idx=meta.get("reflection_idx", 1),
            toggles=meta.get("toggles", {}),
            injected_ids=list(meta.get("injected_ids", [])),
            loaded_skills=list(meta.get("loaded_skills", [])),
            agent_id=meta.get("agent_id", "leo"),
        ))
    # Most-recent first.
    out.sort(key=lambda s: s.last_active or s.started_at, reverse=True)
    return out


def find_last(workspace: Path) -> Session | None:
    sessions = list_sessions(workspace)
    return sessions[0] if sessions else None


def count_messages(session: Session) -> int:
    if not session.messages_path.is_file():
        return 0
    return sum(1 for line in session.messages_path.read_text().splitlines() if line.strip())


def delete_session(workspace: Path, sid: str) -> None:
    sdir = _sessions_root(workspace) / sid
    if not sdir.is_dir():
        raise FileNotFoundError(f"no session {sid!r} in {workspace}")
    shutil.rmtree(sdir)


def derive_title_from_messages(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return _title_from_prompt(m["content"])
    return "(no user turn)"

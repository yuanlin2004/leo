"""Run supervisor — owns the lifecycle of one agent run.

Runs are executed in background threads (Phase 3a; subprocess isolation
is reserved for Phase 6 hardening). Each session has a per-session
threading.Lock that ensures at most one in-flight run per session.

Events emitted by `run_turn` are persisted to `events.jsonl` via
`append_event` and broadcast to all live SSE subscribers via per-session
in-memory asyncio queues. The broadcast hop crosses thread boundaries
using `run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import collections
import fcntl
import os
import threading
import traceback
from typing import AsyncIterator, IO

from leo.core.agent import CancelToken, run_turn
from leo.core.agents import AgentError, filter_skills, get_agent
from leo.core.events import Event
from leo.core.session import (
    Session,
    append_event,
    append_messages,
    derive_title_from_messages,
    load_session,
    read_events,
)
from leo.core.setup import RunContext


class RunBusy(Exception):
    """Raised when a session already has an in-flight run."""


def _acquire_session_flock(session: Session) -> IO[str] | None:
    """Try to grab an exclusive non-blocking lock on <session>/.run.lock.

    Returns the open file object (lock holder) on success, None on
    contention. The lock is process-scoped — closing the file releases
    it. The file itself sticks around; it's a tiny marker.

    Cross-process exclusion catches: CLI running against a workspace
    while the web server is also driving it, or two leo-server processes
    on the same host.
    """
    f = (session.dir / ".run.lock").open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    # Write the pid so an operator inspecting a stuck lock has a clue.
    try:
        f.write(str(os.getpid()))
        f.flush()
    except OSError:
        pass
    return f


def _release_session_flock(f: IO[str]) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        f.close()
    except OSError:
        pass


class RunSupervisor:
    def __init__(self, ctx: RunContext, loop: asyncio.AbstractEventLoop):
        self.ctx = ctx
        self._loop = loop
        # sid -> list of subscribed asyncio.Queues
        self._subscribers: dict[str, list[asyncio.Queue]] = collections.defaultdict(list)
        self._sub_lock = threading.Lock()
        # sid -> per-session run lock
        self._run_locks: dict[str, threading.Lock] = {}
        self._lock_dict_lock = threading.Lock()
        # sid -> CancelToken for the in-flight run (if any)
        self._cancel_tokens: dict[str, CancelToken] = {}

    # -- public API ------------------------------------------------------

    def submit_run(self, sid: str, user_message: str) -> None:
        """Launch a run in a background thread. Returns immediately.

        Raises `RunBusy` if a run for this session is already in flight
        (whether started by this process or another).
        Raises FileNotFoundError if the session does not exist.

        Two-layer exclusion:
        - In-process: `threading.Lock` so two HTTP requests in the same
          server don't both pass the contention check.
        - Cross-process: `fcntl.flock` on `<sid>/.run.lock` so the CLI
          or a second server instance can't drive the same session
          concurrently.
        """
        session, messages = load_session(self.ctx.workspace, sid)
        lock = self._get_or_make_lock(sid)
        if not lock.acquire(blocking=False):
            raise RunBusy(sid)
        flock_file = _acquire_session_flock(session)
        if flock_file is None:
            lock.release()
            raise RunBusy(sid)
        token = CancelToken()
        with self._lock_dict_lock:
            self._cancel_tokens[sid] = token
        t = threading.Thread(
            target=self._run,
            args=(session, messages, user_message, lock, flock_file, token),
            name=f"leo-run-{sid}",
            daemon=True,
        )
        t.start()

    def cancel_run(self, sid: str) -> bool:
        """Signal cancellation for an in-flight run. Returns True if a
        token was found (run was actually in flight), False otherwise."""
        with self._lock_dict_lock:
            token = self._cancel_tokens.get(sid)
        if token is None:
            return False
        token.cancel()
        return True

    def is_running(self, sid: str) -> bool:
        lock = self._run_locks.get(sid)
        return lock is not None and lock.locked()

    async def subscribe(self, sid: str, since: int) -> AsyncIterator[Event]:
        """Yield events for `sid` with seq > since, backfilling from disk
        then switching to live tail. The async iterator runs until the
        consumer disconnects (cancels the task)."""
        q: asyncio.Queue[Event] = asyncio.Queue()
        with self._sub_lock:
            self._subscribers[sid].append(q)
        try:
            # Backfill first. Any event broadcast between subscribe and
            # backfill completion appears in BOTH the queue and the file;
            # the seq-dedup below handles the overlap.
            session, _ = load_session(self.ctx.workspace, sid)
            last = since
            for ev in read_events(session, since=since):
                yield ev
                last = ev.seq
            while True:
                ev = await q.get()
                # seq=0 is the ephemeral-event sentinel (token streams).
                # Forward without dedup, without updating the watermark.
                if ev.seq == 0:
                    yield ev
                    continue
                if ev.seq <= last:
                    continue
                yield ev
                last = ev.seq
        finally:
            with self._sub_lock:
                try:
                    self._subscribers[sid].remove(q)
                except ValueError:
                    pass

    # -- internals -------------------------------------------------------

    def _get_or_make_lock(self, sid: str) -> threading.Lock:
        with self._lock_dict_lock:
            if sid not in self._run_locks:
                self._run_locks[sid] = threading.Lock()
            return self._run_locks[sid]

    def _broadcast(self, sid: str, ev: Event) -> None:
        with self._sub_lock:
            queues = list(self._subscribers.get(sid, []))
        for q in queues:
            # Cross-thread queue put. The future is fire-and-forget; if a
            # subscriber's queue is at capacity (it's unbounded today), or
            # the loop has shut down, we silently drop — the disk copy in
            # events.jsonl remains the source of truth.
            try:
                asyncio.run_coroutine_threadsafe(q.put(ev), self._loop)
            except RuntimeError:
                pass

    def _emit(self, session: Session, type_: str, payload: dict) -> None:
        ev = append_event(session, type_, payload)
        self._broadcast(session.id, ev)

    def _emit_transient(self, session: Session, type_: str, payload: dict) -> None:
        """Broadcast an ephemeral event — not persisted to events.jsonl.

        Uses seq=0 as the sentinel for "no real seq"; the subscribe loop
        recognizes this and forwards the event without consulting the
        dedup high-water-mark. Used for high-frequency token streams
        that aren't meaningful to replay on reconnect.
        """
        from datetime import datetime
        ev = Event(
            seq=0,
            ts=datetime.now().isoformat(timespec="milliseconds"),
            type=type_,
            payload=payload,
        )
        self._broadcast(session.id, ev)

    def _run(
        self,
        session: Session,
        messages: list[dict],
        user_message: str,
        lock: threading.Lock,
        flock_file: IO[str],
        token: CancelToken,
    ) -> None:
        try:
            ctx = self.ctx
            # Bootstrap an empty session with the system message.
            # New sessions are pre-bootstrapped at create time (with the
            # agent-composed system prompt); this branch only fires for
            # the (legacy) case where a session has no system message.
            if not messages:
                messages = [{"role": "system", "content": ctx.system_prompt}]
                append_messages(session, messages)

            # Resolve the agent for this session and apply its skill
            # filter. Falls back to the unfiltered context if the agent
            # has been deleted since session creation.
            session_skills = ctx.skills
            try:
                agent = get_agent(session.agent_id)
                session_skills, _missing = filter_skills(ctx.skills, agent.skills)
            except AgentError:
                pass

            persist_idx = len(messages)
            messages.append({"role": "user", "content": user_message})

            injected_ids: set[str] = set(session.injected_ids) | set(ctx.phase1_ids)
            loaded_skills: set[str] = set(session.loaded_skills)

            # on_prompt lesson injection before the LLM sees the new turn.
            op_text, op_ids = ctx.lessons.apply_on_prompt(
                ctx.lesson_scope, user_message, injected_ids,
            )
            if op_ids:
                messages.append({"role": "user", "content": op_text})
                injected_ids.update(op_ids)
                self._emit(session, "lesson_injected", {
                    "phase": "on_prompt",
                    "ids": list(op_ids),
                    "text": op_text,
                })

            def on_event(type_: str, payload: dict) -> None:
                self._emit(session, type_, payload)

            def on_reply(text: str) -> None:
                # Ephemeral token broadcast — not persisted. Subscribers
                # reconstruct the streaming reply; the final message
                # arrives via the `llm_message` event + messages.jsonl.
                if text:
                    self._emit_transient(session, "llm_token", {
                        "kind": "reply", "text": text,
                    })

            def on_think(text: str) -> None:
                if text:
                    self._emit_transient(session, "llm_token", {
                        "kind": "think", "text": text,
                    })

            run_turn(
                messages,
                llm=ctx.llm,
                skills=session_skills,
                workspace=session.workspace,
                think_on=session.toggles.get("think_on", True),
                net_on=session.toggles.get("net_on", True),
                on_reply=on_reply,
                on_think=on_think,
                on_tool=lambda n, a, r: None,
                lessons=ctx.lessons,
                lesson_scope=ctx.lesson_scope,
                injected_ids=injected_ids,
                on_event=on_event,
                cancel_token=token,
                loaded_skills=loaded_skills,
            )

            # Persist new messages produced this turn.
            if len(messages) > persist_idx:
                append_messages(session, messages[persist_idx:])
            session.injected_ids = sorted(injected_ids)
            session.loaded_skills = sorted(loaded_skills)
            if session.title == "(untitled)":
                session.title = derive_title_from_messages(messages)
            session.write_meta()
        except Exception as e:
            traceback.print_exc()
            try:
                self._emit(session, "run_error", {
                    "error_type": type(e).__name__,
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            with self._lock_dict_lock:
                self._cancel_tokens.pop(session.id, None)
            _release_session_flock(flock_file)
            lock.release()

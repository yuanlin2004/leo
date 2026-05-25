"""Agent loop — one assistant turn (LLM + tool-call cycles).

Extracted from `leo.cli.leo` so that both the CLI REPL and the (forthcoming)
web server can drive the same agent without forking the loop.

The interface here is intentionally unchanged from its prior shape inside
`cli/leo.py`: a synchronous `run_turn(...)` driven by sync callbacks
(`on_reply`, `on_think`, `on_tool`, `on_replan`, `on_lesson_inject`).

The web front-end will adapt these callbacks to a streaming event queue at
its own layer; the agent core stays simple and unaware of transport.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from leo.core.llm import LLM
from leo.core.tools import TOOLS_SCHEMA, ToolContext, dispatch


REPLAN_CAP = 2  # max replans per tool-call boundary, per design doc


class CancelToken:
    """Cooperative cancellation signal for `run_turn`.

    The agent loop checks `is_set()` between major phases (before each
    LLM call and before each tool dispatch). It does not abort a
    streaming LLM call mid-token; the in-flight stream completes, and
    the loop exits before the next LLM/tool step.
    """
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()


class _Cancelled(Exception):
    """Internal sentinel: surface a cancel up to the run loop's exit branch."""


def _split_think(text: str | None) -> tuple[str, str]:
    """Return (think, reply). Handles paired <think>...</think> and orphan </think>."""
    if not text:
        return "", text or ""
    idx = text.rfind("</think>")
    if idx == -1:
        return "", text.strip()
    think = re.sub(r"^\s*<think>\s*", "", text[:idx], count=1).strip()
    reply = text[idx + len("</think>"):].strip()
    return think, reply


class _ThinkStripper:
    """Streams text with <think>...</think> suppressed (or rerouted) across chunk boundaries.

    `on_think_end` (optional): called once per `</think>` close. If it returns
    truthy, subsequent reply bytes are dropped instead of being forwarded to
    `on_reply` — used to abort live emission when a lesson fires on the
    thinking text and we plan to replan.
    """

    def __init__(self, on_reply, on_think=None, on_think_end=None, start_in_think=False):
        self.on_reply = on_reply
        self.on_think = on_think
        self.on_think_end = on_think_end
        self.in_think = start_in_think
        self.buf = ""
        self.suppress_reply = False

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-n:]):
                return n
        return 0

    def _emit_reply(self, s: str) -> None:
        if s and not self.suppress_reply:
            self.on_reply(s)

    def feed(self, chunk: str) -> None:
        text = self.buf + chunk
        self.buf = ""
        while text:
            if self.in_think:
                i = text.find("</think>")
                if i == -1:
                    keep = self._partial_tail(text, "</think>")
                    emit, self.buf = (text[:-keep], text[-keep:]) if keep else (text, "")
                    if emit and self.on_think:
                        self.on_think(emit)
                    return
                if i > 0 and self.on_think:
                    self.on_think(text[:i])
                text = text[i + len("</think>"):]
                self.in_think = False
                if self.on_think_end is not None and not self.suppress_reply:
                    if self.on_think_end():
                        self.suppress_reply = True
            else:
                i = text.find("<think>")
                if i == -1:
                    keep = self._partial_tail(text, "<think>")
                    emit, self.buf = (text[:-keep], text[-keep:]) if keep else (text, "")
                    self._emit_reply(emit)
                    return
                if i > 0:
                    self._emit_reply(text[:i])
                text = text[i + len("<think>"):]
                self.in_think = True

    def flush(self) -> None:
        if self.buf:
            if self.in_think and self.on_think:
                self.on_think(self.buf)
            elif not self.in_think:
                self._emit_reply(self.buf)
            self.buf = ""


def _inject_lesson_message(
    messages: list[dict],
    text: str,
    matched_ids: list[str],
    injected_ids: set[str],
) -> None:
    """Append a mid-loop lesson note and update the dedup set.

    The role is `user`, not `system` — many chat templates (Qwen3 / vLLM
    among them) reject system messages mid-conversation. The rendered text
    starts with `[System note: ...]` so the LLM still recognizes it as
    out-of-band guidance, not user discourse.
    """
    if not text:
        return
    messages.append({"role": "user", "content": text})
    injected_ids.update(matched_ids)


def _tool_call_views(tool_calls) -> list:
    from leo.core.lessons import ToolCallView
    return [
        ToolCallView(name=tc.function.name, arguments=tc.function.arguments or "")
        for tc in tool_calls
    ]


_TOOL_RESULT_PREVIEW_LIMIT = 500


def _make_emit(on_event):
    """Return an `emit(type, payload)` helper that no-ops when `on_event` is None."""
    if on_event is None:
        return lambda _t, _p: None
    def _emit(type: str, payload: dict) -> None:
        on_event(type, payload)
    return _emit


def _llm_message_payload(msg) -> dict:
    return {
        "role": "assistant",
        "has_tool_calls": bool(msg.tool_calls),
        "tool_call_count": len(msg.tool_calls) if msg.tool_calls else 0,
        "content_len": len(msg.content or ""),
    }


def run_turn(
    messages: list[dict],
    *,
    llm: LLM,
    skills,
    workspace: Path,
    think_on: bool,
    net_on: bool,
    on_reply,
    on_think,
    on_tool,
    lessons=None,
    lesson_scope=None,
    injected_ids: set[str] | None = None,
    on_replan=None,
    on_lesson_inject=None,
    on_event=None,
    cancel_token: CancelToken | None = None,
    loaded_skills: set[str] | None = None,
) -> str:
    """Drive LLM + tool-call loop until no more tool calls. Returns final reply text.

    When `lessons` and `lesson_scope` are provided, applies mid-loop hooks:
    - `on_tool_call` matches drive a replan (LLM re-prompted with the matched
      lesson, original draft popped from history).
    - `on_monologue` matches against the thinking text at the `</think>`
      boundary; on a hit, live reply emission is suppressed and the LLM is
      replanned with the lesson injected. Also runs against each tool result.

    When `on_event` is provided, structured observability events are emitted
    alongside the existing sync callbacks. See `core/events.py` for the
    envelope and the design doc for the event vocabulary.
    """
    if injected_ids is None:
        injected_ids = set()

    emit = _make_emit(on_event)
    emit("run_start", {})
    tool_calls_total = 0

    def _check_cancel() -> None:
        if cancel_token is not None and cancel_token.is_set():
            raise _Cancelled

    reply_text = ""
    try:
        while True:
            # Inner replan loop: keep re-calling the LLM until no replan trigger
            # fires (on_monologue at think-boundary or on_tool_call), the cap is
            # hit, or the response has no tool calls.
            replan_count = 0
            while True:
                _check_cancel()
                think_buf: list[str] = []
                mono_match: dict = {"text": "", "ids": []}

                def _accum_think(s: str) -> None:
                    think_buf.append(s)
                    if on_think is not None:
                        on_think(s)

                def _check_think_end() -> bool:
                    if lessons is None or lesson_scope is None:
                        return False
                    text, ids = lessons.apply_on_monologue(
                        lesson_scope, "".join(think_buf), injected_ids,
                    )
                    if not ids:
                        return False
                    mono_match["text"] = text
                    mono_match["ids"] = ids
                    # Suppress live reply only if we can actually replan; if the
                    # budget is exhausted, let the reply stream and just register
                    # the lesson into history below.
                    return replan_count < REPLAN_CAP

                stripper = _ThinkStripper(
                    on_reply=on_reply,
                    on_think=_accum_think,
                    on_think_end=_check_think_end,
                    start_in_think=think_on,
                )
                msg = llm.chat(
                    messages,
                    enable_thinking=think_on,
                    tools=TOOLS_SCHEMA,
                    on_text=stripper.feed,
                    on_reasoning=on_think,
                    should_stop=(cancel_token.is_set if cancel_token else None),
                )
                stripper.flush()
                # If a cancel landed mid-stream, drop the partial assembly
                # — don't commit a half-formed assistant message to history.
                _check_cancel()
                _, reply_text = _split_think(msg.content)
                entry: dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(entry)
                emit("llm_message", _llm_message_payload(msg))

                # on_monologue replan: a lesson fired against the thinking text;
                # the live reply was suppressed by the stripper. Pop the draft,
                # inject the lesson, and re-call the LLM.
                if mono_match["ids"] and replan_count < REPLAN_CAP:
                    messages.pop()
                    _inject_lesson_message(
                        messages, mono_match["text"], mono_match["ids"], injected_ids,
                    )
                    emit("lesson_injected", {
                        "phase": "on_monologue",
                        "ids": list(mono_match["ids"]),
                        "text": mono_match["text"],
                    })
                    emit("replan", {"reason": "on_monologue", "ids": list(mono_match["ids"])})
                    if on_replan is not None:
                        on_replan(mono_match["ids"])
                    replan_count += 1
                    continue
                # Cap hit: reply was allowed to stream; still register the lesson
                # so it influences the next turn.
                if mono_match["ids"]:
                    _inject_lesson_message(
                        messages, mono_match["text"], mono_match["ids"], injected_ids,
                    )
                    emit("lesson_injected", {
                        "phase": "on_monologue",
                        "ids": list(mono_match["ids"]),
                        "text": mono_match["text"],
                    })
                    if on_lesson_inject is not None:
                        on_lesson_inject("on_monologue", mono_match["ids"])

                # on_tool_call replan: only relevant if we have tool calls and a
                # lesson store.
                if (
                    msg.tool_calls
                    and lessons is not None
                    and lesson_scope is not None
                    and replan_count < REPLAN_CAP
                ):
                    text, ids = lessons.apply_on_tool_call(
                        lesson_scope,
                        _tool_call_views(msg.tool_calls),
                        injected_ids,
                    )
                    if ids:
                        messages.pop()
                        _inject_lesson_message(messages, text, ids, injected_ids)
                        emit("lesson_injected", {
                            "phase": "on_tool_call",
                            "ids": list(ids),
                            "text": text,
                        })
                        emit("replan", {"reason": "on_tool_call", "ids": list(ids)})
                        if on_replan is not None:
                            on_replan(ids)
                        replan_count += 1
                        continue
                break

            if not msg.tool_calls:
                emit("run_end", {"tool_calls": tool_calls_total})
                return reply_text

            # Dispatch tool calls; on_monologue runs against each result.
            ctx_obj = ToolContext(
                workspace=workspace, net_on=net_on,
                skills={s.name: s for s in skills},
                loaded_skills=loaded_skills if loaded_skills is not None else set(),
            )
            for tc in msg.tool_calls:
                _check_cancel()
                emit("tool_start", {
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                })
                result = dispatch(tc.function.name, tc.function.arguments, ctx_obj)
                tool_calls_total += 1
                preview = (
                    result if len(result) <= _TOOL_RESULT_PREVIEW_LIMIT
                    else result[:_TOOL_RESULT_PREVIEW_LIMIT]
                )
                emit("tool_end", {
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "result_preview": preview,
                    "result_len": len(result),
                })
                on_tool(tc.function.name, tc.function.arguments, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                if lessons is not None and lesson_scope is not None:
                    blob = (
                        f"{tc.function.name} {tc.function.arguments or ''} {result}"
                    )
                    text, ids = lessons.apply_on_monologue(
                        lesson_scope, blob, injected_ids,
                    )
                    _inject_lesson_message(messages, text, ids, injected_ids)
                    if ids:
                        emit("lesson_injected", {
                            "phase": "on_monologue",
                            "ids": list(ids),
                            "text": text,
                        })
                        if on_lesson_inject is not None:
                            on_lesson_inject("on_monologue", ids)
    except _Cancelled:
        emit("run_cancelled", {"tool_calls": tool_calls_total})
        return reply_text

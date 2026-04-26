from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

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
    """Streams text with <think>...</think> suppressed (or rerouted) across chunk boundaries."""

    def __init__(self, on_reply, on_think=None, start_in_think=False):
        self.on_reply = on_reply
        self.on_think = on_think
        self.in_think = start_in_think
        self.buf = ""

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-n:]):
                return n
        return 0

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
            else:
                i = text.find("<think>")
                if i == -1:
                    keep = self._partial_tail(text, "<think>")
                    emit, self.buf = (text[:-keep], text[-keep:]) if keep else (text, "")
                    if emit:
                        self.on_reply(emit)
                    return
                if i > 0:
                    self.on_reply(text[:i])
                text = text[i + len("<think>"):]
                self.in_think = True

    def flush(self) -> None:
        if self.buf:
            (self.on_think if self.in_think and self.on_think else self.on_reply)(self.buf)
            self.buf = ""

from dotenv import load_dotenv

from leo.cli.banner import render_leo_banner
from leo.core.lessons import LessonStore, SessionContext, WriteError
from leo.core.lessons.reflector import (
    CreateOp,
    ReflectorError,
    SkipOp,
    UpdateOp,
    reflect,
)
from leo.core.llm import LLM
from leo.core.skill_core import discover_skills
from leo.core.tools import TOOLS_SCHEMA, ToolContext, dispatch

try:
    from langsmith import trace as _ls_trace
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def _ls_trace(*_args, **_kwargs):
        class _Noop:
            def end(self, **_k):
                pass
        yield _Noop()

DEFAULT_SYSTEM_PROMPT = "You are Leo, a helpful assistant."
SKILLS_ROOT = Path.home() / ".leo" / "skills"
LESSONS_ROOT = Path.home() / ".leo" / "lessons"

COMMANDS_HELP = (
    "commands:\n"
    "  /help               show this help\n"
    "  /exit, /quit        exit (runs reflection if there's anything to learn)\n"
    "  /exit noref         exit without running reflection\n"
    "  /quit noref         alias for /exit noref\n"
    "  /reset              clear conversation history\n"
    "  /think-on           enable model thinking\n"
    "  /think-off          disable model thinking\n"
    "  /net-on             allow network inside bash sandbox\n"
    "  /net-off            block network inside bash sandbox\n"
    "  /show-toolcall-on   print tool calls and results as they happen\n"
    "  /show-toolcall-off  hide tool-call output (default)\n"
    "  /show-think-on      print model thinking content\n"
    "  /show-think-off     hide model thinking content (default)\n"
    "  /show-lessons-on    print which lessons are injected mid-turn\n"
    "  /show-lessons-off   hide lesson-injection notices (default)\n"
    "  /show-all-on        show think, toolcall, and lessons\n"
    "  /show-all-off       hide think, toolcall, and lessons\n"
    "  /status             show current settings, turn count and token usage\n"
    "  /tools              list installed tools\n"
    "  /skills             list installed skills\n"
    "  /lessons            list installed lessons\n"
    "                      /lessons show <id>   — print full body\n"
    "                      /lessons forget <id> — delete a lesson\n"
    "  /reflect            study the trace and propose lesson updates\n"
    "  /save <file>        save current session to file\n"
    "  /load <file>        load session from file"
)


TOGGLES: dict[str, tuple[dict, str]] = {
    "/think-on":           ({"think_on": True},                                                    "thinking: on"),
    "/think-off":          ({"think_on": False},                                                   "thinking: off"),
    "/net-on":             ({"net_on": True},                                                      "network: on"),
    "/net-off":            ({"net_on": False},                                                     "network: off"),
    "/show-toolcall-on":   ({"show_tool_call": True},                                              "show-toolcall: on"),
    "/show-toolcall-off":  ({"show_tool_call": False},                                             "show-toolcall: off"),
    "/show-think-on":      ({"show_think": True},                                                  "show-think: on"),
    "/show-think-off":     ({"show_think": False},                                                 "show-think: off"),
    "/show-lessons-on":    ({"show_lessons": True},                                                "show-lessons: on"),
    "/show-lessons-off":   ({"show_lessons": False},                                               "show-lessons: off"),
    "/show-all-on":        ({"show_think": True,  "show_tool_call": True,  "show_lessons": True}, "show-think: on, show-toolcall: on, show-lessons: on"),
    "/show-all-off":       ({"show_think": False, "show_tool_call": False, "show_lessons": False}, "show-think: off, show-toolcall: off, show-lessons: off"),
}


def _apply_toggle(state: dict, cmd: str) -> str | None:
    """Apply a toggle command to state. Returns status message or None if unrecognized."""
    entry = TOGGLES.get(cmd)
    if entry is None:
        return None
    state.update(entry[0])
    return entry[1]


def _handle_lessons_command(user_input: str, lessons: LessonStore) -> None:
    """Handle /lessons, /lessons show <id>."""
    parts = user_input.split(maxsplit=2)
    if len(parts) == 1:
        if not lessons.lessons:
            print("(no lessons installed)")
            return
        by_cat: dict[str, list] = {}
        for l in lessons.lessons:
            by_cat.setdefault(l.category, []).append(l)
        for cat in ("preference", "fact", "process", "gotcha"):
            entries = by_cat.get(cat, [])
            if not entries:
                continue
            print(f"  [{cat}]")
            for l in entries:
                print(f"    {l.id}: {l.title} (trigger: {l.trigger.type})")
        return
    sub = parts[1]
    if sub == "show" and len(parts) == 3:
        lesson = lessons.by_id(parts[2])
        if lesson is None:
            print(f"(no lesson with id {parts[2]!r})")
            return
        print(lesson.path.read_text() if lesson.path else "(no path)")
        return
    if sub == "forget" and len(parts) == 3:
        try:
            lessons.forget_lesson(parts[2])
        except WriteError as e:
            print(f"(forget failed: {e})")
            return
        print(f"(forgot {parts[2]})")
        return
    print("usage: /lessons | /lessons show <id> | /lessons forget <id>")


def _parse_exit_command(user_input: str) -> tuple[bool, bool] | None:
    """Recognize /exit, /quit, optionally with a 'noref' suffix.

    Returns (is_exit, skip_reflection) or None if the input isn't an
    exit command. `skip_reflection` is True for '/exit noref' / '/quit noref'.
    """
    parts = user_input.split()
    if not parts or parts[0] not in ("/exit", "/quit"):
        return None
    if len(parts) == 1:
        return True, False
    if len(parts) == 2 and parts[1] == "noref":
        return True, True
    return None  # malformed; fall through to "unknown command"


def _format_proposal(idx: int, op) -> str:
    """One-screen description of a single reflector op for the review UI."""
    if isinstance(op, CreateOp):
        L = op.lesson
        trig = L.get("trigger", {})
        trig_str = trig.get("type", "?")
        if trig.get("tool"):
            trig_str += f"(tool={trig['tool']})"
        if trig.get("keywords"):
            trig_str += f"(keywords={','.join(trig['keywords'])})"
        scope = L.get("scope") or {}
        scope_str = (
            ", ".join(f"{k}={v}" for k, v in scope.items()) if scope else "global"
        )
        return (
            f"[{idx}] CREATE {L.get('category', '?')}/{L.get('title', '<no title>')}\n"
            f"    Trigger: {trig_str}\n"
            f"    Scope:   {scope_str}\n"
            f"    Rule:    {L.get('rule', '')}\n"
            f"    Why:     {L.get('why', '')}"
        )
    if isinstance(op, UpdateOp):
        return (
            f"[{idx}] UPDATE {op.id}\n"
            f"    Fields: {list(op.fields.keys())}"
        )
    if isinstance(op, SkipOp):
        return f"[{idx}] SKIP — {op.reason}"
    return f"[{idx}] {op!r}"


def _apply_op(op, lessons: LessonStore, source_trace: str | None) -> str:
    """Apply a single op. Returns a short status line."""
    if isinstance(op, CreateOp):
        data = dict(op.lesson)
        if source_trace and "source_trace" not in data:
            data["source_trace"] = source_trace
        try:
            new = lessons.create_lesson(data)
        except (WriteError, Exception) as e:
            return f"  CREATE failed: {type(e).__name__}: {e}"
        return f"  CREATE {new.category}/{new.id}"
    if isinstance(op, UpdateOp):
        try:
            updated = lessons.update_lesson(op.id, op.fields)
        except (WriteError, Exception) as e:
            return f"  UPDATE failed: {type(e).__name__}: {e}"
        return f"  UPDATE {updated.id}"
    if isinstance(op, SkipOp):
        return f"  SKIP — {op.reason}"
    return f"  unknown op: {op!r}"


def run_reflection(
    messages: list[dict],
    *,
    llm: LLM,
    lessons: LessonStore,
    session_ctx: SessionContext,
    last_reflection_idx: int,
    auto: bool = False,
) -> int:
    """Run the reflector LLM call, show proposals, apply if confirmed.

    Returns the new `last_reflection_idx` (advanced if anything ran).
    """
    trace = messages[last_reflection_idx:]
    if not any(m.get("role") in ("user", "assistant") for m in trace):
        print("(reflect: nothing to reflect on yet)")
        return last_reflection_idx

    print("(reflecting on the recent trace...)")
    try:
        result = reflect(llm, trace, lessons.in_scope(session_ctx))
    except ReflectorError as e:
        print(f"(reflect: parser error — {e})")
        return last_reflection_idx
    except Exception as e:
        print(f"(reflect: LLM call failed — {type(e).__name__}: {e})")
        return last_reflection_idx

    if not result.ops:
        print("(reflect: nothing to learn)")
        return len(messages)

    print()
    print("=== Reflection proposal ===")
    for i, op in enumerate(result.ops, 1):
        print(_format_proposal(i, op))
    print()

    if auto:
        choice = "y"
    else:
        try:
            choice = input("Apply all? [y/n/skip-<n>] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "n"
            print()

    if choice == "n" or choice == "":
        print("(reflect: discarded)")
        return len(messages)

    skip_idx = None
    if choice.startswith("skip-"):
        try:
            skip_idx = int(choice.split("-", 1)[1])
        except ValueError:
            print(f"(reflect: invalid choice {choice!r}; discarded)")
            return len(messages)

    snapshot_path: str | None = None
    creating_or_updating = any(
        isinstance(op, (CreateOp, UpdateOp)) for op in result.ops
    )
    if creating_or_updating:
        snapshot_path = lessons.write_trace_snapshot(
            trace, slug_hint=_first_title(result.ops),
        )

    for i, op in enumerate(result.ops, 1):
        if skip_idx is not None and i == skip_idx:
            print(f"  [{i}] skipped")
            continue
        print(_apply_op(op, lessons, snapshot_path))
    return len(messages)


def _first_title(ops) -> str:
    for op in ops:
        if isinstance(op, CreateOp):
            return str(op.lesson.get("title", "lesson"))
        if isinstance(op, UpdateOp):
            return op.id
    return "reflection"


def _parse_task_file(text: str) -> tuple[str, list[str]]:
    """Split a task file into (prompt, trailing slash commands)."""
    lines = text.splitlines()
    end = len(lines)
    while end > 0 and lines[end - 1].strip() == "":
        end -= 1
    cmd_start = end
    while cmd_start > 0 and lines[cmd_start - 1].strip().startswith("/"):
        cmd_start -= 1
    cmds = [lines[i].strip() for i in range(cmd_start, end)]
    prompt = "\n".join(lines[:cmd_start]).strip()
    return prompt, cmds


REPLAN_CAP = 2  # max replans per tool-call boundary, per design doc


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
    session_ctx=None,
    injected_ids: set[str] | None = None,
    on_replan=None,
    on_lesson_inject=None,
) -> str:
    """Drive LLM + tool-call loop until no more tool calls. Returns final reply text.

    When `lessons` and `session_ctx` are provided, applies mid-loop hooks:
    - `on_tool_call` matches drive a replan (LLM re-prompted with the matched
      lesson, original draft popped from history).
    - `on_monologue` matches inject a system-role message after the LLM
      response finalizes and after each tool result.
    """
    if injected_ids is None:
        injected_ids = set()

    reply_text = ""
    while True:
        # Inner replan loop: keep re-calling the LLM until either no
        # on_tool_call lessons fire, the cap is hit, or the response has no
        # tool calls.
        replan_count = 0
        while True:
            stripper = _ThinkStripper(
                on_reply=on_reply, on_think=on_think, start_in_think=think_on,
            )
            msg = llm.chat(
                messages,
                enable_thinking=think_on,
                tools=TOOLS_SCHEMA,
                on_text=stripper.feed,
                on_reasoning=on_think,
            )
            stripper.flush()
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

            # Decide whether to replan. Only relevant if we have tool calls
            # and a lesson store.
            if (
                msg.tool_calls
                and lessons is not None
                and session_ctx is not None
                and replan_count < REPLAN_CAP
            ):
                text, ids = lessons.apply_on_tool_call(
                    session_ctx,
                    _tool_call_views(msg.tool_calls),
                    injected_ids,
                )
                if ids:
                    # Pop the just-emitted draft, inject the lesson note,
                    # and loop back to re-call the LLM.
                    messages.pop()
                    _inject_lesson_message(messages, text, ids, injected_ids)
                    if on_replan is not None:
                        on_replan(ids)
                    replan_count += 1
                    continue
            # No replan needed (or cap hit). Run on_monologue against the
            # final assistant content, then exit the inner loop.
            if lessons is not None and session_ctx is not None:
                text, ids = lessons.apply_on_monologue(
                    session_ctx, msg.content or "", injected_ids,
                )
                _inject_lesson_message(messages, text, ids, injected_ids)
                if ids and on_lesson_inject is not None:
                    on_lesson_inject("on_monologue", ids)
            break

        if not msg.tool_calls:
            return reply_text

        # Dispatch tool calls; on_monologue runs against each result.
        ctx_obj = ToolContext(
            workspace=workspace, net_on=net_on,
            skills={s.name: s for s in skills},
        )
        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments, ctx_obj)
            on_tool(tc.function.name, tc.function.arguments, result)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
            if lessons is not None and session_ctx is not None:
                blob = (
                    f"{tc.function.name} {tc.function.arguments or ''} {result}"
                )
                text, ids = lessons.apply_on_monologue(
                    session_ctx, blob, injected_ids,
                )
                _inject_lesson_message(messages, text, ids, injected_ids)
                if ids and on_lesson_inject is not None:
                    on_lesson_inject("on_monologue", ids)


def run_task_mode(
    task_file: str,
    system_prompt: str,
    skills,
    llm: LLM,
    workspace: Path,
    lessons,
    session_ctx,
) -> int:
    prompt, cmds = _parse_task_file(Path(task_file).read_text())
    if not prompt:
        print(f"(task file {task_file} contains no prompt)", file=sys.stderr)
        return 2

    state = {"think_on": True, "net_on": True, "show_tool_call": False, "show_think": False}
    debug = False
    cmd_notes: list[str] = []
    for c in cmds:
        if c == "/debug":
            debug = True
            cmd_notes.append("applied /debug")
        elif _apply_toggle(state, c) is not None:
            cmd_notes.append(f"applied {c}")
        else:
            cmd_notes.append(f"ignored unsupported command: {c}")

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    injected_ids: set[str] = set()
    op_text, op_ids = lessons.apply_on_prompt(session_ctx, prompt, injected_ids)
    if op_ids:
        messages.append({"role": "user", "content": op_text})
        injected_ids.update(op_ids)
    think_chunks: list[str] = []
    tool_records: list[str] = []
    reply_text = ""
    error: str | None = None
    flags = {"reply_started": False, "think_started": False}

    def on_reply(s: str) -> None:
        if not debug:
            return
        if not flags["reply_started"]:
            s = s.lstrip()
            if not s:
                return
            sys.stdout.write("\nleo> ")
            flags["reply_started"] = True
        sys.stdout.write(s)
        sys.stdout.flush()

    def on_think(s: str) -> None:
        think_chunks.append(s)
        if not debug:
            return
        if not flags["think_started"]:
            sys.stdout.write("\n(think) ")
            flags["think_started"] = True
        sys.stdout.write(s)
        sys.stdout.flush()

    def on_tool(name: str, args: str, result: str) -> None:
        preview = result if len(result) <= 200 else result[:200] + "..."
        tool_records.append(f"{name}({args}) -> {preview}")
        if debug:
            print(f"\n(tool {name}({args}) -> {preview})")
            flags["reply_started"] = False
            flags["think_started"] = False

    if debug:
        print("=== Task (debug) ===")
        print(f"file: {task_file}\n")

    def on_replan(ids):
        if debug:
            print(f"\n(replanning: lesson(s) {', '.join(ids)} triggered)")
            flags["reply_started"] = False
            flags["think_started"] = False

    try:
        reply_text = run_turn(
            messages, llm=llm, skills=skills, workspace=workspace,
            think_on=state["think_on"], net_on=state["net_on"],
            on_reply=on_reply, on_think=on_think, on_tool=on_tool,
            lessons=lessons, session_ctx=session_ctx,
            injected_ids=injected_ids, on_replan=on_replan,
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    if debug:
        print()

    turns = sum(1 for m in messages if m["role"] == "assistant")

    print("=== Task ===")
    print(f"file:   {task_file}")
    if cmds:
        print("commands:")
        for note in cmd_notes:
            print(f"  {note}")
    print("\n=== Monologue ===")
    print("".join(think_chunks).strip() or "(no thinking content captured)")
    if tool_records:
        print("\n--- tool calls ---")
        for r in tool_records:
            print(f"- {r}")
    print("\n=== Final Result ===")
    print(reply_text.strip() or "(no final reply)")
    print("\n=== Status ===")
    print(f"outcome:    {'error' if error else 'completed'}")
    if error:
        print(f"error:      {error}")
    print(f"turns:      {turns}")
    print(f"tool_calls: {len(tool_records)}")
    pct = llm.last_total_tokens / llm.max_tokens * 100 if llm.max_tokens else 0.0
    print(f"context:    {llm.last_total_tokens:,} / {llm.max_tokens:,} tokens ({pct:.1f}%)")
    return 1 if error else 0


def main() -> None:
    load_dotenv()
    load_dotenv(Path.home() / ".env")

    parser = argparse.ArgumentParser(prog="leo", allow_abbrev=False)
    parser.add_argument(
        "--sysprompt",
        metavar="FILE",
        help="path to a file whose contents are used as the system prompt",
    )
    parser.add_argument(
        "--task",
        metavar="FILE",
        help="run non-interactively using FILE's contents as the initial user prompt",
    )
    args = parser.parse_args()

    if args.sysprompt:
        system_prompt = Path(args.sysprompt).read_text()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    skills = discover_skills(SKILLS_ROOT)
    if skills:
        lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
        system_prompt = (
            f"{system_prompt}\n\n"
            "Available skills. Before attempting a task, check this list. "
            "If a skill's description matches the task, you MUST call "
            "load_skill(name) FIRST and follow its instructions — do not "
            "try to solve the task ad-hoc. After every tool result, "
            "re-check this list against what you just observed (not just "
            "the original user query) before choosing the next tool — a "
            "skill may match a symptom that only becomes visible after a "
            "fetch or command runs.\n\n"
            f"{lines}"
        )

    llm = LLM()
    workspace = Path.cwd().resolve()

    lessons = LessonStore([LESSONS_ROOT])
    for issue in lessons.issues:
        print(f"(lesson {issue.path.name}: {issue.reason})", file=sys.stderr)
    session_ctx = SessionContext(
        project=os.environ.get("LEO_PROJECT"),
        model=llm.model,
        skills=frozenset(s.name for s in skills),
    )
    lessons_block = lessons.render_session_block(session_ctx)
    if lessons_block:
        system_prompt = f"{system_prompt}\n\n{lessons_block}"

    if args.task:
        sys.exit(
            run_task_mode(
                args.task, system_prompt, skills, llm, workspace,
                lessons, session_ctx,
            )
        )

    state = {
        "think_on": True, "net_on": True,
        "show_tool_call": False, "show_think": False, "show_lessons": False,
    }
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    injected_ids: set[str] = set()
    last_reflection_idx = 1  # everything after the system message

    def print_status() -> None:
        print(f"model:         {llm.model}")
        print(f"base_url:      {llm.base_url}")
        print(f"thinking:      {'on' if state['think_on'] else 'off'}")
        print(f"network:       {'on' if state['net_on'] else 'off'}")
        print(f"show-toolcall: {'on' if state['show_tool_call'] else 'off'}")
        print(f"show-think:    {'on' if state['show_think'] else 'off'}")
        print(f"show-lessons:  {'on' if state['show_lessons'] else 'off'}")
        print(f"workspace:     {workspace}")
        print(f"skills:        {len(skills)} loaded")
        print(f"lessons:       {len(lessons.lessons)} loaded")
        print(f"turns:         {sum(1 for m in messages if m['role'] == 'user')}")
        pct = llm.last_total_tokens / llm.max_tokens * 100 if llm.max_tokens else 0.0
        print(f"context:       {llm.last_total_tokens:,} / {llm.max_tokens:,} tokens ({pct:.1f}%)")

    print(render_leo_banner())
    print_status()
    print("type /help to list commands")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        exit_parsed = _parse_exit_command(user_input)
        if exit_parsed is not None:
            _, skip_reflection = exit_parsed
            if not skip_reflection:
                last_reflection_idx = run_reflection(
                    messages,
                    llm=llm,
                    lessons=lessons,
                    session_ctx=session_ctx,
                    last_reflection_idx=last_reflection_idx,
                )
            break
        toggle_msg = _apply_toggle(state, user_input)
        if toggle_msg is not None:
            print(f"({toggle_msg})")
            continue
        if user_input == "/help":
            print(COMMANDS_HELP)
            continue
        if user_input == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            injected_ids = set()
            last_reflection_idx = 1
            print("(history cleared)")
            continue
        if user_input == "/reflect":
            last_reflection_idx = run_reflection(
                messages,
                llm=llm,
                lessons=lessons,
                session_ctx=session_ctx,
                last_reflection_idx=last_reflection_idx,
            )
            continue
        if user_input == "/tools":
            for t in TOOLS_SCHEMA:
                fn = t["function"]
                print(f"  {fn['name']}: {fn['description']}")
            continue
        if user_input == "/skills":
            if not skills:
                print("(no skills installed)")
            else:
                for s in skills:
                    print(f"  {s.name}: {s.description}")
            continue
        if user_input.startswith("/lessons"):
            _handle_lessons_command(user_input, lessons)
            continue
        if user_input == "/status":
            print_status()
            continue
        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print("usage: /save <file>")
                continue
            path = Path(parts[1]).expanduser()
            path.write_text(
                json.dumps({"messages": messages, "think_on": state["think_on"]}, indent=2)
            )
            print(f"(saved to {path})")
            continue
        if user_input.startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print("usage: /load <file>")
                continue
            path = Path(parts[1]).expanduser()
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"(load failed: {e})")
                continue
            messages = data["messages"]
            state["think_on"] = data.get("think_on", state["think_on"])
            print(f"(loaded from {path})")
            continue

        if user_input.startswith("/"):
            print(COMMANDS_HELP)
            continue

        messages.append({"role": "user", "content": user_input})
        # Phase 2: on_prompt injection before the LLM sees the new turn.
        op_text, op_ids = lessons.apply_on_prompt(
            session_ctx, user_input, injected_ids,
        )
        if op_ids:
            messages.append({"role": "user", "content": op_text})
            injected_ids.update(op_ids)
            if state["show_lessons"]:
                print(f"(lesson on_prompt: {', '.join(op_ids)})")
        with _ls_trace(name="turn", run_type="chain", inputs={"user_input": user_input}) as rt:
            flags = {"reply_started": False, "think_started": False}

            def on_reply(s: str) -> None:
                if not flags["reply_started"]:
                    s = s.lstrip()
                    if not s:
                        return
                    sys.stdout.write("\nleo> ")
                    flags["reply_started"] = True
                sys.stdout.write(s)
                sys.stdout.flush()

            def on_think(s: str) -> None:
                if not state["show_think"]:
                    return
                if not flags["think_started"]:
                    sys.stdout.write("\n(think) ")
                    flags["think_started"] = True
                sys.stdout.write(s)
                sys.stdout.flush()

            def on_tool(name: str, args: str, result: str) -> None:
                if state["show_tool_call"]:
                    preview = result if len(result) <= 200 else result[:200] + "..."
                    print(f"\n(tool {name}({args}) -> {preview})")
                else:
                    print(".", end="", flush=True)
                flags["reply_started"] = False
                flags["think_started"] = False

            def on_replan(ids):
                print(
                    f"\n(replanning: lesson(s) {', '.join(ids)} triggered)",
                    flush=True,
                )
                flags["reply_started"] = False
                flags["think_started"] = False

            def on_lesson_inject(phase, ids):
                if not state["show_lessons"]:
                    return
                print(f"\n(lesson {phase}: {', '.join(ids)})", flush=True)
                flags["reply_started"] = False
                flags["think_started"] = False

            reply_text = run_turn(
                messages, llm=llm, skills=skills, workspace=workspace,
                think_on=state["think_on"], net_on=state["net_on"],
                on_reply=on_reply, on_think=on_think, on_tool=on_tool,
                lessons=lessons, session_ctx=session_ctx,
                injected_ids=injected_ids, on_replan=on_replan,
                on_lesson_inject=on_lesson_inject,
            )
            rt.end(outputs={"reply": reply_text})


if __name__ == "__main__":
    main()

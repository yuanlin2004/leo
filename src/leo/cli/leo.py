from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from prompt_toolkit import PromptSession

from dotenv import load_dotenv

from leo.cli.banner import render_leo_banner
from leo.core.agent import (
    REPLAN_CAP,
    _ThinkStripper,
    _inject_lesson_message,
    _split_think,
    _tool_call_views,
    run_turn,
)
from leo.core.lessons import LessonStore, LessonScope, WriteError
from leo.core.lessons.reflector import (
    CreateOp,
    ReflectorError,
    SkipOp,
    UpdateOp,
    reflect,
)
from leo.core.lessons.schema import SchemaError, parse_lesson_text
from leo.core.llm import LLM
from leo.core.session import (
    Session,
    append_event,
    append_messages,
    count_messages,
    delete_session,
    derive_title_from_messages,
    find_last,
    list_sessions,
    load_session,
    new_session,
)
from leo.core.setup import (
    DEFAULT_SYSTEM_PROMPT,
    LESSONS_ROOT,
    SKILLS_ROOT,
    WORKSPACE_MARKER,
    WORKSPACE_SUBDIRS,
    build_run_context,
)
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


def _cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="leo init",
        description="Create a .leo/ skeleton at PATH (default: current directory).",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="directory to initialize as a Leo workspace (default: cwd)",
    )
    args = parser.parse_args(argv)
    target = Path(args.path).resolve()
    if not target.is_dir():
        print(f"leo init: {target} is not a directory", file=sys.stderr)
        return 1
    leo_dir = target / WORKSPACE_MARKER
    created: list[Path] = []
    for sub in (leo_dir, *(leo_dir / s for s in WORKSPACE_SUBDIRS)):
        if not sub.exists():
            sub.mkdir(parents=True)
            created.append(sub)
    if created:
        print(f"initialized workspace at {target}")
        for p in created:
            print(f"  created {p.relative_to(target)}")
    else:
        print(f"workspace at {target} already initialized")
    return 0


def _resolve_workspace(arg: str | None) -> Path:
    """Return the absolute workspace path or exit with a hint."""
    workspace = Path(arg).resolve() if arg else Path.cwd().resolve()
    if not workspace.is_dir():
        print(f"leo: workspace path {workspace} is not a directory", file=sys.stderr)
        sys.exit(2)
    if not (workspace / WORKSPACE_MARKER).is_dir():
        print(
            f"leo: no workspace found at {workspace}\n"
            f"run `leo init [path]` to create one",
            file=sys.stderr,
        )
        sys.exit(2)
    return workspace


def _fmt_ts(ts: str) -> str:
    """Display ISO timestamp with '-' between date and time instead of 'T'."""
    return ts.replace("T", "-", 1) if ts else ts


def _choose_session_interactively(workspace: Path) -> str | None:
    """Prompt the user to start a new session or resume an existing one.

    Returns the chosen session id to resume, or None to start a new
    session. Exits the process on 'q' / EOF / Ctrl-C.
    """
    sessions = list_sessions(workspace)
    if not sessions:
        return None
    print(f"\nSessions in {workspace}:")
    print("  [n] start a new session  (default)")
    for i, s in enumerate(sessions, 1):
        print(f"  [{i}] {s.id}  {_fmt_ts(s.last_active)}  {s.title}")
    print("  [q] exit")
    while True:
        try:
            raw = input("choose> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
        if raw in ("", "n"):
            return None
        if raw == "q":
            sys.exit(0)
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1].id
        print(f"invalid choice: {raw!r}")


def _cmd_session(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="leo session")
    parser.add_argument(
        "--workspace", metavar="PATH",
        help="workspace directory (default: cwd)",
    )
    sub = parser.add_subparsers(dest="op", required=True)
    # `--workspace` is also accepted after the verb so both orderings work
    # (`leo session --workspace X list` and `leo session list --workspace X`).
    # `default=SUPPRESS` keeps the subparser from clobbering the parent's
    # value when the flag is omitted on the child side.
    ws_kwargs = dict(
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="workspace directory (default: cwd)",
    )
    p_list = sub.add_parser("list", help="list sessions in the workspace")
    p_list.add_argument("--workspace", **ws_kwargs)
    p_show = sub.add_parser("show", help="show meta + message count for a session")
    p_show.add_argument("--workspace", **ws_kwargs)
    p_show.add_argument("id")
    p_rm = sub.add_parser("rm", help="delete a session")
    p_rm.add_argument("--workspace", **ws_kwargs)
    p_rm.add_argument("id")
    args = parser.parse_args(argv)
    workspace = _resolve_workspace(args.workspace)

    if args.op == "list":
        sessions = list_sessions(workspace)
        if not sessions:
            print("(no sessions)")
            return 0
        for s in sessions:
            print(f"  {s.id}  {_fmt_ts(s.last_active)}  {s.title}")
        return 0

    if args.op == "show":
        try:
            session, messages = load_session(workspace, args.id)
        except FileNotFoundError as e:
            print(f"leo session: {e}", file=sys.stderr)
            return 1
        print(f"id:             {session.id}")
        print(f"title:          {session.title}")
        print(f"model:          {session.model}")
        print(f"started_at:     {_fmt_ts(session.started_at)}")
        print(f"last_active:    {_fmt_ts(session.last_active)}")
        print(f"reflection_idx: {session.reflection_idx}")
        print(f"messages:       {len(messages)}")
        print(f"path:           {session.dir}")
        return 0

    if args.op == "rm":
        try:
            delete_session(workspace, args.id)
        except FileNotFoundError as e:
            print(f"leo session: {e}", file=sys.stderr)
            return 1
        print(f"deleted session {args.id}")
        return 0

    return 1

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
    "                      /lessons edit <id>   — open in $EDITOR\n"
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
    if sub == "edit" and len(parts) == 3:
        lesson = lessons.by_id(parts[2])
        if lesson is None:
            print(f"(no lesson with id {parts[2]!r})")
            return
        if lesson.path is None:
            print("(no on-disk path for this lesson)")
            return
        editor = os.environ.get("EDITOR") or "vi"
        try:
            subprocess.run([editor, str(lesson.path)], check=False)
        except FileNotFoundError:
            print(f"(editor {editor!r} not found)")
            return
        lessons.reload()
        # Surface any new validation issue with this file.
        new_issues = [i for i in lessons.issues if i.path == lesson.path]
        if new_issues:
            for issue in new_issues:
                print(f"(edit warning: {issue.reason})")
            return
        if lessons.by_id(parts[2]) is None:
            print(f"(edit removed lesson {parts[2]!r} from store)")
        else:
            print(f"(edited {parts[2]})")
        return
    if sub == "forget" and len(parts) == 3:
        try:
            lessons.forget_lesson(parts[2])
        except WriteError as e:
            print(f"(forget failed: {e})")
            return
        print(f"(forgot {parts[2]})")
        return
    print(
        "usage: /lessons | /lessons show <id> | "
        "/lessons edit <id> | /lessons forget <id>"
    )


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


def _render_draft_lesson(d: dict) -> str:
    """Build a markdown view of a lesson dict, NO validation. Used to seed
    the editor with a possibly-invalid proposal."""
    import yaml as _yaml
    fm: dict = {}
    for k in ("id", "title", "category", "trigger", "scope", "created", "updated"):
        if k in d and d[k] is not None and d[k] != "":
            fm[k] = d[k]
    fm_yaml = _yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    body = (
        f"## Rule\n{(d.get('rule') or '').strip()}\n\n"
        f"## Why\n{(d.get('why') or '').strip()}\n\n"
        f"## How to apply\n{(d.get('how_to_apply') or '').strip()}\n"
    )
    return f"---\n{fm_yaml}\n---\n\n{body}"


def _existing_lesson_to_dict(existing) -> dict:
    """Snapshot of an existing Lesson as a dict suitable for editing."""
    from leo.core.lessons import _scope_to_dict, _trigger_to_dict
    return {
        "id": existing.id,
        "title": existing.title,
        "category": existing.category,
        "trigger": _trigger_to_dict(existing.trigger),
        "scope": _scope_to_dict(existing.scope),
        "created": existing.created,
        "updated": existing.updated,
        "rule": existing.rule,
        "why": existing.why,
        "how_to_apply": existing.how_to_apply,
    }


def _parsed_lesson_to_dict(lesson) -> dict:
    """Dict shape `update_lesson` and `create_lesson` accept."""
    from leo.core.lessons import _scope_to_dict, _trigger_to_dict
    return {
        "title": lesson.title,
        "category": lesson.category,
        "trigger": _trigger_to_dict(lesson.trigger),
        "scope": _scope_to_dict(lesson.scope),
        "rule": lesson.rule,
        "why": lesson.why,
        "how_to_apply": lesson.how_to_apply,
    }


def _edit_op_in_editor(op, lessons: LessonStore):
    """Edit a CreateOp / UpdateOp in $EDITOR. Returns the (possibly
    updated) op, or the original if the user aborts on a validation error.

    SkipOps pass through unchanged.
    """
    if isinstance(op, SkipOp):
        return op
    if isinstance(op, CreateOp):
        seed = dict(op.lesson)
        seed.setdefault("id", "(generated at create time)")
    elif isinstance(op, UpdateOp):
        existing = lessons.by_id(op.id)
        if existing is None:
            print(f"  (cannot edit: no lesson with id {op.id!r})")
            return op
        seed = _existing_lesson_to_dict(existing)
        seed.update(op.fields)
    else:
        return op

    editor = os.environ.get("EDITOR") or "vi"
    text = _render_draft_lesson(seed)
    while True:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, prefix="leo-lesson-",
        ) as tf:
            tf.write(text)
            tmp_path = tf.name
        try:
            subprocess.run([editor, tmp_path], check=False)
            edited = Path(tmp_path).read_text()
        except FileNotFoundError:
            print(f"  (editor {editor!r} not found; keeping original)")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return op
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        try:
            parsed = parse_lesson_text(edited, source="<edit>")
        except SchemaError as e:
            print(f"  (edit failed validation: {e})")
            choice = input("  retry / abort? ").strip().lower() or "abort"
            if choice.startswith("a"):
                print("  (kept original proposal)")
                return op
            text = edited  # carry the user's broken edit forward to fix
            continue

        if isinstance(op, CreateOp):
            return CreateOp(lesson=_parsed_lesson_to_dict(parsed), raw=op.raw)
        return UpdateOp(
            id=op.id, fields=_parsed_lesson_to_dict(parsed), raw=op.raw,
        )


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
    lesson_scope: LessonScope,
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
        result = reflect(llm, trace, lessons.in_scope(lesson_scope))
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
            choice = input(
                "Apply all? [y/n/edit/skip-<n>] "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "n"
            print()

    if choice == "n" or choice == "":
        print("(reflect: discarded)")
        return len(messages)

    if choice == "edit":
        new_ops = []
        for i, op in enumerate(result.ops, 1):
            if isinstance(op, SkipOp):
                new_ops.append(op)
                continue
            print(f"  [{i}] opening in $EDITOR...")
            new_ops.append(_edit_op_in_editor(op, lessons))
        result_ops = new_ops
    else:
        result_ops = result.ops

    skip_idx = None
    if choice.startswith("skip-"):
        try:
            skip_idx = int(choice.split("-", 1)[1])
        except ValueError:
            print(f"(reflect: invalid choice {choice!r}; discarded)")
            return len(messages)

    snapshot_path: str | None = None
    creating_or_updating = any(
        isinstance(op, (CreateOp, UpdateOp)) for op in result_ops
    )
    if creating_or_updating:
        snapshot_path = lessons.write_trace_snapshot(
            trace, slug_hint=_first_title(result_ops),
        )

    for i, op in enumerate(result_ops, 1):
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


def run_task_mode(
    task_file: str,
    system_prompt: str,
    skills,
    llm: LLM,
    workspace: Path,
    lessons,
    lesson_scope,
    phase1_ids: list[str] | None = None,
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
    injected_ids: set[str] = set(phase1_ids or ())
    loaded_skills: set[str] = set()
    op_text, op_ids = lessons.apply_on_prompt(lesson_scope, prompt, injected_ids)
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
            lessons=lessons, lesson_scope=lesson_scope,
            injected_ids=injected_ids, on_replan=on_replan,
            loaded_skills=loaded_skills,
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
    print(f"skills:     {len(skills)} found, {len(loaded_skills)} loaded")
    print(f"lessons:    {lessons.found_count} found, {len(injected_ids)} loaded")
    pct = llm.last_total_tokens / llm.max_tokens * 100 if llm.max_tokens else 0.0
    print(f"context:    {llm.last_total_tokens:,} / {llm.max_tokens:,} tokens ({pct:.1f}%)")
    return 1 if error else 0


def main() -> None:
    load_dotenv()
    load_dotenv(Path.home() / ".env")

    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        sys.exit(_cmd_init(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "session":
        sys.exit(_cmd_session(sys.argv[2:]))

    parser = argparse.ArgumentParser(
        prog="leo",
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Leo — an LLM-based agent. Runs an interactive chat session "
            "by default; use --task for one-shot non-interactive mode."
        ),
        epilog=(
            "subcommands:\n"
            "  leo init [path]         create a .leo/ skeleton (workspace) at path (default: cwd)\n"
            "  leo session list        list sessions in the current workspace\n"
            "  leo session show <id>   print a session's meta + message count\n"
            "  leo session rm <id>     delete a session\n"
            "\n"
            "Each subcommand accepts --workspace PATH and -h for its own help.\n"
            "Type /help inside the REPL for in-session commands."
        ),
    )
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
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        help="workspace directory (must contain a .leo/ folder; default: cwd)",
    )
    parser.add_argument(
        "--session",
        metavar="ID",
        help="resume an existing session by id (or 'last' for the most recent)",
    )
    args = parser.parse_args()

    workspace = _resolve_workspace(args.workspace)

    base_sp = Path(args.sysprompt).read_text() if args.sysprompt else None
    ctx = build_run_context(workspace, base_system_prompt=base_sp)
    system_prompt = ctx.system_prompt
    skills = ctx.skills
    llm = ctx.llm
    lessons = ctx.lessons
    lesson_scope = ctx.lesson_scope
    phase1_ids = ctx.phase1_ids
    for issue in ctx.lesson_issues:
        print(f"(lesson {issue.path.name}: {issue.reason})", file=sys.stderr)

    if args.task:
        if args.session:
            print(
                "leo: --task does not support --session in v1 "
                "(one-shots are ephemeral)", file=sys.stderr,
            )
            sys.exit(2)
        sys.exit(
            run_task_mode(
                args.task, system_prompt, skills, llm, workspace,
                lessons, lesson_scope, phase1_ids,
            )
        )

    # Resolve session: resume if --session, otherwise prompt the user
    # (or create silently if no sessions exist yet).
    state = {
        "think_on": True, "net_on": True,
        "show_tool_call": False, "show_think": False, "show_lessons": False,
    }
    session_choice = args.session
    if session_choice is None:
        session_choice = _choose_session_interactively(workspace)

    session: Session
    if session_choice:
        sid = session_choice
        if sid == "last":
            last = find_last(workspace)
            if last is None:
                print("leo: no sessions in workspace to resume", file=sys.stderr)
                sys.exit(2)
            sid = last.id
        try:
            session, messages = load_session(workspace, sid)
        except FileNotFoundError as e:
            print(f"leo: {e}", file=sys.stderr)
            sys.exit(2)
        if not messages:
            messages = [{"role": "system", "content": system_prompt}]
        injected_ids = set(session.injected_ids) | set(phase1_ids)
        loaded_skills = set(session.loaded_skills)
        last_reflection_idx = session.reflection_idx
        if session.toggles:
            state.update(session.toggles)
        print(f"(resumed session {session.id} — {len(messages)} messages)")
    else:
        messages = [{"role": "system", "content": system_prompt}]
        injected_ids = set(phase1_ids)
        loaded_skills = set()
        last_reflection_idx = 1
        session = new_session(
            workspace, model=llm.model, toggles=state, title="(untitled)",
        )
        append_messages(session, messages)  # persist system message
    persist_idx = len(messages)

    def print_status() -> None:
        print(f"model:         {llm.model}")
        print(f"base_url:      {llm.base_url}")
        print(f"thinking:      {'on' if state['think_on'] else 'off'}")
        print(f"network:       {'on' if state['net_on'] else 'off'}")
        print(f"show-toolcall: {'on' if state['show_tool_call'] else 'off'}")
        print(f"show-think:    {'on' if state['show_think'] else 'off'}")
        print(f"show-lessons:  {'on' if state['show_lessons'] else 'off'}")
        print(f"workspace:     {workspace}")
        print(f"session:       {session.id}")
        print(f"skills:        {len(skills)} found, {len(loaded_skills)} loaded")
        print(
            f"lessons:       {lessons.found_count} found, "
            f"{len(injected_ids)} loaded"
        )
        print(f"turns:         {sum(1 for m in messages if m['role'] == 'user')}")
        pct = llm.last_total_tokens / llm.max_tokens * 100 if llm.max_tokens else 0.0
        print(f"context:       {llm.last_total_tokens:,} / {llm.max_tokens:,} tokens ({pct:.1f}%)")

    print(render_leo_banner())
    print_status()
    print("type /help to list commands")

    prompt_session: PromptSession[str] = PromptSession()

    while True:
        try:
            user_input = prompt_session.prompt("\nyou> ").strip()
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
                    lesson_scope=lesson_scope,
                    last_reflection_idx=last_reflection_idx,
                )
            session.reflection_idx = last_reflection_idx
            session.toggles = dict(state)
            session.injected_ids = sorted(injected_ids)
            session.loaded_skills = sorted(loaded_skills)
            session.write_meta()
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
            injected_ids = set(phase1_ids)
            loaded_skills = set()
            last_reflection_idx = 1
            # Truncate the persisted record and re-seed with the system message.
            session.messages_path.write_text("")
            append_messages(session, messages)
            persist_idx = len(messages)
            print("(history cleared)")
            continue
        if user_input == "/reflect":
            last_reflection_idx = run_reflection(
                messages,
                llm=llm,
                lessons=lessons,
                lesson_scope=lesson_scope,
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
            lesson_scope, user_input, injected_ids,
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

            def on_event(type_: str, payload: dict) -> None:
                # Silent disk-side observability stream. Display behavior is
                # driven by the on_reply / on_think / on_tool / on_replan /
                # on_lesson_inject callbacks above; events.jsonl exists so
                # the web UI (or a future `leo session trace`) can replay
                # the run.
                append_event(session, type_, payload)

            reply_text = run_turn(
                messages, llm=llm, skills=skills, workspace=workspace,
                think_on=state["think_on"], net_on=state["net_on"],
                on_reply=on_reply, on_think=on_think, on_tool=on_tool,
                lessons=lessons, lesson_scope=lesson_scope,
                injected_ids=injected_ids, on_replan=on_replan,
                on_lesson_inject=on_lesson_inject,
                on_event=on_event,
                loaded_skills=loaded_skills,
            )
            rt.end(outputs={"reply": reply_text})
        # Persist any new messages produced this turn (user + assistant +
        # tool messages + any lesson-injected system notes).
        if len(messages) > persist_idx:
            append_messages(session, messages[persist_idx:])
            persist_idx = len(messages)
        if session.title == "(untitled)":
            session.title = derive_title_from_messages(messages)
            session.write_meta()


if __name__ == "__main__":
    main()

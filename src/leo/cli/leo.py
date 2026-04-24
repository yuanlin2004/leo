from __future__ import annotations

import argparse
import json
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

COMMANDS_HELP = (
    "commands:\n"
    "  /help               show this help\n"
    "  /exit, /quit        exit the chatbot\n"
    "  /reset              clear conversation history\n"
    "  /think-on           enable model thinking\n"
    "  /think-off          disable model thinking\n"
    "  /net-on             allow network inside bash sandbox\n"
    "  /net-off            block network inside bash sandbox\n"
    "  /show-toolcall-on   print tool calls and results as they happen\n"
    "  /show-toolcall-off  hide tool-call output (default)\n"
    "  /show-think-on      print model thinking content\n"
    "  /show-think-off     hide model thinking content (default)\n"
    "  /show-all-on        show both think and toolcall\n"
    "  /show-all-off       hide both think and toolcall\n"
    "  /status             show current settings, turn count and token usage\n"
    "  /tools              list installed tools\n"
    "  /skills             list installed skills\n"
    "  /save <file>        save current session to file\n"
    "  /load <file>        load session from file"
)


TOGGLES: dict[str, tuple[dict, str]] = {
    "/think-on":          ({"think_on": True},                              "thinking: on"),
    "/think-off":         ({"think_on": False},                             "thinking: off"),
    "/net-on":            ({"net_on": True},                                "network: on"),
    "/net-off":           ({"net_on": False},                               "network: off"),
    "/show-toolcall-on":  ({"show_tool_call": True},                        "show-toolcall: on"),
    "/show-toolcall-off": ({"show_tool_call": False},                       "show-toolcall: off"),
    "/show-think-on":     ({"show_think": True},                            "show-think: on"),
    "/show-think-off":    ({"show_think": False},                           "show-think: off"),
    "/show-all-on":       ({"show_think": True, "show_tool_call": True},   "show-think: on, show-toolcall: on"),
    "/show-all-off":      ({"show_think": False, "show_tool_call": False}, "show-think: off, show-toolcall: off"),
}


def _apply_toggle(state: dict, cmd: str) -> str | None:
    """Apply a toggle command to state. Returns status message or None if unrecognized."""
    entry = TOGGLES.get(cmd)
    if entry is None:
        return None
    state.update(entry[0])
    return entry[1]


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
) -> str:
    """Drive LLM + tool-call loop until no more tool calls. Returns final reply text."""
    reply_text = ""
    while True:
        stripper = _ThinkStripper(on_reply=on_reply, on_think=on_think, start_in_think=think_on)
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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)
        if not msg.tool_calls:
            return reply_text
        ctx = ToolContext(workspace=workspace, net_on=net_on, skills={s.name: s for s in skills})
        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments, ctx)
            on_tool(tc.function.name, tc.function.arguments, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def run_task_mode(
    task_file: str,
    system_prompt: str,
    skills,
    llm: LLM,
    workspace: Path,
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

    try:
        reply_text = run_turn(
            messages, llm=llm, skills=skills, workspace=workspace,
            think_on=state["think_on"], net_on=state["net_on"],
            on_reply=on_reply, on_think=on_think, on_tool=on_tool,
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

    if args.task:
        sys.exit(run_task_mode(args.task, system_prompt, skills, llm, workspace))

    state = {"think_on": True, "net_on": True, "show_tool_call": False, "show_think": False}
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def print_status() -> None:
        print(f"model:         {llm.model}")
        print(f"base_url:      {llm.base_url}")
        print(f"thinking:      {'on' if state['think_on'] else 'off'}")
        print(f"network:       {'on' if state['net_on'] else 'off'}")
        print(f"show-toolcall: {'on' if state['show_tool_call'] else 'off'}")
        print(f"show-think:    {'on' if state['show_think'] else 'off'}")
        print(f"workspace:     {workspace}")
        print(f"skills:        {len(skills)} loaded")
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
        if user_input in ("/exit", "/quit"):
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
            print("(history cleared)")
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

            reply_text = run_turn(
                messages, llm=llm, skills=skills, workspace=workspace,
                think_on=state["think_on"], net_on=state["net_on"],
                on_reply=on_reply, on_think=on_think, on_tool=on_tool,
            )
            rt.end(outputs={"reply": reply_text})


if __name__ == "__main__":
    main()

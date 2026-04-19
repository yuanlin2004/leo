from __future__ import annotations

import argparse
import json
import re
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

from dotenv import load_dotenv

from leo.cli.banner import render_leo_banner
from leo.core.llm import LLM
from leo.core.skill_core import discover_skills
from leo.core.tools import TOOLS_SCHEMA, ToolContext, dispatch

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
    "  /status             show current settings and turn count\n"
    "  /tools              list installed tools\n"
    "  /skills             list installed skills\n"
    "  /save <file>        save current session to file\n"
    "  /load <file>        load session from file"
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="leo", allow_abbrev=False)
    parser.add_argument(
        "-sysprompt",
        metavar="FILE",
        help="path to a file whose contents are used as the system prompt",
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
    think_on = True
    net_on = True
    show_tool_call = False
    show_think = False
    workspace = Path.cwd().resolve()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def print_status() -> None:
        print(f"model:         {llm.model}")
        print(f"base_url:      {llm.base_url}")
        print(f"thinking:      {'on' if think_on else 'off'}")
        print(f"network:       {'on' if net_on else 'off'}")
        print(f"show-toolcall: {'on' if show_tool_call else 'off'}")
        print(f"show-think:    {'on' if show_think else 'off'}")
        print(f"workspace:     {workspace}")
        print(f"skills:        {len(skills)} loaded")
        print(f"turns:         {sum(1 for m in messages if m['role'] == 'user')}")

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
        if user_input == "/help":
            print(COMMANDS_HELP)
            continue
        if user_input == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("(history cleared)")
            continue
        if user_input == "/think-on":
            think_on = True
            print("(thinking: on)")
            continue
        if user_input == "/think-off":
            think_on = False
            print("(thinking: off)")
            continue
        if user_input == "/net-on":
            net_on = True
            print("(network: on)")
            continue
        if user_input == "/net-off":
            net_on = False
            print("(network: off)")
            continue
        if user_input == "/show-toolcall-on":
            show_tool_call = True
            print("(show-toolcall: on)")
            continue
        if user_input == "/show-toolcall-off":
            show_tool_call = False
            print("(show-toolcall: off)")
            continue
        if user_input == "/show-think-on":
            show_think = True
            print("(show-think: on)")
            continue
        if user_input == "/show-think-off":
            show_think = False
            print("(show-think: off)")
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
                json.dumps(
                    {"messages": messages, "think_on": think_on},
                    indent=2,
                )
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
            think_on = data.get("think_on", think_on)
            print(f"(loaded from {path})")
            continue
        
        if user_input.startswith("/"):
            print(COMMANDS_HELP)
            continue

        messages.append({"role": "user", "content": user_input})
        quiet = not show_think and not show_tool_call
        dot_count = 0
        while True:
            msg = llm.chat(messages, enable_thinking=think_on, tools=TOOLS_SCHEMA)
            think_text, reply_text = _split_think(msg.content)
            if show_think:
                reasoning = getattr(msg, "reasoning_content", None) or think_text
                if reasoning:
                    print(f"\n(think) {reasoning}")
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
            if not msg.tool_calls:
                if dot_count > 0:
                    print("\r" + " " * dot_count + "\r", end="", flush=True)
                print(f"\nleo> {reply_text}")
                break
            if quiet:
                print(".", end="", flush=True)
                dot_count += 1
            ctx = ToolContext(
                workspace=workspace,
                net_on=net_on,
                skills={s.name: s for s in skills},
            )
            for tc in msg.tool_calls:
                result = dispatch(tc.function.name, tc.function.arguments, ctx)
                if show_tool_call:
                    preview = result if len(result) <= 200 else result[:200] + "..."
                    print(f"\n(tool {tc.function.name}({tc.function.arguments}) -> {preview})")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo.cli.banner import render_leo_banner
from leo.core.llm import LLM
from leo.core.tools import TOOLS_SCHEMA, ToolContext, dispatch

DEFAULT_SYSTEM_PROMPT = "You are Leo, a helpful assistant."

COMMANDS_HELP = (
    "commands:\n"
    "  /help         show this help\n"
    "  /exit, /quit  exit the chatbot\n"
    "  /reset        clear conversation history\n"
    "  /think-on     enable model thinking\n"
    "  /think-off    disable model thinking\n"
    "  /net-on       allow network inside bash sandbox\n"
    "  /net-off      block network inside bash sandbox\n"
    "  /status       show model, base_url, thinking/net state, turn count\n"
    "  /tools        list installed tools\n"
    "  /save <file>  save current session to file\n"
    "  /load <file>  load session from file"
)


def main() -> None:
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

    llm = LLM()
    think_on = False
    net_on = True
    workspace = Path.cwd().resolve()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def print_status() -> None:
        print(f"model:     {llm.model}")
        print(f"base_url:  {llm.base_url}")
        print(f"thinking:  {'on' if think_on else 'off'}")
        print(f"network:   {'on' if net_on else 'off'}")
        print(f"workspace: {workspace}")
        print(f"turns:     {sum(1 for m in messages if m['role'] == 'user')}")

    print(render_leo_banner())
    print_status()
    print("type /help to list commands")

    while True:
        try:
            user_input = input("you> ").strip()
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
        if user_input == "/tools":
            for t in TOOLS_SCHEMA:
                fn = t["function"]
                print(f"  {fn['name']}: {fn['description']}")
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

        messages.append({"role": "user", "content": user_input})
        while True:
            msg = llm.chat(messages, enable_thinking=think_on, tools=TOOLS_SCHEMA)
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
                print(f"leo> {msg.content}")
                break
            ctx = ToolContext(workspace=workspace, net_on=net_on)
            for tc in msg.tool_calls:
                result = dispatch(tc.function.name, tc.function.arguments, ctx)
                preview = result if len(result) <= 200 else result[:200] + "..."
                print(f"(tool {tc.function.name}({tc.function.arguments}) -> {preview})")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )


if __name__ == "__main__":
    main()

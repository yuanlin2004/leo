from __future__ import annotations

import argparse
from pathlib import Path

from leo.cli.banner import render_leo_banner
from leo.core.llm import LLM

DEFAULT_SYSTEM_PROMPT = "You are Leo, a helpful assistant."

COMMANDS_HELP = (
    "commands:\n"
    "  /help        show this help\n"
    "  /exit, /quit exit the chatbot\n"
    "  /reset       clear conversation history\n"
    "  /think-on    enable model thinking\n"
    "  /think-off   disable model thinking\n"
    "  /status      show model, base_url, thinking state, turn count"
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

    print(render_leo_banner())
    print(COMMANDS_HELP)

    llm = LLM()
    think_on = True
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

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
        if user_input == "/status":
            print(f"model:    {llm.model}")
            print(f"base_url: {llm.base_url}")
            print(f"thinking: {'on' if think_on else 'off'}")
            print(f"turns:    {sum(1 for m in messages if m['role'] == 'user')}")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = llm.chat(messages, enable_thinking=think_on)
        messages.append({"role": "assistant", "content": reply})
        print(f"leo> {reply}")


if __name__ == "__main__":
    main()

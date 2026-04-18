from __future__ import annotations

from leo.cli.banner import render_leo_banner
from leo.core.llm import LLM

SYSTEM_PROMPT = "You are Leo, a helpful assistant."


def main() -> None:
    print(render_leo_banner())
    print(
        "commands: /exit, /reset, /think-on, /think-off, /status"
    )

    llm = LLM()
    think_on = True
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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

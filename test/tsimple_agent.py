import os

from leo.agents import SimpleAgent
from leo.core import LeoLLMClient, LeoLLMException


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Missing OPENROUTER_API_KEY.")
        return

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("TAVILYKEY"):
        print("Missing TAVILY_API_KEY (or TAVILYKEY fallback).")
        return

    llm = LeoLLMClient(
        model="google/gemini-3-flash-preview",
        provider="openrouter",
        temperature=0.2,
    )
    agent = SimpleAgent(name="leo-simple-agent", llm=llm)

    user_input = "Find the current temperature in San Francisco and that in Shanghai and tell me which is higher today."
    try:
        result = agent.run(user_input=user_input, max_iterations=6)
    except LeoLLMException as exc:
        print(f"LLM error: {exc}")
        return
    except Exception as exc:
        print(f"Agent run failed: {exc}")
        return

    print("Prompt:", user_input)
    print("\nAgent response:\n")
    print(result)


if __name__ == "__main__":
    main()

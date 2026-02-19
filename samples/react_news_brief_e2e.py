from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow running directly from repo root without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leo import LeoLLMClient
from leo.agents import ReActAgent
from leo.tools.registry import ToolsRegistry


class TracingLLM:
    def __init__(self, inner: LeoLLMClient) -> None:
        self._inner = inner
        self._turn = 0
        self._color_enabled = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self._input_color = "\033[94m"
        self._response_color = "\033[92m"
        self._reset = "\033[0m"

    def _styled(self, text: str, *, is_input: bool) -> str:
        if not self._color_enabled:
            return text
        color = self._input_color if is_input else self._response_color
        return f"{color}{text}{self._reset}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self._turn += 1
        print(self._styled(f"\n[request turn {self._turn} messages]", is_input=True))
        print(self._styled(json.dumps(messages, indent=2, default=str), is_input=True))
        response = self._inner.complete(messages=messages, tools=tools)

        print(self._styled(f"\n[assistant turn {self._turn}]", is_input=False))
        if getattr(response, "content", None):
            print(self._styled(response.content, is_input=False))
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(tool_call.function.name for tool_call in tool_calls)
            print(self._styled(f"[tool calls] {names}", is_input=False))

        payload = response.model_dump() if hasattr(response, "model_dump") else {
            "content": response.content,
            "tool_calls": [
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(getattr(call, "function", None), "name", None),
                    "arguments": getattr(getattr(call, "function", None), "arguments", None),
                }
                for call in tool_calls
            ],
        }
        print(self._styled(f"\n[assistant turn {self._turn} full response]", is_input=False))
        print(self._styled(json.dumps(payload, indent=2, default=str), is_input=False))
        return response


def _default_task() -> str:
    return (
        "Prepare a concise AI weekly brief with the most recent signals on "
        "OpenAI, Anthropic, and NVIDIA, including citations."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end ReAct sample chaining web_search, source_normalizer, "
            "date_guard, and brief_writer skills."
        )
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LEO_PROVIDER", "openrouter"),
        help="LLM provider (openrouter or ollama).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LEO_MODEL", "openai/gpt-4o-mini"),
        help="Model ID for the selected provider.",
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(".agents/skills").resolve()),
        help="Path to skills root directory.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=14,
        help="Maximum ReAct loop iterations.",
    )
    parser.add_argument(
        "--task",
        default=_default_task(),
        help="User task prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (os.getenv("TAVILY_API_KEY") or os.getenv("TAVILYKEY")):
        raise SystemExit("Missing Tavily key. Set TAVILY_API_KEY (or TAVILYKEY).")

    registry = ToolsRegistry(skills_root=args.skills_root)
    required = {"web_search", "source_normalizer", "brief_writer", "date_guard"}
    available = {item["name"] for item in registry.list_available_skills()}
    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"Missing required skills under {args.skills_root}: {missing}")

    llm = LeoLLMClient(model=args.model, provider=args.provider, temperature=0.2)
    agent = ReActAgent(
        name="react-news-brief",
        llm=TracingLLM(llm),
        tools_registry=registry,
        extra_system_prompt="""
For this task, run an explicit multi-step ReAct workflow:
1) Call list_available_skills.
2) Load skills with get_skill_details in this order:
   web_search, source_normalizer, date_guard, brief_writer.
3) Call web_search at least 3 times for targeted sub-queries.
4) Use dedupe_sources, filter_by_date, and rank_by_relevance on collected findings.
5) Use validate_recency and resolve_relative_dates where date wording is ambiguous.
6) Build final output with build_brief and format_citations.
7) End with Final Answer.
""",
    )

    print("== ReAct News Brief E2E Sample ==")
    print(f"provider={args.provider} model={args.model}")
    print(f"skills_root={args.skills_root}")
    print(f"task={args.task}")

    final_answer = agent.run(args.task, max_iterations=args.max_iterations)
    print("\n[final answer]")
    print(final_answer)


if __name__ == "__main__":
    main()

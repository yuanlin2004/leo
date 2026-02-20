from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running directly from repo root without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leo import LeoLLMClient
from leo.agents import ReActAgent
from leo.core import configure_leo_logging
from leo.tools.registry import ToolsRegistry


def _default_task() -> str:
    return (
        "Find the latest Python release information and one recent headline "
        "about OpenAI. Compare recency and summarize key points."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end ReAct sample using lazy-loaded web_search skill "
            "with a multi-step tool flow."
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
        default=10,
        help="Maximum ReAct loop iterations.",
    )
    parser.add_argument(
        "--task",
        default=_default_task(),
        help="User task prompt.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LEO_LOG_LEVEL", "INFO"),
        help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_leo_logging(args.log_level, leo_only=True)
    logger = logging.getLogger("leo.samples.react_web_search_e2e")

    if not (os.getenv("TAVILY_API_KEY") or os.getenv("TAVILYKEY")):
        raise SystemExit(
            "Missing Tavily key. Set TAVILY_API_KEY (or TAVILYKEY) before running."
        )

    registry = ToolsRegistry(skills_root=args.skills_root)
    skill_names = {item["name"] for item in registry.list_available_skills()}
    if "web_search" not in skill_names:
        raise SystemExit(
            f"web_search skill not found under {args.skills_root}. "
            "Expected .agents/skills/web_search/SKILL.md."
        )

    llm = LeoLLMClient(model=args.model, provider=args.provider, temperature=0.2)
    agent = ReActAgent(
        name="react-web-search",
        llm=llm,
        tools_registry=registry,
        extra_system_prompt="""
For this task, explicitly use a multi-step ReAct flow:
1) Call list_available_skills to discover skills first.
2) Call get_skill_details with skill_name="web_search".
3) Call web_search at least two times with different targeted queries.
4) Then return Final Answer.
""",
    )

    logger.info("== ReAct E2E Sample ==")
    logger.info("provider=%s model=%s", args.provider, args.model)
    logger.info("skills_root=%s", args.skills_root)
    logger.info("task=%s", args.task)

    final_answer = agent.run(args.task, max_iterations=args.max_iterations)

    logger.info("\n[final answer]")
    logger.info("%s", final_answer)


if __name__ == "__main__":
    main()

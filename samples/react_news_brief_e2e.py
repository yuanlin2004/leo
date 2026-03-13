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
from leo.core.env import load_project_env
from leo.tools.registry import ToolsRegistry


def _default_task() -> str:
    return (
        "Prepare a concise AI weekly brief with the most recent signals on "
        "OpenAI, Anthropic, and NVIDIA, including citations."
    )


def _parse_args() -> argparse.Namespace:
    load_project_env()
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
    parser.add_argument(
        "--log-level",
        default=os.getenv("LEO_LOG_LEVEL", "INFO"),
        help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_leo_logging(args.log_level, leo_only=True)
    logger = logging.getLogger("leo.samples.react_news_brief_e2e")

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
        llm=llm,
        tools_registry=registry,
        extra_system_prompt="""
For this task, run an explicit multi-step ReAct workflow:
1) Call list_available_skills.
2) Activate skills in this order:
   web_search, source_normalizer, date_guard, brief_writer.
3) Call web_search at least 3 times for targeted sub-queries.
4) Use dedupe_sources, filter_by_date, and rank_by_relevance on collected findings.
5) Use validate_recency and resolve_relative_dates where date wording is ambiguous.
6) Build final output with build_brief and format_citations.
7) End with Final Answer.
""",
    )

    logger.info("== ReAct News Brief E2E Sample ==")
    logger.info("provider=%s model=%s", args.provider, args.model)
    logger.info("skills_root=%s", args.skills_root)
    logger.info("task=%s", args.task)

    final_answer = agent.run(args.task, max_iterations=args.max_iterations)
    logger.info("\n[final answer]")
    logger.info("%s", final_answer)


if __name__ == "__main__":
    main()

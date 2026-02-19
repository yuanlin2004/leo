from __future__ import annotations

from leo.tools.web_search import WebSearchError, web_search


def main() -> None:
    try:
        response = web_search(
            "Yuan Lin NVIDIA",
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except WebSearchError as exc:
        print(f"web_search failed: {exc}")
        return

    print(response)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from collections import defaultdict, OrderedDict
from pathlib import Path


INPUT_PATH = Path("code.txt")
OUTPUT_PATH = Path("code-dict.txt")


def extract_solution_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    parts = text.split("###Solution Code")
    for part in parts[1:]:
        block, *_ = part.split("###Request", 1)
        stripped_block = block.strip()
        if stripped_block:
            blocks.append(stripped_block)
    return blocks


def normalize_comment_line(line: str) -> str:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return line.rstrip()
    return stripped[1:].lstrip().rstrip()


def collect_comment_code_pairs(block: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    comment_lines: list[str] = []
    code_lines: list[str] = []

    def flush_pair() -> None:
        if not comment_lines or not code_lines:
            return
        comment = "\n".join(comment_lines).strip()
        code = "\n".join(code_lines).rstrip()
        if comment and code:
            pairs.append((comment, code))

    for raw_line in block.splitlines():
        stripped = raw_line.lstrip()
        is_comment = stripped.startswith("#")

        if is_comment:
            if code_lines:
                flush_pair()
                comment_lines = []
                code_lines = []
            comment_lines.append(normalize_comment_line(raw_line))
            continue

        if comment_lines:
            if stripped == "":
                if code_lines:
                    code_lines.append(raw_line.rstrip())
                continue
            code_lines.append(raw_line.rstrip())

    flush_pair()
    return pairs


def build_comment_dictionary(text: str) -> OrderedDict[str, list[str]]:
    comment_map: defaultdict[str, list[str]] = defaultdict(list)
    for block in extract_solution_blocks(text):
        for comment, code in collect_comment_code_pairs(block):
            if code not in comment_map[comment]:
                comment_map[comment].append(code)

    return OrderedDict((comment, comment_map[comment]) for comment in sorted(comment_map))


def main() -> None:
    text = INPUT_PATH.read_text(encoding="utf-8")
    comment_dictionary = build_comment_dictionary(text)
    OUTPUT_PATH.write_text(
        json.dumps(comment_dictionary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

# Subagent Design v0.01

## Goal

Let the main Leo agent delegate a bounded sub-task to a fresh Leo instance via a
tool call. The parent sees only a single result string; the child runs an
independent conversation under its own turn/depth caps.

Task mode already demonstrates the primitive: a prompt goes in, a final reply
comes out, no user interaction in between. A subagent is the same thing,
invoked by the LLM rather than the CLI.

## Tool surface

New tool: `spawn_subagent`.

```
spawn_subagent(prompt: str, max_turns?: int = 20) -> str
```

Parameters:

- `prompt` (required) — the task the subagent should complete. The subagent
  sees this as its first user message.
- `max_turns` (optional, default 20) — cap on assistant roundtrips. When
  reached, the subagent stops and returns whatever it produced plus a note.

Return value: the subagent's final reply text. On error or truncation, a
string starting with `ERROR:` or `TRUNCATED:` so the parent can branch on it.

v1 keeps the surface minimal. Optional future knobs (`skills`, `think`, `net`,
`system_prompt_override`) can be added once we see real usage.

## Isolation decisions

| Concern          | v1 decision                                  | Rationale                                                                                    |
| ---------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Message history  | Fresh `[system, user]`                       | Parent's history is not visible — the whole point is a clean context.                        |
| System prompt    | Inherit parent's (including skills preamble) | Parent may have been launched with `-sysprompt`; children should run in the same "persona".  |
| Skills           | Inherit all                                  | Skills are metadata — cheap to share, and restricting them silently would be surprising.     |
| Tool set         | Same as parent, including `spawn_subagent`   | Recursion allowed but depth-capped.                                                          |
| LLM / model      | Share parent's `LLM` instance                | One model config per Leo run. `last_total_tokens` already reflects only the most recent call. |
| Workspace        | Share parent's                               | Subagent is expected to read/write the same files. Scratch dir is overkill for v1.           |
| Network (`net_on`)| Inherit                                      | Same trust boundary as parent.                                                               |
| Thinking         | Inherit parent's `think_on`                  | Consistent with task mode's handling.                                                        |
| Depth cap        | Default max depth 2                          | Parent → child is OK; grandchild blocked. Prevents runaway recursion.                        |
| Turn cap         | `max_turns`, default 20                      | Guards against tool-loop pathologies inside the subagent.                                    |
| Visibility to user| Honor parent's `show_tool_call` only        | Same UX as any other tool result. No separate `/show-subagent` flag in v1.                   |
| Visibility to LLM | Final reply only                            | Parent does not see child's monologue — that is the isolation we want.                       |

## Execution flow

`spawn_subagent(prompt, max_turns)` running in `ToolContext` with depth `d`:

1. If `d >= MAX_DEPTH`, return `"ERROR: subagent depth limit (d=<d>) reached"`.
2. Build `messages = [{system: ctx.system_prompt}, {user: prompt}]`.
3. Build child `ToolContext` with `depth = d + 1`, inherited workspace / skills
   / net / llm / system_prompt / think_on.
4. Run a bounded version of `run_turn`: after each assistant roundtrip,
   increment a counter; if it reaches `max_turns`, break and return
   `"TRUNCATED: <last reply text>"`.
5. Collect tool records and think chunks locally. Discard after the call
   returns — parent sees only the final reply string.
6. Catch exceptions, return `"ERROR: <ExceptionType>: <msg>"`.

## Code shape

Changes required (no implementation yet):

1. `ToolContext` (`src/leo/core/tools/__init__.py`) — new fields, all with
   safe defaults so existing call sites still compile:
   - `llm: LLM | None = None`
   - `system_prompt: str = ""`
   - `think_on: bool = True`
   - `depth: int = 0`
   - `MAX_DEPTH` constant (module-level, = 2).

2. `run_turn` (`src/leo/cli/leo.py`) — gains optional `max_turns: int | None`.
   When set, after each assistant message it increments a counter; returns
   early with `(reply_text, "truncated")` if the cap is hit. Signature tweak:
   return `tuple[str, str]` = (reply, status) where status ∈
   `{"done", "truncated"}`. Task mode and chat mode ignore the status.

3. New module `src/leo/core/tools/subagent.py`:
   - `SCHEMA` — OpenAI-style tool schema for `spawn_subagent`.
   - `FUNCTIONS = {"spawn_subagent": _spawn_subagent}`.
   - `_spawn_subagent(ctx, prompt, max_turns=20)` — the handler described in
     *Execution flow* above. Imports `run_turn` lazily to avoid the
     `cli → tools → cli` import cycle (or we lift `run_turn` into
     `src/leo/core/` — see "Alternatives").

4. `tools/__init__.py` — register the new tool in `TOOLS_SCHEMA` /
   `TOOL_FUNCTIONS`.

5. Wire-in sites that build a `ToolContext`:
   - `run_task_mode` (`src/leo/cli/leo.py`)
   - chat-mode turn loop (`src/leo/cli/leo.py`)
   Both need to pass `llm`, `system_prompt`, `think_on`, `depth=0`.

Nothing else moves. `run_turn`, `_parse_task_file`, `TOGGLES`, chat/task modes
— all unchanged in shape.

## Alternatives considered

- **Extract `run_turn` into `src/leo/core/runner.py`.** Cleaner module layout
  (`cli` should not be imported by `tools`) at the cost of one file move. I
  would do this as part of implementation rather than a separate refactor.

- **Separate LLM instance per subagent.** Would give independent token
  accounting but doubles client overhead and complicates config. Reject for
  v1; revisit if we want per-agent model selection.

- **Return a structured result** (e.g. JSON with `final_reply`,
  `tool_calls`, `status`). More informative, but the LLM can only consume
  strings and would then have to parse JSON to react. A plain string reply
  matches how every other tool returns. Reject for v1.

- **Expose a `/subagent <prompt>` CLI command.** Redundant — `--task` already
  does this from the outside. The tool form is specifically for LLM-initiated
  delegation.

## Open questions

1. **System prompt inheritance.** Inheriting means subagents carry whatever
   persona the user set via `-sysprompt`. Is that what we want, or should
   subagents always start from `DEFAULT_SYSTEM_PROMPT + skills`? Leaning
   inherit; easy to change.

2. **Max depth 2 vs 3.** Two is conservative; three allows a planner →
   worker → helper chain. I would start at 2 and raise if we see real need.

3. **Should the subagent's monologue be surfaced to the user (not the LLM)
   when `show_tool_call` is on?** Currently the user would see
   `(tool spawn_subagent(...) -> <final reply preview>)` like any other tool.
   A `/show-subagent` toggle that also dumps the child's think/tool log
   could help debugging — deferrable to v2.

4. **Token budget.** No per-subagent token cap in v1. `max_turns` is the only
   bound. Consider adding `max_tokens` later if we hit context-window pain.

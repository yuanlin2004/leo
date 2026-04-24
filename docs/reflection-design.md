# Reflection Design v0.01

## Goal

Let Leo learn from its own traces. After a task ends (or on demand), Leo
studies the trace — user prompt, its own thinking and tool calls, user
corrections, final outcome — extracts lessons, and writes them to a
persistent lessons database. On future tasks, relevant lessons are
auto-injected into the system prompt so the same mistakes do not repeat.

Reflection is deliberately separate from conversational memory. Memory
answers "what do I know?"; reflection answers "how should I behave?"

## Terminology

- **Trace.** The in-memory `messages` list for the reflected-on span:
  system prompt, user turns, assistant messages (including `<think>`),
  tool calls, tool results.
- **Lesson.** A single behavioral rule derived from a trace. Has a
  category, a scope, triggers, and a rationale. Lives as one markdown
  file on disk plus one index entry.
- **Reflector.** A harness-invoked LLM pass that reads a trace and
  produces lesson writes. Not a tool the main agent calls.
- **Lessons DB.** The on-disk collection under `~/.leo/lessons/`.
  Grep-able, diffable, human-editable.


## Lifecycle — when reflection runs

Reflection fires at exactly two triggers:

1. **Explicit `/reflect` command** in the REPL.
2. **Clean session exit** via `/exit` or `/quit`.

Opt-out forms: `/exit noref` and `/quit noref` skip the end-of-session
pass. Unclean exit (Ctrl-C, EOF, process kill, uncaught exception)
skips reflection — best-effort, not at-all-costs.

Task mode (`--task`) is out of scope for v1. It has no human in the
loop, so approval gating (see below) cannot apply. Revisit once task
mode has a way to attach a reviewer.

## Unit of reflection

The span from the last reflection boundary through the triggering turn.
Boundaries are:

- Start of the session (first user prompt after startup or `/reset`).
- The previous `/reflect` call.

This means an interactive session can produce multiple reflection
passes, each covering only new material. The REPL holds a
`last_reflection_idx: int` into the `messages` list and advances it on
every successful reflection.

## Lesson schema

One markdown file per lesson under `~/.leo/lessons/<id>.md`:

```markdown
---
id: <slug>                          # filename without .md
title: <short phrase>
category: preference | process | gotcha | fact
scope:
  - type: global | project | skill | tool | model
    value: <predicate value>        # e.g. /home/yuan/git/my/leo
triggers: [keyword, keyword, ...]   # cheap retrieval filter
applies_when: <one-line condition>  # human-readable self-gate
created: <ISO date>
updated: <ISO date>
source_trace: artifacts/<ts>-<slug>.json  # relative to the lesson's root
---

Rule: <one sentence>
Why: <the mistake / correction that motivated this>
How to apply: <when this kicks in>
```

Character limits (model-independent): rule ≤ 200 chars, why ≤ 400,
how-to-apply ≤ 400. The reflector is told these limits.

### Categories

Each lesson belongs to exactly one of four categories. Category drives
retrieval behavior, not just taxonomy.

| Category     | Example                                        | Retrieval behavior                                         |
| ------------ | ---------------------------------------------- | ---------------------------------------------------------- |
| `preference` | "User prefers terse replies, no summary."      | Unconditional at session start. Not keyword-gated.         |
| `process`    | "Run tests after every edit to `core/*`."      | Keyword match against user prompt and running monologue.   |
| `gotcha`     | "vLLM drops requests > 4096 tokens."           | Keyword match against next tool call + args, pre-dispatch. |
| `fact`       | "Skills loader lives in `core/skill_core.py`." | Keyword match against user prompt.                         |

The four categories are fixed. Open-vocabulary categories give the
reflector freedom but couple retrieval logic to LLM output.

### Scope

A lesson's `scope` is a list of predicates. All must match the current
context. Empty list = global.

Predicate types:

- `global` — implicit empty list.
- `project` — value is an absolute path; matches when `cwd` is within.
- `skill` — value is a skill name; matches when that skill is loaded.
- `tool` — value is a tool name; matches when that tool is about to
  fire (gotcha category) or has fired recently (other categories).
- `model` — value is a model id; matches when `LEO_LLM_MODEL` equals it.

More predicate types can be added later without a schema break.

## Storage layout

Lessons load from an ordered **list of roots**, not a single folder.
Each root is self-contained — its own `index.json`, its own
`artifacts/` — and has the same internal layout:

```
<root>/
  index.json                  # flat array of {id, title, category, scope, triggers, updated}
  <id>.md                     # one file per lesson
  artifacts/
    <timestamp>-<slug>.json   # trace snapshots referenced by source_trace
```

In v1 the list is `[~/.leo/lessons/]` only. The loader iterates the
list; retrieval merges results across roots. When two roots contain
the same `id`, the earlier root wins.

Writes always target a specific root. v1 writes go to
`~/.leo/lessons/`. The reflector prompt will later include the target
root so it can be chosen per-lesson (e.g. project-scoped lessons →
project root).

This design accommodates two growth paths without schema change:

1. **Project-local lessons.** Add `<cwd>/.leo/lessons/` to the root
   list when present, so project-scoped lessons can live in the repo
   and travel with it. Mirrors the `~/.leo/skills/` vs in-repo
   `skills-repo/` split already used for skills.
2. **Sharding.** When `~/.leo/lessons/` grows uncomfortable, split it
   into subdirs (by scope, by category, by date — whatever) and
   register each subdir as its own root.

The index within a root is read once at startup and refreshed after
every write to that root. Lesson bodies are loaded on demand — only
when a lesson is actually selected for injection. The full DB never
lives in memory.

Trace snapshots are written at reflection time and referenced by the
`source_trace` field for audit. If LangSmith tracing is on, the trace
URL is appended alongside the path.

## Retrieval pipeline

Runs in two places.

### At turn start

On each new user prompt:

1. **Preferences.** Inject every `preference` lesson whose scope
   matches. No keyword check. These go into the system prompt.
2. **Scope filter** the rest of the index.
3. **Keyword match** `process` and `fact` lessons against the user
   prompt (tokenized, case-insensitive substring match against
   `triggers`).
4. **Rank** `(scope_specificity desc, match_count desc, updated desc)`.
   Specificity = length of `scope` list (more predicates = more
   specific).
5. **Cap** top-K (default 5) plus token budget (default ~1500 tokens).
   Preferences do not count against the cap.

Selected lessons are rendered into a system-prompt block and the block
is included in the session's frozen system prompt. This prompt is set
at session start and is not mutated mid-session — prefix cache stays
stable.

### At each loop iteration (mid-task)

After the LLM emits a response but before the harness dispatches the
next tool call:

1. **Gotcha pre-dispatch hook.** For each pending tool call, match
   its `name` and keyword-extracted args against `gotcha` lessons.
2. **Replan if any matched.** Lessons are injected as a system-role
   message and the LLM is re-prompted to revise its action. Without
   this step, an injected lesson would only influence behavior on the
   *next* turn — by which point the gotcha has already fired.
3. **Monologue re-retrieval (post-execution).** After the tool runs,
   keywords from the tool result, assistant content (including
   thinking), and tool call args are matched against all categories.
   New matches (not already injected this turn) are appended as a
   system-role message. No replan here — the next loop iteration
   already calls the LLM, which will see the new lessons naturally.

#### The replan step

When the gotcha hook produces matches:

1. Pop the just-emitted assistant message from `messages` (the one
   containing the pending `tool_calls`).
2. Append a system-role message with the matched lessons, prefixed:

   ```
   [System note: New constraints have been introduced based on your
   pending action. Revise your previous action if needed.]
   ```

3. Call `llm.chat(messages, ...)` again. The LLM either:
   - re-emits revised tool calls — the harness uses these and proceeds;
   - re-emits the same tool calls — the lesson did not apply, proceed;
   - emits text only (no tool calls) — treat as a final reply for the
     turn, like any other tool-free response.
4. The new assistant message goes into `messages` in place of the
   popped one. The original draft is preserved in the trace snapshot
   (for future reflection) but not in conversation history.

Iteration cap: at most **2 replans per tool-call boundary**. After
that the harness logs a warning and dispatches whatever was last
emitted. The per-turn `injected_ids` set prevents the same lesson
from triggering successive replans.

Cost note: replans add an LLM round-trip only when a gotcha actually
matches a pending tool call. In sessions where no gotchas fire, the
loop is unchanged.

#### Cache and dedup

Mid-loop injections never touch the system prompt. This keeps the
prefix cache intact up to the first user message; only the suffix
grows. Borrowed directly from Hermes's frozen-snapshot pattern.

Deduplication: a per-turn `injected_ids: set[str]` prevents the same
lesson from landing twice in one turn (including across replans).

## Injection format

The system-prompt block at session start:

```
## Lessons from prior experience

### Preferences (always apply)
- <rule> — why: <one line>

### Relevant to this task
- <rule> — apply when: <applies_when> — why: <one line>
```

Mid-loop injections use the same rendering but wrapped as a
system-role message:

```
[System note: Additional lessons now in scope based on current
trajectory. Apply where relevant.]
- <rule> — apply when: <applies_when> — why: <one line>
```

Categorized headers give the LLM a usage signal without extra
instruction text.

## Write path — reflector

The reflector is a single LLM call made by the harness, not a tool the
main agent invokes.

### Input

The reflector receives:

1. The trace (messages list for the reflected-on span).
2. The list of existing lessons **in scope** of the current session —
   only `{id, title, category, scope, triggers, applies_when}`, not
   bodies. In-scope means the lesson's scope predicates match the
   current cwd / loaded skills / model.
3. A reflection system prompt that explains the lesson schema, the
   four categories, the character limits, and the output format.

### Output format

The reflector emits zero or more lesson operations, one JSON object
per line:

```json
{"op": "create", "lesson": { ...full lesson yaml as json... }}
{"op": "update", "id": "<existing-id>", "lesson": { ...fields to overwrite... }}
{"op": "skip", "reason": "<why nothing was learned>"}
```

The reflector is allowed — and encouraged — to emit nothing. "Nothing
worth saving" is a valid outcome. Over-eager reflectors pollute the DB.

### Correction detection

The reflector itself decides what counts as a correction from the
trace. No heuristic on user text, no `/correct` command in v1. The
reflection prompt tells it to look for user messages that follow
assistant output and express dissatisfaction, contradiction, or
redirection — and to weight those most heavily.

### Writing

For each operation:

1. **Create.** Generate slug from title (kebab-case, dedup with
   numeric suffix). Write `<slug>.md`, append index entry.
2. **Update.** Load existing lesson body, overlay provided fields,
   bump `updated`, rewrite file, refresh index entry.
3. **Skip.** No-op.

Trace snapshot is always written (one per reflection pass, not one
per lesson) to `artifacts/<timestamp>-<slug>.json`; all created or
updated lessons point at the same snapshot.

## User review

Before any write hits disk, the proposed operations are shown to the
user:

```
=== Reflection proposal ===
[1] CREATE gotcha: mvmb-sheets-need-export-link
    Rule: The "published" Google Sheets URL hides empty rows; use
          /export?format=csv for raw data.
    Triggers: google sheets, mvmb, booster
    Scope: global
    
[2] UPDATE preference: terse-replies
    + append: "No trailing summary after code edits."

Apply all? [y/n/edit/skip-<n>]
```

Interactive defaults:

- `y` — apply all.
- `n` — discard all.
- `edit` — open each proposal in `$EDITOR`.
- `skip-<n>` — drop proposal N, apply the rest.

Non-interactive (task mode, future): `--auto` flag bypasses review.
Not applicable in v1 since reflection does not run in task mode.

## Content safety

Lesson bodies are injected into the system prompt. They are attack
surface. Every write passes a threat scan (borrowed pattern from
Hermes `tools/memory_tool.py`):

- Known prompt-injection phrases ("ignore previous instructions",
  "you are now...", etc.).
- Exfiltration patterns (`curl ... $API_KEY`, `cat ~/.env`, etc.).
- Invisible unicode (ZWSP, BIDI overrides, etc.).

Matches are rejected with an error surfaced during the review step.
The reflector can retry.

## REPL commands

```
/reflect                    trigger reflection now
/lessons                    list all lessons (id, title, category, scope)
/lessons show <id>          print full lesson body
/lessons edit <id>          open lesson in $EDITOR
/lessons forget <id>        delete lesson + index entry (not the trace)
/lessons search <query>     keyword search over titles + triggers
/exit, /quit                clean exit; runs reflection
/exit noref, /quit noref    clean exit; skip reflection
```

## State additions

REPL `state` dict gains:

- `reflect_on: bool = True` — session-level kill switch.
- `last_reflection_idx: int = 1` — index into `messages` marking the
  last reflection boundary. Starts at 1 (after system message).

Nothing in `run_turn` changes. Reflection is a separate function
invoked at lifecycle boundaries, not inside the turn loop.

## Code shape

Rough module layout (not implementation):

```
src/leo/core/
  lessons/
    __init__.py          # LessonStore: multi-root load/save/index/search
    schema.py            # Lesson dataclass, category enum, scope predicate
    retrieval.py         # keyword match, scope filter, ranking
    injection.py         # render system-prompt block + mid-loop messages
    reflector.py         # the LLM pass: build prompt, parse ops, apply
    safety.py            # threat-pattern scan (cf. hermes memory_tool.py)
```

`src/leo/cli/leo.py` grows:

- Retrieval hook on user-prompt receipt, before `run_turn`.
- Reflection dispatcher on `/reflect` and on `/exit` / `/quit`.
- `/lessons ...` command handlers.

`run_turn` gains one optional parameter: a `LessonStore` reference.
When `None` (e.g. in task mode for v1), all mid-loop hooks are
no-ops. When set, the loop body becomes:

1. `llm.chat(...)` → assistant message.
2. If the assistant message has tool calls and gotchas match, run
   the **replan** subloop: pop the message, append the lesson
   system-note, re-call `llm.chat(...)`, repeat until no new gotchas
   match or the replan cap (2) is hit.
3. Dispatch the (possibly revised) tool calls.
4. Run **monologue re-retrieval** against the assistant content +
   tool args + tool results; append matched lessons as a
   system-role message.
5. Loop.

The replan subloop is a new internal helper, not exposed outside
`run_turn`.

## Alternatives considered

- **LLM-driven retrieval via `load_lesson` tool.** Parallel to
  skills. Rejected: lessons are short enough that lazy-loading has no
  value, and relying on the LLM to remember to look risks missed
  lessons. Harness-driven retrieval is deterministic.
- **Embedding-based retrieval.** More robust than keyword match for
  paraphrase, but adds a dependency and an index to maintain. The
  `triggers` field is a clean seam — swap in embeddings later without
  a schema change.
- **Reflection as a tool the LLM calls mid-task.** Tempting but wrong.
  The agent cannot evaluate its own mistakes while still making them;
  reflection is a lifecycle operation.
- **Continuous LLM-driven saves** (Hermes pattern). Shifts quality
  risk to the main-loop model. For local models like Qwen3-35B, a
  dedicated reflector with a single focused prompt is more reliable
  than opportunistic memory writes.
- **Mutating the system prompt mid-session** to re-inject lessons.
  Kills the prefix cache. Suffix-append via system-role messages
  achieves the same visibility without the cache hit.
- **Correction detection via text heuristics** or an explicit
  `/correct` command. Rejected in favor of letting the reflector
  judge from the full trace. Heuristics are brittle; explicit
  commands require user discipline.

## Prior art

Hermes Agent (`NousResearch/hermes-agent`) informed three decisions:

1. **Frozen system-prompt snapshot** for prefix cache stability
   (`tools/memory_tool.py`). Adopted.
2. **Pluggable memory provider abstraction**
   (`agent/memory_provider.py`). Not adopted in v1 (file-backed only),
   but the module layout above leaves room.
3. **Threat scan on injected content** (`tools/memory_tool.py`).
   Adopted — lessons are system-prompt attack surface.

Hermes itself does not have a correction-driven reflection pass. Its
built-in memory is LLM-driven continuous write of facts and
preferences. That is a reasonable design for a larger frontier model
but leaves the "learn from mistakes" loop unfilled — which is what
this design adds.

## Open questions

1. **Preference promotion.** A `process` lesson that recurs across
   sessions ("user corrected this 3 times now") arguably should be
   promoted to `preference`. Is that a responsibility of the reflector
   per-run, or a separate consolidation pass? Leaning separate pass,
   out of scope for v1.
2. **Keyword extraction for mid-loop re-retrieval.** Naive token
   split is probably enough but may be noisy on tool-result blobs.
   Start naive, measure.
3. **Character-limit enforcement.** The reflector may exceed the
   limits. Truncate silently, reject and reprompt, or surface to the
   user in review? Leaning reject-and-reprompt once, then surface if
   still over.
4. **Cross-project lessons.** A lesson learned in project A about
   Google Sheets behavior is probably useful in project B. Default
   scope is `global`, but does the reflector know when to use it vs.
   project-scope? The reflection prompt needs explicit guidance here.
5. **Index corruption.** If `index.json` and the `.md` files disagree
   (hand-edit, crashed write), which wins? Leaning: rebuild index
   from files on startup if parse fails or count mismatch.
6. **Lesson versioning.** Update overwrites history. Do we keep a
   git-backed `~/.leo/lessons/.git/`? Not in v1 — users who want
   history can `git init` the directory themselves.
7. **Replan cap of 2.** Picked by feel. If gotchas are well-scoped,
   one replan should cover almost every case; two is a safety
   margin. Worth measuring. If we see frequent cap hits, that is a
   signal the gotchas themselves are too broad, not that the cap is
   too low.

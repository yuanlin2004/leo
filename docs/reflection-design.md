# Reflection Design v0.02

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
  trigger (when to fire), a scope (when eligible), and a rationale.
  Lives as one markdown file on disk plus one index entry.
- **Trigger.** The condition that activates a lesson and the
  mechanism by which it is injected. One of four types: `always`,
  `on_prompt`, `on_monologue`, `on_tool_call`. Drives operational
  behavior.
- **Scope.** Session-level eligibility predicates (project, skill,
  model). Static for the duration of a session. Independent of
  trigger.
- **Category.** What *kind* of knowledge the lesson is — about the
  user (`preference`), about the world or codebase (`fact`), about
  workflow (`process`), or about a pitfall (`gotcha`). Drives the
  on-disk folder. Independent of trigger; a `process` lesson may
  fire `on_prompt`, `on_monologue`, or `on_tool_call` depending on
  when the workflow rule needs to land.
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

One markdown file per lesson under
`<root>/<category>/<id>.md`. The folder name equals the lesson's
`category` field (see Categories below); `trigger` is a separate
field.

```markdown
---
id: <slug>                          # filename without .md
title: <short phrase>
category: preference | fact | process | gotcha
trigger:
  type: always | on_prompt | on_monologue | on_tool_call
  keywords: [keyword, ...]          # required for on_prompt, on_monologue, on_tool_call
  tool: <tool name>                 # optional, only for on_tool_call
scope:                              # omit or leave empty for global
  project: <string or list of strings>
  skill:   <string or list of strings>
  model:   <string or list of strings>
created: <ISO date>
updated: <ISO date>
source_trace: artifacts/<ts>-<slug>.json  # relative to the lesson's root
---

## Rule
<one sentence>

## Why
<the mistake / correction that motivated this>

## How to apply
<when this kicks in>
```

Character limits (model-independent): rule ≤ 200 chars, why ≤ 400,
how-to-apply ≤ 400. The reflector is told these limits.

### Categories

What kind of knowledge a lesson holds. Used for on-disk grouping and
human-facing organization (`/lessons` listing). The four names are
fixed; the reflector picks one per lesson.

| Category     | Meaning                                          | Typical example                                       |
| ------------ | ------------------------------------------------ | ----------------------------------------------------- |
| `preference` | About the user — style, tone, working habits.    | "User wants terse replies, no trailing summary."      |
| `fact`       | About the world — codebase, APIs, environment.   | "Skills loader lives in `core/skill_core.py`."        |
| `process`    | About workflow — what to do, in what order.      | "Run tests after every edit to `core/*`."             |
| `gotcha`     | About pitfalls — things that bite if ignored.    | "vLLM drops requests > 4096 tokens."                  |

Open-vocabulary categories would give the reflector freedom but make
storage layout depend on LLM output.

### Triggers

When a lesson fires and how it is injected. Independent of category —
any category can use any trigger that fits its activation condition.
The harness dispatches purely on `trigger.type`; categories never
appear in the dispatch logic.

| Trigger          | Evaluated                                      | Injected as                          | Replans? |
| ---------------- | ---------------------------------------------- | ------------------------------------ | -------- |
| `always`         | Once at session start                          | Frozen system-prompt block           | No       |
| `on_prompt`      | Turn start, against the new user message       | Suffix-appended system-role message  | No       |
| `on_monologue`   | After each LLM response and each tool result   | Suffix-appended system-role message  | No       |
| `on_tool_call`   | Pre-dispatch, against pending tool call + args | Suffix-appended system-role message  | **Yes**  |

Trigger semantics:

- `always` — eligibility (scope) is checked once at session start;
  if scope matches, the lesson is folded into the frozen system
  prompt unconditionally.
- `on_prompt` — keyword match (case-insensitive substring) against
  the new user prompt. Always injected as a suffix-appended
  system-role message in v1. (Folding into the frozen system prompt
  on the first user prompt is a future optimization — see Open
  questions.)
- `on_monologue` — keyword match against assistant content
  (including thinking), tool call args, and tool results as they
  accrue. Suffix-appended.
- `on_tool_call` — match by `tool` name (if specified) and/or
  keyword in args (if specified). At least one of `tool` or
  `keywords` must be present. Triggers a replan (see below).

Most category × trigger combinations are sensible. A `preference`
that revises a tool call (`preference` × `on_tool_call`) is unusual
but legitimate; a `gotcha` that fires unconditionally
(`gotcha` × `always`) is essentially a chronic warning. The reflector
chooses both fields independently based on what the trace teaches.

### Scope

A lesson's `scope` is a map from predicate-type name to a value list.
**All listed predicates must match** (AND across keys). Within a
single predicate, the lesson matches if the session value matches
**any** entry in the list (OR within the key). A scalar value is
sugar for a singleton list.

Each entry is matched as a **fnmatch-style glob** (`*` = any run of
characters, `?` = single character, `[...]` = character class). A
literal name with no glob metacharacters matches by equality. Match
is case-sensitive and applied to the whole value (not a substring).

Three "absent" forms compose cleanly:

- **Key omitted** → no constraint on that predicate type. Matches
  any session value.
- **Key with empty list `[]`** → wildcard requiring the type be
  "active" (see per-type column below).
- **`scope:` itself empty or omitted** → vacuous AND, matches every
  session (global).

A **null value** for any key (`project:`, `skill:`, or `model:` with
no value) is a **schema error**, not a silent synonym for `[]` or
for omitting the key. The reflector and hand-edits must commit to
one of the three forms above for every key they include.

YAML gotcha: bare `*` in YAML is an alias-reference token, not a
string. Patterns containing only metacharacters must be quoted:
`model: '*'` or `model: ["*"]`. A name with surrounding text
(`claude-opus-4-*`) needs no quoting.

Scope is independent of trigger — a scope-matching lesson may still
wait on its trigger before firing.

Per-predicate match rules:

| Key       | When list has items                                                         | When list is `[]`                                                                 |
| --------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `project` | matches when `$LEO_PROJECT` matches any listed glob                         | matches whenever `$LEO_PROJECT` is set (any value); never matches if unset       |
| `skill`   | matches when any loaded skill name matches any listed glob                  | matches whenever at least one skill is loaded                                     |
| `model`   | matches when `$LEO_LLM_MODEL` matches any listed glob                       | matches whenever `$LEO_LLM_MODEL` is set (always true since the model is set)    |

`$LEO_PROJECT` is the user-set env var that names the current
project (e.g. via direnv or a project `.env`). It is *not* derived
from `cwd` — running Leo from a subdirectory of a project does not
change the project identity. A lesson with a `project` predicate
(with or without listed values) fails to match when `$LEO_PROJECT`
is unset.

The `[]` form for `model` is effectively redundant (the model is
always set). For `project`, `[]` means "must be inside any project
context." Only `skill` has a non-trivially meaningful wildcard.

Examples:

```yaml
# Specific to one project AND one of two models
scope:
  project: leo
  model: [claude-opus-4-7, claude-sonnet-4-6]

# Any Claude Opus 4.x build (glob)
scope:
  model: claude-opus-4-*

# Applies whenever git or github skill is loaded, on any model/project
scope:
  skill: [git, github]

# Applies whenever any skill at all is loaded
scope:
  skill: []
```

Note: there is no `tool` scope key. Tool involvement is a trigger
concept (`on_tool_call`), not session eligibility. More predicate
types can be added later without a schema break.

## Storage layout

Lessons load from an ordered **list of roots**. Each root is
self-contained — its own `index.json`, its own `artifacts/` — and
has the same internal layout:

```
<root>/
  index.json                  # flat array of {id, title, category, trigger, scope, updated, path}
  preference/                 # category = preference (about the user)
    <id>.md
  fact/                       # category = fact (about the world)
    <id>.md
  process/                    # category = process (workflow rules)
    <id>.md
  gotcha/                     # category = gotcha (pitfalls)
    <id>.md
  artifacts/
    <timestamp>-<slug>.json   # trace snapshots referenced by source_trace
```

A lesson file lives in the folder that matches its `category`
field. The loader rejects category-folder mismatches at startup as
corruption (see *Index corruption* in Open questions). `trigger`
plays no role in folder placement — every folder can hold lessons
of any trigger type.

In v1 the root list is `[~/.leo/lessons/]` only. The loader iterates
the list; retrieval merges results across roots. When two roots
contain the same `id`, the earlier root wins.

Writes always target a specific root. v1 writes go to
`~/.leo/lessons/`. The reflector prompt will later include the target
root so it can be chosen per-lesson (e.g. project-scoped lessons →
project root).

This design accommodates two growth paths without schema change:

1. **Project-local lessons.** Add `<cwd>/.leo/lessons/` to the root
   list when present, so project-scoped lessons can live in the repo
   and travel with it. Mirrors the `~/.leo/skills/` vs in-repo
   `skills-repo/` split already used for skills.
2. **Sharding.** When a root grows uncomfortable, register additional
   roots — split by date, project, source, whatever — without
   changing schema or code paths.

The index within a root is read once at startup and refreshed after
every write to that root. Each index entry stores `path` (relative
to the root, e.g. `gotcha/<id>.md`) so the loader does not have to
glob. Lesson bodies are loaded on demand — only when a lesson is
actually selected for injection. The full DB never lives in memory.

Trace snapshots are written at reflection time and referenced by the
`source_trace` field for audit. If LangSmith tracing is on, the trace
URL is appended alongside the path.

## Retrieval pipeline

The pipeline is a mechanical dispatch on `trigger.type`. Every step
below first applies the **scope filter** — a lesson is considered
only if its scope predicates all match the current session.

### Phase 1 — session start

Run once when the REPL starts (or after `/reset`):

1. **Eligibility** — scope-filter the index across all roots.
2. **Activate `always`** — every eligible lesson with
   `trigger.type = always` is selected.

The selected `always` lessons are rendered into the **frozen
system-prompt block** and included in the session's system prompt.
The system prompt is not mutated again during the session — prefix
cache stays stable. (Hermes's pattern.)

### Phase 2 — turn start

Run when each new user message arrives:

1. **Activate `on_prompt`** — match each eligible `on_prompt`
   lesson's `trigger.keywords` against the user message
   (case-insensitive substring).
2. **Rank** `(scope_specificity desc, match_count desc, updated desc)`.
   Specificity = number of keys present in `scope`.
3. **Cap** top-K (default 5) plus token budget (default ~1500 tokens).

Matched lessons are appended as a single system-role message before
the LLM runs the turn. This is **not** added to the frozen system
prompt — it lives in the message list, so different turns can have
different `on_prompt` selections without invalidating the prefix
cache.

### Phase 3 — inside the loop

`run_turn` repeatedly calls the LLM and dispatches tool calls. Two
trigger types fire here.

#### 3a. `on_tool_call` (pre-dispatch, with replan)

After the LLM emits an assistant message with `tool_calls`, but
*before* the harness dispatches them:

1. For each pending tool call, match each eligible `on_tool_call`
   lesson:
   - if `trigger.tool` is set, the call's `name` must equal it;
   - if `trigger.keywords` is set, at least one must appear in the
     call's serialized args;
   - if both are set, both conditions must hold;
   - at least one of the two must be set (schema invariant).
2. If any new lessons matched, run the **replan** subloop (below).
3. Otherwise dispatch the tool calls as today.

The replan subloop:

1. Pop the just-emitted assistant message from `messages`.
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
4. The new assistant message replaces the popped one in `messages`.
   The original draft is preserved in the trace snapshot but not in
   conversation history.
5. If the new tool calls trigger fresh `on_tool_call` matches (other
   than already-injected lessons), repeat. Cap at **2 replans per
   tool-call boundary**; after that, dispatch whatever was last
   emitted and log a warning.

#### 3b. `on_monologue` (post-LLM-response, post-tool-result)

Run at two points within the loop body, in this order:

- **Right after the LLM responds** (before any tool dispatch or
  replan). Match against the assistant content, including thinking.
- **Right after each tool result lands.** Match against the tool
  call name, serialized args, and result text.

In both cases, matched lessons are appended as a system-role message
in place — the next LLM call (the next loop iteration, or the replan
itself) will see them naturally. **No replan from `on_monologue`** —
we never replan based on the agent's own monologue, only on a
pending tool call.

### Cache and dedup

The frozen system prompt holds only Phase 1 (`always`) lessons.
Everything else lives in the suffix. This keeps the prefix cache
stable from the first user message onward; only the tail grows.

Deduplication: a per-session `injected_ids: set[str]` prevents the
same lesson from landing twice across phases or replans. Its scope
is the entire session, not a single turn — once a lesson has been
shown, it does not appear again until `/reset`.

## Injection format

The frozen system-prompt block (Phase 1 only):

```
## Lessons from prior experience

### Always apply
- <rule> — why: <one line>
```

Suffix-appended messages use the trigger type to label intent.

`on_prompt` (Phase 2):

```
[System note: Lessons relevant to this user prompt.]
- <rule> — why: <one line>
```

`on_monologue` (Phase 3b):

```
[System note: Additional lessons now in scope based on current
trajectory. Apply where relevant.]
- <rule> — why: <one line>
```

`on_tool_call` (Phase 3a, replan):

```
[System note: New constraints have been introduced based on your
pending action. Revise your previous action if needed.]
- <rule> — why: <one line>
```

Each block's heading tells the LLM how to weight the content
(reference vs. directive vs. revision request) without extra prose.

## Write path — reflector

The reflector is a single LLM call made by the harness, not a tool the
main agent invokes.

### Input

The reflector receives:

1. The trace (messages list for the reflected-on span).
2. The list of existing lessons **in scope** of the current session —
   only `{id, title, category, trigger, scope, updated}`, not
   bodies. In-scope means the lesson's scope predicates match the
   current `$LEO_PROJECT` / loaded skills / `$LEO_LLM_MODEL`.
3. A reflection system prompt that explains the lesson schema, the
   four categories, the four trigger types and their semantics, the
   character limits, and the output format. The prompt makes
   explicit that category and trigger are independent choices.

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
   numeric suffix). Pick the destination folder from `category`
   (preference / fact / process / gotcha). Write
   `<root>/<category>/<slug>.md`, append index entry with relative
   `path`.
2. **Update.** Load existing lesson body, overlay provided fields,
   bump `updated`, rewrite file, refresh index entry. Changing
   `category` requires moving the file across folders — handle
   atomically (write new, update index, unlink old). Changing
   `trigger.type` is in-place since the folder is determined by
   category, not trigger.
3. **Skip.** No-op.

Trace snapshot is always written (one per reflection pass, not one
per lesson) to `<root>/artifacts/<timestamp>-<slug>.json`; all
created or updated lessons in that pass point at the same snapshot.

## User review

Before any write hits disk, the proposed operations are shown to the
user:

```
=== Reflection proposal ===
[1] CREATE gotcha/mvmb-sheets-need-export-link
    Rule: The "published" Google Sheets URL hides empty rows; use
          /export?format=csv for raw data.
    Category: gotcha
    Trigger: on_tool_call(tool=fetch_url, keywords=[google sheets, mvmb])
    Scope: global

[2] UPDATE preference/terse-replies
    Category: preference
    Trigger: always
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
/lessons                    list all lessons grouped by category folder
/lessons show <id>          print full lesson body
/lessons edit <id>          open lesson in $EDITOR
/lessons forget <id>        delete lesson + index entry (not the trace)
/lessons search <query>     keyword search over titles + trigger keywords
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
    schema.py            # Lesson dataclass, trigger union, scope predicate
    retrieval.py         # scope filter + per-trigger match dispatch
    injection.py         # render frozen block + suffix messages
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
2. Run `on_monologue` retrieval against the assistant content
   (including thinking); append matches as a system-role message.
3. If the assistant message has tool calls and any `on_tool_call`
   lessons match the pending calls, run the **replan** subloop:
   pop the assistant message, append the lesson system-note, re-call
   `llm.chat(...)`, repeat until no new `on_tool_call` lessons match
   or the replan cap (2) is hit.
4. Dispatch the (possibly revised) tool calls.
5. After each tool result, run `on_monologue` retrieval against the
   tool name, args, and result; append matches as a system-role
   message.
6. Loop.

The replan subloop is a new internal helper, not exposed outside
`run_turn`.

## Alternatives considered

- **LLM-driven retrieval via `load_lesson` tool.** Parallel to
  skills. Rejected: lessons are short enough that lazy-loading has no
  value, and relying on the LLM to remember to look risks missed
  lessons. Harness-driven retrieval is deterministic.
- **Embedding-based retrieval.** More robust than keyword match for
  paraphrase, but adds a dependency and an index to maintain. The
  `trigger.keywords` field is a clean seam — swap in embeddings
  later without a schema change.
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

1. **Trigger promotion.** A recurring lesson ("user corrected this
   3 times now") arguably should have its `trigger` promoted to
   `always` — and possibly its `category` reclassified to
   `preference` if the recurring correction reflects a stable user
   habit. Both promotions are independent. Is this the reflector's
   job per-run, or a separate consolidation pass? Leaning separate
   pass, out of scope for v1.
2. **Keyword extraction for `on_monologue`.** Naive token split is
   probably enough but may be noisy on tool-result blobs. Start
   naive, measure.
3. **Character-limit enforcement.** The reflector may exceed the
   limits. Truncate silently, reject and reprompt, or surface to the
   user in review? Leaning reject-and-reprompt once, then surface if
   still over.
4. **Cross-project lessons.** A lesson learned in project A about
   Google Sheets behavior is probably useful in project B. Default
   scope is empty (global), but does the reflector know when to use
   it vs. project-scope? The reflection prompt needs explicit
   guidance here.
5. **Index corruption.** If `index.json` and the `.md` files
   disagree (hand-edit, crashed write, or a file's `category` does
   not match its containing folder), which wins? Leaning: rebuild
   index from files on startup if parse fails, count mismatch, or
   any file's `category` does not match its folder. A bad `trigger`
   value is a parse error on that one lesson; skip it and warn.
6. **Lesson versioning.** Update overwrites history. Do we keep a
   git-backed `~/.leo/lessons/.git/`? Not in v1 — users who want
   history can `git init` the directory themselves.
7. **Replan cap of 2.** Picked by feel. If `on_tool_call` lessons
   are well-scoped, one replan should cover almost every case; two
   is a safety margin. Worth measuring. If we see frequent cap hits,
   that is a signal the lessons themselves are too broad, not that
   the cap is too low.
8. **`on_prompt` overlap with `always`.** An `on_prompt` lesson
   matched on the *first* user prompt of a session could be folded
   into the frozen system-prompt block instead of the suffix, saving
   tokens on subsequent turns where the prompt cache is hot. v1
   keeps `on_prompt` always in the suffix for simplicity. Worth
   revisiting if measurement shows non-trivial cache churn.

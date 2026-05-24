# Workspace & Session Design v0.01

## Goal

Give Leo a first-class notion of where work happens (**workspace**) and
what work is happening (**session**). Today the "workspace" is just an
implicit `Path.cwd().resolve()` used to bind the bwrap sandbox and
constrain edit paths; skills and lessons live globally under `~/.leo/`,
and conversation history is in-memory with optional manual `/save`.

After this change:

- A **workspace** is a real directory on disk that holds a project's
  Leo-specific state (skills, lessons, memory) alongside the user's
  project files.
- A **session** is one task inside a workspace: its conversation,
  toggles, and trace. Multiple sessions can share a workspace.

## Terminology

- **Workspace.** A project directory containing a `.leo/` subdirectory
  with project-scoped skills, lessons, memory, and session records.
- **Workspace root.** The directory containing `.leo/`. Used as the
  bwrap bind root and the edit/read path boundary.
- **Session.** A persisted record of one task: messages, toggles,
  reflection boundaries, optional artifacts. Belongs to exactly one
  workspace.
- **Interactive session.** Created when Leo starts without `--task`.
  Auto-persisted under `<workspace>/.leo/sessions/`.
- **One-shot run.** `leo --task "..."`. Ephemeral by default — does
  not create a session directory unless `--session <name>` is given.

## Workspace

### Layout

```
<workspace>/
  .leo/
    skills/              # project-specific skills
    lessons/             # project-specific lessons
    memory/              # project memories (added in a later phase)
    sessions/
      <session-id>/      # see Session section
      ...
  (user's project files)
```

The presence of `.leo/` is what makes a directory a workspace. No
hidden state lives outside `.leo/`.

### Resolution

On startup:

1. If `--workspace PATH` is given, use `PATH`.
2. Otherwise, use `Path.cwd()`.
3. If the chosen directory does not contain `.leo/`, exit with:
   ```
   leo: no workspace found at <path>
   run `leo init [path]` to create one
   ```
   No upward search, no auto-create. Explicit is better than magical.

The resolved workspace root replaces the implicit `Path.cwd().resolve()`
at `cli/leo.py:896` and feeds both `ToolContext.workspace` (bwrap bind
and edit-path checks).

### `leo init [path]`

Creates the `.leo/` skeleton at `path` (default: cwd):

```
<path>/.leo/
  skills/
  lessons/
  memory/
  sessions/
```

Idempotent — re-running on an existing workspace creates only missing
subdirs and prints what it did. Does not touch user files.

### Composition with global state

Skills and lessons are **layered**, workspace wins on name collision:

- Skills: `discover_skills(SKILLS_ROOT) + discover_skills(<ws>/.leo/skills)`,
  with later entries overriding earlier on name match.
- Lessons: `LessonStore([<ws>/.leo/lessons, LESSONS_ROOT])`. The store
  already accepts a list; first source wins.

The reflector and lesson writer **always target the workspace store**
when a workspace is active, so newly learned lessons stay project-local
and do not leak to other projects.

### Memory

Deferred to a follow-up. Reserve `<workspace>/.leo/memory/` in `leo
init` so the layout is stable, but no reader/writer in v1.

## Session

### Layout

```
<workspace>/.leo/sessions/<session-id>/
  meta.json              # id, title, started_at, last_active, model,
                         # toggles (think, net, show_*), reflection_idx
  messages.jsonl         # one JSON object per message, append-only
  artifacts/             # optional per-session scratch
```

### Session ID

`YYYY-MM-DD-HHMM-<4 random alphanum>`, e.g. `2026-05-22-1430-a7b3`.
Sorts chronologically, never collides in practice.

### Title

Auto-derived from the first user prompt (first ~60 chars, single line).
Stored in `meta.json`; can be edited by hand or via a future
`leo session rename`.

### Persistence model

- On session start: write `meta.json`, create empty `messages.jsonl`.
- After each assistant turn completes (including any tool-call cycles
  that turn produced): append the new messages to `messages.jsonl` and
  update `last_active` in `meta.json`.
- On `/exit`: run reflection (existing behavior), then write any final
  state, including the post-reflection `reflection_idx` so a resumed
  session does not re-reflect the same trace.

Append-on-turn is durable against crashes (worst case loses the
in-flight turn). It also keeps `messages.jsonl` greppable as a record
of the task.

### Lifecycle

| Trigger                          | Effect                                      |
| -------------------------------- | ------------------------------------------- |
| `leo` (interactive)              | Create new session, persist                 |
| `leo --session <id>`             | Resume that session (load `messages.jsonl`) |
| `leo --session last`             | Resume most recent session in workspace     |
| `leo --task "..."`               | Ephemeral — no session dir created          |
| `leo --task "..." --session <n>` | Persist under the given name                |
| `/exit`                          | Run reflection; session remains on disk     |
| `/exit noref`                    | Skip reflection; session remains on disk    |
| `leo session rm <id>`            | Delete session directory                    |

Sessions are never auto-deleted. The user owns cleanup (today; an
expiry policy can come later if directories pile up).

### Concurrency

Multiple Leo processes may run against the same workspace at the same
time. We do not lock. The workspace filesystem is shared — sessions
that touch the same files can clobber each other, and the user is
expected to coordinate. Lessons/skills/memory reads are safe;
lesson writes during reflection use the existing `LessonStore` write
path and are best-effort under concurrency.

### CLI surface

```
leo                          # new interactive session in cwd workspace
leo --workspace PATH         # use PATH (must contain .leo/)
leo --session <id>           # resume a specific session
leo --session last           # resume most recent
leo --task "..."             # one-shot, ephemeral
leo --task "..." --session N # one-shot, persisted as session N

leo init [path]              # create .leo/ skeleton
leo session list             # list sessions in current workspace
leo session show <id>        # print meta + message count
leo session rm <id>          # delete a session
```

`/save <file>` and `/load <file>` remain for manual export/import.
They are not the primary persistence path anymore.

## Reflection

No mechanical change. Reflection still runs on explicit `/exit` and
operates over the trace slice from `reflection_idx` to end. The only
differences:

- The trace comes from the active session's `messages` list (loaded
  from `messages.jsonl` on resume).
- Lessons produced are written to the workspace lesson store
  (`<ws>/.leo/lessons/`), not the global one.
- After reflection, `meta.json.reflection_idx` is updated so a
  subsequent resume reflects only new turns.

## Runtime objects

The existing `SessionContext` in `core/lessons/retrieval.py` is purely
for lesson scope matching. To free the name for the new runtime
session object, rename it to `LessonScope` (and update its callers in
`core/lessons/`). The new object:

```python
@dataclass
class Session:
    id: str
    workspace: Path
    dir: Path             # <workspace>/.leo/sessions/<id>
    title: str
    messages: list[dict]
    toggles: SessionToggles
    reflection_idx: int
    started_at: str
    last_active: str
```

`ToolContext` gains `session: Session | None` (None for one-shot runs
that did not create a session). Tools that want to write per-session
scratch use `session.dir / "artifacts"`.

## Implementation phases

1. **Workspace plumbing.**
   - `leo init [path]`.
   - `--workspace` flag + resolution + "no .leo/" exit.
   - Layered skill discovery and lesson store.
   - Reflector/writer target the workspace lesson store.
2. **Session plumbing.**
   - `Session` object, ID scheme, `meta.json` + `messages.jsonl`.
   - Append-on-turn persistence.
   - `--session <id>` / `--session last` resume.
   - `leo session list/show/rm`.
   - `--task` stays ephemeral unless `--session` is also given.
   - Rename existing `SessionContext` → `LessonScope`.
3. **Memory.** Separate design doc; uses `<ws>/.leo/memory/`.

## Open items (post-v1)

- Session expiry / cleanup policy.
- Workspace-level config (`<ws>/.leo/config.toml`) for per-project
  model, system prompt, default toggles.
- Cross-workspace session move / export.

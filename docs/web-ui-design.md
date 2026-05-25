# Web UI Design v0.01

## Goal

Add a web front-end for Leo that supports interactive **chat**,
**observability** into runs, **token-streaming**, and **multiple users**.
The CLI stays. The web UI is a second front-end over the same
`leo.core` agent loop — not a fork of it.

## Scope

In scope:

- Browser-based chat against the existing agent loop.
- Live streaming of LLM tokens and tool-call events.
- Per-user authentication (self-contained for v1).
- Per-user workspaces under a per-user data root.
- Multi-session views: list, open, resume, delete.
- Observability panels: tool-call traces, skills, lessons, artifacts.

Out of scope for v1:

- Delegated auth (OAuth/OIDC, reverse-proxy headers).
- Cross-user sharing of sessions or workspaces.
- Mid-stream user injection (only cancel + new turn).
- Mobile-first layout.
- Public-internet hardening (CSRF tokens, rate limits, audit logs).
  v1 targets a LAN / small-team deployment.

## Terminology

- **Server.** The Leo web service process. Single FastAPI app.
- **User.** Authenticated principal, identified by `user_id`.
- **Data root.** Per-user directory on disk under which that user may
  create or open workspaces. Configured in the server config file.
- **Workspace.** Same as in `workspace-session-design.md`: a directory
  containing `.leo/`. Owned by exactly one user (the one whose data
  root contains it).
- **Run.** One assistant turn (LLM call + any tool-call cycles it
  triggers), executed by the run supervisor on behalf of a session.

## Architecture

```
[ React SPA ] ──HTTP + SSE──▶ [ FastAPI app ]
                                    │
                       ┌────────────┼─────────────┐
                       ▼            ▼             ▼
                 server config   leo.core     run supervisor
                 (users.yaml)   (session.py,  (spawns worker
                                 agent loop)   per run)
                       │            │             │
                       │            ▼             ▼
                       │      <ws>/.leo/sessions/<sid>/
                       │        meta.json
                       │        messages.jsonl
                       │        events.jsonl   ◀── new
                       ▼
                 <server>/state/
                   sessions.sqlite       # auth tokens, last-workspace
```

The agent loop is unchanged in behavior; it gains an event-emission
hook and is invoked from the web path via the run supervisor instead
of directly from the CLI REPL.

## Server config

A YAML file the server reads on startup. Path is set by
`LEO_SERVER_CONFIG` env var, defaulting to `~/.leo-server/config.yaml`.

```yaml
# Listen address (defaults shown).
host: 127.0.0.1
port: 8765

# User registry. Self-contained auth in v1.
users:
  - id: alice
    data_root: /home/alice/leo-data
    password_hash: "$2b$12$..."     # bcrypt
  - id: bob
    data_root: /srv/leo/bob
    password_hash: "$2b$12$..."
```

Rules:

- `id` must be a slug (`[a-z0-9_-]+`).
- `data_root` must exist and be a directory readable+writable by the
  server process. Server refuses to start otherwise.
- Two users may not share a `data_root`. Server refuses to start.
- Reloading the config without restart is out of scope for v1.

No registration flow. An admin edits the file and restarts the server.
A `leo-server passwd <user>` helper generates a bcrypt hash so the
admin never types one by hand.

## Authentication

- POST `/api/auth/login` with `{user_id, password}` → sets an
  HTTP-only session cookie holding a random token.
- Server stores `(token, user_id, created_at, last_used)` in
  `sessions.sqlite` (table `auth_tokens`). Tokens expire after 30 days
  of inactivity.
- POST `/api/auth/logout` invalidates the current token.
- All `/api/*` routes except `/api/auth/login` require a valid token.

Self-contained today. The auth layer is a single dependency injection
in FastAPI, swappable later for OIDC without touching route bodies.

## Workspaces in the multi-user model

### Ownership

A workspace belongs to whichever user's `data_root` contains it. The
server never lets user A read user B's workspace, even if file
permissions on disk would allow it. Enforcement: every path the user
supplies is resolved (symlinks followed) and checked to live under
that user's `data_root`. Reject otherwise. This applies to:

- Folder picker listing.
- Workspace open/create.
- Artifact reads.

### Picker flow on login

1. User logs in.
2. Server returns `last_workspace` (from `sessions.sqlite`,
   per-user). If present and the directory still contains `.leo/`,
   the UI opens it directly.
3. Otherwise (or via "switch workspace"), UI shows a folder picker
   rooted at `data_root`:
   - Browse subdirectories at arbitrary depth.
   - Each entry indicates whether it is already a workspace.
   - User picks a folder F:
     - F has `.leo/` → open as workspace.
     - F has no `.leo/` → create one there, then open.
4. Server records the chosen workspace as `last_workspace` for that
   user.

### Nesting rule

A workspace may not be created inside another workspace, nor may a
workspace contain another workspace. On create at folder F under
`data_root R`:

- **Ancestor check.** For every directory D on the path `R → … → F`,
  reject if D has `.leo/`.
- **Descendant check.** Walk F's subtree; reject if any descendant
  has `.leo/`. The walk respects `.gitignore`-style skips of obvious
  bulk dirs (`node_modules`, `.venv`, `.git`) to keep create-time
  bounded; those skip lists are server-configurable.

Both checks are bypassed when *opening* an existing workspace — only
*create* enforces them. A workspace that pre-exists with violations
(created out-of-band) is opened with a warning banner.

### Folder picker security

The picker is served by a single endpoint:

```
GET /api/fs/list?path=<rel-to-data-root>
```

- `path` is resolved against the user's `data_root` with all symlinks
  followed; the resolved real path must remain under `data_root`'s
  real path. Otherwise 403.
- Returns only directories (files are invisible to the picker).
- Each entry includes `{name, is_workspace}`.

The same path-resolution helper guards artifact reads.

## Sessions

Sessions are unchanged from `workspace-session-design.md` on disk.
Web-specific additions:

### Events log

Alongside `messages.jsonl` in each session directory, add:

```
<workspace>/.leo/sessions/<sid>/events.jsonl
```

Append-only, one JSON object per line. Envelope:

```json
{"seq": 42, "ts": "2026-05-23T14:32:11", "type": "tool_start",
 "payload": {"tool": "read", "args": {"path": "..."}}}
```

`seq` is monotonic per session. Event types (v1):

- `run_start`, `run_end`, `run_cancelled`, `run_error`
- `llm_token` (optional — high-frequency; may be coalesced)
- `llm_message` (one per assistant message, mirrors the entry written
  to `messages.jsonl`)
- `tool_start`, `tool_end`
- `reflection_start`, `reflection_end`
- `lesson_injected`, `skill_loaded`

The chat view replays `messages.jsonl`. The observability view
replays `events.jsonl`. The two are independent and the LLM message
format on disk is unchanged.

### Concurrency

Per-session lock. Within a workspace, different sessions may run
concurrently. Within a single session, at most one run at a time.

- Lock implementation: `fcntl.flock` on
  `<sid>/.run.lock`. Held by the run supervisor for the duration of
  the run.
- A second POST to `/messages` on a locked session returns 409 with
  the current run's ID.

Cross-process: two Leo processes (CLI + web, or two web instances
sharing the disk) honor the same lock file.

## Run supervisor

The run supervisor owns the lifecycle of one assistant turn.

- **Isolation.** Each run executes in a worker — initially a thread
  with a dedicated event queue; subprocess isolation is reserved as
  a follow-up if tool crashes prove disruptive. A crashing tool fails
  the run but should not take down the server.
- **Inputs.** `(session, user_message, cancel_token)`.
- **Outputs.** A stream of events posted to a per-session pub/sub
  topic. Events are persisted to `events.jsonl` as they are emitted;
  subscribers receive the same events live.
- **Cancellation.** `POST /api/sessions/{sid}/cancel` sets the
  cancel token. The agent loop checks it between tool calls and at
  the LLM-streaming boundary. A cancelled run emits `run_cancelled`
  and releases the lock.
- **Resource limits (v1).** Per-user cap on simultaneous active runs
  (default 2). Configurable in the server config.

## Streaming

SSE, not WebSockets. Justification:

- Server → client is the only streaming direction we need.
- Client → server is a normal POST (`/messages`) plus a normal POST
  (`/cancel`).
- SSE survives proxies better and reconnects natively in browsers.

Stream endpoint:

```
GET /api/sessions/{sid}/stream?since=<seq>
```

- Subscribes to the session's pub/sub topic.
- Backfills any events with `seq > since` from `events.jsonl` first,
  then switches to live tailing. This makes refresh and reattach
  trivial: the client remembers the last `seq` it saw and resumes
  from there.
- Heartbeat comment line every 15s to keep proxies open.

## API surface

```
POST /api/auth/login              {user_id, password} → set cookie
POST /api/auth/logout
GET  /api/me                      → {user_id, data_root, last_workspace}

GET  /api/fs/list?path=...        → directory entries under data_root

POST /api/workspaces/open         {path}              → workspace info
POST /api/workspaces/create       {path}              → workspace info
GET  /api/workspaces/current      → current workspace
                                     (per-cookie state)

GET  /api/sessions                → list
POST /api/sessions                {title?} → new session
GET  /api/sessions/{sid}          → meta + messages + events
DELETE /api/sessions/{sid}

POST /api/sessions/{sid}/messages {content} → 202, run started
POST /api/sessions/{sid}/cancel
GET  /api/sessions/{sid}/stream?since=<seq>   (SSE)

GET  /api/skills                  → workspace + global skills
GET  /api/lessons                 → workspace + global lessons
GET  /api/artifacts/{sid}/{...}   → file under that session's
                                     artifacts dir
```

All endpoints (except `/auth/login`) require the auth cookie. All
path-bearing endpoints enforce the `data_root` containment check.

## Front-end

React + Vite + TypeScript + Tailwind + **shadcn/ui**.

Justification: htmx is faster to ship but the observability panels —
tool-call trees, JSON viewers, collapsible nested traces, diff views
— are the kinds of UI where React pays back its setup cost quickly.
We will want them.

### Visual style

Take direct cues from [shadcn.com](https://ui.shadcn.com) — the same
look the shadcn/ui project uses for its own site:

- **Minimal, content-first.** No gradients, no shadows on cards by
  default. Borders + spacing do the structural work.
- **Palette: University of Illinois brand colors.** Per
  [brand.illinois.edu/visual-identity/color](https://brand.illinois.edu/visual-identity/color):
  - **Illini Orange** `#FF5F05` — primary accent. Used for primary
    buttons, the streaming indicator, the active session row, focus
    rings, links.
  - **Illini Blue** `#13294B` — deep navy. Used for the top bar,
    headings, and as the dark-theme background base.
  - **Neutrals.** Storm Gray range, white, black for surfaces,
    borders, body text. These supplement; they do not replace the
    primaries.
  - **Distribution.** Follow Illinois' 80/15/5 guidance: primaries
    dominate, neutrals supplement, supporting colors are reserved for
    charts and status badges (e.g. tool success/error glyphs).
  - **shadcn theming.** Override the default CSS-variable theme:
    `--primary` → Illini Orange, `--background` (dark) → Illini Blue,
    `--ring` → Illini Orange. Light-theme background stays near-white;
    dark-theme background is Illini Blue rather than shadcn's default
    near-black.
  - **Accessibility.** Illini Orange on white measures ~3.0:1 contrast
    — passes WCAG AA for large text and non-text UI (buttons,
    indicators, focus rings, the active-session bar) but **fails for
    body text and inline links**. Rules:
    - Body text and headings: Storm Gray / Illini Blue on light;
      white / light gray on dark. Never orange.
    - Inline links: Illini Blue on light theme; a lightened orange
      tint (e.g. `#FF8A3D`) or white-with-orange-underline on dark
      theme. Verify ≥4.5:1 against the resolved background.
    - Primary buttons: orange background with white text (passes AA).
    - The active-session row uses an orange left-border or background
      tint, not orange text.
- **Dark and light themes** via shadcn's CSS-variable scheme. Default
  to system preference. A theme toggle lives in the top-bar menu.
- **Typography.** Inter (or system sans) for UI; JetBrains Mono (or
  another monospace) for code, message content from tools, JSON
  payloads, and session IDs. Tight line-height for chrome, generous
  line-height (~1.6) for chat content.
- **Density.** Comfortable, not compact. Default shadcn sizing for
  buttons and inputs. The chat is a reading surface, not a dashboard
  — give it air.
- **Components, all shadcn/ui primitives:**
  - `Sidebar` for the left session list.
  - `ScrollArea` for the chat and the observability pane.
  - `Tabs` to switch observability sub-views (Trace / Skills /
    Lessons / Artifacts).
  - `Collapsible` for tool-call entries in the trace.
  - `Dialog` for the workspace picker.
  - `Command` (cmdk) for fuzzy session search and "switch workspace".
  - `Toast` for non-blocking errors (lock conflict, run cancelled).
  - `Tooltip` on truncated session titles, model badges, tool names.
- **Icons.** lucide-react only. No mixing icon sets.
- **Code blocks** in chat and tool output: Shiki (or
  react-syntax-highlighter) with the same theme as the rest of the
  UI. No external highlight.js CSS.
- **Empty states.** Centered, single sentence, one primary action —
  match the shadcn/ui docs "no results" pattern.

### Layout sketch

```
┌──────────────┬──────────────────────────────┬─────────────────┐
│ Sessions     │ Chat                         │ Observability   │
│ ───────────  │ ───────────                  │ ─────────────── │
│ • 2026-05-23 │ user: ...                    │ ▾ tool: read    │
│   1430-a7b3  │ assistant: ...               │     args: {...} │
│ • 2026-05-22 │ ▸ tool_call: read(path=...)  │     output: ... │
│ + New        │ assistant: ...               │ ▾ reflection    │
│              │ [input box]                  │ ▾ lessons       │
└──────────────┴──────────────────────────────┴─────────────────┘
```

Three-column flex layout. The left and right columns are
`Resizable` panels (shadcn/ui's resizable primitive); the user's
chosen widths persist in localStorage. The right column collapses
on narrow viewports — chat is the priority surface.

Top bar shows current workspace path, the active model, a
streaming/idle indicator, and a menu (theme toggle, switch
workspace, logout).

## CLI prerequisite refactor

`src/leo/cli/leo.py` is 1367 lines and owns both the REPL and the
agent loop. Before the web UI lands, extract the agent loop into
`src/leo/core/agent.py` with a single async entry point:

```python
async def run_turn(
    session: Session,
    user_message: str,
    *,
    cancel_token: CancelToken,
) -> AsyncIterator[Event]:
    ...
```

The CLI REPL and the FastAPI route both consume this. Without the
refactor we will have two agent loops within a month, and they will
drift.

## Server packaging

- New entry point `leo-server` (`src/leo/server/app.py`) exposed via
  `pyproject.toml` `[project.scripts]`.
- New dependencies: `fastapi`, `uvicorn[standard]`, `sse-starlette`,
  `bcrypt`, `passlib`, `pydantic>=2`, `aiosqlite`.
- Front-end lives under `web/` at the repo root (separate from
  `src/`). Built artifacts are served by FastAPI as static files in
  production; dev runs Vite on its own port with a proxy.

## Implementation phases

1. **Refactor.** Extract the agent loop into `core/agent.py`. CLI
   continues to work unchanged. No web code yet.
2. **Events.** Add `events.jsonl` emission to the agent loop. CLI
   ignores it; this is groundwork for observability.
3. **Single-user web v0.** FastAPI app, no auth, hard-coded
   workspace. Chat works end-to-end with SSE streaming. Sessions
   list + open + new + delete. No observability panels yet.
4. **Observability panels.** Tool-trace tree, skills view, lessons
   view, artifact viewer. Reattach-on-refresh via `?since=<seq>`.
5. **Multi-user.** Server config, login, per-user data root, folder
   picker with nesting checks, `last_workspace` memory, per-user run
   caps.
6. **Hardening.** Per-session lock via `flock`, cancel semantics,
   subprocess isolation if needed.

## Open items (post-v1)

- Delegated auth (OIDC, reverse-proxy header trust).
- Cross-user session sharing (read-only links).
- Public-internet hardening: CSRF tokens, per-IP rate limits, audit
  log, structured server logs.
- Live mid-run user notes (without cancelling the run).
- Mobile layout.
- Hot reload of server config.
- Export / import of a session as a portable bundle.

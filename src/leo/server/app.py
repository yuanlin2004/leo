"""FastAPI app — Phase 3a backend skeleton.

Single user, no auth. Workspace is resolved once at startup from the
LEO_WS env var (or the current working directory) and must contain a
`.leo/` marker. The agent loop is the same `core.agent.run_turn` the
CLI drives.

Endpoints (all under /api):
  GET    /sessions                list sessions in the workspace
  POST   /sessions                create a new session
  GET    /sessions/{sid}          meta + messages + events
  DELETE /sessions/{sid}          delete
  POST   /sessions/{sid}/messages enqueue a user turn (202)
  GET    /sessions/{sid}/stream   SSE event stream (?since=N)

Phase 3a deliberately omits: auth, folder picker, observability panels
beyond raw events, cancellation, and any multi-user concerns. Those land
in Phases 5 and 6.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from leo.core.agents import (
    AGENTS_ROOT,
    AgentError,
    AgentSpec,
    BUILTIN_LEO_ID,
    delete_agent,
    discover_agents,
    get_agent,
    slugify_name,
    unique_id,
    write_agent,
)
from leo.core.lessons import WriteError
from leo.core.lessons.reflector import (
    CreateOp,
    ReflectorError,
    SkipOp,
    UpdateOp,
    reflect,
)
from leo.core.session import (
    append_messages,
    count_messages,
    delete_session,
    list_sessions,
    load_session,
    new_session,
)
from leo.core.setup import (
    WORKSPACE_MARKER,
    WORKSPACE_SUBDIRS,
    build_run_context,
)
from leo.server.supervisor import RunBusy, RunSupervisor


# Skip-list for the descendant nesting walk on workspace create. These
# directories are typically large and never themselves a workspace.
_NESTING_WALK_SKIP = {"node_modules", ".venv", "venv", ".git", "__pycache__", ".tox", "dist", "build", "target"}

# State dir holding `last_workspace.json` for restart persistence.
_SERVER_STATE_DIR = Path.home() / ".leo-server"


# -- request/response models ----------------------------------------------


class NewSessionRequest(BaseModel):
    title: str | None = None
    agent_id: str | None = None        # defaults to "leo" if omitted


class AgentDTO(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    initial_user_prompt: str
    skills: list[str]
    default_think: bool
    builtin: bool


class AgentWriteRequest(BaseModel):
    # `id` is no longer required from the client. POST derives it by
    # slugifying `name` and disambiguating collisions; PUT ignores this
    # field and uses the path parameter.
    id: str | None = None
    name: str
    description: str = ""
    system_prompt: str = ""
    initial_user_prompt: str = ""
    skills: list[str] = []
    default_think: bool = True


class PatchSessionRequest(BaseModel):
    title: str | None = None
    # Agent-behavior toggles (passed into run_turn / sandbox).
    think_on: bool | None = None
    net_on: bool | None = None
    # Display-only toggles (consumed by the web UI; agent ignores them).
    show_think: bool | None = None
    show_tool_use: bool | None = None
    show_reflection: bool | None = None


class SendMessageRequest(BaseModel):
    content: str


class ReflectApplyRequest(BaseModel):
    ops: list[dict]   # see /reflect proposal shape; each entry is one op to apply


class SessionSummary(BaseModel):
    id: str
    title: str
    last_active: str
    started_at: str
    model: str | None
    message_count: int
    is_running: bool
    agent_id: str


class MeResponse(BaseModel):
    """Per-session-of-the-server info: workspace, data_root, model.

    `workspace` is null when the server started without finding a
    workspace (no --workspace, no $LEO_WS, no last-used, no .leo/ in
    cwd or data_root). The client must call /api/workspace/open or
    /api/workspace/create before any session-related endpoint will
    succeed.

    `data_root` bounds the workspace picker; the UI cannot navigate
    above it. Named `/api/me` because once auth lands this endpoint
    grows a `user_id` field — same shape, more data.
    """
    workspace: str | None
    data_root: str
    model: str | None


class FsEntry(BaseModel):
    name: str
    path: str
    is_workspace: bool


class FsListResponse(BaseModel):
    path: str
    data_root: str
    current_is_workspace: bool
    parent: str | None
    entries: list[FsEntry]


class WorkspacePathRequest(BaseModel):
    path: str


class SessionDetail(BaseModel):
    id: str
    title: str
    last_active: str
    started_at: str
    model: str | None
    toggles: dict
    messages: list[dict]
    next_event_seq: int
    is_running: bool
    loaded_skills: list[str]
    injected_lesson_ids: list[str]
    agent_id: str
    # Surfaced so the UI can prefill the chat input on session open.
    # Empty for built-in leo and agents that don't set one.
    initial_user_prompt: str = ""


class SkillInfo(BaseModel):
    name: str
    description: str


class LessonInfo(BaseModel):
    id: str
    title: str
    category: str
    trigger_type: str
    trigger_keywords: list[str]
    trigger_tool: str | None
    scope: dict
    rule: str
    why: str
    how_to_apply: str
    created: str
    updated: str


# -- workspace resolution -------------------------------------------------


def _resolve_data_root(arg: str | None) -> Path:
    raw = arg or os.environ.get("LEO_DATA_ROOT") or str(Path.home())
    droot = Path(raw).resolve()
    if not droot.is_dir():
        print(f"leo-server: data root {droot} is not a directory", file=sys.stderr)
        sys.exit(2)
    return droot


def _read_last_workspace() -> Path | None:
    p = _SERVER_STATE_DIR / "last_workspace.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        ws = Path(data["path"]).resolve()
        if ws.is_dir() and (ws / WORKSPACE_MARKER).is_dir():
            return ws
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def _write_last_workspace(ws: Path) -> None:
    _SERVER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = _SERVER_STATE_DIR / "last_workspace.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"path": str(ws)}, indent=2))
    tmp.replace(p)


def _resolve_workspace(arg: str | None, data_root: Path) -> Path | None:
    """Initial workspace resolution at startup.

    If --workspace or $LEO_WS is set, that path must resolve to a
    workspace; otherwise the server exits (an explicit ask deserves a
    loud failure).

    Otherwise we try the fallback chain:
       last_workspace.json → data_root → cwd
    Returns None if none of those is a workspace; the server starts in
    "no workspace selected" mode and the client must pick one via the
    web UI or POST /api/workspace/open|create.
    """
    explicit = arg or os.environ.get("LEO_WS")
    if explicit:
        p = Path(explicit).resolve()
        if p.is_dir() and (p / WORKSPACE_MARKER).is_dir():
            return p
        print(
            f"leo-server: --workspace / $LEO_WS = {p} is not a workspace.\n"
            f"run `leo init {p}` to create one, or unset the flag/env "
            f"to start with no workspace and pick one via the UI.",
            file=sys.stderr,
        )
        sys.exit(2)
    for c in (
        _read_last_workspace(),
        data_root,
        Path.cwd().resolve(),
    ):
        if c is None:
            continue
        if c.is_dir() and (c / WORKSPACE_MARKER).is_dir():
            return c
    return None


def _ensure_under_data_root(p: Path, data_root: Path) -> Path:
    """Resolve `p` (follow symlinks). Raise 403 if it escapes `data_root`."""
    try:
        resolved = p.resolve()
    except OSError:
        raise HTTPException(status_code=400, detail="invalid path")
    droot = data_root.resolve()
    try:
        resolved.relative_to(droot)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"path {resolved} is outside data_root {droot}",
        )
    return resolved


def _check_no_nested_workspace(folder: Path, data_root: Path) -> None:
    """Reject if any ancestor up to data_root, or any descendant, is a
    workspace already. See web-ui-design.md for the rule."""
    folder = folder.resolve()
    droot = data_root.resolve()
    # Ancestor walk — from folder.parent up to and including data_root.
    cur = folder.parent
    while True:
        if (cur / WORKSPACE_MARKER).is_dir():
            raise HTTPException(
                status_code=409,
                detail=f"ancestor {cur} is already a workspace",
            )
        if cur == droot or cur.parent == cur:
            break
        cur = cur.parent
    # Descendant walk — bounded by skip list.
    for root, dirs, _files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _NESTING_WALK_SKIP]
        if WORKSPACE_MARKER in dirs:
            raise HTTPException(
                status_code=409,
                detail=f"descendant {Path(root) / WORKSPACE_MARKER} is already a workspace",
            )


# -- app factory ----------------------------------------------------------


def create_app(workspace: Path | None, data_root: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        app.state.data_root = data_root
        app.state.loop = loop
        # Guards workspace switches: only one swap at a time, and no
        # swap while a run is in flight.
        app.state.ws_lock = threading.Lock()
        if workspace is not None:
            ctx = build_run_context(workspace)
            for issue in ctx.lesson_issues:
                print(
                    f"(lesson {issue.path.name}: {issue.reason})", file=sys.stderr,
                )
            app.state.ctx = ctx
            app.state.supervisor = RunSupervisor(ctx, loop)
            _write_last_workspace(workspace)
            print(
                f"leo-server ready: workspace={workspace} "
                f"data_root={data_root} model={ctx.llm.model}",
                file=sys.stderr,
            )
        else:
            app.state.ctx = None
            app.state.supervisor = None
            print(
                f"leo-server ready: no workspace selected\n"
                f"  data_root={data_root}\n"
                f"  open one via the web UI or POST /api/workspace/open|create",
                file=sys.stderr,
            )
        yield

    app = FastAPI(title="Leo", version="0.1.0", lifespan=lifespan)
    # Permissive CORS for Phase 3a so the Vite dev server on a different
    # port can reach the API. Tightened in Phase 5 alongside auth.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    def require_ctx(request: Request):
        """Return the active RunContext, or raise 412 if no workspace is
        selected. Use this from any handler that touches workspace state."""
        ctx = request.app.state.ctx
        if ctx is None:
            raise HTTPException(
                status_code=412,
                detail="no workspace selected — open one via /api/workspace/open",
            )
        return ctx

    def require_supervisor(request: Request) -> RunSupervisor:
        sup = request.app.state.supervisor
        if sup is None:
            raise HTTPException(
                status_code=412,
                detail="no workspace selected — open one via /api/workspace/open",
            )
        return sup

    # Aliases for readability in handlers that already had a workspace.
    supervisor = require_supervisor
    ctx_for = require_ctx

    # -- self -------------------------------------------------------------

    @app.get("/api/me", response_model=MeResponse)
    def me(request: Request) -> MeResponse:
        c = request.app.state.ctx  # may be None
        return MeResponse(
            workspace=str(c.workspace) if c is not None else None,
            data_root=str(request.app.state.data_root),
            model=c.llm.model if c is not None else None,
        )

    # -- filesystem + workspace switch -----------------------------------

    @app.get("/api/fs/list", response_model=FsListResponse)
    def fs_list(request: Request, path: str = "") -> FsListResponse:
        """List subdirectories of `path` (default: data_root).

        `path` must resolve to a directory under data_root. Only
        directories are returned (the picker doesn't browse files).
        Hidden directories (dotfiles) are skipped, except that the
        `.leo/` marker is detected on each entry.
        """
        data_root = request.app.state.data_root
        target = Path(path).resolve() if path else data_root.resolve()
        resolved = _ensure_under_data_root(target, data_root)
        if not resolved.is_dir():
            raise HTTPException(status_code=404, detail=f"not a directory: {resolved}")
        entries: list[FsEntry] = []
        try:
            children = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
        except OSError as e:
            raise HTTPException(status_code=403, detail=str(e))
        for child in children:
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            try:
                is_ws = (child / WORKSPACE_MARKER).is_dir()
            except OSError:
                is_ws = False
            entries.append(FsEntry(
                name=child.name, path=str(child), is_workspace=is_ws,
            ))
        parent: str | None = None
        if resolved != data_root.resolve():
            parent = str(resolved.parent)
        return FsListResponse(
            path=str(resolved),
            data_root=str(data_root.resolve()),
            current_is_workspace=(resolved / WORKSPACE_MARKER).is_dir(),
            parent=parent,
            entries=entries,
        )

    @app.post("/api/workspace/open", response_model=MeResponse)
    def workspace_open(
        req: WorkspacePathRequest, request: Request,
    ) -> MeResponse:
        target = Path(req.path)
        data_root = request.app.state.data_root
        resolved = _ensure_under_data_root(target, data_root)
        if not resolved.is_dir():
            raise HTTPException(status_code=404, detail="not a directory")
        if not (resolved / WORKSPACE_MARKER).is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"{resolved} is not a workspace (no .leo/)",
            )
        _switch_workspace(request.app, resolved)
        return MeResponse(
            workspace=str(resolved),
            data_root=str(data_root.resolve()),
            model=request.app.state.ctx.llm.model,
        )

    @app.post("/api/workspace/create", response_model=MeResponse)
    def workspace_create(
        req: WorkspacePathRequest, request: Request,
    ) -> MeResponse:
        target = Path(req.path)
        data_root = request.app.state.data_root
        resolved = _ensure_under_data_root(target, data_root)
        if not resolved.is_dir():
            raise HTTPException(status_code=404, detail="not a directory")
        if (resolved / WORKSPACE_MARKER).is_dir():
            raise HTTPException(status_code=409, detail="already a workspace")
        _check_no_nested_workspace(resolved, data_root)
        # Create the .leo/ skeleton mirroring `leo init`.
        leo_dir = resolved / WORKSPACE_MARKER
        leo_dir.mkdir()
        for sub in WORKSPACE_SUBDIRS:
            (leo_dir / sub).mkdir()
        _switch_workspace(request.app, resolved)
        return MeResponse(
            workspace=str(resolved),
            data_root=str(data_root.resolve()),
            model=request.app.state.ctx.llm.model,
        )

    # -- agents -----------------------------------------------------------

    def _agent_to_dto(spec: AgentSpec) -> AgentDTO:
        return AgentDTO(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            system_prompt=spec.system_prompt,
            initial_user_prompt=spec.initial_user_prompt,
            skills=list(spec.skills),
            default_think=spec.default_think,
            builtin=spec.builtin,
        )

    @app.get("/api/agents", response_model=list[AgentDTO])
    def agents_list() -> list[AgentDTO]:
        # Agents are workspace-independent (single global depot), so no
        # ctx required. Still gated by /api/me's data_root containment
        # at the front-door — agents live under $HOME/.leo/agents.
        return [_agent_to_dto(a) for a in discover_agents()]

    @app.get("/api/agents/{agent_id}", response_model=AgentDTO)
    def agents_get(agent_id: str) -> AgentDTO:
        try:
            return _agent_to_dto(get_agent(agent_id))
        except AgentError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/agents", response_model=AgentDTO, status_code=201)
    def agents_create(req: AgentWriteRequest) -> AgentDTO:
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        existing = discover_agents()
        norm = name.casefold()
        if any(a.name.strip().casefold() == norm for a in existing):
            raise HTTPException(
                status_code=409,
                detail=f"an agent named {name!r} already exists",
            )
        # Derive id from name. Disambiguate against existing ids by
        # appending -2, -3, ... — this only matters when a previous
        # agent shared the same slug but a different name (e.g. accents
        # or punctuation that strips identically).
        base = slugify_name(name)
        taken = {a.id for a in existing}
        new_id = unique_id(base, existing=taken)
        spec = AgentSpec(
            id=new_id, name=name, description=req.description,
            system_prompt=req.system_prompt,
            initial_user_prompt=req.initial_user_prompt,
            skills=list(req.skills), default_think=req.default_think,
        )
        try:
            write_agent(spec)
        except AgentError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _agent_to_dto(get_agent(new_id))

    @app.put("/api/agents/{agent_id}", response_model=AgentDTO)
    def agents_update(agent_id: str, req: AgentWriteRequest) -> AgentDTO:
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        # Reject if another agent already uses this name. Exclude the
        # one being edited from the duplicate check.
        norm = name.casefold()
        for a in discover_agents():
            if a.id == agent_id:
                continue
            if a.name.strip().casefold() == norm:
                raise HTTPException(
                    status_code=409,
                    detail=f"an agent named {name!r} already exists",
                )
        spec = AgentSpec(
            id=agent_id, name=name, description=req.description,
            system_prompt=req.system_prompt,
            initial_user_prompt=req.initial_user_prompt,
            skills=list(req.skills), default_think=req.default_think,
        )
        try:
            write_agent(spec)
        except AgentError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _agent_to_dto(get_agent(agent_id))

    @app.delete("/api/agents/{agent_id}", status_code=204)
    def agents_delete(agent_id: str) -> None:
        try:
            delete_agent(agent_id)
        except AgentError as e:
            # builtin-with-no-override and unknown both raise; both
            # map nicely to 404 from the client's perspective.
            raise HTTPException(status_code=404, detail=str(e))

    # -- skills + lessons -------------------------------------------------

    @app.get("/api/skills", response_model=list[SkillInfo])
    def skills_list(request: Request) -> list[SkillInfo]:
        c = ctx_for(request)
        return [SkillInfo(name=s.name, description=s.description) for s in c.skills]

    @app.get("/api/lessons", response_model=list[LessonInfo])
    def lessons_list(request: Request) -> list[LessonInfo]:
        c = ctx_for(request)
        out: list[LessonInfo] = []
        for L in c.lessons.lessons:
            scope: dict = {}
            if L.scope.project is not None:
                scope["project"] = L.scope.project
            if L.scope.skill is not None:
                scope["skill"] = L.scope.skill
            if L.scope.model is not None:
                scope["model"] = L.scope.model
            out.append(LessonInfo(
                id=L.id,
                title=L.title,
                category=L.category,
                trigger_type=L.trigger.type,
                trigger_keywords=list(L.trigger.keywords),
                trigger_tool=L.trigger.tool,
                scope=scope,
                rule=L.rule,
                why=L.why,
                how_to_apply=L.how_to_apply,
                created=L.created,
                updated=L.updated,
            ))
        return out

    # -- sessions ---------------------------------------------------------

    @app.get("/api/sessions", response_model=list[SessionSummary])
    def sessions_list(request: Request) -> list[SessionSummary]:
        ws = ctx_for(request).workspace
        sup = supervisor(request)
        out: list[SessionSummary] = []
        for s in list_sessions(ws):
            out.append(SessionSummary(
                id=s.id,
                title=s.title,
                last_active=s.last_active,
                started_at=s.started_at,
                model=s.model,
                message_count=count_messages(s),
                is_running=sup.is_running(s.id),
                agent_id=s.agent_id,
            ))
        return out

    @app.post("/api/sessions", response_model=SessionSummary, status_code=201)
    def sessions_create(req: NewSessionRequest, request: Request) -> SessionSummary:
        """Create a session bound to an agent. The agent's system prompt
        is composed (agent + base + skills + lessons) and persisted as
        the system message in messages.jsonl — sessions are immutable
        snapshots, so editing the agent later does NOT mutate them."""
        ctx = ctx_for(request)
        agent_id = req.agent_id or "leo"
        try:
            per_session = build_run_context(
                ctx.workspace, agent_id=agent_id,
            )
        except AgentError as e:
            raise HTTPException(status_code=404, detail=str(e))
        for missing in per_session.missing_agent_skills:
            print(
                f"(agent {agent_id}: references missing skill {missing!r}, "
                f"skipping)", file=sys.stderr,
            )
        # default_think from the agent seeds session.toggles.
        default_think = (
            per_session.agent.default_think if per_session.agent else True
        )
        s = new_session(
            ctx.workspace,
            model=ctx.llm.model,
            toggles={"think_on": default_think},
            title=req.title or "(untitled)",
            agent_id=agent_id,
        )
        # Bootstrap the messages log with the composed system prompt.
        append_messages(
            s, [{"role": "system", "content": per_session.system_prompt}],
        )
        # Carry the workspace's session-start lesson firings onto this
        # session so the supervisor doesn't re-inject them.
        s.injected_ids = list(per_session.phase1_ids)
        s.write_meta()
        return SessionSummary(
            id=s.id, title=s.title, last_active=s.last_active,
            started_at=s.started_at, model=s.model,
            message_count=count_messages(s), is_running=False,
            agent_id=s.agent_id,
        )

    @app.get("/api/sessions/{sid}", response_model=SessionDetail)
    def sessions_get(sid: str, request: Request) -> SessionDetail:
        ctx = ctx_for(request)
        try:
            s, messages = load_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        # Look up the agent's initial_user_prompt; fall back to "" if the
        # agent has been deleted since session creation.
        try:
            ag = get_agent(s.agent_id)
            initial = ag.initial_user_prompt
        except AgentError:
            initial = ""
        return SessionDetail(
            id=s.id, title=s.title, last_active=s.last_active,
            started_at=s.started_at, model=s.model, toggles=s.toggles,
            messages=messages, next_event_seq=s.next_event_seq,
            is_running=supervisor(request).is_running(sid),
            loaded_skills=list(s.loaded_skills),
            injected_lesson_ids=list(s.injected_ids),
            agent_id=s.agent_id,
            initial_user_prompt=initial,
        )

    @app.patch("/api/sessions/{sid}", response_model=SessionSummary)
    def sessions_patch(
        sid: str, req: PatchSessionRequest, request: Request,
    ) -> SessionSummary:
        """Update title and/or session-level toggles. Only the fields you
        send are touched; others retain their current value."""
        ctx = ctx_for(request)
        try:
            s, _msgs = load_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        # Refuse to mutate a session that's actively running — toggle
        # changes mid-turn would be observed inconsistently.
        if supervisor(request).is_running(sid):
            raise HTTPException(
                status_code=409,
                detail="cannot patch a session with a run in progress",
            )
        if req.title is not None:
            s.title = req.title
        # Apply only the toggles the client sent — leaves others untouched.
        for field in ("think_on", "net_on", "show_think", "show_tool_use", "show_reflection"):
            val = getattr(req, field)
            if val is not None:
                s.toggles = {**s.toggles, field: val}
        s.write_meta()
        return SessionSummary(
            id=s.id, title=s.title, last_active=s.last_active,
            started_at=s.started_at, model=s.model,
            message_count=count_messages(s),
            is_running=supervisor(request).is_running(sid),
            agent_id=s.agent_id,
        )

    @app.delete("/api/sessions/{sid}", status_code=204)
    def sessions_delete(sid: str, request: Request) -> None:
        ctx = ctx_for(request)
        if supervisor(request).is_running(sid):
            raise HTTPException(
                status_code=409,
                detail=(
                    "cannot delete a session with a run in progress; "
                    "cancel the run first"
                ),
            )
        try:
            delete_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")

    @app.post("/api/sessions/{sid}/messages", status_code=202)
    def sessions_send(
        sid: str, req: SendMessageRequest, request: Request,
    ) -> dict[str, Any]:
        sup = supervisor(request)
        try:
            sup.submit_run(sid, req.content)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        except RunBusy:
            raise HTTPException(
                status_code=409,
                detail="a run is already in progress for this session",
            )
        return {"status": "accepted"}

    # -- reflection -------------------------------------------------------

    @app.post("/api/sessions/{sid}/reflect")
    def sessions_reflect(sid: str, request: Request) -> dict[str, Any]:
        """Run the reflector over the session's trace slice (from
        reflection_idx to end). Returns proposals; nothing is persisted.

        Stateless: the client gets the proposals, lets the user choose,
        and posts the chosen ops back to /reflect/apply.
        """
        ctx = ctx_for(request)
        try:
            s, messages = load_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        if supervisor(request).is_running(sid):
            raise HTTPException(
                status_code=409,
                detail="cannot reflect on a session with a run in progress",
            )
        trace = messages[s.reflection_idx:]
        if not any(m.get("role") in ("user", "assistant") for m in trace):
            return {"proposals": [], "reason": "nothing to reflect on yet"}
        try:
            result = reflect(ctx.llm, trace, ctx.lessons.in_scope(ctx.lesson_scope))
        except ReflectorError as e:
            # Don't 502 — the model just produced unusable output. Return
            # a clean "nothing applied" with a reason the user can act on
            # (retry usually fixes it; the reflector is non-deterministic).
            return {
                "proposals": [],
                "reason": f"reflector returned malformed output ({e}); try again",
            }
        proposals: list[dict] = []
        for i, op in enumerate(result.ops):
            if isinstance(op, CreateOp):
                proposals.append({
                    "index": i, "kind": "create", "lesson": op.lesson,
                })
            elif isinstance(op, UpdateOp):
                proposals.append({
                    "index": i, "kind": "update", "id": op.id, "fields": op.fields,
                })
            elif isinstance(op, SkipOp):
                proposals.append({
                    "index": i, "kind": "skip", "reason": op.reason,
                })
        return {"proposals": proposals}

    @app.post("/api/sessions/{sid}/reflect/apply")
    def sessions_reflect_apply(
        sid: str, req: ReflectApplyRequest, request: Request,
    ) -> dict[str, Any]:
        """Apply the ops the user chose to keep. Each op is one of:
          {kind: "create", lesson: {...}}
          {kind: "update", id: "...", fields: {...}}
        SkipOps are ignored (they were informational). Snapshots the
        current trace slice once if any create/update is applied.
        """
        ctx = ctx_for(request)
        try:
            s, messages = load_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        if supervisor(request).is_running(sid):
            raise HTTPException(
                status_code=409,
                detail="cannot apply reflection during an active run",
            )
        trace = messages[s.reflection_idx:]
        creates_or_updates = [
            o for o in req.ops if o.get("kind") in ("create", "update")
        ]
        snapshot_path: str | None = None
        if creates_or_updates:
            slug = "reflection"
            first = creates_or_updates[0]
            if first["kind"] == "create":
                slug = str(first.get("lesson", {}).get("title", slug))
            else:
                slug = str(first.get("id", slug))
            snapshot_path = ctx.lessons.write_trace_snapshot(trace, slug_hint=slug)
        results: list[dict] = []
        for op in req.ops:
            kind = op.get("kind")
            try:
                if kind == "create":
                    data = dict(op.get("lesson") or {})
                    if snapshot_path and "source_trace" not in data:
                        data["source_trace"] = snapshot_path
                    created = ctx.lessons.create_lesson(data)
                    results.append({
                        "kind": "create", "status": "ok", "id": created.id,
                    })
                elif kind == "update":
                    updated = ctx.lessons.update_lesson(
                        str(op.get("id", "")), dict(op.get("fields") or {}),
                    )
                    results.append({
                        "kind": "update", "status": "ok", "id": updated.id,
                    })
                elif kind == "skip":
                    results.append({"kind": "skip", "status": "ignored"})
                else:
                    results.append({"kind": kind, "status": "unknown"})
            except WriteError as e:
                results.append({"kind": kind, "status": "error", "error": str(e)})
            except Exception as e:
                results.append({
                    "kind": kind, "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                })
        # Advance reflection_idx so the next /reflect doesn't see the
        # same trace.
        s.reflection_idx = len(messages)
        s.write_meta()
        return {"results": results, "reflection_idx": s.reflection_idx}

    @app.delete("/api/lessons/{lesson_id}", status_code=204)
    def lessons_forget(lesson_id: str, request: Request) -> None:
        ctx = ctx_for(request)
        try:
            ctx.lessons.forget_lesson(lesson_id)
        except WriteError as e:
            # forget_lesson raises WriteError when the id is unknown too.
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/sessions/{sid}/cancel", status_code=200)
    def sessions_cancel(sid: str, request: Request) -> dict[str, Any]:
        sup = supervisor(request)
        # Verify the session exists; 404 otherwise.
        try:
            load_session(ctx_for(request).workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")
        cancelled = sup.cancel_run(sid)
        if not cancelled:
            raise HTTPException(
                status_code=409,
                detail="no run in progress for this session",
            )
        # Cancellation is cooperative — the agent loop will exit at the
        # next safe checkpoint and emit `run_cancelled`. We don't wait
        # here; the SSE stream surfaces the event when it lands.
        return {"status": "cancelling"}

    @app.get("/api/sessions/{sid}/stream")
    async def sessions_stream(sid: str, request: Request, since: int = 0):
        sup = supervisor(request)
        ctx = ctx_for(request)
        # Cheap existence probe before opening the stream.
        try:
            load_session(ctx.workspace, sid)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no session {sid}")

        async def gen():
            try:
                async for ev in sup.subscribe(sid, since=since):
                    yield {
                        "event": ev.type,
                        "id": str(ev.seq),
                        "data": _json(ev.to_dict()),
                    }
                    if await request.is_disconnected():
                        return
            except asyncio.CancelledError:
                return

        # ping_message_factory keeps proxies from killing idle connections;
        # 15s matches the design doc.
        return EventSourceResponse(gen(), ping=15)

    # -- static front-end (production build) -----------------------------
    #
    # The built SPA lives at <repo>/web/dist. If it's present, mount it at
    # /assets/ for hashed bundles and serve index.html for any non-/api
    # path. Registered last so the API routes above take precedence. In
    # dev the Vite server runs on 5173 and proxies /api here, so this
    # mount only matters after `npm run build`.
    web_dist = _find_web_dist()
    if web_dist is not None:
        app.mount(
            "/assets",
            StaticFiles(directory=web_dist / "assets"),
            name="spa-assets",
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str):
            # /api/* is already matched above; the dynamic catch-all
            # would otherwise eat unknown /api/* URLs and serve HTML.
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)
            candidate = web_dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    return app


# -- json helper ---------------------------------------------------------


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _switch_workspace(app: FastAPI, new_workspace: Path) -> None:
    """Swap the active workspace under app.state.ws_lock.

    Refuses if any session in the current supervisor has a run in flight
    — the user must cancel first. If there is no current supervisor (the
    server started with no workspace), this is the first activation and
    no lock check is needed. Open SSE subscribers on the previous
    workspace are orphaned (their queues stop receiving events); clients
    will time out and reconnect against the new workspace.
    """
    state = app.state
    with state.ws_lock:
        sup: RunSupervisor | None = state.supervisor
        if sup is not None:
            # Reach into the supervisor only to scan locks — we own the
            # supervisor we're about to discard, so this is fine.
            for sid, lock in sup._run_locks.items():  # noqa: SLF001
                if lock.locked():
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"session {sid} has a run in progress — "
                            f"cancel it before switching workspace"
                        ),
                    )
        new_ctx = build_run_context(new_workspace)
        for issue in new_ctx.lesson_issues:
            print(
                f"(lesson {issue.path.name}: {issue.reason})", file=sys.stderr,
            )
        new_sup = RunSupervisor(new_ctx, state.loop)
        state.ctx = new_ctx
        state.supervisor = new_sup
        _write_last_workspace(new_workspace)
        print(
            f"leo-server: switched workspace to {new_workspace}",
            file=sys.stderr,
        )


def _find_web_dist() -> Path | None:
    """Locate the built front-end. Looks at <repo>/web/dist relative to
    the installed `leo` package; returns None if not built yet."""
    # leo/server/app.py → leo/server → leo → src → <repo>
    here = Path(__file__).resolve()
    repo = here.parents[3]
    cand = repo / "web" / "dist"
    return cand if cand.is_dir() else None


# -- entry point ----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="leo-server",
        description="Run the Leo web server (single user, no auth).",
    )
    parser.add_argument(
        "--workspace", metavar="PATH",
        help=(
            "initial workspace directory (must contain .leo/). "
            "Default: $LEO_WS, then last-used, then data_root, then cwd."
        ),
    )
    parser.add_argument(
        "--data-root", metavar="PATH",
        help=(
            "root of the workspace picker — the browsable area. "
            "Default: $LEO_DATA_ROOT, then $HOME."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    data_root = _resolve_data_root(args.data_root)
    workspace = _resolve_workspace(args.workspace, data_root)
    app = create_app(workspace, data_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

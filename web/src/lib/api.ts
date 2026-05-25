/** Typed client for the Leo HTTP API.
 *
 * All routes are same-origin in production (the Python server serves the
 * built SPA) and same-origin via the Vite proxy in development. So bare
 * `/api/...` paths work in both modes — no base URL needed.
 */

export type Me = {
  /** null when the server has no workspace selected yet — the UI must
   * open one via the picker before most endpoints will work. */
  workspace: string | null
  data_root: string
  model: string | null
}

export type FsEntry = {
  name: string
  path: string
  is_workspace: boolean
}

export type FsListing = {
  path: string
  data_root: string
  current_is_workspace: boolean
  parent: string | null
  entries: FsEntry[]
}

export type SessionSummary = {
  id: string
  title: string
  last_active: string
  started_at: string
  model: string | null
  message_count: number
  is_running: boolean
  agent_id: string
}

export type AgentDTO = {
  id: string
  name: string
  description: string
  system_prompt: string
  initial_user_prompt: string
  skills: string[]
  default_think: boolean
  builtin: boolean
}

export type Message = {
  role: "system" | "user" | "assistant" | "tool"
  content: string | null
  tool_calls?: Array<{
    id: string
    type: "function"
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
}

export type SessionDetail = {
  id: string
  title: string
  last_active: string
  started_at: string
  model: string | null
  toggles: Record<string, unknown>
  messages: Message[]
  next_event_seq: number
  is_running: boolean
  loaded_skills: string[]
  injected_lesson_ids: string[]
  agent_id: string
  initial_user_prompt: string
}

export type SkillInfo = {
  name: string
  description: string
}

export type LessonInfo = {
  id: string
  title: string
  category: "preference" | "fact" | "process" | "gotcha"
  trigger_type: "always" | "on_prompt" | "on_monologue" | "on_tool_call"
  trigger_keywords: string[]
  trigger_tool: string | null
  scope: Record<string, string[]>
  rule: string
  why: string
  how_to_apply: string
  created: string
  updated: string
}

export type AgentEvent = {
  seq: number
  ts: string
  type: string
  payload: Record<string, unknown>
}

async function http<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const resp = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  })
  if (!resp.ok) {
    const body = await resp.text().catch(() => "")
    throw new ApiError(resp.status, resp.statusText, body)
  }
  if (resp.status === 204) return undefined as T
  return resp.json() as Promise<T>
}

export class ApiError extends Error {
  status: number
  body: string
  constructor(status: number, statusText: string, body: string) {
    super(`${status} ${statusText}: ${body}`)
    this.status = status
    this.body = body
  }
}

export type ReflectionProposal =
  | { index: number; kind: "create"; lesson: Record<string, unknown> }
  | {
      index: number
      kind: "update"
      id: string
      fields: Record<string, unknown>
    }
  | { index: number; kind: "skip"; reason: string }

export type ReflectionApplyResult = {
  kind: string
  status: string
  id?: string
  error?: string
}

export const api = {
  me: () => http<Me>("/api/me"),
  listSessions: () => http<SessionSummary[]>("/api/sessions"),
  patchSession: (
    sid: string,
    body: {
      title?: string
      think_on?: boolean
      net_on?: boolean
      show_think?: boolean
      show_tool_use?: boolean
      show_reflection?: boolean
    },
  ) =>
    http<SessionSummary>(`/api/sessions/${sid}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  reflect: (sid: string) =>
    http<{ proposals: ReflectionProposal[]; reason?: string }>(
      `/api/sessions/${sid}/reflect`,
      { method: "POST" },
    ),
  applyReflection: (sid: string, ops: object[]) =>
    http<{ results: ReflectionApplyResult[]; reflection_idx: number }>(
      `/api/sessions/${sid}/reflect/apply`,
      { method: "POST", body: JSON.stringify({ ops }) },
    ),
  forgetLesson: (id: string) =>
    http<void>(`/api/lessons/${id}`, { method: "DELETE" }),
  createSession: (opts: { title?: string; agent_id?: string } = {}) =>
    http<SessionSummary>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        title: opts.title ?? null,
        agent_id: opts.agent_id ?? null,
      }),
    }),
  getSession: (sid: string) => http<SessionDetail>(`/api/sessions/${sid}`),
  deleteSession: (sid: string) =>
    http<void>(`/api/sessions/${sid}`, { method: "DELETE" }),
  sendMessage: (sid: string, content: string) =>
    http<{ status: string }>(`/api/sessions/${sid}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  cancelRun: (sid: string) =>
    http<{ status: string }>(`/api/sessions/${sid}/cancel`, { method: "POST" }),
  listSkills: () => http<SkillInfo[]>("/api/skills"),
  listLessons: () => http<LessonInfo[]>("/api/lessons"),
  fsList: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ""
    return http<FsListing>(`/api/fs/list${qs}`)
  },
  listAgents: () => http<AgentDTO[]>("/api/agents"),
  getAgent: (id: string) => http<AgentDTO>(`/api/agents/${id}`),
  createAgent: (a: Omit<AgentDTO, "builtin">) =>
    http<AgentDTO>("/api/agents", {
      method: "POST",
      body: JSON.stringify(a),
    }),
  updateAgent: (id: string, a: Omit<AgentDTO, "id" | "builtin">) =>
    http<AgentDTO>(`/api/agents/${id}`, {
      method: "PUT",
      body: JSON.stringify(a),
    }),
  deleteAgent: (id: string) =>
    http<void>(`/api/agents/${id}`, { method: "DELETE" }),
  openWorkspace: (path: string) =>
    http<Me>("/api/workspace/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  createWorkspace: (path: string) =>
    http<Me>("/api/workspace/create", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
}

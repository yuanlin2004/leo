import { useCallback, useEffect, useState } from "react"
import { TopBar } from "@/components/TopBar"
import { SessionList } from "@/components/SessionList"
import { ChatColumn } from "@/components/ChatColumn"
import { ObservabilityColumn } from "@/components/ObservabilityColumn"
import { SessionSettingsDialog } from "@/components/SessionSettingsDialog"
import { ReflectionDialog } from "@/components/ReflectionDialog"
import { WorkspacePickerDialog } from "@/components/WorkspacePickerDialog"
import { AgentPickerDialog } from "@/components/AgentPickerDialog"
import { AgentsDialog } from "@/components/AgentsDialog"
import { api } from "@/lib/api"
import type { AgentEvent, Me, Message, SessionDetail, SessionSummary } from "@/lib/api"
import { useEventStream } from "@/lib/sse"

export default function App() {
  // -- dialogs (declared early so workspace-init can flip the picker) ---
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [reflectOpen, setReflectOpen] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [agentPickerOpen, setAgentPickerOpen] = useState(false)
  const [agentsBuilderOpen, setAgentsBuilderOpen] = useState(false)
  // After a session is created with an agent that has an initial_user_prompt,
  // we want to prefill the chat input. ChatColumn reads this and clears it.
  const [pendingInputSeed, setPendingInputSeed] = useState<string>("")

  // -- server identity (workspace + model) ------------------------------
  const [me, setMe] = useState<Me | null>(null)
  useEffect(() => {
    api.me()
      .then((m) => {
        setMe(m)
        // Auto-open the picker when the server has no workspace yet —
        // otherwise nothing else will work and the user has no signal.
        if (m.workspace === null) setPickerOpen(true)
      })
      .catch((e) => console.error("me", e))
  }, [])

  const hasWorkspace = me?.workspace != null

  // -- sessions list -----------------------------------------------------
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeSid, setActiveSid] = useState<string | null>(null)

  const refreshSessions = useCallback(async () => {
    const list = await api.listSessions()
    setSessions(list)
    setActiveSid((cur) => {
      if (cur && list.some((s) => s.id === cur)) return cur
      return list[0]?.id ?? null
    })
  }, [])

  // Don't touch session endpoints until a workspace is selected.
  useEffect(() => {
    if (!hasWorkspace) return
    refreshSessions().catch((e) => console.error("listSessions", e))
  }, [hasWorkspace, refreshSessions])

  // Light polling of the sessions list so the sidebar's `is_running`
  // indicator stays fresh on sessions other than the active one. Gated
  // on hasWorkspace so we don't spam 412s in the no-workspace state.
  useEffect(() => {
    if (!hasWorkspace) return
    const id = window.setInterval(() => {
      refreshSessions().catch(() => {})
    }, 3000)
    return () => window.clearInterval(id)
  }, [hasWorkspace, refreshSessions])

  // -- active session detail --------------------------------------------
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [events, setEvents] = useState<AgentEvent[]>([])
  /** Buffer for the in-progress LLM stream. Reset on each llm_message
   * (turn boundary) and on session switch. Two separate text streams
   * because reply and think tokens come from the same callback chain
   * but render in different places. */
  const [streamDraft, setStreamDraft] = useState<{
    reply: string
    think: string
  }>({ reply: "", think: "" })

  const refreshDetail = useCallback(async (sid: string) => {
    const d = await api.getSession(sid)
    setDetail(d)
    return d
  }, [])

  useEffect(() => {
    if (!activeSid) {
      setDetail(null)
      setEvents([])
      setStreamDraft({ reply: "", think: "" })
      return
    }
    setEvents([])
    setStreamDraft({ reply: "", think: "" })
    refreshDetail(activeSid).catch((e) => console.error("getSession", e))
  }, [activeSid, refreshDetail])

  // -- live event stream -------------------------------------------------
  // Each event also triggers a session-detail refetch so messages.jsonl
  // contents (which we don't ship through SSE) appear in the chat view.
  // On turn-terminal events we also refresh the sessions list so the
  // left column reflects updated title (derived from first user prompt)
  // and message_count.
  useEventStream(activeSid, (ev) => {
    if (ev.type === "llm_token") {
      // High-frequency ephemeral event — buffer in state, don't push
      // to the events list (would explode it; not in events.jsonl).
      const p = ev.payload as { kind?: string; text?: string }
      const text = p.text ?? ""
      if (!text) return
      setStreamDraft((prev) =>
        p.kind === "think"
          ? { ...prev, think: prev.think + text }
          : { ...prev, reply: prev.reply + text },
      )
      return
    }
    setEvents((prev) => [...prev, ev])
    if (
      ev.type === "llm_message" ||
      ev.type === "tool_end" ||
      ev.type === "run_end" ||
      ev.type === "run_error" ||
      ev.type === "run_cancelled" ||
      ev.type === "lesson_injected"
    ) {
      if (activeSid) refreshDetail(activeSid).catch(() => {})
    }
    if (
      ev.type === "llm_message" ||
      ev.type === "run_cancelled" ||
      ev.type === "run_error"
    ) {
      // Turn boundary or interrupted run: the persisted message takes
      // over from the streaming draft.
      setStreamDraft({ reply: "", think: "" })
    }
    if (ev.type === "run_start") {
      // Fresh turn starting — start with an empty draft.
      setStreamDraft({ reply: "", think: "" })
    }
    if (
      ev.type === "run_end" ||
      ev.type === "run_error" ||
      ev.type === "run_cancelled"
    ) {
      refreshSessions().catch(() => {})
    }
  })

  // -- actions -----------------------------------------------------------
  const messages: Message[] = detail?.messages ?? []
  const isRunning = detail?.is_running ?? false

  const handleCreate = useCallback(async () => {
    // Open the picker. Actual creation happens in handleCreateWithAgent.
    setAgentPickerOpen(true)
  }, [])

  const handleCreateWithAgent = useCallback(
    async (agent_id: string, title: string | null) => {
      const s = await api.createSession({
        agent_id,
        title: title ?? undefined,
      })
      await refreshSessions()
      setActiveSid(s.id)
      // If the chosen agent has an initial user prompt, seed the input
      // box once. The detail GET happens via the activeSid effect.
      try {
        const detail = await api.getSession(s.id)
        if (detail.initial_user_prompt) {
          setPendingInputSeed(detail.initial_user_prompt)
        }
      } catch {
        /* non-fatal */
      }
    },
    [refreshSessions],
  )

  const handleDelete = useCallback(
    async (sid: string) => {
      await api.deleteSession(sid)
      await refreshSessions()
    },
    [refreshSessions],
  )

  const handleSend = useCallback(
    async (content: string) => {
      if (!activeSid) return
      try {
        await api.sendMessage(activeSid, content)
        setDetail((d) => (d ? { ...d, is_running: true } : d))
      } catch (e) {
        console.error("sendMessage", e)
      }
    },
    [activeSid],
  )

  const handleCancel = useCallback(async () => {
    if (!activeSid) return
    try {
      await api.cancelRun(activeSid)
      // is_running stays true until the run_cancelled event arrives via SSE
      // and triggers a detail refetch — that's the source of truth.
    } catch (e) {
      console.error("cancelRun", e)
    }
  }, [activeSid])

  // -- post-switch refresh ----------------------------------------------

  const handleWorkspaceSwitched = useCallback(async () => {
    // Force a full state reset: the new workspace has its own sessions,
    // skills, and lessons. The current active session and event stream
    // are no longer valid.
    setActiveSid(null)
    setDetail(null)
    setEvents([])
    try {
      setMe(await api.me())
      await refreshSessions()
    } catch (e) {
      console.error("post-switch refresh", e)
    }
  }, [refreshSessions])

  const handlePatchSession = useCallback(
    async (patch: { title?: string; think_on?: boolean; net_on?: boolean }) => {
      if (!activeSid) return
      await api.patchSession(activeSid, patch)
      await refreshDetail(activeSid)
      await refreshSessions()
    },
    [activeSid, refreshDetail, refreshSessions],
  )

  // `canReflect` is true when there's at least one assistant message in the
  // session — i.e. something happened. The server enforces the precise
  // semantic (any message after reflection_idx); this is a UI hint.
  const canReflect =
    messages.some((m) => m.role === "assistant") && !isRunning

  // -- theme -------------------------------------------------------------
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window === "undefined") return false
    const saved = localStorage.getItem("leo-theme")
    if (saved) return saved === "dark"
    return window.matchMedia("(prefers-color-scheme: dark)").matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark)
    localStorage.setItem("leo-theme", isDark ? "dark" : "light")
  }, [isDark])

  return (
    <div className="h-full flex flex-col">
      <TopBar
        workspace={me?.workspace ?? null}
        model={detail?.model ?? me?.model ?? null}
        isRunning={isRunning}
        isDark={isDark}
        onToggleTheme={() => setIsDark(!isDark)}
        onOpenWorkspacePicker={() => setPickerOpen(true)}
        onOpenAgentsBuilder={() => setAgentsBuilderOpen(true)}
      />
      <div className="flex-1 flex min-h-0">
        <SessionList
          sessions={sessions}
          activeSid={activeSid}
          onSelect={setActiveSid}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
        <ChatColumn
          title={detail?.title ?? null}
          messages={messages}
          events={events}
          streamDraft={streamDraft}
          isRunning={isRunning}
          onSend={handleSend}
          onCancel={handleCancel}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenReflect={() => setReflectOpen(true)}
          canReflect={canReflect}
          showThink={Boolean((detail?.toggles as Record<string, unknown>)?.show_think ?? true)}
          showToolUse={Boolean((detail?.toggles as Record<string, unknown>)?.show_tool_use ?? true)}
          showReflection={Boolean((detail?.toggles as Record<string, unknown>)?.show_reflection ?? false)}
          inputSeed={pendingInputSeed}
          onConsumeInputSeed={() => setPendingInputSeed("")}
        />
        <ObservabilityColumn events={events} detail={detail} />
      </div>
      {detail && (
        <SessionSettingsDialog
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          detail={detail}
          onSave={handlePatchSession}
        />
      )}
      {activeSid && (
        <ReflectionDialog
          open={reflectOpen}
          onClose={() => setReflectOpen(false)}
          sid={activeSid}
          onApplied={() => {
            // Lessons list will be re-fetched the next time the Lessons
            // tab mounts; nothing to update here right now.
          }}
        />
      )}
      <WorkspacePickerDialog
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        currentWorkspace={me?.workspace ?? null}
        onSwitched={handleWorkspaceSwitched}
      />
      <AgentPickerDialog
        open={agentPickerOpen}
        onClose={() => setAgentPickerOpen(false)}
        onCreate={handleCreateWithAgent}
      />
      <AgentsDialog
        open={agentsBuilderOpen}
        onClose={() => setAgentsBuilderOpen(false)}
      />
    </div>
  )
}

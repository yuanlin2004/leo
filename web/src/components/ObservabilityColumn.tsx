import { useEffect, useState } from "react"
import { Activity, Wrench, BookOpen, ChevronRight, ChevronDown, AlertCircle, Trash2 } from "lucide-react"
import { api } from "@/lib/api"
import type { AgentEvent, LessonInfo, SessionDetail, SkillInfo } from "@/lib/api"
import { cn } from "@/lib/utils"

type Tab = "trace" | "skills" | "lessons"

type Props = {
  events: AgentEvent[]
  detail: SessionDetail | null
  /** Drawer state below the lg breakpoint. Ignored on lg+. */
  mobileOpen: boolean
}

export function ObservabilityColumn({ events, detail, mobileOpen }: Props) {
  const [tab, setTab] = useState<Tab>("trace")
  return (
    <aside
      className={cn(
        // Drawer (default): solid card background.
        "w-96 max-w-[90vw] border-l flex flex-col bg-card z-40",
        // Desktop: static, in-flow, original tinted background.
        "lg:static lg:translate-x-0 lg:z-auto shrink-0 lg:bg-muted/30",
        "fixed top-12 bottom-0 right-0 transition-transform duration-200",
        mobileOpen ? "translate-x-0 shadow-xl" : "translate-x-full",
      )}
    >
      <div className="flex border-b">
        <TabButton active={tab === "trace"} onClick={() => setTab("trace")}>
          <Activity className="w-3.5 h-3.5" />
          Trace
          <span className="text-[10px] text-muted-foreground">{events.length}</span>
        </TabButton>
        <TabButton active={tab === "skills"} onClick={() => setTab("skills")}>
          <Wrench className="w-3.5 h-3.5" />
          Skills
        </TabButton>
        <TabButton active={tab === "lessons"} onClick={() => setTab("lessons")}>
          <BookOpen className="w-3.5 h-3.5" />
          Lessons
        </TabButton>
      </div>
      <div className="flex-1 overflow-y-auto">
        {tab === "trace" && <TraceTab events={events} />}
        {tab === "skills" && <SkillsTab loadedNames={detail?.loaded_skills ?? []} />}
        {tab === "lessons" && (
          <LessonsTab injectedIds={detail?.injected_lesson_ids ?? []} />
        )}
      </div>
    </aside>
  )
}

function TabButton({
  active, onClick, children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex-1 flex items-center justify-center gap-1.5 px-3 h-10 font-mono text-xs uppercase tracking-wider transition-colors",
        active
          ? "text-foreground border-b-2 border-[var(--color-illini-orange)] -mb-px"
          : "text-muted-foreground hover:text-foreground hover:bg-accent/30",
      )}
    >
      {children}
    </button>
  )
}

// -- Trace tab ------------------------------------------------------------

type Turn = {
  events: AgentEvent[]
  status: "running" | "ok" | "error" | "cancelled"
}

function groupTurns(events: AgentEvent[]): Turn[] {
  const turns: Turn[] = []
  let cur: Turn | null = null
  for (const e of events) {
    if (e.type === "run_start") {
      cur = { events: [e], status: "running" }
      turns.push(cur)
    } else if (cur) {
      cur.events.push(e)
      if (e.type === "run_end") cur.status = "ok"
      else if (e.type === "run_error") cur.status = "error"
      else if (e.type === "run_cancelled") cur.status = "cancelled"
    } else {
      // Orphan event (shouldn't happen with current backend)
      turns.push({ events: [e], status: "ok" })
    }
  }
  return turns
}

function TraceTab({ events }: { events: AgentEvent[] }) {
  const turns = groupTurns(events)
  if (turns.length === 0) {
    return <Empty>no events yet</Empty>
  }
  return (
    <div className="p-2 space-y-2">
      {turns.map((t, i) => (
        <TurnBlock key={i} turn={t} index={i + 1} />
      ))}
    </div>
  )
}

function TurnBlock({ turn, index }: { turn: Turn; index: number }) {
  const [open, setOpen] = useState(true)
  const startSeq = turn.events[0]?.seq
  const endSeq = turn.events[turn.events.length - 1]?.seq
  const toolCount = turn.events.filter((e) => e.type === "tool_start").length
  const replans = turn.events.filter((e) => e.type === "replan").length
  return (
    <div className="rounded border bg-background">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1.5 px-2 py-1.5 font-mono text-[11px] hover:bg-accent/30 transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span className="font-semibold">turn #{index}</span>
        <span className="text-muted-foreground">
          seq {startSeq}–{endSeq}
        </span>
        <span className="ml-auto flex items-center gap-2 text-muted-foreground">
          {toolCount > 0 && <span>{toolCount} tools</span>}
          {replans > 0 && <span className="text-[var(--color-illini-orange)]">{replans} replan</span>}
          <StatusDot status={turn.status} />
        </span>
      </button>
      {open && (
        <div className="border-t px-2 py-1 space-y-0.5">
          {turn.events.map((e) => (
            <EventRow key={e.seq} ev={e} />
          ))}
        </div>
      )}
    </div>
  )
}

function StatusDot({ status }: { status: Turn["status"] }) {
  const colors = {
    running: "bg-[var(--color-illini-orange)] animate-pulse",
    ok: "bg-emerald-500",
    error: "bg-red-500",
    cancelled: "bg-muted-foreground",
  } as const
  return <span className={cn("inline-block w-2 h-2 rounded-full", colors[status])} />
}

function EventRow({ ev }: { ev: AgentEvent }) {
  const summary = summarizeEvent(ev)
  return (
    <div className="flex items-baseline gap-2 py-0.5 font-mono text-[11px]">
      <span className="text-muted-foreground w-8 shrink-0 text-right">
        {ev.seq.toString().padStart(3, "0")}
      </span>
      <span className={cn("shrink-0", typeColor(ev.type))}>{ev.type}</span>
      {summary && (
        <span className="text-muted-foreground truncate">{summary}</span>
      )}
    </div>
  )
}

function typeColor(type: string): string {
  if (type === "tool_start" || type === "tool_end") return "text-[var(--color-illini-orange)]"
  if (type === "replan" || type === "lesson_injected") return "text-amber-600 dark:text-amber-400"
  if (type === "run_error") return "text-red-600 dark:text-red-400"
  if (type === "run_end") return "text-emerald-600 dark:text-emerald-400"
  if (type === "run_cancelled") return "text-muted-foreground"
  return "text-foreground"
}

function summarizeEvent(ev: AgentEvent): string {
  const p = ev.payload as Record<string, unknown>
  switch (ev.type) {
    case "tool_start":
    case "tool_end": {
      const name = String(p.name ?? "")
      const args = String(p.arguments ?? "")
      if (ev.type === "tool_start") {
        return `${name}(${truncate(args, 60)})`
      }
      const len = Number(p.result_len ?? 0)
      return `${name} → ${len} chars`
    }
    case "llm_message": {
      const tcs = Number(p.tool_call_count ?? 0)
      const cl = Number(p.content_len ?? 0)
      return tcs > 0 ? `${tcs} tool call${tcs > 1 ? "s" : ""}` : `${cl} chars`
    }
    case "lesson_injected": {
      const phase = String(p.phase ?? "")
      const ids = (p.ids as string[]) ?? []
      return `${phase}: ${ids.join(", ")}`
    }
    case "replan": {
      const reason = String(p.reason ?? "")
      const ids = (p.ids as string[]) ?? []
      return `${reason}: ${ids.join(", ")}`
    }
    case "run_error": {
      return `${String(p.error_type ?? "")}: ${truncate(String(p.message ?? ""), 80)}`
    }
    case "run_end": {
      const tc = Number(p.tool_calls ?? 0)
      return `${tc} tool calls`
    }
    default:
      return ""
  }
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n) + "…"
}

// -- Skills tab -----------------------------------------------------------

function SkillsTab({ loadedNames }: { loadedNames: string[] }) {
  const [skills, setSkills] = useState<SkillInfo[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.listSkills().then(setSkills).catch((e) => setErr(String(e)))
  }, [])
  if (err) return <ErrorBox text={err} />
  if (!skills) return <Empty>loading…</Empty>
  if (skills.length === 0) return <Empty>no skills installed</Empty>
  const loadedSet = new Set(loadedNames)
  return (
    <div className="p-2 space-y-1.5">
      {skills.map((s) => (
        <div key={s.name} className="rounded border bg-background px-2 py-1.5">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-semibold">{s.name}</span>
            {loadedSet.has(s.name) && (
              <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--color-illini-orange)] text-white">
                loaded
              </span>
            )}
          </div>
          <div className="text-xs text-muted-foreground mt-1 leading-snug">
            {s.description}
          </div>
        </div>
      ))}
    </div>
  )
}

// -- Lessons tab ----------------------------------------------------------

const LESSON_CATEGORIES: LessonInfo["category"][] = [
  "preference", "fact", "process", "gotcha",
]

function LessonsTab({ injectedIds }: { injectedIds: string[] }) {
  const [lessons, setLessons] = useState<LessonInfo[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const refresh = () => {
    api.listLessons().then(setLessons).catch((e) => setErr(String(e)))
  }
  useEffect(() => {
    refresh()
  }, [])

  const handleForget = async (id: string) => {
    if (!confirm(`Forget lesson ${id}? This deletes the markdown file.`)) return
    try {
      await api.forgetLesson(id)
      refresh()
    } catch (e) {
      setErr(String(e))
    }
  }

  if (err) return <ErrorBox text={err} />
  if (!lessons) return <Empty>loading…</Empty>
  if (lessons.length === 0) return <Empty>no lessons installed</Empty>
  const byCat = new Map<string, LessonInfo[]>()
  for (const l of lessons) {
    const arr = byCat.get(l.category) ?? []
    arr.push(l)
    byCat.set(l.category, arr)
  }
  const injectedSet = new Set(injectedIds)
  return (
    <div className="p-2 space-y-3">
      {LESSON_CATEGORIES.map((cat) => {
        const items = byCat.get(cat) ?? []
        if (items.length === 0) return null
        return (
          <div key={cat}>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground px-1 py-1">
              {cat} · {items.length}
            </div>
            <div className="space-y-1">
              {items.map((l) => (
                <LessonRow
                  key={l.id}
                  lesson={l}
                  injected={injectedSet.has(l.id)}
                  onForget={() => handleForget(l.id)}
                />
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function LessonRow({
  lesson, injected, onForget,
}: {
  lesson: LessonInfo
  injected: boolean
  onForget: () => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="group rounded border bg-background">
      <div className="flex items-start">
        <button
          onClick={() => setOpen(!open)}
          className="flex-1 flex items-start gap-1.5 px-2 py-1.5 text-left hover:bg-accent/30 transition-colors min-w-0"
        >
          {open ? (
            <ChevronDown className="w-3 h-3 mt-0.5 shrink-0" />
          ) : (
            <ChevronRight className="w-3 h-3 mt-0.5 shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium truncate">{lesson.title}</div>
            <div className="font-mono text-[10px] text-muted-foreground mt-0.5">
              {lesson.id} · {lesson.trigger_type}
              {lesson.trigger_keywords.length > 0 && (
                <> · {lesson.trigger_keywords.join(", ")}</>
              )}
            </div>
          </div>
          {injected && (
            <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--color-illini-orange)] text-white shrink-0">
              in scope
            </span>
          )}
        </button>
        <button
          onClick={onForget}
          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity p-1.5"
          aria-label="Forget lesson"
          title="Forget this lesson (delete the file)"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
      {open && (
        <div className="border-t px-2 py-2 text-xs space-y-2">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
              Rule
            </div>
            <div className="leading-snug">{lesson.rule}</div>
          </div>
          {lesson.why && (
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                Why
              </div>
              <div className="text-muted-foreground leading-snug">
                {lesson.why}
              </div>
            </div>
          )}
          {lesson.how_to_apply && (
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                How to apply
              </div>
              <div className="text-muted-foreground leading-snug">
                {lesson.how_to_apply}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// -- shared ---------------------------------------------------------------

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="p-4 text-xs text-muted-foreground font-mono">
      {children}
    </div>
  )
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="p-3 m-2 rounded border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 text-xs">
      <div className="flex items-center gap-1.5 text-red-600 dark:text-red-400 font-mono mb-1">
        <AlertCircle className="w-3.5 h-3.5" />
        error
      </div>
      <div className="text-muted-foreground break-all">{text}</div>
    </div>
  )
}

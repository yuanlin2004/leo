import { useEffect, useRef, useState } from "react"
import { Send, Square, ChevronRight, ChevronDown, Settings, Sparkles, BookOpen } from "lucide-react"
import type { AgentEvent, Message } from "@/lib/api"
import { cn } from "@/lib/utils"

type Props = {
  title: string | null
  messages: Message[]
  events: AgentEvent[]
  /** Live token stream for the in-flight LLM call. Cleared at turn end. */
  streamDraft: { reply: string; think: string }
  isRunning: boolean
  onSend: (content: string) => void
  onCancel: () => void
  onOpenSettings: () => void
  onOpenReflect: () => void
  /** True iff there is anything meaningful since the last reflection. */
  canReflect: boolean
  /** Display toggles read from session.toggles. */
  showThink: boolean
  showToolUse: boolean
  showReflection: boolean
  /** Optional text to prefill the chat input. Caller is responsible for
   * clearing it when consumed; the column writes it into local state on
   * receipt. */
  inputSeed: string
  onConsumeInputSeed: () => void
}

export function ChatColumn({
  title, messages, events, streamDraft, isRunning, onSend, onCancel,
  onOpenSettings, onOpenReflect, canReflect,
  showThink, showToolUse, showReflection,
  inputSeed, onConsumeInputSeed,
}: Props) {
  const [draft, setDraft] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Consume an inbound seed once (e.g. agent's initial_user_prompt).
  // Only replace the draft if the user hasn't started typing.
  useEffect(() => {
    if (inputSeed && !draft) {
      setDraft(inputSeed)
      textareaRef.current?.focus()
    }
    if (inputSeed) onConsumeInputSeed()
    // Intentional: only re-run when inputSeed changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputSeed])

  // Auto-scroll to bottom on new message OR streaming progress.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length, streamDraft.reply.length, streamDraft.think.length])

  const visible = messages.filter((m) => {
    if (m.role === "tool") return showToolUse
    return m.role === "user" || m.role === "assistant"
  })

  const lessonChips = showReflection
    ? events.filter((e) => e.type === "lesson_injected")
    : []

  const submit = () => {
    const text = draft.trim()
    if (!text || isRunning) return
    onSend(text)
    setDraft("")
  }

  return (
    <section className="flex-1 flex flex-col min-w-0">
      <div className="flex items-center justify-between px-4 h-10 border-b shrink-0 bg-card">
        <h2 className="font-medium text-sm truncate">
          {title ?? <span className="text-muted-foreground">no session</span>}
        </h2>
        <div className="flex items-center gap-1">
          <button
            onClick={onOpenReflect}
            disabled={!canReflect || isRunning}
            className="flex items-center gap-1 text-xs font-mono px-2 py-1 rounded hover:bg-accent transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title={
              canReflect
                ? "Study the recent trace and propose lessons"
                : "Nothing new to reflect on yet"
            }
          >
            <Sparkles className="w-3.5 h-3.5" />
            reflect
          </button>
          <button
            onClick={onOpenSettings}
            disabled={isRunning}
            className="text-muted-foreground hover:text-foreground transition-colors p-1.5 rounded hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed"
            aria-label="Session settings"
            title="Session settings"
          >
            <Settings className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        {visible.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {visible.map((m, i) => (
              <MessageView key={i} m={m} showThink={showThink} showToolUse={showToolUse} />
            ))}
            {(streamDraft.reply || streamDraft.think) && (
              <StreamingDraft
                draft={streamDraft}
                showThink={showThink}
              />
            )}
            {lessonChips.length > 0 && <ReflectionStrip events={lessonChips} />}
            {isRunning && !streamDraft.reply && !streamDraft.think && (
              <StreamingIndicator />
            )}
          </div>
        )}
      </div>
      <div className="border-t px-4 sm:px-6 py-3 sm:py-4 bg-card">
        <div className="max-w-3xl mx-auto flex flex-col sm:flex-row gap-2 sm:items-end">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={
              isRunning ? "Waiting for response…" : "Send a message (Enter to send, Shift+Enter for newline)"
            }
            disabled={isRunning}
            rows={2}
            className="flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 font-sans"
          />
          {isRunning ? (
            <button
              onClick={onCancel}
              className="flex items-center justify-center gap-1 w-full sm:w-auto px-4 py-2 rounded-md border border-[var(--color-illini-orange)] text-[var(--color-illini-orange)] hover:bg-[var(--color-illini-orange)]/10 transition-colors font-mono text-sm"
              title="Cancel the in-flight run"
            >
              <Square className="w-4 h-4" />
              cancel
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!draft.trim()}
              className="flex items-center justify-center gap-1 w-full sm:w-auto px-4 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed font-mono text-sm"
            >
              <Send className="w-4 h-4" />
              send
            </button>
          )}
        </div>
      </div>
    </section>
  )
}

function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-sm">
      no messages yet — start the conversation below
    </div>
  )
}

function StreamingIndicator() {
  return (
    <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
      <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-illini-orange)] animate-pulse" />
      thinking…
    </div>
  )
}

/** The live, in-progress assistant message. Replaced by the persisted
 * message in `detail.messages` once `llm_message` fires. */
function StreamingDraft({
  draft, showThink,
}: {
  draft: { reply: string; think: string }
  showThink: boolean
}) {
  return (
    <div className="space-y-2">
      {draft.think && showThink && (
        <StreamingThinkBlock text={draft.think} />
      )}
      {draft.reply && (
        <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {draft.reply}
          <span className="inline-block w-1.5 h-4 ml-0.5 align-text-bottom bg-[var(--color-illini-orange)] animate-pulse" />
        </div>
      )}
      {!draft.reply && draft.think && (
        // Thinking phase only — show a small live indicator beneath the
        // think block so the user sees activity even before reply tokens
        // start flowing.
        <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-illini-orange)] animate-pulse" />
          thinking…
        </div>
      )}
    </div>
  )
}

function StreamingThinkBlock({ text }: { text: string }) {
  // Auto-expanded during streaming so the user can watch reasoning live.
  return (
    <div className="rounded border bg-muted/40 text-xs">
      <div className="flex items-center gap-1.5 px-3 py-1.5 font-mono text-muted-foreground border-b">
        <ChevronDown className="w-3 h-3" />
        thinking
        <span className="ml-auto inline-block w-2 h-2 rounded-full bg-[var(--color-illini-orange)] animate-pulse" />
      </div>
      <pre className="px-3 pb-2 pt-1 whitespace-pre-wrap break-words font-mono text-muted-foreground">
        {text}
      </pre>
    </div>
  )
}

function splitThink(content: string | null): { think: string; reply: string } {
  if (!content) return { think: "", reply: "" }
  const idx = content.lastIndexOf("</think>")
  if (idx === -1) return { think: "", reply: content.trim() }
  const think = content
    .slice(0, idx)
    .replace(/^\s*<think>\s*/, "")
    .trim()
  const reply = content.slice(idx + "</think>".length).trim()
  return { think, reply }
}

function MessageView({
  m, showThink, showToolUse,
}: {
  m: Message
  showThink: boolean
  showToolUse: boolean
}) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[var(--color-illini-orange)] text-white px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
          {m.content}
        </div>
      </div>
    )
  }
  if (m.role === "assistant") {
    const { think, reply } = splitThink(m.content)
    const hasToolCalls = (m.tool_calls?.length ?? 0) > 0
    return (
      <div className="space-y-2">
        {think && showThink && <ThinkBlock text={think} />}
        {reply && (
          <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
            {reply}
          </div>
        )}
        {hasToolCalls && showToolUse &&
          m.tool_calls!.map((tc) => (
            <ToolCallBadge key={tc.id} name={tc.function.name} args={tc.function.arguments} />
          ))}
      </div>
    )
  }
  if (m.role === "tool") {
    // showToolUse=false already filters tool messages out of `visible`;
    // this branch is reached only when display is enabled.
    return <ToolResultBlock content={m.content ?? ""} />
  }
  return null
}

function ReflectionStrip({ events }: { events: AgentEvent[] }) {
  return (
    <div className="border-t pt-3">
      <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
        <BookOpen className="w-3 h-3" />
        lessons fired this session
      </div>
      <div className="flex flex-wrap gap-1.5">
        {events.map((e) => {
          const p = e.payload as Record<string, unknown>
          const phase = String(p.phase ?? "")
          const ids = (p.ids as string[]) ?? []
          return (
            <span
              key={e.seq}
              className="inline-flex items-center gap-1.5 rounded border bg-muted/40 px-2 py-1 text-[11px] font-mono"
              title={String(p.text ?? "")}
            >
              <span className="text-[var(--color-illini-orange)] font-semibold">{phase}</span>
              <span className="text-muted-foreground truncate max-w-xs">{ids.join(", ")}</span>
            </span>
          )
        })}
      </div>
    </div>
  )
}

function ThinkBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded border bg-muted/40 text-xs">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 hover:bg-muted/60 transition-colors font-mono text-muted-foreground"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        thinking
      </button>
      {open && (
        <pre className="px-3 pb-2 pt-0 whitespace-pre-wrap break-words font-mono text-muted-foreground">
          {text}
        </pre>
      )}
    </div>
  )
}

function ToolCallBadge({ name, args }: { name: string; args: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded border bg-muted/40 px-2 py-1 text-xs font-mono">
      <span className="text-[var(--color-illini-orange)] font-semibold">{name}</span>
      <span className="text-muted-foreground truncate max-w-md">{args}</span>
    </div>
  )
}

function ToolResultBlock({ content }: { content: string }) {
  const [open, setOpen] = useState(false)
  const preview = content.length > 200 ? content.slice(0, 200) + "…" : content
  return (
    <div className="rounded border bg-muted/40 text-xs">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 w-full px-3 py-1.5 hover:bg-muted/60 transition-colors font-mono text-muted-foreground",
        )}
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        tool result · {content.length} chars
      </button>
      <pre className="px-3 pb-2 whitespace-pre-wrap break-words font-mono text-muted-foreground">
        {open ? content : preview}
      </pre>
    </div>
  )
}

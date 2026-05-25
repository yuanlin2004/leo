import { Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import type { SessionSummary } from "@/lib/api"

type Props = {
  sessions: SessionSummary[]
  activeSid: string | null
  onSelect: (sid: string) => void
  onCreate: () => void
  onDelete: (sid: string) => void
}

export function SessionList({
  sessions, activeSid, onSelect, onCreate, onDelete,
}: Props) {
  return (
    <aside className="w-72 border-r flex flex-col shrink-0 bg-muted/30">
      <div className="flex items-center justify-between px-3 h-10 border-b">
        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
          Sessions
        </h2>
        <button
          onClick={onCreate}
          className="flex items-center gap-1 text-xs font-mono px-2 py-1 rounded hover:bg-accent transition-colors"
          title="New session"
        >
          <Plus className="w-3.5 h-3.5" />
          new
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <div className="p-4 text-xs text-muted-foreground font-mono">
            no sessions yet — click "new"
          </div>
        ) : (
          <ul>
            {sessions.map((s) => (
              <li key={s.id}>
                <div
                  className={cn(
                    "group flex items-start gap-2 px-3 py-2.5 border-l-2 border-transparent cursor-pointer hover:bg-accent/50 transition-colors",
                    activeSid === s.id &&
                      "bg-accent border-l-[var(--color-illini-orange)]",
                  )}
                  onClick={() => onSelect(s.id)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      {s.is_running && (
                        <span
                          className="inline-block w-2 h-2 rounded-full bg-[var(--color-illini-orange)] animate-pulse shrink-0"
                          title="Run in progress"
                          aria-label="Run in progress"
                        />
                      )}
                      <span className="text-sm truncate">{s.title}</span>
                    </div>
                    <div className="font-mono text-[10px] text-muted-foreground mt-0.5 flex items-center gap-1.5">
                      <span>{s.id} · {s.message_count} msgs</span>
                      {s.agent_id && s.agent_id !== "leo" && (
                        <span className="px-1 py-0.5 rounded border text-[9px] uppercase tracking-wider"
                              title={`Agent: ${s.agent_id}`}>
                          {s.agent_id}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (s.is_running) {
                        alert(
                          "This session has a run in progress. Cancel it first, then delete.",
                        )
                        return
                      }
                      if (confirm(`Delete session ${s.id}?`)) onDelete(s.id)
                    }}
                    disabled={s.is_running}
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity p-0.5 disabled:opacity-40 disabled:hover:text-muted-foreground"
                    aria-label="Delete"
                    title={
                      s.is_running
                        ? "Cancel the run before deleting"
                        : "Delete this session"
                    }
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}

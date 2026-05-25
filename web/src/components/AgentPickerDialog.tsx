import { useEffect, useState } from "react"
import { Bot } from "lucide-react"
import { Modal } from "@/components/ui/Modal"
import { api } from "@/lib/api"
import type { AgentDTO } from "@/lib/api"
import { cn } from "@/lib/utils"

type Props = {
  open: boolean
  onClose: () => void
  onCreate: (agent_id: string, title: string | null) => Promise<void>
}

export function AgentPickerDialog({ open, onClose, onCreate }: Props) {
  const [agents, setAgents] = useState<AgentDTO[] | null>(null)
  const [selectedId, setSelectedId] = useState<string>("leo")
  const [title, setTitle] = useState<string>("")
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setBusy(false)
    setErr(null)
    setTitle("")
    api.listAgents()
      .then((a) => {
        setAgents(a)
        setSelectedId((cur) => (a.some((x) => x.id === cur) ? cur : "leo"))
      })
      .catch((e) => setErr(String(e)))
  }, [open])

  const create = async () => {
    setBusy(true)
    setErr(null)
    try {
      await onCreate(selectedId, title.trim() || null)
      onClose()
    } catch (e) {
      setErr(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="New session"
      widthClass="max-w-xl"
      footer={
        <>
          <button
            onClick={onClose}
            disabled={busy}
            className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm disabled:opacity-50"
          >
            cancel
          </button>
          <button
            onClick={create}
            disabled={busy || !agents}
            className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
          >
            {busy ? "creating…" : "create"}
          </button>
        </>
      }
    >
      <div className="space-y-3 text-sm">
        <div>
          <label className="font-mono text-xs uppercase tracking-wider text-muted-foreground block mb-1">
            Title (optional)
          </label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="leave blank to derive from first message"
            className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-wider text-muted-foreground block mb-1">
            Agent
          </label>
          {err && (
            <div className="rounded border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-xs text-muted-foreground mb-2 break-all">
              {err}
            </div>
          )}
          {!agents ? (
            <div className="py-4 text-center text-xs text-muted-foreground font-mono">
              loading…
            </div>
          ) : (
            <ul className="rounded border bg-background divide-y max-h-64 overflow-y-auto">
              {agents.map((a) => (
                <li
                  key={a.id}
                  onClick={() => setSelectedId(a.id)}
                  className={cn(
                    "flex items-start gap-2 px-3 py-2 cursor-pointer hover:bg-accent/40 transition-colors",
                    selectedId === a.id && "bg-accent/60",
                  )}
                >
                  <Bot className={cn(
                    "w-4 h-4 mt-0.5 shrink-0",
                    selectedId === a.id ? "text-[var(--color-illini-orange)]" : "text-muted-foreground",
                  )} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate flex items-center gap-2">
                      {a.name}
                      {a.builtin && (
                        <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border text-muted-foreground">
                          builtin
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5 leading-snug">
                      {a.description || "(no description)"}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Modal>
  )
}

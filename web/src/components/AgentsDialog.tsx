import { useEffect, useState } from "react"
import { Bot, Plus, Trash2, AlertCircle, ChevronLeft } from "lucide-react"
import { Modal } from "@/components/ui/Modal"
import { api } from "@/lib/api"
import type { AgentDTO, SkillInfo } from "@/lib/api"

type Props = {
  open: boolean
  onClose: () => void
}

type EditState =
  | { mode: "list" }
  | { mode: "edit"; agent: AgentDTO }     // existing agent
  | { mode: "new" }                        // creating

export function AgentsDialog({ open, onClose }: Props) {
  const [view, setView] = useState<EditState>({ mode: "list" })
  const [agents, setAgents] = useState<AgentDTO[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const refresh = () => {
    api.listAgents()
      .then(setAgents)
      .catch((e) => setErr(String(e)))
  }

  useEffect(() => {
    if (!open) return
    setView({ mode: "list" })
    setErr(null)
    refresh()
  }, [open])

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={
        view.mode === "list"
          ? "Agents"
          : view.mode === "new"
            ? "New agent"
            : `Edit · ${view.agent.name}`
      }
      widthClass="max-w-3xl"
      footer={null /* edit/list views have their own buttons */}
    >
      {err && (
        <div className="flex items-start gap-2 rounded border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-xs mb-3">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 text-red-600 dark:text-red-400 shrink-0" />
          <span className="text-muted-foreground break-all">{err}</span>
        </div>
      )}

      {view.mode === "list" && (
        <ListView
          agents={agents}
          onNew={() => setView({ mode: "new" })}
          onEdit={(a) => setView({ mode: "edit", agent: a })}
          onDelete={async (id) => {
            if (!confirm(`Delete agent "${id}"?`)) return
            try {
              await api.deleteAgent(id)
              refresh()
            } catch (e) {
              setErr(String(e))
            }
          }}
          onClose={onClose}
        />
      )}

      {(view.mode === "edit" || view.mode === "new") && (
        <EditView
          initial={view.mode === "edit" ? view.agent : null}
          onCancel={() => setView({ mode: "list" })}
          onSaved={() => {
            setView({ mode: "list" })
            refresh()
          }}
          onError={(e) => setErr(e)}
        />
      )}
    </Modal>
  )
}

// -- list ----------------------------------------------------------------

function ListView({
  agents, onNew, onEdit, onDelete, onClose,
}: {
  agents: AgentDTO[] | null
  onNew: () => void
  onEdit: (a: AgentDTO) => void
  onDelete: (id: string) => void
  onClose: () => void
}) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <p className="text-xs text-muted-foreground">
          Agents are reusable templates for starting sessions — system prompt,
          skill subset, default toggles, initial user prompt.
        </p>
        <button
          onClick={onNew}
          className="flex items-center gap-1 text-xs font-mono px-2 py-1 rounded hover:bg-accent transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          new
        </button>
      </div>
      {!agents ? (
        <div className="py-6 text-center text-xs text-muted-foreground font-mono">
          loading…
        </div>
      ) : (
        <ul className="rounded border bg-background divide-y">
          {agents.map((a) => (
            <li key={a.id} className="group flex items-start gap-2 px-3 py-2.5">
              <Bot className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
              <div
                className="flex-1 min-w-0 cursor-pointer"
                onClick={() => onEdit(a)}
              >
                <div className="text-sm font-medium flex items-center gap-2">
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
                <div className="font-mono text-[10px] text-muted-foreground mt-1">
                  {a.id}
                  {a.skills.length > 0 && <> · skills: {a.skills.join(", ")}</>}
                  {a.initial_user_prompt && <> · has initial prompt</>}
                </div>
              </div>
              {!a.builtin && (
                <button
                  onClick={() => onDelete(a.id)}
                  className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity p-1"
                  aria-label="Delete"
                  title="Delete this agent"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="flex justify-end pt-2">
        <button
          onClick={onClose}
          className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm"
        >
          close
        </button>
      </div>
    </div>
  )
}

// -- edit / new ----------------------------------------------------------

function EditView({
  initial, onCancel, onSaved, onError,
}: {
  initial: AgentDTO | null
  onCancel: () => void
  onSaved: () => void
  onError: (e: string) => void
}) {
  const isNew = initial === null
  const [name, setName] = useState(initial?.name ?? "")
  const [description, setDescription] = useState(initial?.description ?? "")
  const [systemPrompt, setSystemPrompt] = useState(initial?.system_prompt ?? "")
  const [initialUserPrompt, setInitialUserPrompt] = useState(
    initial?.initial_user_prompt ?? "",
  )
  const [skills, setSkills] = useState<string[]>(initial?.skills ?? [])
  const [defaultThink, setDefaultThink] = useState(initial?.default_think ?? true)
  const [allSkills, setAllSkills] = useState<SkillInfo[] | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.listSkills().then(setAllSkills).catch(() => setAllSkills([]))
  }, [])

  const canSave = !!name.trim()

  const save = async () => {
    setSaving(true)
    onError("")
    try {
      const body = {
        name: name.trim(),
        description: description.trim(),
        system_prompt: systemPrompt,
        initial_user_prompt: initialUserPrompt,
        skills,
        default_think: defaultThink,
      }
      if (isNew) {
        // Server derives the id from name.
        await api.createAgent({ id: "", ...body })
      } else {
        await api.updateAgent(initial!.id, body)
      }
      onSaved()
    } catch (e) {
      onError(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-4 text-sm">
      <button
        onClick={onCancel}
        className="flex items-center gap-1 text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
      >
        <ChevronLeft className="w-3.5 h-3.5" />
        back to list
      </button>

      <div>
        <Label>Name</Label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoFocus={isNew}
          placeholder="e.g. Research agent"
          className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
        />
        {!isNew && (
          <div className="mt-1 font-mono text-[10px] text-muted-foreground">
            id: {initial!.id} (auto-generated; cannot be changed)
          </div>
        )}
      </div>

      <div>
        <Label>Description</Label>
        <input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="one-line summary shown in the picker"
          className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>

      <div>
        <Label>System prompt</Label>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          rows={6}
          placeholder="Prepended to LEO's base system prompt at session creation."
          className="w-full rounded-md border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>

      <div>
        <Label>Initial user prompt (optional)</Label>
        <textarea
          value={initialUserPrompt}
          onChange={(e) => setInitialUserPrompt(e.target.value)}
          rows={2}
          placeholder="Prefilled into the chat input when a session opens."
          className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>

      <div>
        <Label>Allowed skills (empty = all installed)</Label>
        {!allSkills ? (
          <div className="text-xs text-muted-foreground font-mono py-2">
            loading…
          </div>
        ) : allSkills.length === 0 ? (
          <div className="text-xs text-muted-foreground font-mono py-2">
            no skills installed
          </div>
        ) : (
          <div className="rounded border bg-background p-2 space-y-1 max-h-40 overflow-y-auto">
            {allSkills.map((s) => {
              const checked = skills.includes(s.name)
              return (
                <label
                  key={s.name}
                  className="flex items-start gap-2 cursor-pointer text-xs"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      setSkills((cur) =>
                        e.target.checked
                          ? [...cur, s.name]
                          : cur.filter((n) => n !== s.name),
                      )
                    }}
                    className="mt-0.5 w-3.5 h-3.5 accent-[var(--color-illini-orange)]"
                  />
                  <div className="flex-1 min-w-0">
                    <span className="font-mono font-medium">{s.name}</span>
                    <span className="text-muted-foreground ml-2">
                      {s.description}
                    </span>
                  </div>
                </label>
              )
            })}
          </div>
        )}
      </div>

      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={defaultThink}
          onChange={(e) => setDefaultThink(e.target.checked)}
          className="mt-0.5 w-4 h-4 accent-[var(--color-illini-orange)]"
        />
        <div className="flex-1 min-w-0">
          <div className="font-medium">Default thinking on</div>
          <div className="text-xs text-muted-foreground mt-0.5">
            New sessions created with this agent start with `think_on=true`.
          </div>
        </div>
      </label>

      <div className="flex justify-end gap-2 pt-2 border-t">
        <button
          onClick={onCancel}
          disabled={saving}
          className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm disabled:opacity-50"
        >
          cancel
        </button>
        <button
          onClick={save}
          disabled={!canSave || saving}
          className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
        >
          {saving ? "saving…" : "save"}
        </button>
      </div>
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-1">
      {children}
    </div>
  )
}

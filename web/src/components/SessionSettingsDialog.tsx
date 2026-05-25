import { useEffect, useState } from "react"
import { Modal } from "@/components/ui/Modal"
import type { SessionDetail } from "@/lib/api"

type Patch = {
  title?: string
  think_on?: boolean
  net_on?: boolean
  show_think?: boolean
  show_tool_use?: boolean
  show_reflection?: boolean
}

type Props = {
  open: boolean
  onClose: () => void
  detail: SessionDetail
  onSave: (patch: Patch) => Promise<void>
}

/** Defaults: behavior toggles default-on; show_reflection default-off
 * (lesson-injection chips are opt-in noise). */
function readToggles(detail: SessionDetail) {
  const t = detail.toggles as Record<string, unknown>
  return {
    think_on: Boolean(t.think_on ?? true),
    net_on: Boolean(t.net_on ?? true),
    show_think: Boolean(t.show_think ?? true),
    show_tool_use: Boolean(t.show_tool_use ?? true),
    show_reflection: Boolean(t.show_reflection ?? false),
  }
}

export function SessionSettingsDialog({ open, onClose, detail, onSave }: Props) {
  const [title, setTitle] = useState(detail.title)
  const [v, setV] = useState(() => readToggles(detail))
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    setTitle(detail.title)
    setV(readToggles(detail))
    setErr(null)
  }, [open, detail])

  const submit = async () => {
    setSaving(true)
    setErr(null)
    const cur = readToggles(detail)
    const patch: Patch = {}
    if (title.trim() && title !== detail.title) patch.title = title.trim()
    for (const k of [
      "think_on", "net_on", "show_think", "show_tool_use", "show_reflection",
    ] as const) {
      if (v[k] !== cur[k]) patch[k] = v[k]
    }
    try {
      if (Object.keys(patch).length > 0) await onSave(patch)
      onClose()
    } catch (e) {
      setErr(String(e))
    } finally {
      setSaving(false)
    }
  }

  const setKey = <K extends keyof typeof v>(k: K) => (val: boolean) =>
    setV((prev) => ({ ...prev, [k]: val }))

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Session settings"
      footer={
        <>
          <button
            onClick={onClose}
            disabled={saving}
            className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm disabled:opacity-50"
          >
            cancel
          </button>
          <button
            onClick={submit}
            disabled={saving}
            className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
          >
            {saving ? "saving…" : "save"}
          </button>
        </>
      }
    >
      <div className="space-y-5 text-sm">
        <div>
          <label className="font-mono text-xs uppercase tracking-wider text-muted-foreground block mb-1">
            Title
          </label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>

        <section>
          <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-2">
            Agent behavior
          </div>
          <div className="space-y-3">
            <ToggleRow
              label="Thinking"
              description="Allow the model to use <think>…</think> blocks before its reply."
              checked={v.think_on}
              onChange={setKey("think_on")}
            />
            <ToggleRow
              label="Network"
              description="Allow the bash tool to reach the network from inside the sandbox."
              checked={v.net_on}
              onChange={setKey("net_on")}
            />
          </div>
        </section>

        <section>
          <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-2">
            Display
          </div>
          <div className="space-y-3">
            <ToggleRow
              label="Show thinking"
              description="Render the model's thinking content (collapsed by default) in the chat."
              checked={v.show_think}
              onChange={setKey("show_think")}
            />
            <ToggleRow
              label="Show tool use"
              description="Render tool-call badges and tool-result blocks inline in the chat."
              checked={v.show_tool_use}
              onChange={setKey("show_tool_use")}
            />
            <ToggleRow
              label="Show reflection"
              description="Render a chip in the chat each time a lesson fires (on_prompt, on_monologue, on_tool_call)."
              checked={v.show_reflection}
              onChange={setKey("show_reflection")}
            />
          </div>
        </section>

        {detail.is_running && (
          <div className="text-xs font-mono text-amber-600 dark:text-amber-400">
            A run is in progress — settings cannot be saved until it finishes.
          </div>
        )}
        {err && (
          <div className="text-xs font-mono text-red-600 dark:text-red-400 break-all">
            {err}
          </div>
        )}
      </div>
    </Modal>
  )
}

function ToggleRow({
  label, description, checked, onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 w-4 h-4 accent-[var(--color-illini-orange)]"
      />
      <div className="flex-1 min-w-0">
        <div className="font-medium">{label}</div>
        <div className="text-xs text-muted-foreground mt-0.5 leading-snug">
          {description}
        </div>
      </div>
    </label>
  )
}

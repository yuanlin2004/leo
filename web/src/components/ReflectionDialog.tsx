import { useEffect, useState } from "react"
import { Modal } from "@/components/ui/Modal"
import { api } from "@/lib/api"
import type { ReflectionApplyResult, ReflectionProposal } from "@/lib/api"

type Props = {
  open: boolean
  onClose: () => void
  sid: string
  onApplied: () => void
}

type Phase = "loading" | "review" | "applying" | "results" | "empty" | "error"

export function ReflectionDialog({ open, onClose, sid, onApplied }: Props) {
  const [phase, setPhase] = useState<Phase>("loading")
  const [proposals, setProposals] = useState<ReflectionProposal[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [reason, setReason] = useState<string | null>(null)
  const [results, setResults] = useState<ReflectionApplyResult[]>([])
  const [err, setErr] = useState<string | null>(null)

  // Kick off /reflect on open.
  useEffect(() => {
    if (!open) return
    setPhase("loading")
    setProposals([])
    setSelected(new Set())
    setReason(null)
    setResults([])
    setErr(null)
    api
      .reflect(sid)
      .then((r) => {
        if (!r.proposals.length) {
          setReason(r.reason ?? "nothing to learn")
          setPhase("empty")
          return
        }
        setProposals(r.proposals)
        // Default: pre-select all create/update, skip the skips.
        setSelected(
          new Set(
            r.proposals
              .filter((p) => p.kind !== "skip")
              .map((p) => p.index),
          ),
        )
        setPhase("review")
      })
      .catch((e) => {
        setErr(String(e))
        setPhase("error")
      })
  }, [open, sid])

  const toggle = (idx: number) => {
    setSelected((cur) => {
      const next = new Set(cur)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const apply = async () => {
    setPhase("applying")
    setErr(null)
    const ops: object[] = proposals
      .filter((p) => selected.has(p.index) && p.kind !== "skip")
      .map((p) => {
        if (p.kind === "create") return { kind: "create", lesson: p.lesson }
        if (p.kind === "update") return { kind: "update", id: p.id, fields: p.fields }
        return p
      })
    try {
      const res = await api.applyReflection(sid, ops)
      setResults(res.results)
      setPhase("results")
      onApplied()
    } catch (e) {
      setErr(String(e))
      setPhase("error")
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Reflection"
      widthClass="max-w-2xl"
      footer={
        phase === "review" ? (
          <>
            <span className="mr-auto text-xs text-muted-foreground font-mono">
              {selected.size} of {proposals.filter((p) => p.kind !== "skip").length} selected
            </span>
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm"
            >
              cancel
            </button>
            <button
              onClick={apply}
              disabled={selected.size === 0}
              className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
            >
              apply {selected.size}
            </button>
          </>
        ) : (
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm"
          >
            close
          </button>
        )
      }
    >
      {phase === "loading" && (
        <Status>analyzing the recent trace…</Status>
      )}
      {phase === "applying" && <Status>applying…</Status>}
      {phase === "empty" && <Status>{reason ?? "nothing to learn"}</Status>}
      {phase === "error" && (
        <Status tone="error">{err ?? "unknown error"}</Status>
      )}
      {phase === "review" && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Reflection proposed {proposals.length} ops. Check the ones to apply
            (skips are informational and never persist).
          </p>
          {proposals.map((p) => (
            <ProposalRow
              key={p.index}
              proposal={p}
              checked={selected.has(p.index)}
              onToggle={() => toggle(p.index)}
            />
          ))}
        </div>
      )}
      {phase === "results" && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Applied {results.filter((r) => r.status === "ok").length} of{" "}
            {results.filter((r) => r.kind !== "skip").length} ops.
          </p>
          {results.map((r, i) => (
            <ResultRow key={i} result={r} />
          ))}
        </div>
      )}
    </Modal>
  )
}

function Status({
  children, tone = "info",
}: {
  children: React.ReactNode
  tone?: "info" | "error"
}) {
  const cls =
    tone === "error"
      ? "text-red-600 dark:text-red-400"
      : "text-muted-foreground"
  return (
    <div className={`py-6 text-center text-sm font-mono ${cls}`}>{children}</div>
  )
}

function ProposalRow({
  proposal: p, checked, onToggle,
}: {
  proposal: ReflectionProposal
  checked: boolean
  onToggle: () => void
}) {
  if (p.kind === "skip") {
    return (
      <div className="rounded border bg-muted/30 px-3 py-2">
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          skip
        </div>
        <div className="text-xs text-muted-foreground mt-0.5">{p.reason}</div>
      </div>
    )
  }
  const isCreate = p.kind === "create"
  const data = isCreate
    ? (p.lesson as Record<string, unknown>)
    : (p.fields as Record<string, unknown>)
  const title = isCreate
    ? String(data.title ?? "")
    : `update ${(p as { id: string }).id}`
  const category = isCreate ? String(data.category ?? "") : ""
  const trigger = (data.trigger as Record<string, unknown>) ?? {}
  return (
    <label className="flex gap-2 rounded border bg-background px-3 py-2 cursor-pointer hover:bg-accent/30 transition-colors">
      <input
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        className="mt-0.5 w-4 h-4 accent-[var(--color-illini-orange)]"
      />
      <div className="flex-1 min-w-0 text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-illini-orange)]">
            {p.kind}
          </span>
          {category && (
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              {category}
            </span>
          )}
          <span className="font-medium truncate">{title}</span>
        </div>
        {data.rule != null && (
          <div className="mt-1 leading-snug">{String(data.rule)}</div>
        )}
        {data.why != null && (
          <div className="mt-1 text-muted-foreground leading-snug">
            <span className="font-mono text-[10px] uppercase tracking-wider mr-1">
              why
            </span>
            {String(data.why)}
          </div>
        )}
        {Boolean(trigger.type || trigger.keywords) && (
          <div className="mt-1 font-mono text-[10px] text-muted-foreground">
            trigger: {String(trigger.type ?? "?")}
            {Array.isArray(trigger.keywords) && trigger.keywords.length > 0 && (
              <> · keywords: {(trigger.keywords as string[]).join(", ")}</>
            )}
          </div>
        )}
      </div>
    </label>
  )
}

function ResultRow({ result }: { result: ReflectionApplyResult }) {
  const okColor =
    result.status === "ok"
      ? "text-emerald-600 dark:text-emerald-400"
      : result.status === "ignored"
        ? "text-muted-foreground"
        : "text-red-600 dark:text-red-400"
  return (
    <div className="rounded border bg-background px-3 py-2 text-xs font-mono">
      <span className={okColor}>{result.status}</span>
      <span className="ml-2 text-muted-foreground">{result.kind}</span>
      {result.id && <span className="ml-2">{result.id}</span>}
      {result.error && (
        <span className="ml-2 text-red-600 dark:text-red-400">
          {result.error}
        </span>
      )}
    </div>
  )
}

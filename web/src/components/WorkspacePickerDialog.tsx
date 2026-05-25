import { useEffect, useState } from "react"
import { Folder, FolderCheck, ChevronLeft, Home, AlertCircle } from "lucide-react"
import { Modal } from "@/components/ui/Modal"
import { api } from "@/lib/api"
import type { FsListing } from "@/lib/api"
import { cn } from "@/lib/utils"

type Props = {
  open: boolean
  onClose: () => void
  /** Path of the workspace currently active, so we can highlight it. */
  currentWorkspace: string | null
  /** Called after a successful open/create with the new workspace path. */
  onSwitched: () => void
}

export function WorkspacePickerDialog({
  open, onClose, currentWorkspace, onSwitched,
}: Props) {
  const [listing, setListing] = useState<FsListing | null>(null)
  const [path, setPath] = useState<string>("")  // empty = data_root
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Load the listing whenever the dialog opens or the path changes.
  useEffect(() => {
    if (!open) return
    setErr(null)
    api
      .fsList(path || undefined)
      .then(setListing)
      .catch((e) => setErr(String(e)))
  }, [open, path])

  // On open, default to the current workspace's parent so the user
  // lands somewhere familiar.
  useEffect(() => {
    if (!open) return
    if (currentWorkspace && path === "") {
      // Use parent so the current ws is visible in the listing.
      const parent = currentWorkspace.replace(/\/[^/]+$/, "") || currentWorkspace
      setPath(parent)
    }
  }, [open])  // intentional one-shot on dialog open

  const goTo = (p: string) => {
    setListing(null)
    setPath(p)
  }

  const handleOpenCurrent = async () => {
    if (!listing) return
    setBusy(true)
    setErr(null)
    try {
      await api.openWorkspace(listing.path)
      onSwitched()
      onClose()
    } catch (e) {
      setErr(String(e))
    } finally {
      setBusy(false)
    }
  }

  const handleCreateHere = async () => {
    if (!listing) return
    if (
      !confirm(
        `Create a Leo workspace at:\n\n${listing.path}\n\nThis adds a .leo/ folder.`,
      )
    ) {
      return
    }
    setBusy(true)
    setErr(null)
    try {
      await api.createWorkspace(listing.path)
      onSwitched()
      onClose()
    } catch (e) {
      setErr(String(e))
    } finally {
      setBusy(false)
    }
  }

  const handleEntry = async (entry: { path: string; is_workspace: boolean }) => {
    // Single click navigates in; the Open/Create buttons act on the current
    // folder. This matches Finder/Files behavior — drill-in, then act.
    goTo(entry.path)
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Select workspace"
      widthClass="max-w-2xl"
      footer={
        listing ? (
          <>
            <button
              onClick={onClose}
              disabled={busy}
              className="px-3 py-1.5 rounded-md border hover:bg-accent transition-colors font-mono text-sm disabled:opacity-50"
            >
              cancel
            </button>
            {listing.current_is_workspace ? (
              <button
                onClick={handleOpenCurrent}
                disabled={busy || listing.path === currentWorkspace}
                className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
              >
                {listing.path === currentWorkspace ? "already active" : "open this folder"}
              </button>
            ) : (
              <button
                onClick={handleCreateHere}
                disabled={busy}
                className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity font-mono text-sm disabled:opacity-50"
              >
                create workspace here
              </button>
            )}
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
      <div className="space-y-3 text-sm">
        {listing && (
          <div className="flex items-center gap-2 font-mono text-xs">
            <button
              onClick={() => goTo("")}
              className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded hover:bg-accent"
              title="Go to data root"
            >
              <Home className="w-3.5 h-3.5" />
            </button>
            {listing.parent && (
              <button
                onClick={() => goTo(listing.parent!)}
                className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded hover:bg-accent flex items-center gap-1"
                title="Up one level"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
            )}
            <span className="truncate text-muted-foreground">{listing.path}</span>
            {listing.current_is_workspace && (
              <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--color-illini-orange)] text-white shrink-0">
                workspace
              </span>
            )}
          </div>
        )}

        {err && (
          <div className="flex items-start gap-2 rounded border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-xs">
            <AlertCircle className="w-3.5 h-3.5 mt-0.5 text-red-600 dark:text-red-400 shrink-0" />
            <span className="text-muted-foreground break-all">{err}</span>
          </div>
        )}

        {!listing && !err && (
          <div className="py-6 text-center text-xs text-muted-foreground font-mono">
            loading…
          </div>
        )}

        {listing && listing.entries.length === 0 && (
          <div className="py-6 text-center text-xs text-muted-foreground font-mono">
            no subdirectories
          </div>
        )}

        {listing && listing.entries.length > 0 && (
          <p className="text-[11px] text-muted-foreground font-mono">
            click a folder to enter it · then "open" if it's a workspace,
            or "create" to make a new one
          </p>
        )}
        {listing && listing.entries.length > 0 && (
          <ul className="rounded border bg-background divide-y max-h-96 overflow-y-auto">
            {listing.entries.map((e) => (
              <li
                key={e.path}
                onClick={() => handleEntry(e)}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-accent/40 transition-colors",
                  e.path === currentWorkspace && "bg-accent/30",
                )}
              >
                {e.is_workspace ? (
                  <FolderCheck className="w-4 h-4 text-[var(--color-illini-orange)] shrink-0" />
                ) : (
                  <Folder className="w-4 h-4 text-muted-foreground shrink-0" />
                )}
                <span className="text-sm flex-1 truncate">{e.name}</span>
                {e.is_workspace && (
                  <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border text-muted-foreground shrink-0">
                    workspace
                  </span>
                )}
                {e.path === currentWorkspace && (
                  <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--color-illini-orange)] text-white shrink-0">
                    active
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Modal>
  )
}

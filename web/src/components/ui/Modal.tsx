import { useEffect } from "react"
import { X } from "lucide-react"

type Props = {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  /** Optional footer actions row, rendered to the right. */
  footer?: React.ReactNode
  /** Width in px or any tailwind max-w utility. Default 480. */
  widthClass?: string
}

/** Minimal centered modal. Backdrop click + ESC close. Designed to
 * match the rest of the UI (Illini accent for the close hover) without
 * pulling in a full headless-ui dependency. */
export function Modal({
  open, onClose, title, children, footer, widthClass = "max-w-md",
}: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className={`relative w-full ${widthClass} max-h-[85vh] flex flex-col bg-card text-card-foreground border rounded-lg shadow-xl`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-4 h-12 border-b">
          <h2 className="font-mono text-sm font-semibold">{title}</h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 rounded"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </header>
        <div className="flex-1 overflow-y-auto px-4 py-3">{children}</div>
        {footer && (
          <footer className="flex items-center justify-end gap-2 px-4 h-12 border-t">
            {footer}
          </footer>
        )}
      </div>
    </div>
  )
}

import { Sun, Moon, Activity, Circle, FolderTree, Bot, Menu, PanelRight } from "lucide-react"
import { cn } from "@/lib/utils"
import leoLogo from "@/assets/leo-logo.svg"

type Props = {
  workspace: string | null
  model: string | null
  isRunning: boolean
  isDark: boolean
  onToggleTheme: () => void
  onOpenWorkspacePicker: () => void
  onOpenAgentsBuilder: () => void
  /** Drawer toggles — only visible below lg. */
  onToggleSessionsDrawer: () => void
  onToggleObsDrawer: () => void
}

export function TopBar({
  workspace, model, isRunning, isDark, onToggleTheme, onOpenWorkspacePicker,
  onOpenAgentsBuilder, onToggleSessionsDrawer, onToggleObsDrawer,
}: Props) {
  return (
    <header
      className="flex items-center justify-between px-3 sm:px-4 h-12 border-b shrink-0"
      style={{ backgroundColor: "var(--color-illini-blue)", color: "white" }}
    >
      <div className="flex items-center gap-2 sm:gap-3 min-w-0">
        <button
          onClick={onToggleSessionsDrawer}
          className="lg:hidden text-white/70 hover:text-white transition-colors p-1 rounded shrink-0"
          aria-label="Open sessions list"
          title="Sessions"
        >
          <Menu className="w-5 h-5" />
        </button>
        <img
          src={leoLogo}
          alt="LEO 力行智能"
          className="h-6 sm:h-7 w-auto select-none shrink-0"
          draggable={false}
        />
        <span className="text-white/40 hidden sm:inline">·</span>
        <button
          onClick={onOpenWorkspacePicker}
          className="flex items-center gap-1.5 font-mono text-xs text-white/80 hover:text-white transition-colors min-w-0 max-w-xl"
          title={workspace ?? "Pick a workspace"}
        >
          <FolderTree className="w-3.5 h-3.5 shrink-0" />
          {/* Icon-only on mobile to save space — full path from sm up. */}
          <span className="hidden sm:inline truncate">
            {workspace ?? "(no workspace)"}
          </span>
        </button>
      </div>
      <div className="flex items-center gap-4">
        {model && (
          <span className="font-mono text-xs text-white/70 hidden md:inline">
            {model}
          </span>
        )}
        <span
          className={cn(
            "flex items-center gap-1.5 text-xs font-mono",
            isRunning ? "text-[var(--color-illini-orange)]" : "text-white/50",
          )}
          title={isRunning ? "Run in progress" : "Idle"}
        >
          {isRunning ? (
            <Activity className="w-3.5 h-3.5 animate-pulse" />
          ) : (
            <Circle className="w-3.5 h-3.5" />
          )}
          {isRunning ? "running" : "idle"}
        </span>
        <button
          onClick={onOpenAgentsBuilder}
          className="flex items-center gap-1 text-white/70 hover:text-white transition-colors p-1 rounded text-xs font-mono"
          title="Manage agents"
        >
          <Bot className="w-4 h-4" />
          <span className="hidden md:inline">agents</span>
        </button>
        <button
          onClick={onToggleTheme}
          className="text-white/70 hover:text-white transition-colors p-1 rounded"
          aria-label="Toggle theme"
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
        <button
          onClick={onToggleObsDrawer}
          className="lg:hidden text-white/70 hover:text-white transition-colors p-1 rounded"
          aria-label="Toggle observability panel"
          title="Trace · Skills · Lessons"
        >
          <PanelRight className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}

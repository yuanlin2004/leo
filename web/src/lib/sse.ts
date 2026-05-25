/** SSE subscription hook for the session event stream.
 *
 * Uses the browser's native EventSource. The server-side endpoint
 * (`GET /api/sessions/{sid}/stream?since=N`) backfills disk events
 * with seq > N then switches to live tail; the client just needs to
 * remember the last seq it saw across reconnects.
 */

import { useEffect, useRef } from "react"
import type { AgentEvent } from "./api"

export type SseStatus = "idle" | "connecting" | "open" | "closed" | "error"

export function useEventStream(
  sid: string | null,
  onEvent: (ev: AgentEvent) => void,
  onStatus?: (s: SseStatus) => void,
) {
  // Latch the latest callbacks so we don't tear down on every render.
  const evRef = useRef(onEvent)
  const stRef = useRef(onStatus)
  evRef.current = onEvent
  stRef.current = onStatus

  // Track last-seen seq so a reconnect resumes from where we left off.
  const sinceRef = useRef(0)

  useEffect(() => {
    if (!sid) {
      stRef.current?.("idle")
      return
    }

    let stopped = false
    let es: EventSource | null = null
    let retryTimer: number | null = null

    const open = (since: number) => {
      if (stopped) return
      stRef.current?.("connecting")
      const url = `/api/sessions/${sid}/stream?since=${since}`
      es = new EventSource(url)

      // The server attaches `event: <type>` to each frame, so we have to
      // register handlers for the types we care about. Easiest: register
      // a wildcard fallback via the underlying `message` event isn't
      // enough — we listen on common event types and also keep a generic
      // listener via `addEventListener` for each. Simpler: parse the
      // type from the JSON payload and route from there. We add explicit
      // listeners for every emitted event type.
      const types = [
        "run_start", "llm_message", "tool_start", "tool_end",
        "lesson_injected", "replan", "run_end", "run_error",
        "run_cancelled", "llm_token",
      ]
      for (const t of types) {
        es.addEventListener(t, (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data) as AgentEvent
            sinceRef.current = Math.max(sinceRef.current, data.seq)
            evRef.current(data)
          } catch (err) {
            console.warn("sse parse error", err, e.data)
          }
        })
      }

      es.onopen = () => stRef.current?.("open")
      es.onerror = () => {
        // EventSource auto-reconnects on transient failures. We forcibly
        // close and reopen with the latest `since` so the resume hint is
        // current. Native auto-reconnect would resume with the original
        // ?since=0 query, replaying every event.
        es?.close()
        if (stopped) return
        stRef.current?.("error")
        retryTimer = window.setTimeout(() => open(sinceRef.current), 1500)
      }
    }

    open(0)

    return () => {
      stopped = true
      if (retryTimer) window.clearTimeout(retryTimer)
      es?.close()
      stRef.current?.("closed")
    }
  }, [sid])
}

import { useSyncExternalStore } from "react";
import { getStreams, type ConnState, type StreamManager } from "../../api/sse";
import { Icon, type IconName } from "./Icon";
import { TermTipBody } from "./InfoTip";
import { useTooltip } from "./Tooltip";

/** Each connection state: icon, the pill's word, and what it means for the page (the tooltip). */
const INFO: Record<ConnState, { icon: IconName; text: string; tip: string }> = {
  live: { icon: "wifi", text: "live", tip: "Live updates are on: progress and new results appear on the page as they happen." },
  connecting: { icon: "wifi", text: "connecting…", tip: "Connecting to SPARC Studio for live updates." },
  reconnecting: { icon: "refresh", text: "reconnecting…", tip: "The live connection dropped. Studio is retrying, waiting a little longer each time." },
  polling: { icon: "wifiOff", text: "polling", tip: "Live updates failed three times, so the page now checks for news every 5 seconds instead." },
  closed: { icon: "wifiOff", text: "offline", tip: "Not connected to SPARC Studio. Pages will not update until the Studio server is running again." },
};

/**
 * Live / reconnecting / polling indicator for the SSE streams. What the state means shows on
 * hover, keyboard focus or tap (a tooltip, not a native title).
 */
export function ConnectionPill({ manager }: { manager?: StreamManager }) {
  const m = manager ?? getStreams();
  const state = useSyncExternalStore(m.subscribeState, m.getState, m.getState);
  const i = INFO[state];
  const t = useTooltip<HTMLSpanElement>({ content: <TermTipBody tip={i.tip} technical={`Server-sent events: ${state}`} />, side: "bottom", bubbleClassName: "tip-term" });
  return (
    <>
      <span ref={t.anchorRef} className="pill conn-pill" data-state={state} role="status" aria-label={`Connection: ${i.text}`} tabIndex={0} {...t.triggerProps}>
        <Icon name={i.icon} />
        <span className="hide-narrow">{i.text}</span>
      </span>
      {t.bubble}
    </>
  );
}

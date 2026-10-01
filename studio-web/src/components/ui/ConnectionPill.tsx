import { useSyncExternalStore } from "react";
import { getStreams, type ConnState, type StreamManager } from "../../api/sse";
import { Icon, type IconName } from "./Icon";

const INFO: Record<ConnState, { icon: IconName; text: string; title: string }> = {
  live: { icon: "wifi", text: "live", title: "Receiving live updates" },
  connecting: { icon: "wifi", text: "connecting…", title: "Connecting to the Studio server" },
  reconnecting: { icon: "refresh", text: "reconnecting…", title: "Connection lost; retrying with backoff" },
  polling: { icon: "wifiOff", text: "polling", title: "Live updates failed three times; checking every 5 s" },
  closed: { icon: "wifiOff", text: "offline", title: "Not connected to the Studio server" },
};

/** Live / reconnecting / polling indicator for the SSE streams. */
export function ConnectionPill({ manager }: { manager?: StreamManager }) {
  const m = manager ?? getStreams();
  const state = useSyncExternalStore(m.subscribeState, m.getState, m.getState);
  const i = INFO[state];
  return (
    <span className="pill conn-pill" data-state={state} title={i.title} role="status" aria-label={`Connection: ${i.text}`}>
      <Icon name={i.icon} />
      <span className="hide-narrow">{i.text}</span>
    </span>
  );
}

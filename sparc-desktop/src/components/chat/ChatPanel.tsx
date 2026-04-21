import { useState, useRef, useEffect } from "react";
import CubeLogo from "@/components/brand/CubeLogo";
import { useAnthropicChat } from "@/hooks/useAnthropicChat";
import type { ClaudeAction } from "@/lib/types";

interface ChatPanelProps {
  onClose?: () => void;
  onAction?: (action: ClaudeAction) => void;
  systemPrompt?: string;
}

const DEMO_REPLIES = [
  "Proposing DAG edges: Canopy→AAT, Impervious→AAT, Canopy→NDVI→AAT. Open DAG view to accept.",
  "Added monotone constraint: Pct_Canopy (−) on AAT_z. Verified against Providence UHI priors.",
  "Scenario written: canopy +10 pp · predicted mean ΔAAT_z = −0.258 (σ = 0.154).",
];

export default function ChatPanel({ onClose, onAction, systemPrompt }: ChatPanelProps) {
  const apiKey = localStorage.getItem("anthropic-api-key") ?? "";
  const hasKey = apiKey.length > 0;

  const { messages, isLoading, error, sendMessage } = useAnthropicChat(
    systemPrompt ?? "You are a spatial analysis assistant.",
    onAction,
  );

  // Demo mode fallback state (no API key)
  const [demoMsgs, setDemoMsgs] = useState<Array<{ role: "user" | "assistant"; text: string }>>([
    {
      role: "assistant",
      text: "I can wire up your DAG from natural language. Try: 'Canopy and impervious affect air temperature, mediated by NDVI.'",
    },
  ]);

  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, demoMsgs.length]);

  const send = () => {
    if (!input.trim()) return;
    if (hasKey) {
      sendMessage(input, apiKey);
    } else {
      const u = { role: "user" as const, text: input };
      setDemoMsgs((m) => [
        ...m,
        u,
        { role: "assistant", text: DEMO_REPLIES[m.length % DEMO_REPLIES.length] },
      ]);
    }
    setInput("");
  };

  // Determine which messages to display
  const displayMsgs = hasKey
    ? messages.map((m) => ({ role: m.role, text: m.content }))
    : demoMsgs;

  return (
    <div
      style={{
        position: "absolute",
        left: 228,
        bottom: 0,
        width: 360,
        height: 420,
        background: "#fff",
        border: "1px solid var(--line)",
        borderRadius: "8px 8px 0 0",
        display: "flex",
        flexDirection: "column",
        zIndex: 40,
        boxShadow: "0 -8px 24px rgba(0,0,0,0.08)",
        animation: "slideUp 0.22s ease-out",
      }}
    >
      <div
        style={{
          padding: "10px 14px",
          borderBottom: "1px solid var(--line)",
          display: "flex",
          alignItems: "center",
        }}
      >
        <div style={{ width: 22, height: 22, marginRight: 8 }}>
          <CubeLogo size={22} />
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700 }}>SPARC Assistant</div>
          <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>
            {hasKey ? "claude-sonnet-4.6 · live" : "demo mode · no api key"}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            marginLeft: "auto",
            border: "none",
            background: "transparent",
            fontSize: 16,
            cursor: "pointer",
            color: "var(--muted)",
          }}
        >
          ×
        </button>
      </div>

      <div
        ref={scrollRef}
        className="scroll"
        style={{ flex: 1, overflowY: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}
      >
        {!hasKey && (
          <div
            style={{
              background: "#fff8ef",
              border: "1px solid var(--amber)",
              borderRadius: 6,
              padding: "8px 10px",
              fontSize: 11,
              color: "var(--ink-2)",
              marginBottom: 4,
            }}
          >
            No API key set. Go to <strong>Settings</strong> to add your Anthropic key for live responses.
          </div>
        )}
        {displayMsgs.map((m, i) => (
          <div
            key={i}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "85%",
              background: m.role === "user" ? "var(--ink)" : "#f7f4ee",
              color: m.role === "user" ? "#fff" : "var(--ink-2)",
              padding: "7px 10px",
              borderRadius: 7,
              fontSize: 12,
              lineHeight: 1.45,
            }}
          >
            {m.text}
          </div>
        ))}
        {isLoading && (
          <div
            style={{
              alignSelf: "flex-start",
              maxWidth: "85%",
              background: "#f7f4ee",
              color: "var(--muted)",
              padding: "7px 10px",
              borderRadius: 7,
              fontSize: 12,
            }}
          >
            thinking…
          </div>
        )}
        {error && (
          <div
            style={{
              alignSelf: "flex-start",
              maxWidth: "85%",
              background: "#fff0f0",
              color: "var(--crimson)",
              padding: "7px 10px",
              borderRadius: 7,
              fontSize: 11,
            }}
          >
            {error}
          </div>
        )}
      </div>

      <div style={{ padding: 10, borderTop: "1px solid var(--line)", display: "flex", gap: 6 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !isLoading && send()}
          placeholder="Ask about your DAG, physics, or scenarios…"
          style={{
            flex: 1,
            padding: "7px 10px",
            border: "1px solid var(--line)",
            borderRadius: 5,
            fontSize: 12,
            fontFamily: "inherit",
          }}
        />
        <button
          onClick={send}
          disabled={isLoading}
          style={{
            background: "var(--ink)",
            color: "#fff",
            border: "none",
            padding: "0 12px",
            borderRadius: 5,
            fontSize: 12,
            fontWeight: 600,
            cursor: isLoading ? "not-allowed" : "pointer",
            fontFamily: "inherit",
            opacity: isLoading ? 0.5 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

import { useState, useRef, useEffect } from "react";
import { useAnthropicChat } from "@/hooks/useAnthropicChat";
import type { ClaudeAction } from "@/lib/types";
import CubeLogo from "@/components/brand/CubeLogo";

import { buildSystemPrompt } from "@/lib/prompts";

const FALLBACK_SYSTEM_PROMPT = buildSystemPrompt("general");

interface ChatPanelProps {
  onAction?: (action: ClaudeAction) => void;
  systemPrompt?: string;
  onClose?: () => void;
}

export default function ChatPanel({ onAction, systemPrompt, onClose }: ChatPanelProps) {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("anthropic-api-key") ?? "");
  const [input, setInput] = useState("");
  const [showKeyInput, setShowKeyInput] = useState(!apiKey);
  const bottomRef = useRef<HTMLDivElement>(null);

  const { messages, isLoading, error, sendMessage } = useAnthropicChat(systemPrompt ?? FALLBACK_SYSTEM_PROMPT, onAction);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || !apiKey) return;
    sendMessage(input.trim(), apiKey);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const saveKey = () => {
    localStorage.setItem("anthropic-api-key", apiKey);
    setShowKeyInput(false);
  };

  if (showKeyInput || !apiKey) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24, textAlign: 'center' }}>
        <CubeLogo size={48} animate={false} hue="ink" />
        <div>
          <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-sparc-ink)' }}>Connect AI Assistant</p>
          <p style={{ fontSize: 11, color: 'var(--color-sparc-muted)', marginTop: 4 }}>Enter your Anthropic API key to enable AI-assisted project setup.</p>
        </div>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-ant-..."
          style={{
            width: '100%',
            padding: '7px 10px',
            border: '1px solid var(--color-sparc-line)',
            borderRadius: 5,
            fontSize: 12,
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={saveKey}
          disabled={!apiKey}
          style={{
            background: 'var(--color-sparc-ink)',
            color: '#fff',
            border: 'none',
            padding: '7px 14px',
            borderRadius: 5,
            fontSize: 12,
            fontWeight: 600,
            cursor: apiKey ? 'pointer' : 'not-allowed',
            fontFamily: 'inherit',
            opacity: apiKey ? 1 : 0.4,
          }}
        >
          Save Key
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fff' }}>
      {/* Header */}
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--color-sparc-line)',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <div style={{ width: 28, height: 28, marginRight: 10 }}>
          <CubeLogo size={28} animate={false} hue="ink" />
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-sparc-ink)' }}>SPARC Assistant</div>
          <div className="mono" style={{ fontSize: 9.5, color: 'var(--color-sparc-muted)' }}>claude-haiku-4.5 · dag mode</div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{
              marginLeft: 'auto',
              border: 'none',
              background: 'transparent',
              fontSize: 16,
              cursor: 'pointer',
              color: 'var(--color-sparc-muted)',
              padding: '0 4px',
            }}
          >
            ×
          </button>
        )}
      </div>

      {/* Messages */}
      <div
        className="scroll"
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        {messages.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, paddingTop: 32, paddingBottom: 16 }}>
            <CubeLogo size={48} animate hue="ink" intensity={0.4} />
            <p style={{ fontSize: 12, color: 'var(--color-sparc-muted)', textAlign: 'center', lineHeight: 1.5, padding: '0 16px' }}>
              Ask Claude to help configure your project, build a DAG, or set physics constraints.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              background: msg.role === 'user' ? 'var(--color-sparc-ink)' : '#f7f4ee',
              color: msg.role === 'user' ? '#fff' : 'var(--color-sparc-ink-2)',
              padding: '7px 10px',
              borderRadius: 7,
              fontSize: 12,
              lineHeight: 1.45,
            }}
          >
            <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
            {msg.actions && msg.actions.length > 0 && (
              <div
                style={{
                  marginTop: 6,
                  borderTop: `1px solid ${msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'var(--color-sparc-line)'}`,
                  paddingTop: 6,
                  fontSize: 10,
                  color: msg.role === 'user' ? 'rgba(255,255,255,0.7)' : 'var(--color-sparc-muted)',
                }}
              >
                {msg.actions.length} action(s) applied
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div
            style={{
              alignSelf: 'flex-start',
              maxWidth: '85%',
              background: '#f7f4ee',
              padding: '7px 10px',
              borderRadius: 7,
              fontSize: 12,
              color: 'var(--color-sparc-muted)',
            }}
          >
            <span style={{ display: 'inline-flex', gap: 2 }}>
              <span className="animate-bounce" style={{ animationDelay: '0ms' }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: '150ms' }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: '300ms' }}>·</span>
            </span>
          </div>
        )}
        {error && (
          <div style={{ border: '1px solid rgba(231,60,37,0.3)', background: 'rgba(231,60,37,0.06)', padding: '7px 10px', borderRadius: 5, fontSize: 11, color: 'var(--color-sparc-crimson)' }}>
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: 10, borderTop: '1px solid var(--color-sparc-line)', display: 'flex', gap: 6 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your DAG, physics, or scenarios…"
          style={{
            flex: 1,
            padding: '7px 10px',
            border: '1px solid var(--color-sparc-line)',
            borderRadius: 5,
            fontSize: 12,
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          style={{
            background: 'var(--color-sparc-ink)',
            color: '#fff',
            border: 'none',
            padding: '0 12px',
            borderRadius: 5,
            fontSize: 12,
            fontWeight: 600,
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
            opacity: isLoading || !input.trim() ? 0.4 : 1,
          }}
        >
          Send
        </button>
      </div>
      <div style={{ padding: '0 10px 6px' }}>
        <button
          onClick={() => setShowKeyInput(true)}
          style={{
            background: 'none',
            border: 'none',
            fontSize: 10,
            color: 'var(--color-sparc-muted)',
            cursor: 'pointer',
            fontFamily: 'inherit',
            padding: 0,
          }}
        >
          Change API key
        </button>
      </div>
    </div>
  );
}

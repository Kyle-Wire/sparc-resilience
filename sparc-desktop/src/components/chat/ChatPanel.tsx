import { useState, useRef, useEffect } from "react";
import { useAnthropicChat } from "@/hooks/useAnthropicChat";
import type { ClaudeAction } from "@/lib/types";

import { buildSystemPrompt } from "@/lib/prompts";

const FALLBACK_SYSTEM_PROMPT = buildSystemPrompt("general");

interface ChatPanelProps {
  onAction?: (action: ClaudeAction) => void;
  systemPrompt?: string;
}

export default function ChatPanel({ onAction, systemPrompt }: ChatPanelProps) {
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
      <div className="flex h-full flex-col items-center justify-center gap-4 p-4 text-center">
        <p className="text-sm text-sparc-gray-600">Enter your Anthropic API key to enable AI-assisted project setup.</p>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-ant-..."
          className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm"
        />
        <button onClick={saveKey} disabled={!apiKey} className="rounded bg-black px-4 py-2 text-sm text-white disabled:opacity-40">
          Save Key
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {messages.length === 0 && (
          <p className="text-xs text-sparc-gray-600">Ask Claude to help configure your project, build a DAG, or set physics constraints.</p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`rounded p-3 text-sm ${msg.role === "user" ? "bg-sparc-gray-100 ml-6" : "bg-black text-white mr-6"}`}
          >
            <div className="whitespace-pre-wrap">{msg.content}</div>
            {msg.actions && msg.actions.length > 0 && (
              <div className="mt-2 border-t border-sparc-gray-600 pt-2 text-xs opacity-70">
                {msg.actions.length} action(s) applied
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="mr-6 rounded bg-sparc-gray-100 p-3 text-sm animate-pulse text-sparc-gray-600">Thinking...</div>
        )}
        {error && <div className="rounded border border-sparc-crimson p-2 text-xs text-sparc-crimson">{error}</div>}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-sparc-gray-200 p-3">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Claude..."
            rows={2}
            className="flex-1 resize-none rounded border border-sparc-gray-300 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="rounded bg-black px-3 py-2 text-sm text-white disabled:opacity-40"
          >
            Send
          </button>
        </div>
        <button onClick={() => setShowKeyInput(true)} className="mt-1 text-xs text-sparc-gray-600 hover:underline">
          Change API key
        </button>
      </div>
    </div>
  );
}

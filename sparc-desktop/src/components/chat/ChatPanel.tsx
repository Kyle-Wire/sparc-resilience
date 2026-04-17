import { useState, useRef, useEffect } from "react";
import { useAnthropicChat } from "@/hooks/useAnthropicChat";
import type { ClaudeAction } from "@/lib/types";

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
      <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-sparc-purple/10">
          <span className="text-lg">💬</span>
        </div>
        <div>
          <p className="text-sm font-medium text-sparc-gray-800">Connect AI Assistant</p>
          <p className="mt-1 text-xs text-sparc-gray-500">Enter your Anthropic API key to enable AI-assisted project setup.</p>
        </div>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-ant-..."
          className="w-full rounded-lg border border-sparc-gray-200 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none focus:ring-1 focus:ring-sparc-purple/30"
        />
        <button onClick={saveKey} disabled={!apiKey} className="rounded-lg bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-40">
          Save Key
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-sparc-gray-200 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-sparc-purple/10">
            <span className="text-xs">💬</span>
          </div>
          <span className="text-sm font-semibold text-sparc-gray-800">AI Assistant</span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="flex h-6 w-6 items-center justify-center rounded text-sparc-gray-400 hover:bg-sparc-gray-100 hover:text-sparc-gray-600"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M11 3L3 11M3 3l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center gap-3 pt-8 pb-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-sparc-purple/10">
              <span className="text-xl">✦</span>
            </div>
            <p className="text-xs text-sparc-gray-500 text-center leading-relaxed px-4">
              Ask Claude to help configure your project, build a DAG, or set physics constraints.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2.5 text-sm leading-relaxed ${
              msg.role === "user"
                ? "ml-8 bg-sparc-purple text-white"
                : "mr-8 bg-sparc-gray-50 text-sparc-gray-800 border border-sparc-gray-100"
            }`}
          >
            <div className="whitespace-pre-wrap">{msg.content}</div>
            {msg.actions && msg.actions.length > 0 && (
              <div className={`mt-2 border-t pt-2 text-xs ${
                msg.role === "user" ? "border-white/20 text-white/70" : "border-sparc-gray-200 text-sparc-gray-400"
              }`}>
                {msg.actions.length} action(s) applied
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="mr-8 rounded-lg bg-sparc-gray-50 border border-sparc-gray-100 px-3 py-2.5 text-sm text-sparc-gray-400">
            <span className="inline-flex gap-1">
              <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
            </span>
          </div>
        )}
        {error && <div className="rounded-lg border border-red-200 bg-red-50 p-2.5 text-xs text-red-600">{error}</div>}
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
            className="flex-1 resize-none rounded-lg border border-sparc-gray-200 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none focus:ring-1 focus:ring-sparc-purple/30"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="self-end rounded-lg bg-sparc-purple px-3 py-2 text-sm font-medium text-white hover:bg-sparc-magenta disabled:opacity-40"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M14 2L7 9M14 2l-4.5 12-2-5.5L2 6.5 14 2z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
          </button>
        </div>
        <button onClick={() => setShowKeyInput(true)} className="mt-1.5 text-[11px] text-sparc-gray-400 hover:text-sparc-gray-600 hover:underline">
          Change API key
        </button>
      </div>
    </div>
  );
}

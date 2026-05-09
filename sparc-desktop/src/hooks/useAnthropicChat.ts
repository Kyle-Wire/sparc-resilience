import { useState, useCallback } from "react";
import type { ChatMessage, ClaudeAction } from "@/lib/types";
import { getToken } from "@/stores/tokenStore";

const SIDECAR_BASE = "http://127.0.0.1:8008";

/**
 * Extract structured JSON action blocks from Claude's response text.
 * Looks for ```json ... ``` fenced blocks containing an "action" field.
 */
function extractActions(text: string): ClaudeAction[] {
  const actions: ClaudeAction[] = [];
  const regex = /```json\s*([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1]);
      if (parsed && typeof parsed.action === "string") {
        actions.push(parsed as ClaudeAction);
      }
    } catch {
      // not valid JSON — skip
    }
  }
  return actions;
}

/**
 * Chat hook for the Anthropic Claude API.
 * All API calls are proxied through the SPARC sidecar at /ai/chat.
 * The API key is stored in the OS keychain — the webview never touches it.
 *
 * @param systemPrompt - The system prompt for this conversation mode.
 * @param onAction - Callback invoked for each structured action parsed from Claude's response.
 */
export function useAnthropicChat(
  systemPrompt: string,
  onAction?: (action: ClaudeAction) => void,
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      setError(null);
      const userMsg: ChatMessage = { role: "user", content };
      const updated = [...messages, userMsg];
      setMessages(updated);
      setIsLoading(true);

      try {
        const token = getToken();
        const res = await fetch(`${SIDECAR_BASE}/ai/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { "x-sparc-token": token } : {}),
          },
          body: JSON.stringify({
            system: systemPrompt,
            messages: updated.map((m) => ({ role: m.role, content: m.content })),
          }),
        });

        if (!res.ok) {
          const errBody = await res.text();
          throw new Error(`API error ${res.status}: ${errBody}`);
        }

        const data = await res.json();
        const assistantText: string = data.content[0]?.text ?? "";
        const actions = extractActions(assistantText);

        // Dispatch actions
        if (onAction) {
          actions.forEach(onAction);
        }

        const assistantMsg: ChatMessage = {
          role: "assistant",
          content: assistantText,
          actions,
        };
        setMessages([...updated, assistantMsg]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
      }
    },
    [messages, systemPrompt, onAction],
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearChat };
}


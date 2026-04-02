import { useState, useCallback } from "react";
import type { ChatMessage, ClaudeAction } from "@/lib/types";

const CLAUDE_MODEL = "claude-sonnet-4-6-20250514";

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
 * API calls go directly from the webview to api.anthropic.com.
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
    async (content: string, apiKey: string) => {
      setError(null);
      const userMsg: ChatMessage = { role: "user", content };
      const updated = [...messages, userMsg];
      setMessages(updated);
      setIsLoading(true);

      try {
        const res = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
          },
          body: JSON.stringify({
            model: CLAUDE_MODEL,
            max_tokens: 1024,
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

import { useState, useEffect } from "react";
import { useNotification } from "@/hooks/useNotifications";

export default function SettingsView() {
  const [apiKey, setApiKey] = useState("");
  const { notify } = useNotification();
  const [serverPort, setServerPort] = useState("8008");

  useEffect(() => {
    setApiKey(localStorage.getItem("anthropic-api-key") ?? "");
    setServerPort(localStorage.getItem("sparc-server-port") ?? "8008");
  }, []);

  const saveApiKey = () => {
    if (apiKey) {
      localStorage.setItem("anthropic-api-key", apiKey);
    } else {
      localStorage.removeItem("anthropic-api-key");
    }
    notify("success", "API key saved");
  };

  const savePort = () => {
    localStorage.setItem("sparc-server-port", serverPort);
    notify("success", "Server port saved (restart required)");
  };

  const clearAllData = () => {
    if (!confirm("Clear all stored settings? This cannot be undone.")) return;
    localStorage.clear();
    setApiKey("");
    setServerPort("8008");
  };

  return (
    <div className="p-6 space-y-8 max-w-2xl">
      <div>
        <h2 className="text-lg font-bold">Settings</h2>
        <p className="text-sm text-sparc-gray-600">
          Configure AI assistant, server, and application preferences.
        </p>
      </div>

      {/* API Key */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Anthropic API Key</h3>
        <p className="text-xs text-sparc-gray-500">
          Required for the AI assistant. Your key is stored locally and sent
          directly to api.anthropic.com — never to SPARC servers.
        </p>
        <div className="flex gap-2">
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-ant-api03-..."
            className="flex-1 rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
          <button
            onClick={saveApiKey}
            className="rounded bg-sparc-purple px-4 py-2 text-sm font-medium text-white hover:bg-sparc-magenta"
          >
            Save
          </button>
        </div>
        {apiKey && (
          <p className="text-xs text-green-600">
            Key set: {apiKey.slice(0, 10)}...{apiKey.slice(-4)}
          </p>
        )}
      </section>

      {/* Server */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Server Port</h3>
        <p className="text-xs text-sparc-gray-500">
          The local port for the SPARC FastAPI server. Change requires restart.
        </p>
        <div className="flex gap-2">
          <input
            type="number"
            value={serverPort}
            onChange={(e) => setServerPort(e.target.value)}
            className="w-32 rounded border border-sparc-gray-300 px-3 py-2 text-sm font-mono focus:border-sparc-purple focus:outline-none"
          />
          <button
            onClick={savePort}
            className="rounded border border-sparc-gray-300 px-4 py-2 text-sm hover:bg-sparc-gray-100"
          >
            Save
          </button>
        </div>
      </section>

      {/* About */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">About</h3>
        <div className="rounded border border-sparc-gray-200 p-4 text-xs space-y-1">
          <p><span className="font-medium">App:</span> SPARC Labs Desktop v1.0.0</p>
          <p><span className="font-medium">Pipeline:</span> SPARC v2.1.0</p>
          <p><span className="font-medium">AI Model:</span> Claude Sonnet 4.6</p>
          <p><span className="font-medium">Runtime:</span> Tauri v2 + FastAPI</p>
        </div>
      </section>

      {/* Danger zone */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-red-600">Danger Zone</h3>
        <button
          onClick={clearAllData}
          className="rounded border border-red-300 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
        >
          Clear All Stored Data
        </button>
      </section>
    </div>
  );
}

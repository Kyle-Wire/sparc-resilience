import { useState, useEffect, useRef } from "react";
import { useNotification } from "@/hooks/useNotifications";
import EasterEgg from "@/components/common/EasterEgg";

const KONAMI = ["ArrowUp","ArrowUp","ArrowDown","ArrowDown","ArrowLeft","ArrowRight","ArrowLeft","ArrowRight","b","a"];

const DOCS = [
  { name: "User Manual", file: "MANUAL.md" },
  { name: "Pipeline Guide", file: "PIPELINE_GUIDE.md" },
  { name: "Interpretation Guide", file: "INTERPRETATION_GUIDE.md" },
  { name: "Contributing", file: "CONTRIBUTING.md" },
];

export default function SettingsView() {
  const [apiKey, setApiKey] = useState("");
  const { notify } = useNotification();
  const [serverPort, setServerPort] = useState("8008");
  const [showEgg, setShowEgg] = useState(false);
  const konamiIdx = useRef(0);

  useEffect(() => {
    setApiKey(localStorage.getItem("anthropic-api-key") ?? "");
    setServerPort(localStorage.getItem("sparc-server-port") ?? "8008");
  }, []);

  // Konami code listener
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === KONAMI[konamiIdx.current]) {
        konamiIdx.current++;
        if (konamiIdx.current === KONAMI.length) {
          setShowEgg(true);
          konamiIdx.current = 0;
        }
      } else {
        konamiIdx.current = 0;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
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

      {/* About SPARC Labs */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">About SPARC Labs</h3>
        <div className="rounded border border-sparc-gray-200 p-4 text-xs space-y-2">
          <p className="text-sm font-medium">SPARC – Spatial Analysis and Research Core</p>
          <p className="text-sparc-gray-600">
            SPARC is a geospatial machine-learning pipeline that integrates
            geographically weighted models, causal inference, and
            physics-constrained scenario simulation for environmental and
            infrastructure analysis.
          </p>
          <div className="border-t border-sparc-gray-200 pt-2 space-y-1">
            <p><span className="font-medium">Version:</span> 2.1.0</p>
            <p><span className="font-medium">Desktop App:</span> v1.0.0 (Tauri v2 + FastAPI)</p>
            <p><span className="font-medium">AI Model:</span> Claude Sonnet 4.6</p>
            <p><span className="font-medium">Author:</span> Kyle Wire</p>
          </div>
          <p className="text-sparc-gray-600 italic">
            If you use this software in research, please cite using the included CITATION.cff.
          </p>
        </div>
      </section>

      {/* Contact Us */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Contact Us</h3>
        <div className="rounded border border-sparc-gray-200 p-4 text-xs space-y-2">
          <p>
            <span className="font-medium">GitHub Issues:</span>{" "}
            <a
              href="https://github.com/SPARCLabs/GW3C_v2.0/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sparc-purple hover:underline"
            >
              github.com/SPARCLabs/GW3C_v2.0/issues
            </a>
          </p>
          <p>
            <span className="font-medium">Repository:</span>{" "}
            <a
              href="https://github.com/SPARCLabs/GW3C_v2.0"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sparc-purple hover:underline"
            >
              github.com/SPARCLabs/GW3C_v2.0
            </a>
          </p>
        </div>
      </section>

      {/* Documentation */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Documentation</h3>
        <div className="rounded border border-sparc-gray-200 divide-y divide-sparc-gray-200">
          {DOCS.map((doc) => (
            <button
              key={doc.file}
              onClick={() => {
                // Open docs folder relative to install — for now, copy to clipboard
                navigator.clipboard.writeText(doc.file);
                notify("success", `${doc.name} filename copied`);
              }}
              className="flex w-full items-center justify-between px-4 py-3 text-xs hover:bg-sparc-gray-100 transition-colors"
            >
              <span className="font-medium">{doc.name}</span>
              <span className="text-sparc-gray-600">{doc.file}</span>
            </button>
          ))}
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

      {/* Easter egg */}
      {showEgg && <EasterEgg onClose={() => setShowEgg(false)} />}
    </div>
  );
}

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

const LOGO_HUES = [
  { key: "ink", label: "Ink", color: "#1a1416" },
  { key: "red", label: "Red", color: "#e73c25" },
  { key: "purple", label: "Purple", color: "#602468" },
  { key: "amber", label: "Amber", color: "#e79024" },
] as const;

const PAPER_TONES = [
  { key: "warm", label: "Warm", paper: "#f7f4ee", line: "#c9c2b3" },
  { key: "cool", label: "Cool", paper: "#f3f4f2", line: "#c5cac4" },
  { key: "white", label: "White", paper: "#ffffff", line: "#d8d4cb" },
] as const;

export type LogoHue = typeof LOGO_HUES[number]["key"];
export type PaperTone = typeof PAPER_TONES[number]["key"];

export interface ThemeSettings {
  logoHue: LogoHue;
  paperTone: PaperTone;
}

export function loadTheme(): ThemeSettings {
  try {
    const raw = localStorage.getItem("sparc-theme");
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { logoHue: "ink", paperTone: "warm" };
}

export function applyTheme(theme: ThemeSettings) {
  const tone = PAPER_TONES.find((t) => t.key === theme.paperTone) ?? PAPER_TONES[0];
  document.documentElement.style.setProperty("--color-sparc-paper", tone.paper);
  document.documentElement.style.setProperty("--color-sparc-line", tone.line);
  document.documentElement.style.setProperty("--color-sparc-gray-100", tone.paper);
  document.documentElement.style.setProperty("--color-sparc-gray-300", tone.line);
  localStorage.setItem("sparc-theme", JSON.stringify(theme));
}

export default function SettingsView() {
  const [apiKey, setApiKey] = useState("");
  const { notify } = useNotification();
  const [serverPort, setServerPort] = useState("8008");
  const [showEgg, setShowEgg] = useState(false);
  const konamiIdx = useRef(0);
  const [theme, setTheme] = useState<ThemeSettings>(loadTheme);

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

      {/* Theme */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold">Theme</h3>
        <p className="text-xs text-sparc-gray-500">
          Customize the application appearance.
        </p>

        {/* Logo Hue */}
        <div>
          <label className="text-xs font-medium block mb-2">Logo Colour</label>
          <div className="flex gap-2">
            {LOGO_HUES.map((h) => (
              <button
                key={h.key}
                onClick={() => {
                  const next = { ...theme, logoHue: h.key };
                  setTheme(next);
                  applyTheme(next);
                  // Dispatch custom event for CubeLogo instances to pick up
                  window.dispatchEvent(new CustomEvent("sparc-theme-change", { detail: next }));
                  notify("success", `Logo colour: ${h.label}`);
                }}
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 6,
                  border: theme.logoHue === h.key ? "2px solid var(--color-sparc-ink)" : "1px solid var(--color-sparc-line)",
                  background: h.color,
                  cursor: "pointer",
                  boxShadow: theme.logoHue === h.key ? "0 0 0 2px rgba(0,0,0,0.1)" : "none",
                }}
                title={h.label}
              />
            ))}
          </div>
        </div>

        {/* Paper Tone */}
        <div>
          <label className="text-xs font-medium block mb-2">Paper Tone</label>
          <div className="flex gap-2">
            {PAPER_TONES.map((t) => (
              <button
                key={t.key}
                onClick={() => {
                  const next = { ...theme, paperTone: t.key };
                  setTheme(next);
                  applyTheme(next);
                  window.dispatchEvent(new CustomEvent("sparc-theme-change", { detail: next }));
                  notify("success", `Paper tone: ${t.label}`);
                }}
                style={{
                  padding: "6px 14px",
                  borderRadius: 5,
                  border: theme.paperTone === t.key ? "2px solid var(--color-sparc-ink)" : "1px solid var(--color-sparc-line)",
                  background: t.paper,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: "pointer",
                  fontFamily: "inherit",
                  color: "var(--color-sparc-ink)",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
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

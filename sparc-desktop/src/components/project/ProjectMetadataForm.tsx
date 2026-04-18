import { useState, useEffect } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig } from "@/lib/types";

export default function ProjectMetadataForm() {
  const [meta, setMeta] = useState({
    name: "",
    description: "",
    domain: "",
    author: "",
    version: "",
    response_units: "",
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((cfg: ProjectConfig) => {
        const p = cfg.project ?? {};
        setMeta({
          name: p.name ?? "",
          description: p.description ?? "",
          domain: p.domain ?? "",
          author: p.author ?? "",
          version: p.version ?? "",
          response_units: p.response_units ?? "",
        });
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveConfig({ project: meta });
      notify("success", "Project metadata saved");
    } catch (e: any) {
      notify("error", e.message || "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const field = (
    label: string,
    key: keyof typeof meta,
    opts?: { multiline?: boolean; placeholder?: string }
  ) => (
    <div>
      <label className="mb-1 block text-xs font-medium text-sparc-gray-600">{label}</label>
      {opts?.multiline ? (
        <textarea
          value={meta[key]}
          onChange={(e) => setMeta((m) => ({ ...m, [key]: e.target.value }))}
          placeholder={opts?.placeholder}
          rows={3}
          className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none"
        />
      ) : (
        <input
          type="text"
          value={meta[key]}
          onChange={(e) => setMeta((m) => ({ ...m, [key]: e.target.value }))}
          placeholder={opts?.placeholder}
          className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm focus:border-sparc-purple focus:outline-none"
        />
      )}
    </div>
  );

  if (loading) return <div className="animate-pulse p-4 text-sm text-sparc-gray-500">Loading…</div>;

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Project Metadata</h3>
        <p className="text-xs text-sparc-gray-500">
          This information appears in generated reports and exports.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {field("Project Name", "name", { placeholder: "Urban Heat Island Analysis" })}
        {field("Domain", "domain", { placeholder: "uhi, air_quality, wildfire…" })}
        {field("Author", "author", { placeholder: "Jane Smith" })}
        {field("Version", "version", { placeholder: "1.0" })}
        {field("Response Units", "response_units", { placeholder: "°C, μg/m³, dB…" })}
      </div>
      {field("Description", "description", {
        multiline: true,
        placeholder: "Brief description of the study objectives and scope…",
      })}

      <button
        onClick={handleSave}
        disabled={saving}
        className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50"
      >
        {saving ? "Saving…" : "Save Metadata"}
      </button>
    </div>
  );
}

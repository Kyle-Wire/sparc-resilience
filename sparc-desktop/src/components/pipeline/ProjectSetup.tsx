import { useState, useEffect } from "react";
import { listTemplates, initProject, loadProject } from "@/lib/api";
import type { TemplateInfo } from "@/lib/types";

interface Props {
  onProjectLoaded: () => void;
}

export default function ProjectSetup({ onProjectLoaded }: Props) {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [selected, setSelected] = useState("blank");
  const [outputDir, setOutputDir] = useState("");
  const [projectPath, setProjectPath] = useState("");
  const [mode, setMode] = useState<"create" | "open">("create");
  const [status, setStatus] = useState("");

  useEffect(() => {
    listTemplates().then((r) => setTemplates(r.templates)).catch(() => {});
  }, []);

  const handleCreate = async () => {
    if (!outputDir) return;
    setStatus("Creating project...");
    try {
      const res = await initProject(selected, outputDir);
      setStatus(`Created. Loading ${res.project_yml}...`);
      await loadProject(res.project_yml);
      setStatus("Project loaded.");
      onProjectLoaded();
    } catch (err) {
      setStatus(`Error: ${err}`);
    }
  };

  const handleOpen = async () => {
    if (!projectPath) return;
    setStatus("Loading project...");
    try {
      await loadProject(projectPath);
      setStatus("Project loaded.");
      onProjectLoaded();
    } catch (err) {
      setStatus(`Error: ${err}`);
    }
  };

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="mb-6 text-2xl font-bold">Project Setup</h1>

      {/* Mode toggle */}
      <div className="mb-6 flex gap-4 border-b border-sparc-gray-200 pb-2">
        <button
          onClick={() => setMode("create")}
          className={`pb-1 text-sm ${mode === "create" ? "border-b-2 border-black font-medium" : "text-sparc-gray-600"}`}
        >
          New Project
        </button>
        <button
          onClick={() => setMode("open")}
          className={`pb-1 text-sm ${mode === "open" ? "border-b-2 border-black font-medium" : "text-sparc-gray-600"}`}
        >
          Open Existing
        </button>
      </div>

      {mode === "create" ? (
        <div className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium">Domain Template</label>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm"
            >
              {templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Output Directory</label>
            <input
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="/path/to/new/project"
              className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm"
            />
          </div>

          <button
            onClick={handleCreate}
            className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800"
          >
            Create Project
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium">Project YAML Path</label>
            <input
              type="text"
              value={projectPath}
              onChange={(e) => setProjectPath(e.target.value)}
              placeholder="/path/to/project.yml"
              className="w-full rounded border border-sparc-gray-300 px-3 py-2 text-sm"
            />
          </div>

          <button
            onClick={handleOpen}
            className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800"
          >
            Open Project
          </button>
        </div>
      )}

      {status && <p className="mt-4 text-sm text-sparc-gray-600">{status}</p>}
    </div>
  );
}

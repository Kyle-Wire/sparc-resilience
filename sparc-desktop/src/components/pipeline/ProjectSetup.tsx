import { useState, useEffect } from "react";
import { listTemplates, initProject } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { getRecents, removeRecent, type RecentProject } from "@/lib/recentProjects";
import { pickFolder, pickYaml } from "@/lib/fileDialogs";
import type { TemplateInfo } from "@/lib/types";

interface Props {
  projectPath: string | null;
  onProjectLoaded: (path: string, meta?: { name?: string; template?: string }) => Promise<void>;
}

export default function ProjectSetup({ projectPath, onProjectLoaded }: Props) {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [selected, setSelected] = useState("blank");
  const [outputDir, setOutputDir] = useState("");
  const [yamlPath, setYamlPath] = useState("");
  const [mode, setMode] = useState<"create" | "open">("create");
  const [status, setStatus] = useState("");
  const [recents, setRecents] = useState<RecentProject[]>([]);
  const { notify } = useNotification();

  useEffect(() => {
    listTemplates().then((r) => setTemplates(r.templates)).catch(() => {});
    setRecents(getRecents());
  }, []);

  const handleCreate = async () => {
    if (!outputDir) return;
    setStatus("Creating project...");
    try {
      const res = await initProject(selected, outputDir);
      setStatus(`Created. Loading...`);
      await onProjectLoaded(res.project_yml, { name: selected, template: selected });
    } catch (err: any) {
      setStatus("");
      notify("error", err.message ?? String(err));
    }
  };

  const handleOpen = async (path?: string) => {
    const target = path ?? yamlPath;
    if (!target) return;
    setStatus("Loading project...");
    try {
      await onProjectLoaded(target);
    } catch (err: any) {
      setStatus("");
      notify("error", err.message ?? String(err));
    }
  };

  const handleBrowseFolder = async () => {
    const folder = await pickFolder();
    if (folder) setOutputDir(folder);
  };

  const handleBrowseYaml = async () => {
    const file = await pickYaml();
    if (file) {
      setYamlPath(file);
      // Auto-open after selection
      setStatus("Loading project...");
      try {
        await onProjectLoaded(file);
      } catch (err: any) {
        setStatus("");
        notify("error", err.message ?? String(err));
      }
    }
  };

  const handleRemoveRecent = (path: string) => {
    removeRecent(path);
    setRecents(getRecents());
  };

  // Handle YAML file drop
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (!file.name.endsWith(".yml") && !file.name.endsWith(".yaml")) {
      notify("error", "Please drop a .yml or .yaml file");
      return;
    }
    // In Tauri, dropped files have a path property
    const path = (file as any).path as string | undefined;
    if (path) {
      await handleOpen(path);
    } else {
      notify("error", "Could not determine file path from drop");
    }
  };

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="mb-6 text-2xl font-bold">Project Setup</h1>

      {/* Current project indicator */}
      {projectPath && (
        <div className="mb-6 rounded border border-green-200 bg-green-50 p-3">
          <div className="text-xs font-medium text-green-800">Current Project</div>
          <div className="mt-1 truncate font-mono text-xs text-green-700">{projectPath}</div>
        </div>
      )}

      {/* Recent projects */}
      {recents.length > 0 && (
        <div className="mb-6">
          <h2 className="mb-2 text-sm font-medium">Recent Projects</h2>
          <ul className="space-y-1 rounded border border-sparc-gray-200 bg-sparc-gray-50 p-2">
            {recents.map((r) => (
              <li key={r.path} className="flex items-center gap-2">
                <button
                  onClick={() => handleOpen(r.path)}
                  className="flex-1 truncate rounded px-2 py-1.5 text-left text-sm hover:bg-sparc-gray-200"
                >
                  <span className="font-medium">{r.name}</span>
                  {r.template && <span className="ml-2 text-xs text-sparc-gray-500">{r.template}</span>}
                  <div className="truncate font-mono text-xs text-sparc-gray-500">{r.path}</div>
                </button>
                <button
                  onClick={() => handleRemoveRecent(r.path)}
                  className="shrink-0 rounded p-1 text-xs text-sparc-gray-400 hover:bg-sparc-gray-200 hover:text-sparc-gray-700"
                  title="Remove from recents"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

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
            <div className="flex gap-2">
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="/path/to/new/project"
                className="flex-1 rounded border border-sparc-gray-300 px-3 py-2 text-sm"
                readOnly={false}
              />
              <button
                onClick={handleBrowseFolder}
                className="shrink-0 rounded border border-sparc-gray-300 bg-white px-3 py-2 text-sm font-medium hover:bg-sparc-gray-100"
              >
                Browse…
              </button>
            </div>
          </div>

          <button
            onClick={handleCreate}
            disabled={!outputDir}
            className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Create Project
          </button>
        </div>
      ) : (
        <div
          className="space-y-4"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <div>
            <label className="mb-1 block text-sm font-medium">Project YAML File</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={yamlPath}
                onChange={(e) => setYamlPath(e.target.value)}
                placeholder="Select a project.yml file"
                className="flex-1 rounded border border-sparc-gray-300 px-3 py-2 text-sm"
                readOnly={false}
              />
              <button
                onClick={handleBrowseYaml}
                className="shrink-0 rounded border border-sparc-gray-300 bg-white px-3 py-2 text-sm font-medium hover:bg-sparc-gray-100"
              >
                Browse…
              </button>
            </div>
          </div>

          <button
            onClick={() => handleOpen()}
            disabled={!yamlPath}
            className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Open Project
          </button>

          <div className="mt-4 flex items-center justify-center rounded-lg border-2 border-dashed border-sparc-gray-300 p-8 text-center">
            <p className="text-sm text-sparc-gray-500">
              Or drag-and-drop a <span className="font-medium">.yml</span> file here
            </p>
          </div>
        </div>
      )}

      {status && <p className="mt-4 text-sm text-sparc-gray-600">{status}</p>}
    </div>
  );
}

// "Link into config" (SPEC §9.3 inputs): preview the YAML diff the server would apply for an
// input file (`POST /link {apply: false}`), then apply it (`apply: true`). The diff is the
// server's unified diff text, rendered with +/− marks as well as colour.
import { useEffect, useState } from "react";
import { errorMessage } from "../../../api/client";
import { linkInput, type LinkKind, type LinkResult } from "../../../api/projects";
import { invalidate } from "../../../api/resource";
import { Button } from "../../../components/ui/Button";
import { Dialog } from "../../../components/ui/Dialog";
import { Link } from "../../../router";
import { toast } from "../../../stores/ui";

/** A unified diff string (`--- a`, `+++ b`, `@@ … @@`, `+`, `-`, ` `) as diff rows. */
export function UnifiedDiff({ text, label = "Config changes" }: { text: string; label?: string }) {
  const lines = text.replace(/\n$/, "").split("\n");
  const rows = lines.filter((l) => !l.startsWith("---") && !l.startsWith("+++"));
  const changed = rows.some((l) => (l.startsWith("+") || l.startsWith("-")) && l.length > 0);
  return (
    <div className="diff" role="region" aria-label={label}>
      {!changed ? <div className="skip">No changes: the config already uses this file.</div> : null}
      {changed
        ? rows.map((l, k) => {
            if (l.startsWith("@@"))
              return (
                <div key={k} className="skip">
                  <span />
                  <span>{l}</span>
                </div>
              );
            const op = l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : "ctx";
            return (
              <div key={k} className={op}>
                <span aria-hidden="true">{op === "add" ? "+" : op === "del" ? "−" : " "}</span>
                <span>
                  <span className="sr-only">{op === "add" ? "added: " : op === "del" ? "removed: " : ""}</span>
                  {l.slice(op === "ctx" && !l.startsWith(" ") ? 0 : 1) || " "}
                </span>
              </div>
            );
          })
        : null}
    </div>
  );
}

const TITLES: Record<LinkKind, string> = {
  forcing: "Link the campaign forcing (physics.forcing)",
  climate: "Link the CMIP6 change factors (climate.table)",
  layers: "Link people & land cover (planner.layers)",
  features_join: "Use the open predictors in this project",
  features_new_project: "Create the _open project",
};

export function LinkDialog({
  pid,
  kind,
  path,
  open,
  onClose,
  onApplied,
}: {
  pid: string;
  kind: LinkKind;
  path: string;
  open: boolean;
  onClose: () => void;
  onApplied?: (r: LinkResult) => void;
}) {
  const [preview, setPreview] = useState<LinkResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    let live = true;
    setPreview(null);
    setError(null);
    linkInput(pid, { kind, path, apply: false }).then(
      (r) => live && setPreview(r),
      (e: unknown) => live && setError(e),
    );
    return () => {
      live = false;
    };
  }, [open, pid, kind, path]);

  const apply = async () => {
    setBusy(true);
    try {
      const r = await linkInput(pid, { kind, path, apply: true });
      invalidate(`project:${pid}`);
      if (r.new_project_id) {
        invalidate("projects");
        toast("success", "Created the _open project", { href: `/p/${encodeURIComponent(r.new_project_id)}`, linkLabel: "Open it" });
      } else toast("success", r.version ? `Linked into the config (version ${r.version})` : "Linked into the config");
      onApplied?.(r);
      onClose();
    } catch (e) {
      toast("error", "Linking failed", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      busy={busy}
      wide
      title={TITLES[kind]}
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" onClick={apply} busy={busy} disabled={!preview}>
            {kind === "features_new_project" ? "Create project" : "Apply to config"}
          </Button>
        </>
      }
    >
      <div className="stack" style={{ gap: 8 }}>
        <p className="cap">
          File <span className="mono">{path}</span>. The change is saved as a new config version; earlier versions stay in the{" "}
          <Link to={`/p/${encodeURIComponent(pid)}/config`}>config history</Link>.
        </p>
        {error ? (
          <div className="callout" data-tone="crit" role="alert">
            {errorMessage(error)}
          </div>
        ) : preview ? (
          <UnifiedDiff text={preview.yaml_diff} />
        ) : (
          <p className="cap" role="status">
            <span className="spinner" aria-hidden="true" /> Preparing the YAML diff…
          </p>
        )}
      </div>
    </Dialog>
  );
}

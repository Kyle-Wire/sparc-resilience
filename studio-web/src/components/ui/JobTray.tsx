import { useEffect, useRef, useState } from "react";
import { Link } from "../../router";
import { activeJobs, useJobs } from "../../stores/jobs";
import { fmtEta, fmtPct } from "../../theme/format";
import { Icon } from "./Icon";
import { ProgressBar } from "./ProgressBar";
import { StatusChip } from "./StatusChip";

/** Top-bar job tray: running/queued counts, mini progress bars, links to Mission Control. */
export function JobTray() {
  const jobs = useJobs((s) => s.jobs);
  const progress = useJobs((s) => s.progress);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const list = activeJobs(jobs);
  const running = list.filter((j) => j.status === "running" || j.status === "starting" || j.status === "cancelling").length;
  const queued = list.length - running;

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const label = `${running} running, ${queued} queued`;
  return (
    <div className="jobtray" ref={ref} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button type="button" className="btn small" aria-expanded={open} aria-haspopup="true" aria-label={`Jobs: ${label}`} onClick={() => setOpen((o) => !o)}>
        <Icon name={running ? "play" : "clock"} />
        <span className="num">{running}</span>
        <span className="hide-narrow">running</span>
        {queued ? <span className="muted num">+{queued}</span> : null}
      </button>
      {open ? (
        <div className="jobtray-panel" role="dialog" aria-label="Active jobs">
          {list.length === 0 ? <p className="cap" style={{ padding: 8 }}>No jobs are running or queued.</p> : null}
          {list.slice(0, 12).map((j) => {
            const p = progress[j.id];
            const frac = p?.frac ?? j.progress;
            return (
              <Link key={j.id} to={`/jobs/${j.id}`} className="jobtray-item" onClick={() => setOpen(false)}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span style={{ fontWeight: 600, fontSize: "0.86rem" }}>{j.label || j.kind}</span>
                  <StatusChip status={j.status} meta={frac !== null && frac !== undefined ? fmtPct(frac) : undefined} />
                </div>
                <ProgressBar value={j.status === "queued" || j.status === "blocked" ? 0 : frac} label={`${j.label} progress`} />
                <span className="cap">
                  {p?.stage ?? j.stage ?? j.kind}
                  {p?.eta_s !== undefined || j.eta_s !== null ? ` · ${fmtEta(p?.eta_s ?? j.eta_s, p?.eta_lo ?? j.eta_lo, p?.eta_hi ?? j.eta_hi)}` : ""}
                </span>
              </Link>
            );
          })}
          <Link to="/jobs" className="btn small ghost" onClick={() => setOpen(false)}>
            Open Activity
          </Link>
        </div>
      ) : null}
    </div>
  );
}

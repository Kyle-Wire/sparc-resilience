// Right pane, top: the live quick estimate (the emulator preview) in plain words. Kicker "QUICK
// ESTIMATE"; the big number "About −0.42 °F" where you made changes with "−0.05 °F" citywide under
// it; three short facts (area changed with residents, estimated cost, heat-risk shift); a plain trust
// line with exactAdvice() reasons rewritten by words.plainAdvice. Without an emulator: a calm empty
// state and "Build quick estimate (≈1 min)" (the server's build_emulator action, as "Build emulator"
// was). Once the full model's result is up to date, the estimate steps back (smaller, muted, "before
// the full model"). All wording comes from model/quick.ts (pure) and model/words.ts; this file is layout only.
import { useState } from "react";
import { errorMessage, type ApiError } from "../../../../api/client";
import type { Action, Job } from "../../../../api/types";
import { Button } from "../../../../components/ui/Button";
import { runAction } from "../../../../components/ui/EmptyState";
import { InfoTip } from "../../../../components/ui/InfoTip";
import { JobStrip } from "../../../../components/ui/JobStrip";
import { Tooltip } from "../../../../components/ui/Tooltip";
import { useRunDetail } from "../../../../layouts/resources";
import { Link } from "../../../../router";
import { useJobs } from "../../../../stores/jobs";
import { toast } from "../../../../stores/ui";
import { QUICK_WORDS, approxTemp, buildLabel, quickView } from "../../model/quick";
import type { WorkbenchModel } from "../../model/useWorkbench";
import { LAB_WORDS, noQuickEstimateText } from "../../model/words";
import { useFullModelUpToDate } from "./RunFullModel";
import "./quick.css";

/**
 * Props of the preview pane: the whole workbench model (read-only use).
 *
 * Reads: `model.preview` (summary: edited_mean, mean, n_edited, hatched, reasons, trust),
 * `model.previewBusy`, `model.previewError` (+ its `.action`), `model.noEmulator` /
 * `model.emulator.action` (build the emulator), `model.compile` (people, per-lever est_cost),
 * `model.doc.costs`, `model.cellAreaKm2`, `model.grid.n`, `model.heatPreview`, `model.advice`,
 * `model.levers`, `model.unit`.
 *
 * DOM contract (tests and e2e):
 * - the root is a region `aria-label="Preview"` (exactly one on the page: the page passes
 *   `showPreviewCard={false}` to DesignMap);
 * - the row carrying the changed area has `data-n-edited={summary.n_edited}`;
 * - the big number has `data-edited-mean={summary.edited_mean}` (full precision) and the hero
 *   `data-city-mean={summary.mean}`; both are `<dt>`/`<dd>` rows ("where you made changes",
 *   "citywide") like the facts ("Area changed", "Estimated cost", "Heat risk"). The citywide row is
 *   left out when it would repeat the big number (the changes cover the whole city, which the area
 *   line says, or the two round to the same figure);
 * - the root has `data-hatched="true"` when the trust line is a rough one and `data-trust` =
 *   good | rough | hatched (the map hatches the changed places only for "hatched");
 * - once the full model's result is up to date (RunFullModel useFullModelUpToDate) the root and the
 *   hero carry `data-superseded="true"`: the kicker reads "Quick estimate (before the full model)" and
 *   the big number is smaller and muted (the answer card's number leads);
 * - without an emulator the box `[data-no-emulator]` says what building it gives, worded by the engine
 *   state (words.noQuickEstimateText) so it never contradicts the engine line under Run full model;
 *   on a run that saved no trained models it carries `data-no-checkpoint="true"` and a "Launch a run"
 *   link to /p/{pid}/launch instead of the build button.
 */
export type PreviewPaneProps = { model: WorkbenchModel };

const BUILDING = new Set(["queued", "starting", "running", "cancelling", "blocked"]);

/** The run's emulator build in progress, if any (it was started here or anywhere else). */
function useEmulatorJob(rid: string): Job | null {
  return useJobs((s) => Object.values(s.jobs).find((j) => j.run_id === rid && j.kind === "post.emulator" && BUILDING.has(j.status)) ?? null);
}

/** "Build quick estimate (≈1 min)", wrapping before the duration rather than inside it in a narrow pane. */
function BuildLabel({ text }: { text: string }) {
  const at = text.lastIndexOf(" (");
  if (at < 0) return <>{text}</>;
  return (
    <span>
      {text.slice(0, at)} <span className="qe-nowrap">{text.slice(at + 1)}</span>
    </span>
  );
}

/**
 * No emulator yet: say what the quick estimate is for and offer to build it. A run that saved no
 * trained models (engine state no_checkpoint) cannot get one either (the build unpickles
 * checkpoint.pkl), so the box links to Launch instead of offering a build that would fail.
 */
function BuildQuickEstimate({ model, action }: { model: WorkbenchModel; action: Action | null }) {
  const [busy, setBusy] = useState(false);
  const run = useRunDetail(model.rid).data?.run;
  const mode = run?.mode ?? null;
  const pid = model.pid ?? run?.project_id ?? null;
  const job = useEmulatorJob(model.rid);
  const noModels = model.engineState === "no_checkpoint" && !job;
  return (
    <div className="qe-empty" data-no-emulator="true" data-no-checkpoint={noModels || undefined}>
      <p className="qe-empty-title">{job ? QUICK_WORDS.building : LAB_WORDS.noQuickEstimateTitle}</p>
      <p className="cap">{noQuickEstimateText(model.engineState)}</p>
      {noModels ? (
        pid ? (
          <Link className="btn" to={`/p/${encodeURIComponent(pid)}/launch`}>
            {LAB_WORDS.launchRun}
          </Link>
        ) : null
      ) : action && !job ? (
        <Button
          busy={busy}
          icon="bolt"
          onClick={async () => {
            setBusy(true);
            try {
              await runAction(action);
            } catch (e) {
              toast("error", "Could not start the quick estimate", { body: errorMessage(e) });
            } finally {
              setBusy(false);
            }
          }}
        >
          <BuildLabel text={buildLabel(mode, model.levers.length)} />
        </Button>
      ) : null}
      {job ? <JobStrip jobId={job.id} /> : null}
    </div>
  );
}

/** The preview request failed: one plain line, the server's reason muted, and its action if any. */
function PreviewFailed({ error }: { error: ApiError }) {
  const [busy, setBusy] = useState(false);
  const action = error.action;
  return (
    <div className="qe-empty" role="status">
      <p className="qe-empty-title">{QUICK_WORDS.unavailable}</p>
      <p className="cap">{error.message}</p>
      {action ? (
        <Button
          size="small"
          busy={busy}
          onClick={async () => {
            setBusy(true);
            try {
              await runAction(action);
            } catch (e) {
              toast("error", "Could not start it", { body: errorMessage(e) });
            } finally {
              setBusy(false);
            }
          }}
        >
          {action.label}
        </Button>
      ) : null}
    </div>
  );
}

export function PreviewPane({ model }: PreviewPaneProps) {
  // Once the full model has the firm answer, the quick estimate steps back: a smaller, muted figure
  // under "Quick estimate (before the full model)", so one temperature leads the page.
  const superseded = useFullModelUpToDate(model);
  const s = model.preview?.summary ?? null;
  const err = model.previewError;
  const noEmulator = model.noEmulator || err?.code === "no_emulator";
  const v = s
    ? quickView({
        summary: s,
        unit: model.unit,
        compile: model.compile,
        levers: model.levers,
        costs: model.doc.costs,
        cellAreaKm2: model.cellAreaKm2,
        nTotal: model.grid.n,
        heat: model.heatPreview?.shift ?? null,
        advice: model.advice,
      })
    : null;
  const live = !noEmulator && !err && v;
  // The citywide figure only when it says something the big number does not.
  const showCity = !!s && !!v && s.n_edited > 0 && s.n_edited < model.grid.n && v.city.toLowerCase() !== approxTemp(s.edited_mean, model.unit).toLowerCase();
  const rough = live ? v.trust.tone !== "good" : false;
  const caveats = model.heatPreview?.caveats ?? [];
  return (
    <section
      className="qe card"
      aria-label="Preview"
      aria-busy={model.previewBusy || undefined}
      data-hatched={rough || undefined}
      data-trust={live ? v.trust.tone : undefined}
      data-superseded={(live && superseded) || undefined}
    >
      <header className="qe-head">
        <span className="kicker">{live && superseded ? LAB_WORDS.quickEstimateBefore : LAB_WORDS.quickEstimateKicker}</span>
        <InfoTip term="emulator_preview" />
        {model.previewBusy && !noEmulator ? <span className="spinner qe-spin" role="status" aria-label="Updating the quick estimate" /> : null}
      </header>
      {noEmulator ? (
        <BuildQuickEstimate model={model} action={model.emulator?.action ?? err?.action ?? null} />
      ) : err ? (
        <PreviewFailed error={err} />
      ) : s && v ? (
        <>
          <dl className="qe-hero" data-tone={v.big.tone} data-numeric={v.big.numeric || undefined} data-superseded={superseded || undefined} data-city-mean={String(s.mean)}>
            <div className="qe-hero-main">
              <dt>
                {LAB_WORDS.whereChanged}
                <InfoTip term="where_changed" />
              </dt>
              <dd className="qe-big" data-edited-mean={String(s.edited_mean)}>
                {v.big.text}
              </dd>
            </div>
            {showCity ? (
              <div className="qe-hero-city">
                <dt>
                  {LAB_WORDS.citywide}
                  <InfoTip term="citywide" />
                </dt>
                <dd>{v.city}</dd>
              </div>
            ) : null}
          </dl>
          {s.n_edited > 0 ? null : <p className="cap">{QUICK_WORDS.noChangeYetHint}</p>}
          <dl className="qe-facts">
            <div className="qe-fact" data-n-edited={s.n_edited}>
              <dt>{QUICK_WORDS.area}</dt>
              <dd>{v.area}</dd>
            </div>
            {v.cost ? (
              <div className="qe-fact" data-cost="true">
                <dt>{QUICK_WORDS.cost}</dt>
                <dd>
                  <span className="qe-value">{v.cost}</span>
                  <InfoTip label="the estimated cost" text={QUICK_WORDS.costTip} className="qe-info" />
                </dd>
              </div>
            ) : null}
            {v.heat ? (
              <div className="qe-fact" data-heat="true">
                <dt>
                  {QUICK_WORDS.heat}
                  <InfoTip term="heat_risk_level" label={QUICK_WORDS.heat.toLowerCase()} text={QUICK_WORDS.heatTip} />
                </dt>
                <dd>
                  <span className="qe-value">{v.heat}</span>
                  {caveats.length ? (
                    <>
                      {" "}
                      <Tooltip content={caveats.join(" ")}>
                        <span className="pill qe-flag" data-tone="warn" tabIndex={0}>
                          {QUICK_WORDS.heatUnproven}
                        </span>
                      </Tooltip>
                    </>
                  ) : null}
                </dd>
              </div>
            ) : null}
          </dl>
          <div className="qe-trust" data-tone={v.trust.tone} role="note">
            <span className="qe-trust-mark" aria-hidden="true" />
            <div className="qe-trust-text">
              <p className="lab-trust-line">
                {v.trust.line}
                {v.trust.tone === "hatched" ? <InfoTip term="hatched" /> : null}
              </p>
              {v.trust.reasons.length ? (
                <ul className="qe-reasons cap">
                  {v.trust.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          </div>
        </>
      ) : (
        <div className="qe-empty qe-idle">
          <p className="cap">{model.previewBusy ? LAB_WORDS.previewBusy : LAB_WORDS.previewIdle}</p>
        </div>
      )}
    </section>
  );
}

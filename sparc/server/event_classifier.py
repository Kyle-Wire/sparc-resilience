"""Pure pipeline-event classifier.

``classify(line)`` is a pure function: given a raw (stripped, non-empty)
stdout line it returns a ``PipelineEvent`` dict.  It never touches
``sys.stdout``, asyncio, or any I/O — only string matching.

The ``StdoutCapture`` adapter in ``stream.py`` calls this for every line it
captures; other callers (CLI, tests) can call it directly.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_STAGE_RE = re.compile(r"Stage\s+(\d+)", re.IGNORECASE)
_FOLD_RE = re.compile(r"[Ff]old\s+(\d+)")
_METRIC_RE = re.compile(r"(r2|rmse|mae|mape)\s*[=:]\s*([\d.]+)", re.IGNORECASE)
# Match a tqdm-style progress percentage: "  50%|██▌       |" or "100%|###|"
# Restricting to tqdm-shaped contexts prevents matching NUTS "accept: 69.7%".
_PCT_RE = re.compile(r"(?:^|\s)(\d{1,3})%\s*\|")
_CAPACITY_RE = re.compile(r"hidden_dim=(\d+):\s*CV\s+R[²2]=\s*([\d.]+)")
_EPOCH_RE = re.compile(r"(?:Epoch|Retrain|SWA epoch)\s+(\d+)/(\d+)\s+loss=([\d.]+)")
_COMPONENTS_RE = re.compile(r"\[([\w=.\s]+)\]")
_CURRICULUM_RE = re.compile(r"\[CURRICULUM\]\s+(Stage\s+\w+):\s*(.+)")
_CONVERGENCE_RE = re.compile(r"\[CONVERGENCE\]\s+(\w+)")
_DAG_GATE_RE = re.compile(r"\[DAG_APPROVAL_REQUESTED\](?:\s*(\{.*\}))?")
_MODEL_START_RE = re.compile(r"\[MODEL_START\]\s+(\w+)\s+\((\d+)/(\d+)\)")
_MODEL_DONE_RE = re.compile(r"\[MODEL_DONE\]\s+(\w+)\s+\((\d+)/(\d+)\)")
_MODEL_RESULT_RE = re.compile(
    r"^(\w+):\s*R[\u00b22]\s*=\s*([\d.]+),\s*RMSE\s*=\s*([\d.]+)",
    re.IGNORECASE,
)
_NEURAL_FOLD_RE = re.compile(
    r"[Ff]old\s+(\d+)\s*/\s*(\d+)\s*\((\d+)\s+train,\s*(\d+)\s+test\)"
)
_FOLD_DONE_RE = re.compile(r"[Ff]old\s+(\d+)\s+training\s+done\s+in\s+([\d.]+)s")

# Stage 2 model weight map (% of total stage-2 progress)
_MODEL_WEIGHTS: dict[str, tuple[int, int]] = {
    "ols":    (5,  15),
    "gwr":    (15, 35),
    "gwrf":   (35, 55),
    "ggpgam": (55, 70),
}

# Phase-based progress markers (pattern → label displayed in the UI)
_PHASE_RE: list[tuple[re.Pattern[str], str]] = [
    # Correlogram
    (re.compile(r"Correlogram Analysis", re.IGNORECASE), "Correlogram analysis"),
    (re.compile(r"Analyzing\s+(\S+)"), "Analyzing variable"),
    (re.compile(r"Pipeline Configuration", re.IGNORECASE), "Pipeline configuration"),
    # GWEN
    (re.compile(r"GWEN Variable Selection", re.IGNORECASE), "GWEN variable selection"),
    (re.compile(r"GWEN SELECTION RATIONALE", re.IGNORECASE), "GWEN results"),
    # Spatial CV
    (re.compile(r"Loading and Preprocessing", re.IGNORECASE), "Loading data"),
    (re.compile(r"Loading Spatial Folds", re.IGNORECASE), "Loading spatial folds"),
    (re.compile(r"Generating Spatial Folds", re.IGNORECASE), "Generating spatial folds"),
    (re.compile(r"Training\s+(\S+)\s+across all folds", re.IGNORECASE), "Training model"),
    (re.compile(r"Training\s+(\S+)\s+on\s+\d+\s+samples", re.IGNORECASE), "Training model"),
    (re.compile(r"completed.*folds successful", re.IGNORECASE), "Model complete"),
    (re.compile(r"Generating OOF predictions", re.IGNORECASE), "OOF predictions"),
    (re.compile(r"Retraining Base Models", re.IGNORECASE), "Retraining base models"),
    (re.compile(r"Spatial autocorrelation analysis", re.IGNORECASE), "Spatial autocorrelation"),
    (re.compile(r"Spatial CV Complete", re.IGNORECASE), "Spatial CV complete"),
    # Neural meta-learner
    (re.compile(r"Neural Meta.?Learner", re.IGNORECASE), "Neural meta-learner"),
    (re.compile(r"Capacity sweep", re.IGNORECASE), "Capacity sweep"),
    # Causal
    (re.compile(r"Causal Validation", re.IGNORECASE), "Causal validation"),
    # Scenarios
    (re.compile(r"Scenario Simulation", re.IGNORECASE), "Scenario simulation"),
]

# Checkpoint patterns: (regex, checkpoint_id, curated_message, level)
# {0}, {1}, ... are replaced with capture groups.
_CHECKPOINTS: list[tuple[re.Pattern[str], str, str, str]] = [
    # Stage 0 — Correlogram
    (re.compile(r"Computing spatial correlogram for (\w+)"),
     "correlogram_var", "Computing correlogram for {0}", "info"),
    (re.compile(r"Optimal CV block size:\s*(\S+)"),
     "block_size", "Optimal CV block size: {0}", "info"),
    (re.compile(r"Correlogram-Based Spatial Analysis Complete"),
     "correlogram_done", "Correlogram analysis complete", "success"),
    (re.compile(r"Pipeline configuration saved"),
     "config_saved", "Pipeline configuration saved", "info"),
    # Stage 1 — GWEN
    (re.compile(r"Fitting GWEN model"),
     "gwen_fitting", "Fitting GWEN model \u2014 evaluating variable stability", "info"),
    (re.compile(r"\[OK\] GWEN model fitted"),
     "gwen_fitted", "GWEN model fitted successfully", "success"),
    (re.compile(r"Selected (\d+) predictors"),
     "gwen_selected", "{0} predictor variables selected", "success"),
    (re.compile(r"GWEN VARIABLE SELECTION COMPLETE"),
     "gwen_done", "GWEN variable selection complete", "success"),
    # Stage 2 — Spatial CV
    (re.compile(r"Generating Spatial Folds"),
     "folds_gen", "Generating spatial validation folds", "info"),
    (re.compile(r"Fold (\d+):\s*Train=(\d+),\s*Test=(\d+)"),
     "fold_size", "Fold {0} prepared \u2014 {1} train / {2} test samples", "info"),
    (re.compile(r"Optimized OOF Performance"),
     "oof_perf", "Computing out-of-fold performance metrics", "info"),
    (re.compile(r"Spatial CV Complete"),
     "cv_done", "Spatial cross-validation complete", "success"),
    (re.compile(r"Retraining Base Models on Full Dataset"),
     "retrain_start", "Retraining models on full dataset", "info"),
    (re.compile(r"\[OK\] Saved (\S+\.pkl)"),
     "model_saved", "Model saved: {0}", "success"),
    (re.compile(r"Final Results"),
     "final_results", "Compiling final ensemble results", "info"),
    (re.compile(r"V2 Neural Meta-Learner"),
     "meta_learner", "Training neural meta-learner ensemble", "info"),
    (re.compile(r"All surrogates passed validation"),
     "surrogates_ok", "All surrogates passed validation", "success"),
    # Stage 3 — Causal
    (re.compile(r"Running DAG assumption diagnostics"),
     "dag_diag", "Running DAG assumption diagnostics", "info"),
    (re.compile(r"NUTS:\s*warmup phase \((\d+)"),
     "nuts_warmup", "NUTS warmup \u2014 {0} transitions", "info"),
    (re.compile(r"NUTS:\s*sampling phase \((\d+)"),
     "nuts_sampling", "NUTS sampling \u2014 drawing {0} posterior samples", "info"),
    (re.compile(r"NUTS convergence FAILED"),
     "nuts_conv_fail", "NUTS convergence check needs attention", "warn"),
    (re.compile(r"MC.{1,3}:\s*(\d+)\s*/\s*(\d+)\s*accepted"),
     "mc3_result", "MC\u00b3 complete \u2014 {0}/{1} proposals accepted", "success"),
    # Stage 4 — Scenarios
    (re.compile(r"PHYSICS-CONSTRAINED SCENARIO PREDICTOR"),
     "scenario_engine", "Initializing physics-constrained predictor", "info"),
    (re.compile(r"Variable:\s+(\S+)\s+\(direction=(\w+)\)"),
     "scenario_var", "Simulating {0} ({1})", "info"),
    (re.compile(r"RUNNING JOINT SCENARIOS"),
     "joint_scenarios", "Running joint intervention scenarios", "info"),
    (re.compile(r"GENERATING INTERPRETATION MAPS"),
     "gen_maps", "Generating spatial interpretation maps", "info"),
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify(line: str) -> dict[str, Any]:
    """Classify a raw stdout line into a PipelineEvent dict.

    The input must already be stripped and non-empty (caller's responsibility).
    Carriage-return overwrite sequences (tqdm's ``\\r``) are handled: the last
    segment after any ``\\r`` is used.

    Always returns a dict with at least ``{"type": ..., "message": ...}``.
    """
    # Handle tqdm \r overwrites — take the last segment
    if "\r" in line:
        line = line.rsplit("\r", 1)[-1].strip()
        if not line:
            return {"type": "log", "message": ""}

    event: dict[str, Any] = {"type": "log", "message": line}

    # Stage / fold tagging (additive — don't change type)
    m = _STAGE_RE.search(line)
    if m:
        event["stage"] = int(m.group(1))

    m = _FOLD_RE.search(line)
    if m:
        event["fold"] = int(m.group(1))

    # Model result line: "OLS: R² = 0.2942, RMSE = 1.4375"
    # Check before generic metric extraction — more specific, takes priority.
    m_mr_early = _MODEL_RESULT_RE.search(line)
    if m_mr_early:
        event.update(
            type="model_result",
            model=m_mr_early.group(1).upper(),
            r2=float(m_mr_early.group(2)),
            rmse=float(m_mr_early.group(3)),
        )
        return event

    # Metric extraction
    m = _METRIC_RE.search(line)
    if m:
        event["metric"] = m.group(1).lower()
        event["value"] = float(m.group(2))
        event["type"] = "metric"

    # tqdm progress bar → dedicated run_progress event
    m = _PCT_RE.search(line)
    if m:
        event["type"] = "run_progress"
        event["pct"] = int(m.group(1))

    # MODEL_START — checkpoint + weight-based progress
    m_start = _MODEL_START_RE.search(line)
    if m_start:
        model_name = m_start.group(1)
        model_idx = int(m_start.group(2))
        model_total = int(m_start.group(3))
        event.update(
            type="checkpoint",
            checkpoint_id="model_start",
            level="info",
            phase=f"Training {model_name.upper()}",
            model=model_name,
            model_index=model_idx,
            model_total=model_total,
            message=f"Training {model_name.upper()} ({model_idx}/{model_total})",
        )
        w = _MODEL_WEIGHTS.get(model_name)
        if w:
            event["pct"] = w[0]

    # MODEL_DONE — checkpoint + weight-based progress
    m_done = _MODEL_DONE_RE.search(line)
    if m_done:
        model_name = m_done.group(1)
        model_idx = int(m_done.group(2))
        model_total = int(m_done.group(3))
        event.update(
            type="checkpoint",
            checkpoint_id="model_done",
            level="success",
            phase=f"{model_name.upper()} complete",
            model=model_name,
            model_index=model_idx,
            model_total=model_total,
            message=f"{model_name.upper()} training complete ({model_idx}/{model_total})",
        )
        w = _MODEL_WEIGHTS.get(model_name)
        if w:
            event["pct"] = w[1]

    # Phase label (if not already set by MODEL_START/DONE)
    if "phase" not in event:
        for pattern, label in _PHASE_RE:
            if pattern.search(line):
                event["phase"] = label
                break

    # Capacity sweep
    m_cap = _CAPACITY_RE.search(line)
    if m_cap:
        event.update(
            type="capacity_result",
            hidden_dim=int(m_cap.group(1)),
            r2=float(m_cap.group(2)),
        )

    # Epoch / Retrain / SWA epoch update
    m_ep = _EPOCH_RE.search(line)
    if m_ep:
        event.update(
            type="epoch_update",
            epoch=int(m_ep.group(1)),
            n_epochs=int(m_ep.group(2)),
            total_loss=float(m_ep.group(3)),
        )
        if line.lstrip().startswith("Retrain"):
            event["train_phase"] = "retrain"
        elif line.lstrip().startswith("SWA"):
            event["train_phase"] = "swa"
        else:
            event["train_phase"] = "cv"
        m_comp = _COMPONENTS_RE.search(line)
        if m_comp:
            components: dict[str, float] = {}
            for pair in m_comp.group(1).split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        components[k] = float(v)
                    except ValueError:
                        pass
            event["components"] = components

    # Curriculum stage
    m_cur = _CURRICULUM_RE.search(line)
    if m_cur:
        event.update(
            type="curriculum_stage",
            curriculum=m_cur.group(1),
            label=m_cur.group(2),
        )

    # Convergence
    m_conv = _CONVERGENCE_RE.search(line)
    if m_conv:
        event.update(type="convergence", status=m_conv.group(1).lower())

    # DAG approval gate
    m_gate = _DAG_GATE_RE.search(line)
    if m_gate:
        raw = m_gate.group(1) or ""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        event.update(
            type="dag_approval_requested",
            n_edges=payload.get("n_edges", 0),
            n_nodes=payload.get("n_nodes", 0),
        )

    # Fold boundary and model-result lines (only if still a plain log)
    if event["type"] == "log":
        m_nf = _NEURAL_FOLD_RE.search(line)
        if m_nf:
            event.update(
                type="fold_start",
                fold=int(m_nf.group(1)),
                n_folds=int(m_nf.group(2)),
                n_train=int(m_nf.group(3)),
                n_test=int(m_nf.group(4)),
            )
        elif (m_fd := _FOLD_DONE_RE.search(line)):
            event.update(
                type="fold_complete",
                fold=int(m_fd.group(1)),
                elapsed_seconds=float(m_fd.group(2)),
            )
        elif (m_mr := _MODEL_RESULT_RE.search(line)):
            event.update(
                type="model_result",
                model=m_mr.group(1).upper(),
                r2=float(m_mr.group(2)),
                rmse=float(m_mr.group(3)),
            )

    # Checkpoint promotion (only if still a plain log)
    if event["type"] == "log":
        for cp_pat, cp_id, cp_msg, cp_lvl in _CHECKPOINTS:
            cm = cp_pat.search(line)
            if cm:
                event.update(type="checkpoint", checkpoint_id=cp_id, level=cp_lvl)
                try:
                    event["message"] = cp_msg.format(*cm.groups())
                except (IndexError, KeyError):
                    event["message"] = cp_msg
                break

    return event

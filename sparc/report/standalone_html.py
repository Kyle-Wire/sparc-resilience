"""
Self-contained HTML report bundler (Phase 17)
==============================================

Reads the JSON artifacts from a completed run directory and builds a
single, self-contained HTML file with:
    - inline CSS (warm-paper palette to match the desktop)
    - all data embedded as inline ``<script type="application/json">`` blocks
    - small vanilla JS for tab switching and table rendering
    - optional chat history block

No external CDNs or relative file references — the file can be emailed,
hosted on a static bucket, or opened from disk on a machine that has
never run SPARC.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparc.registry.provenance import load_provenance


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_load_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_first(run_dir: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = run_dir / c
        if p.exists():
            return p
    return None


def _collect_artifacts(run_dir: Path) -> dict[str, Any]:
    """Pull the canonical JSON artifacts we know about into a dict.

    Tries the new ``stageN_topic/`` short-name layout first, then falls back
    to the legacy mixed-case directories for older runs that haven't been
    re-run on the new pipeline.
    """
    bundle: dict[str, Any] = {}
    # Stage 0 — correlogram results
    bundle["correlogram"] = (
        _safe_load_json(run_dir / "stage0_correlogram" / "results.json")
        or _safe_load_json(run_dir / "Stage_0_Correlogram" / "correlogram_analysis_results.json")
        or _safe_load_json(run_dir / "Stage_0_Correlogram" / "correlogram_results.json")
    )
    # Stage 1 — GWEN
    bundle["gwen"] = (
        _safe_load_json(run_dir / "stage1_gwen" / "results.json")
        or _safe_load_json(run_dir / "Stage_1_GWEN" / "gwen_results.json")
    )
    # Stages 2-4 — directory names update in PR3; keep legacy paths for now.
    bundle["model_performance"] = _safe_load_json(run_dir / "Stage_2_Spatial_CV" / "model_performance.json")
    bundle["scenario_coefficients"] = _safe_load_json(run_dir / "Stage_3_Causal_Validation" / "scenario_coefficients.json")
    bundle["mc3"] = _safe_load_json(run_dir / "Stage_3_Causal_Validation" / "mc3_results.json")
    bundle["dose_response"] = _safe_load_json(run_dir / "Stage_3_Causal_Validation" / "dose_response_curves.json")
    bundle["scenarios"] = _safe_load_json(run_dir / "Stage_4_Scenarios" / "scenario_results.json")
    bundle["dag"] = _safe_load_json(run_dir / "Stage_3_Causal_Validation" / "dag_discovery_report.json")
    return bundle


# ------------------------------------------------------------------
# HTML scaffolding
# ------------------------------------------------------------------

_CSS = """
:root {
    --paper:#FAF7F0; --ink:#2A1F1A; --crimson:#C0392B; --purple:#5B3A8C;
    --amber:#D4A017; --muted:#8B7B6E; --line:#E5DCD0; --good:#3F7D3A;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 'Iowan Old Style',Georgia,serif}
header{padding:24px 32px;border-bottom:1px solid var(--line);background:#fff}
h1{margin:0 0 4px;font-size:24px}
h2{margin:24px 0 8px;font-size:18px;color:var(--purple)}
h3{margin:16px 0 4px;font-size:14px;color:var(--ink);text-transform:uppercase;letter-spacing:.05em}
.kicker{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
nav{display:flex;gap:4px;padding:0 32px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:10}
nav button{background:none;border:0;padding:12px 16px;font:inherit;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent}
nav button.active{color:var(--ink);border-bottom-color:var(--crimson)}
main{padding:24px 32px;max-width:1200px}
.card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:16px;margin-bottom:16px}
.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);text-transform:uppercase;font-size:11px;letter-spacing:.05em}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:rgba(91,58,140,.1);color:var(--purple)}
.tag.crimson{background:rgba(192,57,43,.1);color:var(--crimson)}
.tag.amber{background:rgba(212,160,23,.15);color:#7d5e0e}
pre{background:#fff;border:1px solid var(--line);border-radius:4px;padding:12px;font-size:12px;overflow:auto}
.section{display:none}
.section.active{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px}
.stat{background:#fff;border:1px solid var(--line);border-radius:6px;padding:12px}
.stat .v{font-size:20px;font-weight:600;color:var(--ink)}
.stat .l{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.chat-msg{padding:8px 12px;border-radius:6px;margin-bottom:8px}
.chat-msg.user{background:rgba(91,58,140,.06);border-left:3px solid var(--purple)}
.chat-msg.assistant{background:rgba(192,57,43,.04);border-left:3px solid var(--crimson)}
"""

_JS = """
function showSection(id){
    document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    document.querySelector('nav button[data-target="'+id+'"]').classList.add('active');
}
function renderTable(rows, container){
    if(!rows||!rows.length){container.innerHTML='<p class="muted">No data.</p>';return;}
    var keys = Object.keys(rows[0]);
    var html='<table><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>';
    rows.forEach(r=>{
        html+='<tr>'+keys.map(k=>{var v=r[k];if(typeof v==='number')v=Number.isInteger(v)?v:v.toFixed(4);return '<td>'+(v==null?'':v)+'</td>';}).join('')+'</tr>';
    });
    html+='</tbody></table>';
    container.innerHTML=html;
}
function init(){
    var data = JSON.parse(document.getElementById('sparc-data').textContent);
    if(data.model_performance && data.model_performance.models){
        var rows = Object.entries(data.model_performance.models).map(([n,m])=>({model:n, r2:m.r2, rmse:m.rmse, mae:m.mae}));
        renderTable(rows, document.getElementById('model-table'));
    }
    if(data.scenario_coefficients){
        var sc = data.scenario_coefficients;
        var arr = Array.isArray(sc.effects) ? sc.effects : Object.entries(sc).map(([k,v])=>({treatment:k, ...v}));
        renderTable(arr.slice(0,50), document.getElementById('causal-table'));
    }
    if(data.scenarios && data.scenarios.summary){
        renderTable(data.scenarios.summary.slice(0,50), document.getElementById('scenarios-table'));
    }
    showSection('overview');
}
window.addEventListener('DOMContentLoaded', init);
"""


def _render_provenance_block(prov: dict | None) -> str:
    if not prov:
        return '<p class="muted">No provenance recorded for this run.</p>'
    git = prov.get("git") or {}
    data = prov.get("data") or {}
    env = prov.get("env") or {}
    env_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in sorted(env.items())
    )
    return (
        '<div class="grid">'
        f'<div class="stat"><div class="l">Config hash</div><div class="v" style="font-size:13px;font-family:monospace">{html.escape((prov.get("config_hash") or "")[:16])}…</div></div>'
        f'<div class="stat"><div class="l">Data sha256</div><div class="v" style="font-size:13px;font-family:monospace">{html.escape((data.get("sha256") or "")[:16])}…</div></div>'
        f'<div class="stat"><div class="l">Git commit</div><div class="v" style="font-size:13px;font-family:monospace">{html.escape((git.get("commit") or "")[:8])}{"+dirty" if git.get("dirty") else ""}</div></div>'
        f'<div class="stat"><div class="l">Frozen at</div><div class="v" style="font-size:13px">{html.escape(prov.get("timestamp_utc") or "?")}</div></div>'
        '</div>'
        '<h3>Environment</h3>'
        f'<table><thead><tr><th>Package</th><th>Version</th></tr></thead><tbody>{env_rows}</tbody></table>'
    )


def _render_chat(chat: list[dict] | None) -> str:
    if not chat:
        return '<p class="muted">No chat history attached to this snapshot.</p>'
    parts = []
    for m in chat:
        role = (m.get("role") or "user").lower()
        cls = "user" if role == "user" else "assistant"
        content = html.escape(str(m.get("content") or ""))
        parts.append(f'<div class="chat-msg {cls}"><strong>{html.escape(role)}:</strong><br>{content}</div>')
    return "\n".join(parts)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def build_standalone_html(
    run_dir: str | Path,
    *,
    chat_history: list[dict] | None = None,
    project_name: str | None = None,
) -> str:
    """Build a fully self-contained HTML snapshot for a run directory."""
    run_dir = Path(run_dir)
    bundle = _collect_artifacts(run_dir)
    prov = load_provenance(run_dir)

    title = project_name or run_dir.name or "SPARC Run"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Pre-render some sections server-side so it's still readable without JS
    perf = bundle.get("model_performance") or {}
    models = perf.get("models") or {}
    if isinstance(models, dict):
        model_rows = [(name, m.get("r2"), m.get("rmse"), m.get("mae")) for name, m in models.items() if isinstance(m, dict)]
    else:
        model_rows = []

    static_models = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{r2:.4f}</td><td>{rmse:.4f}</td><td>{mae:.4f}</td></tr>"
        for (n, r2, rmse, mae) in model_rows
        if all(isinstance(x, (int, float)) for x in (r2, rmse, mae))
    )

    chat_html = _render_chat(chat_history)
    prov_html = _render_provenance_block(prov)

    # Scrub anything that might already contain </script>
    data_json = json.dumps(bundle, default=str).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SPARC Snapshot — {html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
    <div class="kicker">SPARC Resilience snapshot</div>
    <h1>{html.escape(title)}</h1>
    <div class="muted">Generated {html.escape(generated)} · <span class="tag">self-contained</span></div>
</header>
<nav>
    <button data-target="overview" onclick="showSection('overview')">Overview</button>
    <button data-target="models" onclick="showSection('models')">Models</button>
    <button data-target="causal" onclick="showSection('causal')">Causal</button>
    <button data-target="scenarios" onclick="showSection('scenarios')">Scenarios</button>
    <button data-target="provenance" onclick="showSection('provenance')">Provenance</button>
    <button data-target="chat" onclick="showSection('chat')">Chat</button>
</nav>
<main>
    <section id="overview" class="section card">
        <h2>Overview</h2>
        <p class="muted">This file embeds every key artifact from the run so it can be
        opened offline. Use the tabs above to inspect models, causal estimates,
        scenarios, and reproducibility metadata.</p>
        <div class="grid">
            <div class="stat"><div class="l">Models</div><div class="v">{len(model_rows)}</div></div>
            <div class="stat"><div class="l">Has causal</div><div class="v">{"yes" if bundle.get("scenario_coefficients") else "no"}</div></div>
            <div class="stat"><div class="l">Has DAG</div><div class="v">{"yes" if bundle.get("dag") else "no"}</div></div>
            <div class="stat"><div class="l">Chat msgs</div><div class="v">{len(chat_history or [])}</div></div>
        </div>
    </section>
    <section id="models" class="section card">
        <h2>Spatial-CV model performance</h2>
        <table><thead><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th></tr></thead>
        <tbody>{static_models or '<tr><td colspan="4" class="muted">No model performance JSON found.</td></tr>'}</tbody></table>
        <h3 style="margin-top:24px">Full payload</h3>
        <div id="model-table"></div>
    </section>
    <section id="causal" class="section card">
        <h2>Causal estimates</h2>
        <div id="causal-table"></div>
    </section>
    <section id="scenarios" class="section card">
        <h2>Scenarios</h2>
        <div id="scenarios-table"></div>
    </section>
    <section id="provenance" class="section card">
        <h2>Reproducibility</h2>
        {prov_html}
    </section>
    <section id="chat" class="section card">
        <h2>Chat history</h2>
        {chat_html}
    </section>
</main>
<script type="application/json" id="sparc-data">{data_json}</script>
<script>{_JS}</script>
</body>
</html>"""


__all__ = ["build_standalone_html"]

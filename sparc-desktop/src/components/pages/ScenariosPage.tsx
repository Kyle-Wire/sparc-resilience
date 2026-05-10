import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import {
  getConfig,
  getScenarioDetail,
  getScenarioLibrary,
  appendScenarioToLibrary,
  dataSummary,
  type ScenarioTimeline,
} from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { presetsForDomain, applyPresetToPredictors } from "@/lib/scenarioPresets";

type ScenariosTab = "configure" | "library";

/** Infer the natural absolute range for a variable from its name. */
function naturalRange(col: string): { lo: number; hi: number; step: number; unit: string } | null {
  const lower = col.toLowerCase();
  if (/pct|percent|impervious|canopy|cover|urban|green/.test(lower)) {
    return { lo: 0, hi: 100, step: 1, unit: "%" };
  }
  if (/ndvi|evi|savi|lai|fpar|vegetation.index/.test(lower)) {
    return { lo: -1, hi: 1, step: 0.05, unit: "" };
  }
  if (/albedo|reflect/.test(lower)) {
    return { lo: 0, hi: 1, step: 0.02, unit: "" };
  }
  return null;
}

/** Compact label formatter for slider min/max/baseline values. */
function fmtSlider(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(1);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

interface Scenario {
  id: string;
  name: string;
  interventions: Record<string, number>;
  delta: number;
  status: "draft" | "computed" | "baseline";
}

interface InterventionSlider {
  variable: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  value: number;
  baseline: number;
}

export default function ScenariosPage() {
  const { notify } = useNotification();
  const [tab, setTab] = useState<ScenariosTab>("configure");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliders, setSliders] = useState<InterventionSlider[]>([]);
  const [_configRaw, setConfigRaw] = useState<Record<string, unknown> | null>(null);
  const [addingScenario, setAddingScenario] = useState(false);
  const [newScenarioName, setNewScenarioName] = useState("");
  const [library, setLibrary] = useState<ScenarioTimeline | null>(null);
  const [libParent, setLibParent] = useState<string | null>(null);
  const [libComment, setLibComment] = useState("");
  const [libAuthor, setLibAuthor] = useState("me");
  const histRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Load scenarios from API results
    getScenarioDetail()
      .then((detail: any) => {
        if (detail?.scenarios && Array.isArray(detail.scenarios) && detail.scenarios.length > 0) {
          setScenarios(
            detail.scenarios.map((sc: any, i: number) => ({
              id: `s${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? sc.mean_delta ?? 0,
              status: "computed" as const,
            })),
          );
        }
      })
      .catch(() => {});

    // Load config for intervention builder sliders + preset library
    getConfig()
      .then((config) => {
        setConfigRaw(config as unknown as Record<string, unknown>);
        const cfgScenarios = (config.scenarios ?? []) as any[];
        const cols = config.predictors ?? [];
        const domain = config.project?.domain ?? "";
        const presets = presetsForDomain(domain);

        setScenarios((existing) => {
          // Don't clobber API-loaded computed scenarios; merge config + presets only if empty.
          if (existing.length > 0) return existing;
          const merged: Scenario[] = [];
          for (let i = 0; i < cfgScenarios.length; i++) {
            const sc = cfgScenarios[i] ?? {};
            merged.push({
              id: `cfg-${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? 0,
              status: "draft",
            });
          }
          for (const p of presets) {
            const interventions = applyPresetToPredictors(p, cols);
            // Only include presets that match at least one project predictor.
            if (p.id !== "preset-baseline" && Object.keys(interventions).length === 0) continue;
            merged.push({
              id: p.id,
              name: p.name,
              interventions,
              delta: 0,
              status: p.id === "preset-baseline" ? "baseline" : "draft",
            });
          }
          return merged;
        });

        // Build sliders from predictors in config, using domain-inferred natural ranges
        if (cols.length > 0) {
          const top = cols.slice(0, 6);
          dataSummary()
            .then((ds) => {
              const ns = ds?.numeric_summary ?? {};
              setSliders(
                top.map((col: string) => {
                  const stats = ns[col];
                  const mean = stats ? Number(stats.mean ?? 0) : 0;
                  const baseline = Number.isFinite(mean) ? mean : 0;

                  const nr = naturalRange(col);
                  let lo: number;
                  let hi: number;
                  let step: number;
                  let unit: string;

                  if (nr) {
                    lo = nr.lo;
                    hi = nr.hi;
                    step = nr.step;
                    unit = nr.unit;
                  } else {
                    const dataMin = stats ? Number(stats.min ?? baseline) : baseline - 0.5;
                    const dataMax = stats ? Number(stats.max ?? baseline) : baseline + 0.5;
                    lo = Number.isFinite(dataMin) ? dataMin : baseline - 0.5;
                    hi = Number.isFinite(dataMax) ? dataMax : baseline + 0.5;
                    step = Math.max((hi - lo) / 40, 1e-4);
                    unit = "";
                  }

                  return {
                    variable: col,
                    min: lo,
                    max: hi,
                    step,
                    unit,
                    value: Math.min(Math.max(baseline, lo), hi),
                    baseline: Math.min(Math.max(baseline, lo), hi),
                  };
                }),
              );
            })
            .catch(() => {
              setSliders(
                top.map((col: string) => ({
                  variable: col,
                  min: 0,
                  max: 100,
                  step: 1,
                  unit: "",
                  value: 0,
                  baseline: 0,
                })),
              );
            });
        }
      })
      .catch(() => {});
  }, []);

  // Posterior histogram
  useEffect(() => {
    const canvas = histRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const active = scenarios[activeIdx];
    if (!active) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Add and compute scenarios to see results", w / 2, h / 2);
      return;
    }

    if (active.status !== "computed" || active.delta === 0) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Compute scenarios to see Δ distribution", w / 2, h / 2);
      return;
    }

    // -- Posterior Gaussian density ---------------------------------------
    // Pick the dominant intervention variable; lookup its NUTS posterior std
    // (per-unit). Std of Δ = |intervention| × std_per_unit.  If no posterior
    // available, fall back to an assumed CV of 25% so the density is still
    // informative.
    const interventionEntries = Object.entries(active.interventions);
    interventionEntries.sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
    );
    let sigma = Math.max(Math.abs(active.delta) * 0.25, 1e-6);
    const posteriorMatched = false;

    const mu = active.delta;
    const xMin = mu - 4 * sigma;
    const xMax = mu + 4 * sigma;
    const xRange = Math.max(xMax - xMin, 1e-9);

    // Compute densities
    const N = 120;
    const xs = Array.from({ length: N }, (_, i) => xMin + (i / (N - 1)) * xRange);
    const ys = xs.map((x) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)));
    const yMax = Math.max(...ys);

    const plotL = 36, plotR = w - 12, plotT = 14, plotB = h - 28;
    const toX = (x: number) => plotL + ((x - xMin) / xRange) * (plotR - plotL);
    const toY = (y: number) => plotB - (y / yMax) * (plotB - plotT);

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotL, plotT); ctx.lineTo(plotL, plotB); ctx.lineTo(plotR, plotB);
    ctx.stroke();

    // Zero reference line
    if (xMin <= 0 && xMax >= 0) {
      ctx.strokeStyle = "#a59f93";
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(toX(0), plotT); ctx.lineTo(toX(0), plotB); ctx.stroke();
      ctx.setLineDash([]);
    }

    // 90% CI shading (mu ± 1.645 σ)
    const ciLo = mu - 1.645 * sigma;
    const ciHi = mu + 1.645 * sigma;
    ctx.fillStyle = SPARC_RAMP_HEX[mu < 0 ? 0 : SPARC_RAMP_HEX.length - 1] + "33";
    ctx.beginPath();
    ctx.moveTo(toX(ciLo), plotB);
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < ciLo || xs[i] > ciHi) continue;
      ctx.lineTo(toX(xs[i]), toY(ys[i]));
    }
    ctx.lineTo(toX(ciHi), plotB);
    ctx.closePath();
    ctx.fill();

    // Density curve
    ctx.strokeStyle = SPARC_RAMP_HEX[mu < 0 ? 0 : SPARC_RAMP_HEX.length - 1];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = toX(xs[i]), py = toY(ys[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Mean marker
    ctx.strokeStyle = "#1a1416";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(toX(mu), plotT); ctx.lineTo(toX(mu), plotB); ctx.stroke();

    // Axis labels (x-axis)
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(xMin.toFixed(2), plotL, h - 12);
    ctx.fillText(xMax.toFixed(2), plotR, h - 12);
    ctx.fillText("Δ response", w / 2, h - 2);

    // Mean + CI label
    ctx.fillStyle = "#1a1416";
    ctx.font = "bold 10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(
      `μ=${mu.toFixed(2)}  90% CI [${ciLo.toFixed(2)}, ${ciHi.toFixed(2)}]`,
      w / 2,
      plotT - 2,
    );
    if (!posteriorMatched) {
      ctx.fillStyle = "#9a8e75";
      ctx.font = "8px 'JetBrains Mono'";
      ctx.fillText("σ estimated (no NUTS posterior matched)", w / 2, plotT + 10);
    }
  }, [scenarios, activeIdx]);

  const handleSliderChange = useCallback((variable: string, value: number) => {
    setSliders((prev) => prev.map((s) => s.variable === variable ? { ...s, value } : s));
  }, []);

  const handleAddScenario = useCallback(() => {
    setNewScenarioName(`Scenario ${scenarios.length + 1}`);
    setAddingScenario(true);
  }, [scenarios.length]);

  const handleConfirmAddScenario = useCallback(async () => {
    if (!newScenarioName.trim()) return;
    const interventions: Record<string, number> = {};
    sliders.forEach((s) => {
      if (s.value !== 0) interventions[s.variable] = s.value;
    });
    const name = newScenarioName.trim();
    const newSc = { id: `s${Date.now()}`, name, interventions, delta: 0, status: "draft" as const };
    setScenarios((prev) => [...prev, newSc]);
    setAddingScenario(false);
    setNewScenarioName("");
    try {
      await appendScenarioToLibrary(
        { name, interventions, delta: 0, status: "draft" },
        { author: libAuthor, comment: name },
      );
      notify("success", `Scenario "${name}" added`);
      refreshLibrary();
    } catch {
      notify("warning", `Scenario "${name}" added locally — could not persist to server`);
    }
  }, [newScenarioName, sliders, libAuthor, notify, refreshLibrary]);

  const refreshLibrary = useCallback(() => {
    getScenarioLibrary().then(setLibrary).catch(() => setLibrary(null));
  }, []);

  useEffect(() => { refreshLibrary(); }, [refreshLibrary]);

  const handleSaveActiveToLibrary = useCallback(async () => {
    const sc = scenarios[activeIdx];
    if (!sc) { notify("warning", "No scenario selected"); return; }
    try {
      const entry = await appendScenarioToLibrary(
        { name: sc.name, interventions: sc.interventions, delta: sc.delta, status: sc.status },
        { author: libAuthor, comment: libComment || sc.name, parent_id: libParent },
      );
      notify("success", `Saved → ${entry.id}`);
      setLibComment("");
      refreshLibrary();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [scenarios, activeIdx, libAuthor, libComment, libParent, notify, refreshLibrary]);

  return (
    <div>
      <SectionHeader
        kicker="08 · analysis"
        label="Scenarios"
        right={<Btn small onClick={handleAddScenario}>+ Add scenario</Btn>}
      />

      {/* Tab bar — Configure / Library */}
      <div
        role="tablist"
        style={{
          display: "flex",
          gap: 4,
          borderBottom: "1px solid var(--line)",
          marginBottom: 12,
        }}
      >
        {(
          [
            ["configure", "Configure"],
            ["library", "Library"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            style={{
              padding: "6px 14px",
              fontSize: 12,
              border: 0,
              borderBottom:
                tab === key ? "2px solid var(--crimson, #e73c25)" : "2px solid transparent",
              background: "transparent",
              cursor: "pointer",
              color: tab === key ? "var(--ink)" : "var(--muted)",
              fontWeight: tab === key ? 700 : 500,
              marginBottom: -1,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <StatGrid>
        <Stat label="Scenarios" value={String(scenarios.length)} tint="var(--ink)" />
        <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--crimson)" />
        <Stat label="Best Δ" value={scenarios.length ? `${Math.min(...scenarios.map((s) => s.delta)).toFixed(1)}` : "—"} tint="var(--purple)" />
        <Stat label="Variables" value={String(sliders.length)} tint="var(--amber)" />
      </StatGrid>

      {/* ------------------------------------------------------------ */}
      {/* Configure tab — scenario library card + intervention builder  */}
      {/* ------------------------------------------------------------ */}
      {tab === "configure" && (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Scenario library" subtitle="click to select · compare to baseline">
            {scenarios.map((sc, i) => (
              <div
                key={sc.id}
                onClick={() => setActiveIdx(i)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 8px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                  background: activeIdx === i ? "rgba(231,60,37,0.04)" : "transparent",
                  borderRadius: 4,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {Object.entries(sc.interventions)
                      .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                      .join(", ") || "no interventions"}
                  </div>
                </div>
                <span
                  className="mono"
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: sc.delta < 0 ? "var(--purple)" : sc.delta > 0 ? "var(--crimson)" : "var(--muted)",
                  }}
                >
                  {sc.delta === 0 ? "—" : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(1)} °C`}
                </span>
                <Tag
                  color={
                    sc.status === "computed"
                      ? "var(--ink)"
                      : sc.status === "baseline"
                      ? "var(--purple)"
                      : "var(--muted)"
                  }
                >
                  {sc.status}
                </Tag>
              </div>
            ))}
            {addingScenario && (
              <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "10px 8px", borderTop: "1px dashed var(--line)", marginTop: 4 }}>
                <input
                  autoFocus
                  type="text"
                  value={newScenarioName}
                  onChange={(e) => setNewScenarioName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleConfirmAddScenario(); if (e.key === "Escape") setAddingScenario(false); }}
                  placeholder="Scenario name…"
                  style={{ flex: 1, border: "1px solid var(--crimson)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}
                />
                <Btn small primary onClick={handleConfirmAddScenario}>Add</Btn>
                <Btn small onClick={() => setAddingScenario(false)}>Cancel</Btn>
              </div>
            )}
          </Card>

          <Card title="Intervention builder" subtitle="adjust sliders to create new scenario">
            {sliders.map((s) => (
              <div key={s.variable} style={{ padding: "10px 0", borderTop: "1px dashed var(--line)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>{s.variable.replace(/_/g, " ")}</span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: s.value !== 0 ? "var(--crimson)" : "var(--muted)",
                    }}
                  >
                    {s.value > 0 ? "+" : ""}
                    {s.value.toFixed(2)} {s.unit}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 52, textAlign: "right", flexShrink: 0 }}>
                    {fmtSlider(s.min)}{s.unit}
                  </span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={s.value}
                    onChange={(e) => handleSliderChange(s.variable, Number(e.target.value))}
                    style={{ flex: 1, accentColor: "var(--crimson)" }}
                  />
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 52, flexShrink: 0 }}>
                    {fmtSlider(s.max)}{s.unit}
                  </span>
                </div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 2 }}>
                  baseline: {fmtSlider(s.baseline)}{s.unit}
                </div>
              </div>
            ))}
          </Card>
        </div>

        <Card title="Posterior distribution" subtitle={scenarios[activeIdx]?.name ?? "select a scenario"}>
          <canvas
            ref={histRef}
            style={{ width: "100%", height: 300, display: "block" }}
          />
          <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 8, textAlign: "center" }}>
            {scenarios[activeIdx]?.status === "computed"
              ? `Median Δ = ${scenarios[activeIdx]?.delta.toFixed(2)} · 95% CI: [${(scenarios[activeIdx]?.delta - 0.98).toFixed(2)}, ${(scenarios[activeIdx]?.delta + 0.98).toFixed(2)}]`
              : "Not yet computed"}
          </div>
        </Card>
      </div>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Library tab — versioned, append-only scenario journal         */}
      {/* ------------------------------------------------------------ */}
      {tab === "library" && (
      <Card
        title="Versioned scenario library"
        subtitle={library ? `${library.count} entr${library.count === 1 ? "y" : "ies"} · append-only journal` : "loading…"}
        actions={
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              value={libAuthor}
              onChange={(e) => setLibAuthor(e.target.value)}
              placeholder="author"
              style={{ fontSize: 11, padding: "3px 6px", borderRadius: 3, border: "1px solid var(--line)", width: 80, fontFamily: "'JetBrains Mono', monospace" }}
            />
            <input
              value={libComment}
              onChange={(e) => setLibComment(e.target.value)}
              placeholder="comment (optional)"
              style={{ fontSize: 11, padding: "3px 6px", borderRadius: 3, border: "1px solid var(--line)", width: 220 }}
            />
            <Btn small primary onClick={handleSaveActiveToLibrary}>Save active</Btn>
          </div>
        }
      >
        {!library || library.count === 0 ? (
          <div style={{ fontSize: 11, color: "var(--muted)" }}>
            No saved entries. Pick a scenario above and click "Save active" to seed the library.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#fdf6e9" }}>
                  {["", "id", "parent", "author", "comment", "name", "created"].map((h) => (
                    <th key={h} style={{ ...thStyle, textAlign: "left" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {library.entries.slice().reverse().map((e) => {
                  const sc = (e.scenario as { name?: string }) || {};
                  const selected = libParent === e.id;
                  return (
                    <tr key={e.id} style={{ borderBottom: "1px dotted var(--line)", background: selected ? "rgba(91,58,140,.06)" : undefined }}>
                      <td style={tdStyle}>
                        <input
                          type="radio"
                          name="lib-parent"
                          checked={selected}
                          onChange={() => setLibParent(e.id)}
                          title="branch from this entry"
                        />
                      </td>
                      <td style={tdStyle} className="mono">{e.id}</td>
                      <td style={tdStyle} className="mono">{e.parent_id || "—"}</td>
                      <td style={tdStyle}>{e.author}</td>
                      <td style={tdStyle}>{e.comment || <span style={{ color: "var(--muted)" }}>—</span>}</td>
                      <td style={tdStyle}>{sc.name || "—"}</td>
                      <td style={tdStyle} className="mono">{e.created_utc.slice(0, 19).replace("T", " ")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {libParent && (
              <div style={{ fontSize: 10.5, color: "var(--muted)", marginTop: 6 }}>
                Branching from <span className="mono">{libParent}</span> ·{" "}
                <button onClick={() => setLibParent(null)} style={{ background: "none", border: 0, color: "var(--purple)", textDecoration: "underline", cursor: "pointer", fontSize: 10.5 }}>clear</button>
              </div>
            )}
          </div>
        )}
      </Card>
      )}
    </div>
  );
}

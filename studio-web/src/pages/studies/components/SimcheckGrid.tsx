// The simcheck generator × seed grid (SPEC §5.11): each replicate is a cell coloured by its
// effect share on the diverging ramp centred on 1, grey while pending, red on error, with a
// dot for a gate redraw. Status is also in each cell's label and marker, never colour alone.
import type { SimcheckView } from "../../../api/studies";
import { useDark } from "../../../stores/ui";
import { fmtDuration, fmtNum } from "../../../theme/format";
import { rampColor, TOKEN_VALUES } from "../../../theme/palette";
import { cellKey, simCellLook, simGrid, simProgress } from "../model/simcheck";

export function SimcheckGrid({ view, workers = 1 }: { view: SimcheckView | null | undefined; workers?: number }) {
  const dark = useDark();
  const grid = simGrid(view);
  if (!grid.generators.length) return <p className="cap">No replicates planned yet.</p>;
  const prog = simProgress(grid, workers, view?.eta_s ?? null);
  const tok = TOKEN_VALUES[dark ? "dark" : "light"];
  const ramp = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => rampColor("div", t, dark)).join(", ")})`;
  const step = grid.seeds.length > 30 ? 10 : 5;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="sx-meta">
        <span>
          Replicates <b>{prog.done}</b> of <b>{prog.total}</b>
        </span>
        {prog.running ? (
          <span>
            Running <b>{prog.running}</b>
          </span>
        ) : null}
        {prog.errors ? (
          <span>
            Errors <b>{prog.errors}</b>
          </span>
        ) : null}
        {prog.eta_s ? (
          <span>
            ETA <b>≈{fmtDuration(prog.eta_s)}</b>
          </span>
        ) : null}
      </div>
      <div className="sg-wrap">
        <table className="sg-grid" aria-label="Simulation check: generator by seed (effect share; 1 = recovered exactly)">
          <thead>
            <tr>
              <th scope="col">
                <span className="sr-only">Generator</span>
              </th>
              {grid.seeds.map((s) => (
                <th key={s} scope="col">
                  {s % step === 0 ? s : <span className="sr-only">{s}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.generators.map((g) => (
              <tr key={g}>
                <th scope="row">{g}</th>
                {grid.seeds.map((s) => {
                  const c = grid.cells.get(cellKey(g, s));
                  if (!c) return <td key={s} data-status="none" aria-hidden="true" />;
                  const look = simCellLook(c, g, s, grid.span, dark);
                  // backgroundColor, not the `background` shorthand: the stylesheet hatches gate failures with background-image.
                  return (
                    <td key={s} data-status={c.status} data-redraw={look.redraw || undefined} data-fill={look.fill} style={{ backgroundColor: look.fill, color: look.ink }} title={look.label} aria-label={look.label}>
                      {look.marker}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="sg-legend" aria-label="Grid legend">
        <span className="sg-ramp">
          share {fmtNum(1 - grid.span, 1)}
          <span className="bar" style={{ background: ramp }} aria-hidden="true" />
          {fmtNum(1 + grid.span, 1)} (1 = exact)
        </span>
        <span>
          <i style={{ background: tok["--gray-mark"] }} aria-hidden="true" />
          pending
        </span>
        <span>
          <i style={{ background: tok["--critical"] }} aria-hidden="true" />
          error (!)
        </span>
        <span>• gate redraw</span>
        <span>– no planted effect (null)</span>
        <span>hatched: failed the data gate</span>
      </div>
    </div>
  );
}

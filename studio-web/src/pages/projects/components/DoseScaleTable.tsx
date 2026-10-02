// Dose-scale table of one lever (SPEC §9.3 levers): each dose as a multiple of the layer's
// standard deviation and the percentile a median cell reaches with it. Doses beyond 1 sd are
// amber and say so in words ("beyond 1 sd"), so the warning does not rely on colour.
import type { DoseScaleEntry } from "../../../api/projects";
import { Icon } from "../../../components/ui/Icon";
import { fmtNum, fmtSig, unitLabel } from "../../../theme/format";
import { doseScaleRows } from "../model/doseScale";

export function DoseScaleTable({ lever, entry, unit, stale }: { lever: string; entry: DoseScaleEntry | null | undefined; unit?: string | null; stale?: boolean }) {
  const rows = doseScaleRows(entry);
  if (!entry || !rows.length) return <p className="cap">No dose scale for {lever} yet: run the data check.</p>;
  const u = unitLabel(unit);
  return (
    <div className="stack" style={{ gap: 4 }}>
      <table className="tbl dose-table" aria-label={`Dose scale of ${lever}`}>
        <thead>
          <tr>
            <th scope="col">Dose{u ? ` (${u})` : ""}</th>
            <th scope="col" className="r">
              Dose ÷ sd
            </th>
            <th scope="col" className="r">
              Median cell reaches
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} data-beyond-sd={r.beyondSd ? "true" : undefined} className={r.beyondSd ? "amber" : undefined}>
              <td className="num">{fmtSig(r.dose, 4)}</td>
              <td className="r">
                {r.inSd === null ? "—" : `${fmtNum(r.inSd, 2)} sd`}
                {r.beyondSd ? (
                  <span className="amber-flag">
                    <Icon name="alert" size={12} /> beyond 1 sd
                  </span>
                ) : null}
              </td>
              <td className="r">{r.percentile === null ? "—" : `p${fmtNum(r.percentile, 0)}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <span className="cap">
        sd of {lever} {fmtSig(entry.sd, 3)}
        {u ? ` ${u}` : ""}. Doses beyond 1 sd push most cells outside the observed range, so their effects lean on extrapolation.
        {stale ? " The levers changed since the last check: run it again to refresh." : ""}
      </span>
    </div>
  );
}

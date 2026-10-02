// Dose-scale rows of the levers step (SPEC §9.3): each dose of a lever as a multiple of the
// layer's standard deviation and the percentile a median cell reaches. A dose beyond 1 sd is
// flagged (amber, with text): the response there is mostly extrapolation.
import type { DoseScaleEntry } from "../../../api/projects";

export type DoseRow = { dose: number; inSd: number | null; percentile: number | null; beyondSd: boolean };

/** Threshold above which a dose is flagged, in standard deviations. */
export const DOSE_SD_LIMIT = 1;

export function doseScaleRows(entry: DoseScaleEntry | null | undefined): DoseRow[] {
  if (!entry) return [];
  const sd = typeof entry.sd === "number" && entry.sd > 0 ? entry.sd : null;
  return entry.doses.map((dose, i) => {
    const given = entry.doses_in_sd?.[i];
    const inSd = typeof given === "number" && Number.isFinite(given) ? given : sd !== null ? Math.abs(dose) / sd : null;
    const p = entry.percentile_reached?.[i];
    return { dose, inSd, percentile: typeof p === "number" && Number.isFinite(p) ? p : null, beyondSd: inSd !== null && inSd > DOSE_SD_LIMIT };
  });
}

/** Doses of a lever that exceed 1 sd (for a summary line). */
export function dosesBeyondSd(entry: DoseScaleEntry | null | undefined): number[] {
  return doseScaleRows(entry)
    .filter((r) => r.beyondSd)
    .map((r) => r.dose);
}

// Configured scenario names exactly as core generates them (sparc.core.scenarios
// .specs_from_config): a ladder "Canopy Increase" with increments [5, 10] gives
// "Canopy Increase +5", "Canopy Increase +10"; a decrease ladder uses U+2212 ("−10"). The
// increment is printed with Python's `{inc:g}`. Packages keep their own name. The slug rule
// is sparc.core.catalog.scenario_slug (headline scenarios and `sc:<slug>` layers use it).
import { MINUS } from "../../../theme/format";

export type Ladder = { name?: unknown; variable?: unknown; direction?: unknown; increments?: unknown };
export type Package = { name?: unknown; interventions?: unknown };

/** Python's `format(v, "g")`: 6 significant digits, trailing zeros dropped, exponent outside [1e-4, 1e6). */
export function pyG(v: number): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "nan" : v > 0 ? "inf" : "-inf";
  if (v === 0) return Object.is(v, -0) ? "-0" : "0";
  const [mant, expStr] = v.toExponential(5).split("e");
  const exp = Number(expStr);
  if (exp < -4 || exp >= 6) {
    const m = mant.replace(/\.?0+$/, "");
    const sign = exp < 0 ? "-" : "+";
    return `${m}e${sign}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  const s = v.toFixed(Math.max(0, 5 - exp));
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null);

export function isDecrease(direction: unknown): boolean {
  return String(direction ?? "increase").toLowerCase() === "decrease";
}

/** The scenario names one ladder generates (one per increment). */
export function ladderNames(l: Ladder): string[] {
  const name = String(l.name ?? "");
  const sign = isDecrease(l.direction) ? MINUS : "+";
  const incs = Array.isArray(l.increments) ? l.increments : [];
  const out: string[] = [];
  for (const raw of incs) {
    const inc = num(raw);
    if (inc === null) continue;
    out.push(`${name} ${sign}${pyG(inc)}`);
  }
  return out;
}

/** Every configured scenario name of a raw config, in core's order (ladders, then packages). */
export function configuredScenarioNames(raw: unknown): string[] {
  const r = (raw ?? {}) as { scenarios?: unknown; joint_scenarios?: unknown };
  const names: string[] = [];
  for (const l of Array.isArray(r.scenarios) ? (r.scenarios as Ladder[]) : []) names.push(...ladderNames(l ?? {}));
  for (const p of Array.isArray(r.joint_scenarios) ? (r.joint_scenarios as Package[]) : []) names.push(String(p?.name ?? ""));
  return names.filter((n) => n.trim() !== "");
}

/**
 * Configured-scenario id: lower case; "−"/"-" before a number → "minus-", "+" → "plus-";
 * every other run of non-[a-z0-9] → "-"; "-" stripped; "-2", "-3"… on collision with `taken`.
 */
export function scenarioSlug(name: string, taken: Iterable<string> = []): string {
  let s = name.toLowerCase();
  s = s.replace(/[−-](?=\s*\d)/g, " minus-");
  s = s.replace(/\+(?=\s*\d)/g, " plus-");
  s = s.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "scenario";
  const used = new Set(taken);
  if (!used.has(s)) return s;
  let i = 2;
  while (used.has(`${s}-${i}`)) i++;
  return `${s}-${i}`;
}

/** Name → slug for every configured scenario, with the collision suffixes core assigns. */
export function scenarioSlugs(names: string[]): { name: string; slug: string }[] {
  const taken: string[] = [];
  return names.map((name) => {
    const slug = scenarioSlug(name, taken);
    taken.push(slug);
    return { name, slug };
  });
}

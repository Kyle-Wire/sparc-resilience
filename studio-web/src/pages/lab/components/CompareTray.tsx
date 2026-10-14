// The compare tray (SPEC §7.4, §7.12): up to four pinned items at the bottom of the Design
// workbench, with "Compare" once two are pinned. Nothing shows until the first "Pin to compare"
// (an empty bar would only cover the page).
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { Link } from "../../../router";
import { TRAY_MAX, compareHref, trayRefs, useTray } from "../model/tray";

export function CompareTray({ rid }: { rid: string }) {
  const items = trayRefs(useTray((s) => s.byRun[rid]));
  const unpin = useTray((s) => s.unpin);
  const clear = useTray((s) => s.clear);
  const canPinBaseline = !items.some((i) => i.ref.kind === "baseline") && items.length < TRAY_MAX;
  if (!items.length) return null;
  return (
    <section className="compare-tray" aria-label="Compare tray">
      <span className="eyebrow">Compare tray</span>
      {items.map((i) => (
        <span key={i.enc} className="chip">
          {i.label}
          <IconButton icon="x" size="small" label={`Remove ${i.label} from the tray`} onClick={() => unpin(rid, i.enc)} />
        </span>
      ))}
      {items.length < 2 ? <span className="cap">Pin one more result, scenario or plan to compare them (up to 4).</span> : null}
      {canPinBaseline ? (
        <Button size="small" variant="ghost" onClick={() => useTray.getState().pin(rid, { kind: "baseline" }, "Baseline (no change)")}>
          + baseline
        </Button>
      ) : null}
      <span className="spacer" />
      <Button size="small" variant="ghost" onClick={() => clear(rid)}>
        Clear
      </Button>
      {items.length >= 2 ? (
        <Link className="btn small primary" to={compareHref(rid, items.map((i) => i.ref))}>
          Compare {items.length}
        </Link>
      ) : null}
    </section>
  );
}

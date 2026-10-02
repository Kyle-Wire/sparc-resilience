// The compare tray (SPEC §7.4, §7.12): up to four pinned items at the bottom of the Design
// workbench, with "Compare" once two are pinned.
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { Link } from "../../../router";
import { TRAY_MAX, compareHref, trayRefs, useTray } from "../model/tray";

export function CompareTray({ rid }: { rid: string }) {
  const items = trayRefs(useTray((s) => s.byRun[rid]));
  const unpin = useTray((s) => s.unpin);
  const clear = useTray((s) => s.clear);
  const canPinBaseline = !items.some((i) => i.ref.kind === "baseline") && items.length < TRAY_MAX;
  return (
    <section className="compare-tray" aria-label="Compare tray">
      <span className="eyebrow">Compare tray</span>
      {items.length ? (
        items.map((i) => (
          <span key={i.enc} className="chip">
            {i.label}
            <IconButton icon="x" size="small" label={`Remove ${i.label} from the tray`} onClick={() => unpin(rid, i.enc)} />
          </span>
        ))
      ) : (
        <span className="cap">Pin results, configured scenarios or plans to compare them (2–4 items).</span>
      )}
      {canPinBaseline && items.length ? (
        <Button size="small" variant="ghost" onClick={() => useTray.getState().pin(rid, { kind: "baseline" }, "Baseline (no change)")}>
          + baseline
        </Button>
      ) : null}
      <span className="spacer" />
      {items.length ? (
        <Button size="small" variant="ghost" onClick={() => clear(rid)}>
          Clear
        </Button>
      ) : null}
      {items.length >= 2 ? (
        <Link className="btn small primary" to={compareHref(rid, items.map((i) => i.ref))}>
          Compare {items.length}
        </Link>
      ) : null}
    </section>
  );
}

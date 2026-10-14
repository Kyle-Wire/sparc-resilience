// Host pre-check for a network input (SPEC §9.3 inputs, api.md §2 netcheck): lists the hosts
// the job will contact and whether each answers (latency or error), so a blocked proxy or
// offline laptop is visible before a job is queued. Runs once on mount, then on demand.
import { useCallback, useEffect, useState } from "react";
import { errorMessage } from "../../../api/client";
import { netcheck, type NetcheckRow } from "../../../api/inputs";
import { Button } from "../../../components/ui/Button";
import { Pill } from "../../../components/ui/Pill";
import { TipSpan } from "../../../layouts/tips";
import { fmtInt } from "../../../theme/format";

export function NetCheck({ hosts, auto = true }: { hosts: readonly string[]; auto?: boolean }) {
  const [rows, setRows] = useState<NetcheckRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const key = hosts.join(",");

  const run = useCallback(async () => {
    if (!key) return;
    setBusy(true);
    setError(null);
    try {
      const r = await netcheck(key.split(","));
      setRows(r.results);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }, [key]);

  useEffect(() => {
    if (auto) void run();
  }, [auto, run]);

  if (!hosts.length) return <p className="cap">This input needs no network access.</p>;
  const byHost = new Map((rows ?? []).map((r) => [r.host, r]));
  return (
    <div className="netcheck stack" style={{ gap: 6 }} aria-label="Network hosts">
      <div className="row" style={{ gap: 6 }}>
        <span className="eyebrow">Servers it downloads from</span>
        {hosts.map((h) => {
          const r = byHost.get(h);
          if (!r)
            return (
              <TipSpan key={h} tip="Not checked yet.">
                <Pill icon="dot">{h}</Pill>
              </TipSpan>
            );
          return r.ok ? (
            <TipSpan key={h} tip={`${h} answered${r.ms !== null ? ` in ${fmtInt(r.ms)} ms` : ""}.`}>
              <Pill tone="good" icon="check">
                {h} {r.ms !== null ? <span className="num">{fmtInt(r.ms)} ms</span> : null}
              </Pill>
            </TipSpan>
          ) : (
            <TipSpan key={h} tip={`${h} did not answer.`} technical={r.error ?? null}>
              <Pill tone="crit" icon="x">
                {h} unreachable
              </Pill>
            </TipSpan>
          );
        })}
        <Button size="small" variant="ghost" icon="refresh" busy={busy} onClick={() => void run()}>
          Check hosts
        </Button>
      </div>
      {error ? <span className="cap">Host check failed: {error}</span> : null}
      {rows?.some((r) => !r.ok) ? <span className="cap">Some hosts did not answer: the job may fail. Check the network or proxy, or try again later.</span> : null}
    </div>
  );
}

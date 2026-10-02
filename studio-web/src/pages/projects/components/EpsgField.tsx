// CRS field of the data step (SPEC §9.3): an EPSG search box over a curated offline list
// (any typed "EPSG:<code>" is accepted too), the server's CRS hint, and a centroid inset
// that places the data on a small world map once a check has given it lon/lat.
import { useId, useState } from "react";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { fmtNum } from "../../../theme/format";
import { normaliseEpsg, searchEpsg, utmForLonLat } from "../model/epsg";
import { IssueLines, useCfg } from "./cfg";

/** "41.826° N, 71.403° W". */
export function fmtLatLon(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${fmtNum(Math.abs(lat), 3)}° ${ns}, ${fmtNum(Math.abs(lon), 3)}° ${ew}`;
}

/** A 120×60 equirectangular world box with the data centroid marked. */
export function CentroidInset({ lon, lat }: { lon: number; lat: number }) {
  const x = ((lon + 180) / 360) * 120;
  const y = ((90 - lat) / 180) * 60;
  return (
    <figure className="centroid-inset">
      <svg viewBox="0 0 120 60" width={120} height={60} role="img" aria-label={`Data centroid at ${fmtLatLon(lat, lon)}`}>
        <rect x={0} y={0} width={120} height={60} rx={4} className="inset-bg" />
        {[-60, -30, 0, 30, 60].map((g) => (
          <line key={`lat${g}`} x1={0} x2={120} y1={((90 - g) / 180) * 60} y2={((90 - g) / 180) * 60} className="inset-grid" />
        ))}
        {[-120, -60, 0, 60, 120].map((g) => (
          <line key={`lon${g}`} y1={0} y2={60} x1={((g + 180) / 360) * 120} x2={((g + 180) / 360) * 120} className="inset-grid" />
        ))}
        <circle cx={x} cy={y} r={5} className="inset-halo" />
        <circle cx={x} cy={y} r={2.2} className="inset-dot" />
      </svg>
      <figcaption className="cap num">{fmtLatLon(lat, lon)}</figcaption>
    </figure>
  );
}

export function EpsgField({ hint, centroid, confidence }: { hint?: string | null; centroid?: { lon: number; lat: number } | null; confidence?: number | null }) {
  const c = useCfg();
  const id = useId();
  const cur = (c.value("data.crs") as string | null | undefined) ?? "";
  const [query, setQuery] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const text = query ?? cur;
  const results = open ? searchEpsg(text === cur ? "" : text, 8) : [];
  const utm = centroid ? utmForLonLat(centroid.lon, centroid.lat) : null;
  const choose = (code: string | null) => {
    c.set("data.crs", code ?? undefined);
    setQuery(null);
    setOpen(false);
  };
  return (
    <div data-cfg-path="data.crs" className="epsg-field">
      <div className="field">
        <label htmlFor={id}>
          Coordinate reference system (EPSG)
          {c.suggested("data.crs") ? (
            <>
              {" "}
              <Badge tone="accent" title={confidence !== null && confidence !== undefined ? `Suggested from the data (confidence ${Math.round(confidence * 100)}%)` : "Suggested from the data"}>
                suggested
              </Badge>
            </>
          ) : null}
        </label>
        <div className="row" style={{ gap: 6, alignItems: "start" }}>
          <div className="epsg-combo">
            <input
              id={id}
              type="text"
              role="combobox"
              aria-expanded={open}
              aria-controls={`${id}-list`}
              aria-autocomplete="list"
              placeholder="Search: 3438, UTM 19N, Rhode Island…"
              value={text}
              onFocus={() => setOpen(true)}
              onChange={(e) => {
                setQuery(e.target.value);
                setOpen(true);
              }}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setOpen(false);
                  setQuery(null);
                }
                if (e.key === "Enter") {
                  e.preventDefault();
                  const code = normaliseEpsg(text) ?? results[0]?.code ?? null;
                  if (text.trim() === "") choose(null);
                  else if (code) choose(code);
                }
              }}
              onBlur={() => window.setTimeout(() => setOpen(false), 150)}
            />
            {open && results.length ? (
              <ul className="epsg-list" id={`${id}-list`} role="listbox" aria-label="Matching coordinate systems">
                {results.map((r) => (
                  <li key={r.code} role="option" aria-selected={r.code === cur}>
                    <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => choose(r.code)}>
                      <span className="mono">{r.code}</span> {r.name} <span className="muted">· {r.unit}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
          {cur ? (
            <Button size="small" variant="ghost" onClick={() => choose(null)}>
              Clear
            </Button>
          ) : null}
        </div>
        <span className="hint">
          Needed for open data, the CMIP6 site, GeoTIFF exports and the people objective.
          {hint && hint !== cur ? (
            <>
              {" "}
              The data looks like <button type="button" className="linklike" onClick={() => choose(hint)}>{hint}</button>.
            </>
          ) : null}
          {utm && utm !== cur ? (
            <>
              {" "}
              The data centroid is in <button type="button" className="linklike" onClick={() => choose(utm)}>{utm}</button> (UTM).
            </>
          ) : null}
        </span>
        <IssueLines issues={c.issues("data.crs")} />
      </div>
      {centroid ? <CentroidInset lon={centroid.lon} lat={centroid.lat} /> : null}
    </div>
  );
}

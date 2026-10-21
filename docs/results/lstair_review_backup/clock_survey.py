"""Per downloaded city: readings, share with a timestamp, windows, sensor ids, clock kind (no network)."""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sparc-resilience")
from pathlib import Path
import pandas as pd
from sparc.core.identify.traverses import scan_traverses
from sparc.core.lstair.dataset import city_spec, SKIP_FILES
raw = Path(sys.argv[1]); out = {}
dates = {"boise_id": "2019-08-14", "boston_ma": "2019-07-29"}
for d in sorted(p for p in raw.iterdir() if p.is_dir()):
    rec = json.loads((d / "_download.json").read_text())
    name = next((n for n in json.load(open(sys.argv[2])) if city_spec(n).slug == d.name), d.name)
    s = city_spec(name)
    try:
        pts, rep = scan_traverses(d, timezone=s.tz, exclude=SKIP_FILES, campaign_date=dates.get(d.name))
    except Exception as exc:
        out[name] = {"error": str(exc)[:200]}; print(name, "ERROR", exc); continue
    t = pd.to_datetime(pts.get("time"), errors="coerce") if len(pts) else pd.Series([], dtype="datetime64[ns]")
    r = {"n": int(len(pts)), "timed": float(t.notna().mean()) if len(pts) else 0.0,
         "windows": sorted(map(str, pts["window"].dropna().unique())) if len(pts) else [],
         "dates": sorted({str(x.date()) for x in t.dropna()})[:6],
         "vehicles": int(pts["vehicle_key"].nunique()) if len(pts) else 0,
         "clocks": sorted({str(x.get("clock")) for x in rep if x.get("rows")})[:3],
         "files": [x["file"].rsplit("/", 1)[-1] for x in rep if x.get("rows")][:8]}
    out[name] = r
    print(f"{name:24s} n={r['n']:7d} timed={r['timed']:.2f} win={r['windows']} dates={r['dates'][:3]} veh={r['vehicles']}", flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)

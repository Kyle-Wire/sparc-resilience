"""List each CAPA catalog city's OSF files (no downloads): what the lstair download would keep, and sizes."""
import json, sys, concurrent.futures as cf
from sparc.data.collect import capa_catalog as cat
from sparc.data.collect.capa import _scan_storage, TRAVERSE_FILE_EXTS, RASTER_ARCHIVE_PATTERNS
from sparc.core.lstair.dataset import SKIP_FILES

out = sys.argv[1]
def one(name):
    _, e = cat.lookup_city(name)
    node, hint = cat.traverse_node(e)
    node = node or e.get("osf_node"); hint = hint if node != e.get("osf_node") else e.get("folder_hint")
    info = {}
    try:
        files = _scan_storage(node, top_level_filter=hint, info=info, root_files=not hint)
    except Exception as exc:
        return name, {"node": node, "hint": hint, "error": f"{type(exc).__name__}: {exc}"[:200]}
    kept, skipped = [], []
    for f in files:
        p = str(f.get("path") or f["name"]).lstrip("/")
        low = p.lower()
        if not low.endswith(TRAVERSE_FILE_EXTS):
            skipped.append(p); continue
        if any(q.lower() in low for q in RASTER_ARCHIVE_PATTERNS + SKIP_FILES):
            skipped.append(p); continue
        kept.append({"path": p, "size": f.get("size")})
    return name, {"node": node, "hint": hint, "matched": info.get("matched"), "kept": kept,
                  "n_skipped": len(skipped), "skipped_sample": skipped[:15],
                  "kept_mb": round(sum((k["size"] or 0) for k in kept) / 1e6, 1)}

res = {}
with cf.ThreadPoolExecutor(6) as ex:
    for name, r in ex.map(one, cat.list_cities()):
        res[name] = r
        print(f"{name:26s} {r.get('node')} kept={len(r.get('kept', []))} {r.get('kept_mb','')}MB {r.get('error','')}", flush=True)
json.dump(res, open(out, "w"), indent=1)

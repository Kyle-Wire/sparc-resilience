"""Query IEM and Open-Meteo for non-airport stations near Philadelphia."""
import json
import urllib.request

LAT, LON = 39.952, -75.165  # Center City Philadelphia

NETWORKS = [
    "PA_RWIS",    # Road Weather
    "PA_DCP",     # Data Collection Platform
    "PACLIMATE",  # Climate network
]

for net in NETWORKS:
    url = f"https://mesonet.agron.iastate.edu/geojson/network.php?network={net}"
    print(f"\n=== {net} ===")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        features = data.get("features", [])
        print(f"  {len(features)} total stations")
        for f in features:
            try:
                slat = f["geometry"]["coordinates"][1]
                slon = f["geometry"]["coordinates"][0]
                if abs(slat - LAT) < 0.7 and abs(slon - LON) < 0.7:
                    props = f["properties"]
                    name = props.get("sname", props.get("name", "?"))
                    sid = props.get("id", "?")
                    print(f"  NEAR: [{sid}] {name}  ({slat:.4f}, {slon:.4f})")
            except Exception:
                pass
    except Exception as e:
        print(f"  Error: {e}")

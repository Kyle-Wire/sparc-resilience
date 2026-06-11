"""
Query IEM for historical temperature data from non-airport urban stations
near Philadelphia for the CAPA campaign date 2022-09-21.

Networks checked:
- PA_RWIS: Road Weather Information Systems (urban highway sensors)
- PACLIMATE: Historical Cooperative Observer stations
"""
import json
import urllib.request
from datetime import datetime

CAMPAIGN_DATE = "2022-09-21"
LAT, LON = 39.952, -75.165  # Center City Philly
RADIUS_DEG = 0.35  # ~30km

# Urban PACLIMATE stations of interest (from find_urban_stations.py output)
URBAN_CANDIDATE_IDS = [
    # PACLIMATE network - historical co-op stations
    "PACLIMATE_DREXEL_UNIV",
    "PACLIMATE_PHILADELPHIA_FRANKLIN_INSTITUTE",
]

# PA_RWIS has station IDs we need to discover
# Let me query the IEM station list API to get proper IDs
def get_network_stations(network: str) -> list[dict]:
    url = f"https://mesonet.agron.iastate.edu/geojson/network.php?network={network}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    stations = []
    for f in data.get("features", []):
        try:
            slat = f["geometry"]["coordinates"][1]
            slon = f["geometry"]["coordinates"][0]
            if abs(slat - LAT) < RADIUS_DEG and abs(slon - LON) < RADIUS_DEG:
                props = f["properties"]
                stations.append({
                    "network": network,
                    "id": props.get("id", props.get("sid", "?")),
                    "name": props.get("sname", props.get("name", "?")),
                    "lat": slat,
                    "lon": slon,
                })
        except Exception:
            pass
    return stations


def query_iem_data(station_id: str, network: str, date: str) -> dict | None:
    """Query IEM for a station's hourly data on a specific date."""
    # IEM one-minute/hourly data endpoint
    start = f"{date}T00:00:00"
    end = f"{date}T23:59:59"
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station_id}&data=tmpf&year1={date[:4]}&month1={date[5:7]}&day1={date[8:10]}"
        f"&year2={date[:4]}&month2={date[5:7]}&day2={date[8:10]}"
        f"&tz=America%2FNew_York&format=onlycomma&latlon=no&missing=null&trace=null"
        f"&direct=no&report_type=3"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8")
        lines = [l for l in text.strip().split("\n") if l and not l.startswith("#")]
        if len(lines) < 2:
            return None
        temps = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[2] not in ("null", "M", ""):
                try:
                    temps.append(float(parts[2]))
                except ValueError:
                    pass
        if not temps:
            return None
        return {
            "n_obs": len(temps),
            "morning_f": sum(temps[6:9]) / max(1, len(temps[6:9])),   # ~6-9am
            "midday_f":  sum(temps[11:14]) / max(1, len(temps[11:14])), # ~11am-2pm
            "evening_f": sum(temps[19:22]) / max(1, len(temps[19:22])), # ~7-10pm
        }
    except Exception as e:
        return None


print(f"Campaign date: {CAMPAIGN_DATE}\n")

for network in ["PA_RWIS", "PACLIMATE"]:
    stations = get_network_stations(network)
    print(f"\n=== {network} ({len(stations)} near Philly) ===")
    for st in stations:
        sid = st["id"]
        if sid == "?":
            continue
        result = query_iem_data(sid, network, CAMPAIGN_DATE)
        if result and result["n_obs"] >= 10:
            print(
                f"  [{sid}] {st['name'][:45]:<45}  "
                f"({st['lat']:.4f},{st['lon']:.4f})  "
                f"n={result['n_obs']}  "
                f"morn={result['morning_f']:.1f}F  "
                f"mid={result['midday_f']:.1f}F  "
                f"eve={result['evening_f']:.1f}F"
            )
        elif result:
            print(f"  [{sid}] {st['name'][:40]:<40}  sparse: n={result['n_obs']}")

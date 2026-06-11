"""
Fetch ASOS hourly temperatures for each training city on its campaign date.
Used to compute airport UHI background for relative-UHI ANP retraining.
"""
import json
import math
import urllib.request

CITIES = [
    # slug,                lat,      lon,     campaign_date,  IEM network
    ("raleigh_nc",     35.800, -78.660, "2021-06-17", "NC_ASOS"),
    ("chicago_il",     41.830, -87.731, "2023-07-28", "IL_ASOS"),
    ("atlanta_ga",     33.770, -84.420, "2021-09-26", "GA_ASOS"),
    ("seattle_wa",     47.610,-122.350, "2020-09-09", "WA_ASOS"),
    ("albuquerque_nm", 35.130,-106.650, "2021-06-06", "NM_ASOS"),
    ("charlotte_nc",   35.225, -80.855, "2024-08-06", "NC_ASOS"),
    ("milwaukee_wi",   43.090, -87.965, "2022-08-06", "WI_ASOS"),
]

# UTC hours corresponding to each window (local time varies but these cover morning/midday/evening)
# UTC hours for window approximations (conservative, covers most US timezones)
WINDOW_HOURS_UTC = {
    "morning": [10, 11, 12],     # ~5-8am local
    "midday":  [16, 17, 18],     # ~11am-2pm local
    "evening": [23, 0, 1],       # ~6-9pm local
}


def fetch_asos_hourly(sid: str, date: str) -> dict[int, float]:
    """Return {utc_hour: tmpf} for a station on a date."""
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={sid}&data=tmpf"
        f"&year1={date[:4]}&month1={date[5:7]}&day1={date[8:10]}"
        f"&year2={date[:4]}&month2={date[5:7]}&day2={date[8:10]}"
        f"&tz=UTC&format=onlycomma&latlon=no&missing=null&trace=null&direct=no&report_type=3"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode()
    lines = [l for l in text.strip().split("\n") if l and not l.startswith("#")]
    temps: dict[int, float] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("null", "M", ""):
            continue
        try:
            hour = int(parts[1][11:13])
            temps[hour] = float(parts[2])
        except Exception:
            pass
    return temps


def window_avg(temps: dict, hours: list[int]) -> float | None:
    vals = [temps[h] for h in hours if h in temps]
    return sum(vals) / len(vals) if vals else None


def get_nearest_stations(net: str, lat: float, lon: float, n: int = 4) -> list[tuple]:
    url = f"https://mesonet.agron.iastate.edu/geojson/network.php?network={net}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    dists = []
    for f in data.get("features", []):
        slat = f["geometry"]["coordinates"][1]
        slon = f["geometry"]["coordinates"][0]
        d = math.sqrt((slat - lat) ** 2 + (slon - lon) ** 2)
        sid = f["properties"].get("sid", "")
        name = f["properties"].get("sname", "")
        if sid:
            dists.append((d, sid, name, slat, slon))
    dists.sort()
    return dists[:n]


# ERA5 background for each city window (need this to compute UHI_airport)
# We'll fetch from Open-Meteo archive API
def fetch_era5_temps(lat: float, lon: float, date: str) -> dict[str, float]:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit&models=era5"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    temps = data["hourly"]["temperature_2m"]
    # UTC hours 10/11/12 for morning, 15/16/17 for midday, 23/0 for evening
    return {
        "morning": sum(temps[h] for h in [10, 11, 12]) / 3,
        "midday":  sum(temps[h] for h in [15, 16, 17]) / 3,
        "evening": sum(temps[h] for h in [22, 23]) / 2,
    }


results = {}
print("Fetching ASOS + ERA5 for training cities...\n")

for slug, lat, lon, date, net in CITIES:
    print(f"\n=== {slug}  ({date}) ===")
    stations = get_nearest_stations(net, lat, lon, n=3)

    era5 = fetch_era5_temps(lat, lon, date)
    print(f"  ERA5 background:  morn={era5['morning']:.1f}F  mid={era5['midday']:.1f}F  eve={era5['evening']:.1f}F")

    city_uhis: dict[str, list] = {"morning": [], "midday": [], "evening": []}
    for _, sid, name, slat, slon in stations:
        try:
            temps = fetch_asos_hourly(sid, date)
            if not temps:
                print(f"  [{sid}] {name[:40]:<40}  no data")
                continue
            m  = window_avg(temps, WINDOW_HOURS_UTC["morning"])
            md = window_avg(temps, WINDOW_HOURS_UTC["midday"])
            ev = window_avg(temps, WINDOW_HOURS_UTC["evening"])
            uhi_m  = (m  - era5["morning"]) if m  else None
            uhi_md = (md - era5["midday"])  if md else None
            uhi_ev = (ev - era5["evening"]) if ev else None
            print(
                f"  [{sid}] {name[:38]:<38}"
                f"  morn={m or 0:.1f}F(UHI={uhi_m or 0:+.1f})"
                f"  mid={md or 0:.1f}F(UHI={uhi_md or 0:+.1f})"
                f"  eve={ev or 0:.1f}F(UHI={uhi_ev or 0:+.1f})"
            )
            if uhi_m  is not None: city_uhis["morning"].append(uhi_m)
            if uhi_md is not None: city_uhis["midday"].append(uhi_md)
            if uhi_ev is not None: city_uhis["evening"].append(uhi_ev)
        except Exception as e:
            print(f"  [{sid}] error: {e}")

    bg = {}
    for win, vals in city_uhis.items():
        if vals:
            bg[win] = sum(vals) / len(vals)
            print(f"  >> {win} airport background UHI = {bg[win]:+.2f}F")
    results[slug] = {"era5": era5, "airport_bg": bg}

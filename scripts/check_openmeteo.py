"""Check Open-Meteo ERA5 coverage and urban station availability near Philadelphia."""
import json
import urllib.request

points = [
    ("CenterCity", 39.952, -75.165),
    ("NorthPhilly", 40.020, -75.155),
    ("WestPhilly",  39.955, -75.220),
    ("SouthPhilly", 39.930, -75.175),
    ("Airport",     39.873, -75.227),
]

date = "2022-09-21"
print(f"Open-Meteo ERA5 @ {date}\n")
print(f"{'Location':<14}  {'elev':>6}  {'morn':>6}  {'mid':>6}  {'eve':>6}")
print("-" * 50)

for name, lat, lon in points:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m"
        f"&temperature_unit=fahrenheit"
        f"&models=era5"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        temps = data["hourly"]["temperature_2m"]
        elev = data.get("elevation", "?")
        print(f"{name:<14}  {elev:>6}  {temps[7]:>6.1f}  {temps[12]:>6.1f}  {temps[20]:>6.1f}")
    except Exception as e:
        print(f"{name}: Error - {e}")

"""
Compare ERA5 (9km) vs HRRR (3km urban-aware) temperature at urban vs suburban Philadelphia.
HRRR uses WRF urban canopy model which should capture UHI.
"""
import json
import urllib.request

date = "2022-09-21"

# Urban vs suburban comparison points
points = [
    ("Center_City",  39.952, -75.165, "urban core"),
    ("N_Philly",     40.020, -75.155, "dense urban"),
    ("S_Philly",     39.920, -75.175, "dense urban"),
    ("Kensington",   39.990, -75.120, "dense urban"),
    ("Penn_Landing", 39.950, -75.140, "urban waterfront"),
    ("Ardmore",      40.005, -75.285, "suburban"),
    ("Doylestown",   40.310, -75.130, "rural/small town"),
    ("Airport",      39.873, -75.227, "airport"),
    ("SJ_Airport",   39.941, -74.840, "airport/rural"),
]

print(f"{'Location':<16}  {'Context':<16}  {'ERA5_mid':>9}  {'HRRR_mid':>9}  {'diff':>6}")
print("-" * 65)

for name, lat, lon, context in points:
    era5_mid = None
    hrrr_mid = None

    # ERA5
    url_era5 = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit&models=era5"
    )
    try:
        req = urllib.request.Request(url_era5, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        temps = d["hourly"]["temperature_2m"]
        era5_mid = (temps[11] + temps[12] + temps[13]) / 3
    except Exception:
        pass

    # HRRR  (requires hrrr_conus or best_match)
    url_hrrr = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit&models=best_match"
    )
    try:
        req = urllib.request.Request(url_hrrr, headers={"User-Agent": "SPARC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        temps = d["hourly"]["temperature_2m"]
        hrrr_mid = (temps[11] + temps[12] + temps[13]) / 3
    except Exception:
        pass

    era5_str = f"{era5_mid:>9.1f}" if era5_mid is not None else "      NaN"
    hrrr_str = f"{hrrr_mid:>9.1f}" if hrrr_mid is not None else "      NaN"
    diff_str = f"{hrrr_mid - era5_mid:>6.1f}" if era5_mid and hrrr_mid else "    NaN"
    print(f"{name:<16}  {context:<16}  {era5_str}  {hrrr_str}  {diff_str}")

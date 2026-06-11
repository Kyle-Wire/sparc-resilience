"""
Query Netatmo public station data near Philadelphia.
Netatmo has ~50k+ personal weather stations in cities, often in neighborhoods.
Public API: no key needed for getpublicdata endpoint.
"""
import json
import urllib.request

# Philadelphia bounding box (roughly)
LAT_NE, LON_NE = 40.15, -74.9
LAT_SW, LON_SW = 39.85, -75.3
DATE = "2022-09-21"  # campaign date

print("Querying Netatmo public stations near Philadelphia...")
url = (
    f"https://api.netatmo.com/api/getpublicdata?"
    f"lat_ne={LAT_NE}&lon_ne={LON_NE}"
    f"&lat_sw={LAT_SW}&lon_sw={LON_SW}"
    f"&required_data=temperature"
    f"&filter=false"
)
try:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        status = r.status
        text = r.read().decode("utf-8")
    data = json.loads(text)
    print(f"Status: {status}")
    print(f"Response keys: {list(data.keys())}")
    body = data.get("body", [])
    print(f"Stations found: {len(body)}")
    if body:
        st = body[0]
        print(f"First station sample: {st}")
except Exception as e:
    print(f"Error: {e}")

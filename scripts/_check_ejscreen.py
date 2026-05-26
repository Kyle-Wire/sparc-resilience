"""Probe EJScreen REST endpoints to find the working URL and field names."""
import json
import urllib.request
import urllib.parse

BBOX = (-71.46, 41.80, -71.38, 41.86)
minx, miny, maxx, maxy = BBOX

CANDIDATES = [
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen/MapServer/0/query",
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen_2023/MapServer/0/query",
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen_2024/MapServer/0/query",
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen/FeatureServer/0/query",
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen_2023/FeatureServer/0/query",
    "https://geodata.epa.gov/arcgis/rest/services/OEI/EJScreen_2024/FeatureServer/0/query",
]

for url in CANDIDATES:
    try:
        params = urllib.parse.urlencode({
            "f": "json",
            "geometry": json.dumps({
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326},
            }),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "false",
            "resultRecordCount": "1",
        })
        req = urllib.request.Request(
            url + "?" + params,
            headers={"User-Agent": "SPARC/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)

        if "features" in data and data["features"]:
            fields = list(data["features"][0].get("attributes", {}).keys())
            print(f"OK  {url}")
            print(f"    Fields: {fields[:15]}")
            vuln_fields = [f for f in fields if "VULN" in f.upper() or "PCT" in f.upper() or "P_" in f[:3]]
            if vuln_fields:
                print(f"    Vulnerability candidates: {vuln_fields[:8]}")
        elif "error" in data:
            print(f"ERR {data['error']['code']}  {url}")
        else:
            print(f"EMPTY  {url}  keys={list(data.keys())}")
    except Exception as exc:
        print(f"FAIL  {url}  -> {exc}")

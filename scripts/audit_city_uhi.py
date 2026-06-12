"""Quick audit: print ERA5 and CAPA UHI for all collected cities."""
import geopandas as gpd
import glob
import os

for path in sorted(glob.glob("output/cities/*/data.geoparquet")):
    city = path.split(os.sep)[-2]
    try:
        gdf = gpd.read_parquet(path)
        date = str(gdf["campaign_date"].dropna().unique()[0]) if "campaign_date" in gdf.columns else "?"
        lines = []
        for win, col in [("morning", "aat_morning"), ("midday", "aat_midday"), ("evening", "aat_evening")]:
            era5_col = f"era5_{win}_t2m"
            if col in gdf.columns and era5_col in gdf.columns:
                aat = gdf[col].dropna().mean()
                era5 = gdf[era5_col].dropna().mean()
                if aat > 0 and era5 != 0:
                    era5_f = era5 * 9 / 5 + 32
                    uhi = aat - era5_f
                    flag = " ⚠ SUSPICIOUS" if uhi > 15 or era5_f < 55 else ""
                    lines.append(f"  {win:8s} AAT={aat:.0f}F  ERA5={era5_f:.0f}F  UHI={uhi:+.1f}F{flag}")
        if lines:
            print(f"{city:30s} date={date}")
            for l in lines:
                print(l)
    except Exception as e:
        print(f"{city}: ERROR {e}")

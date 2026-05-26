#!/usr/bin/env python3
"""
Convert multiple LAZ/LAS files into a single GeoTIFF using laspy + rasterio.

Usage:
    python laz_to_geotiff.py file1.laz file2.laz file3.laz file4.laz output.tif
    python laz_to_geotiff.py *.laz output.tif --resolution 0.5 --output-type max

    # Clean DTM (ground only, gap-filled, smoothed):
    python laz_to_geotiff.py *.laz dtm.tif --ground-only --fill-gaps --smooth-sigma 1.0

Requirements:
    pip install "laspy[lazrs]" rasterio numpy
    (scipy optional but recommended for gap filling)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


NODATA = -9999.0
OUTPUT_TYPES = ["mean", "min", "max", "count", "stdev"]


# ASPRS LAS classification codes
CLASS_GROUND = 2
NOISE_CLASSES = [7, 18]


def fill_gaps(grid: np.ndarray, nodata: float) -> np.ndarray:
    """Fill nodata holes via nearest-neighbour propagation."""
    mask = grid == nodata
    if not mask.any():
        return grid
    try:
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(mask, return_indices=True)
        filled = grid[idx[0], idx[1]]
        return filled
    except ImportError:
        # Fallback: iterative 3x3 mean fill (slower but no scipy needed)
        result = grid.copy()
        for _ in range(20):
            still_empty = result == nodata
            if not still_empty.any():
                break
            from numpy.lib.stride_tricks import sliding_window_view
            padded = np.pad(result, 1, constant_values=np.nan)
            padded[padded == nodata] = np.nan
            windows = sliding_window_view(padded, (3, 3))
            local_mean = np.nanmean(windows, axis=(-2, -1))
            result[still_empty] = np.where(
                np.isfinite(local_mean[still_empty]),
                local_mean[still_empty],
                nodata,
            )
        return result


def smooth(grid: np.ndarray, nodata: float, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing, ignoring nodata cells."""
    try:
        from scipy.ndimage import gaussian_filter
        mask = grid == nodata
        tmp = grid.copy()
        tmp[mask] = 0.0
        smoothed = gaussian_filter(tmp, sigma=sigma)
        # normalise by the blurred weight mask so edges stay correct
        weight = gaussian_filter((~mask).astype(float), sigma=sigma)
        weight = np.where(weight > 1e-6, weight, 1.0)
        result = smoothed / weight
        result[mask] = nodata
        return result
    except ImportError:
        print("  WARNING: scipy not available, skipping smoothing.")
        return grid


def sigma_clip(z: np.ndarray, n_sigma: float) -> np.ndarray:
    """Return boolean mask of inlier points (within n_sigma of mean)."""
    mean, std = z.mean(), z.std()
    return np.abs(z - mean) <= n_sigma * std


def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmin: float,
    ymax: float,
    ncols: int,
    nrows: int,
    resolution: float,
    output_type: str,
) -> np.ndarray:
    """Bin point cloud Z values into a 2-D raster grid."""
    col = ((x - xmin) / resolution).astype(np.int64)
    row = ((ymax - y) / resolution).astype(np.int64)

    valid = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    col, row, z = col[valid], row[valid], z[valid]
    flat = row * ncols + col
    size = nrows * ncols

    if output_type == "count":
        grid = np.zeros(size, dtype=np.float32)
        np.add.at(grid, flat, 1)

    elif output_type == "max":
        grid = np.full(size, -np.inf, dtype=np.float64)
        np.maximum.at(grid, flat, z)
        grid[grid == -np.inf] = NODATA

    elif output_type == "min":
        grid = np.full(size, np.inf, dtype=np.float64)
        np.minimum.at(grid, flat, z)
        grid[grid == np.inf] = NODATA

    elif output_type in ("mean", "stdev"):
        count = np.zeros(size, dtype=np.float64)
        total = np.zeros(size, dtype=np.float64)
        np.add.at(count, flat, 1)
        np.add.at(total, flat, z)
        has = count > 0
        grid = np.full(size, NODATA, dtype=np.float64)
        grid[has] = total[has] / count[has]

        if output_type == "stdev":
            sq = np.zeros(size, dtype=np.float64)
            np.add.at(sq, flat, z ** 2)
            var = np.full(size, NODATA, dtype=np.float64)
            var[has] = sq[has] / count[has] - (total[has] / count[has]) ** 2
            grid = np.full(size, NODATA, dtype=np.float64)
            grid[has] = np.sqrt(np.maximum(var[has], 0))

    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    return grid.reshape(nrows, ncols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LAZ/LAS files and write a GeoTIFF raster (no PDAL required)."
    )
    parser.add_argument(
        "laz_files",
        nargs="+",
        metavar="INPUT.laz",
        help="Input LAZ or LAS files.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT.tif",
        help="Output GeoTIFF path.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Cell size in CRS units (default: 1.0).",
    )
    parser.add_argument(
        "--output-type",
        default="mean",
        choices=OUTPUT_TYPES,
        help="Aggregation method per cell (default: mean).",
    )
    parser.add_argument(
        "--dimension",
        default="Z",
        choices=["Z", "intensity", "return_number"],
        help="Point attribute to rasterise (default: Z).",
    )
    parser.add_argument(
        "--ground-only",
        action="store_true",
        default=False,
        help="Keep only ASPRS ground points (classification 2). "
             "Removes buildings, trees, and noise. Recommended for DTM/terrain.",
    )
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        default=True,
        help="Drop ASPRS noise points (classification 7 & 18). Always on when --ground-only is set.",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=None,
        metavar="N",
        help="Remove points whose Z is more than N standard deviations from the mean "
             "(e.g. --sigma-clip 3). Useful for catching stray outlier spikes.",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        default=False,
        help="Interpolate empty cells using nearest-neighbour propagation "
             "(recommended with --ground-only since ground returns are sparse).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Apply Gaussian smoothing with this sigma after rasterising "
             "(e.g. --smooth-sigma 1.0). Reduces remaining jagged edges.",
    )
    args = parser.parse_args()

    # --- validate inputs ---
    laz_files = [Path(f) for f in args.laz_files]
    for f in laz_files:
        if not f.exists():
            sys.exit(f"ERROR: File not found: {f}")

    output_tif = Path(args.output)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    try:
        import laspy
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError as e:
        sys.exit(
            f"ERROR: Missing dependency — {e}\n"
            'Install with:  pip install "laspy[lazrs]" rasterio'
        )

    print(f"Reading {len(laz_files)} file(s)...")
    if args.ground_only:
        print("  Mode: ground-only (ASPRS class 2)")

    all_x, all_y, all_z = [], [], []
    crs = None

    for path in laz_files:
        print(f"  {path.name}", end="", flush=True)
        with laspy.open(str(path)) as f:
            las = f.read()

        # Try to extract CRS from the file header (first file wins)
        if crs is None:
            try:
                crs = las.header.parse_crs()
            except Exception:
                pass

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)

        if args.dimension == "Z":
            vals = np.asarray(las.z, dtype=np.float64)
        elif args.dimension == "intensity":
            vals = np.asarray(las.intensity, dtype=np.float64)
        elif args.dimension == "return_number":
            vals = np.asarray(las.return_number, dtype=np.float64)

        if hasattr(las, "classification"):
            cls = np.asarray(las.classification, dtype=np.uint8)
            if args.ground_only:
                keep = cls == CLASS_GROUND
            elif args.exclude_noise:
                keep = ~np.isin(cls, NOISE_CLASSES)
            else:
                keep = np.ones(len(x), dtype=bool)
            x, y, vals = x[keep], y[keep], vals[keep]

        all_x.append(x)
        all_y.append(y)
        all_z.append(vals)
        print(f" — {len(x):,} pts")

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    print(f"Total points: {len(x):,}")

    # --- optional sigma clipping ---
    if args.sigma_clip is not None:
        inliers = sigma_clip(z, args.sigma_clip)
        n_removed = (~inliers).sum()
        x, y, z = x[inliers], y[inliers], z[inliers]
        print(f"Sigma clip ({args.sigma_clip}σ): removed {n_removed:,} outlier pts")

    # --- compute grid dimensions ---
    res = args.resolution
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1

    print(f"Grid: {nrows} rows × {ncols} cols  |  resolution: {res}")
    print(f"Aggregation: {args.output_type}  |  dimension: {args.dimension}")

    # --- rasterise ---
    grid = rasterize(x, y, z, xmin, ymax, ncols, nrows, res, args.output_type)

    # --- gap fill ---
    if args.fill_gaps:
        empty_before = (grid == NODATA).sum()
        print(f"Filling {empty_before:,} empty cells...")
        grid = fill_gaps(grid, NODATA)

    # --- smooth ---
    if args.smooth_sigma is not None:
        print(f"Smoothing (σ={args.smooth_sigma})...")
        grid = smooth(grid, NODATA, args.smooth_sigma)

    # --- write GeoTIFF ---
    transform = from_origin(xmin, ymax, res, res)

    rasterio_crs = None
    if crs is not None:
        try:
            rasterio_crs = CRS.from_wkt(crs.to_wkt())
        except Exception:
            try:
                rasterio_crs = CRS.from_epsg(crs.to_epsg())
            except Exception:
                print("  WARNING: Could not parse CRS from point cloud headers.")

    with rasterio.open(
        str(output_tif),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=np.float32,
        crs=rasterio_crs,
        transform=transform,
        nodata=NODATA,
        compress="lzw",
        predictor=2,
        bigtiff="IF_SAFER",
    ) as dst:
        dst.write(grid, 1)

    print(f"Done. Output: {output_tif.resolve()}")


if __name__ == "__main__":
    main()

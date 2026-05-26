#!/usr/bin/env python3
"""
Convert a GeoTIFF elevation raster to an Unreal Engine 5 heightmap.

UE5 requirements:
  - 16-bit grayscale PNG  (.png)  OR  16-bit little-endian RAW  (.r16 / .raw)
  - Dimensions must match a valid UE5 landscape size:
      127, 253, 505, 1009, 2017, 4033, 8129  (any W x H combination of these)
  - Values 0–65535 represent the full elevation range

Usage:
    python geotiff_to_ue5_heightmap.py merged_dtm.tif heightmap.png
    python geotiff_to_ue5_heightmap.py merged_dtm.tif heightmap.r16 --format raw
    python geotiff_to_ue5_heightmap.py merged_dtm.tif heightmap.png --size 4033x2017
    python geotiff_to_ue5_heightmap.py merged_dtm.tif heightmap.png --size 2017x2017 --padding edge

Requirements:
    pip install rasterio numpy
    (both already present in your Python 3.12 environment)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# Valid UE5 landscape dimension values for each axis
UE5_VALID_SIZES = [127, 253, 505, 1009, 2017, 4033, 8129]


def best_ue5_size(src_width: int, src_height: int) -> tuple[int, int]:
    """Pick the largest valid W×H that fits within the source dimensions."""
    best = None
    for w in UE5_VALID_SIZES:
        for h in UE5_VALID_SIZES:
            if w <= src_width and h <= src_height:
                if best is None or (w * h) > (best[0] * best[1]):
                    best = (w, h)
    if best is None:
        # source is smaller than smallest valid size — use smallest
        best = (UE5_VALID_SIZES[0], UE5_VALID_SIZES[0])
    return best


def parse_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must be WxH, e.g. 4033x2017")
    w, h = int(parts[0]), int(parts[1])
    if w not in UE5_VALID_SIZES or h not in UE5_VALID_SIZES:
        valid = ", ".join(str(v) for v in UE5_VALID_SIZES)
        raise argparse.ArgumentTypeError(
            f"Both dimensions must be one of: {valid}"
        )
    return w, h


def resample(data: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Bilinear resample a 2-D array to (target_h, target_w)."""
    try:
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        import rasterio
        from rasterio.io import MemoryFile

        src_h, src_w = data.shape
        src_transform = from_bounds(0, 0, src_w, src_h, src_w, src_h)
        dst_transform = from_bounds(0, 0, src_w, src_h, target_w, target_h)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff", height=src_h, width=src_w,
                count=1, dtype=data.dtype, transform=src_transform,
            ) as src_ds:
                src_ds.write(data, 1)

            with memfile.open() as src_ds:
                out = src_ds.read(
                    1,
                    out_shape=(target_h, target_w),
                    resampling=Resampling.bilinear,
                )
        return out
    except Exception:
        # Fallback: simple nearest-neighbour via numpy
        row_idx = (np.linspace(0, data.shape[0] - 1, target_h)).astype(int)
        col_idx = (np.linspace(0, data.shape[1] - 1, target_w)).astype(int)
        return data[np.ix_(row_idx, col_idx)]


def pad_or_crop(data: np.ndarray, target_w: int, target_h: int, mode: str) -> np.ndarray:
    """Crop or pad a 2-D array to exactly (target_h, target_w)."""
    src_h, src_w = data.shape
    # Crop if needed
    data = data[:min(src_h, target_h), :min(src_w, target_w)]
    h, w = data.shape
    if h < target_h or w < target_w:
        pad_h = target_h - h
        pad_w = target_w - w
        if mode == "edge":
            data = np.pad(data, ((0, pad_h), (0, pad_w)), mode="edge")
        else:  # zero / nodata
            data = np.pad(data, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF DTM/DSM to an Unreal Engine 5 heightmap."
    )
    parser.add_argument("input", metavar="INPUT.tif", help="Source GeoTIFF (any resolution).")
    parser.add_argument("output", metavar="OUTPUT.png|r16", help="Output heightmap file.")
    parser.add_argument(
        "--format",
        choices=["png", "raw"],
        default=None,
        help="Output format: png (16-bit PNG) or raw (16-bit .r16). "
             "Inferred from file extension if omitted.",
    )
    parser.add_argument(
        "--size",
        metavar="WxH",
        type=parse_size,
        default=None,
        help="Target UE5 landscape size, e.g. 4033x2017. "
             "Auto-selected from valid UE5 sizes if omitted.",
    )
    parser.add_argument(
        "--padding",
        choices=["edge", "zero"],
        default="edge",
        help="How to fill edges if source is smaller than target (default: edge).",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Crop/pad instead of resampling to the target size.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    fmt = args.format
    if fmt is None:
        ext = output_path.suffix.lower()
        fmt = "raw" if ext in (".r16", ".raw") else "png"

    try:
        import rasterio
    except ImportError:
        sys.exit("ERROR: rasterio not installed.  pip install rasterio")

    # --- read source ---
    print(f"Reading {input_path.name}...")
    with rasterio.open(str(input_path)) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        src_w, src_h = src.width, src.height
        profile = src.profile

    if nodata is not None:
        data[data == nodata] = np.nan

    src_h_actual, src_w_actual = data.shape
    print(f"  Source size: {src_w_actual} × {src_h_actual}")

    # --- choose target size ---
    if args.size:
        target_w, target_h = args.size
    else:
        target_w, target_h = best_ue5_size(src_w_actual, src_h_actual)
        print(f"  Auto-selected UE5 size: {target_w} × {target_h}")

    # --- fill NaN with interpolated or edge values ---
    nan_mask = np.isnan(data)
    if nan_mask.any():
        # Simple fill: replace with column median
        col_medians = np.nanmedian(data, axis=0)
        data = np.where(nan_mask, col_medians, data)

    # --- resample or crop/pad ---
    if not args.no_resample:
        print(f"  Resampling to {target_w} × {target_h} (bilinear)...")
        data = resample(data, target_w, target_h)
    else:
        data = pad_or_crop(data, target_w, target_h, args.padding)

    # --- normalise to uint16 ---
    z_min = float(np.nanmin(data))
    z_max = float(np.nanmax(data))
    z_range = z_max - z_min if z_max != z_min else 1.0

    norm = (data - z_min) / z_range          # 0.0 – 1.0
    uint16 = (norm * 65535.0).round().astype(np.uint16)

    # --- write output ---
    if fmt == "png":
        with rasterio.open(
            str(output_path),
            "w",
            driver="PNG",
            height=target_h,
            width=target_w,
            count=1,
            dtype=np.uint16,
        ) as dst:
            dst.write(uint16, 1)
        print(f"  Written: {output_path.resolve()}")

    else:  # raw 16-bit little-endian
        raw_bytes = uint16.astype("<u2").tobytes()
        output_path.write_bytes(raw_bytes)
        print(f"  Written: {output_path.resolve()}")

    # --- print UE5 import settings ---
    z_range_m = z_max - z_min
    z_scale = (z_range_m / 512.0) * 100.0          # UE5 Z scale (cm units)
    z_offset_m = z_min + z_range_m / 2.0            # center of elevation range
    z_offset_cm = z_offset_m * 100.0

    print()
    print("=" * 56)
    print("  Unreal Engine 5 import settings")
    print("=" * 56)
    print(f"  Heightmap size : {target_w} × {target_h}")
    print(f"  Elevation min  : {z_min:.2f} m")
    print(f"  Elevation max  : {z_max:.2f} m")
    print(f"  Elevation range: {z_range_m:.2f} m")
    print()
    print("  In the Landscape Import dialog:")
    print(f"    Z Scale        = {z_scale:.4f}")
    print()
    print("  After import, set the Landscape Actor's Z position to:")
    print(f"    Z = {z_offset_cm:.2f} cm  ({z_offset_m:.2f} m)")
    print("=" * 56)


if __name__ == "__main__":
    main()

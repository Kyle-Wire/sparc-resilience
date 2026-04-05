#!/usr/bin/env python3
"""
Build the SPARC sidecar binary for the Tauri desktop app.

Usage:
    python build-sidecar.py              # Build for current platform
    python build-sidecar.py --target-dir sparc-desktop/src-tauri/binaries

The output binary is named per Tauri's sidecar convention:
    sparc-sidecar-{target-triple}
e.g. sparc-sidecar-aarch64-apple-darwin, sparc-sidecar-x86_64-pc-windows-msvc
"""

import platform
import subprocess
import shutil
import sys
from pathlib import Path

# PyInstaller entry point: starts the FastAPI server
ENTRY_SCRIPT = Path(__file__).parent / "sparc" / "__main__.py"

# Hidden imports that PyInstaller often misses
HIDDEN_IMPORTS = [
    "sparc.server",
    "sparc.server.app",
    "sparc.server.state",
    "sparc.server.stream",
    "sparc.config",
    "sparc.config.config",
    "sparc.data",
    "sparc.data.data_utils",
    "sparc.data.processing",
    "sparc.run",
    "sparc.run.pipeline_paths",
    "sparc.run.pipeline_utils",
    "sparc.run.result_store",
    "sparc.causal",
    "sparc.causal.dag_definition",
    "sparc.causal.causal_discovery",
    "sparc.causal.counterfactual_engine",
    "sparc.causal.spatial_cate",
    "sparc.interventions",
    "sparc.interventions.scenario_simulator",
    "sparc.interventions.extrapolation_guard",
    "sparc.interventions.physics_priors",
    "sparc.report",
    "sparc.features",
    "sparc.features.laplacian",
    "sparc.features.fold_aware_laplacian",
    "sparc.models",
    "sparc.models.meta_ensemble",
    "sparc.models.deep_kriging_v2",
    "sparc.models.ggpgam",
    "sparc.models.gwen",
    "sparc.models.gwr",
    "sparc.models.gwrf",
    "sparc.models.ols",
    "sparc.models.fingerprint",
    "sparc.evaluation",
    "sparc.evaluation.evaluation",
    "sparc.evaluation.causal_evaluation",
    "sparc.evaluation.sensitivity",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette",
    "websockets",
    "multipart",
    "torch",
    "geopandas",
    "fiona",
    "rasterio",
    "rasterstats",
    "mgwr",
    "libpysal",
    "esda",
    "dowhy",
    "econml",
    "lightgbm",
    "optuna",
    "pygam",
    "pyproj",
    "pyarrow",
    "yaml",
    "jinja2",
    "jsonschema",
]

# Data files to include (templates, schemas)
DATA_DIRS = [
    ("templates", "templates"),
    ("sparc/config/project_schema.json", "sparc/config/project_schema.json"),
    ("sparc/report/templates", "sparc/report/templates"),
]


def get_target_triple() -> str:
    """Determine the target triple for the current platform."""
    machine = platform.machine().lower()
    system = platform.system().lower()

    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    arch = arch_map.get(machine, machine)

    if system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "windows":
        return f"{arch}-pc-windows-msvc"
    elif system == "linux":
        return f"{arch}-unknown-linux-gnu"
    else:
        return f"{arch}-{system}"


def build(target_dir: str | None = None):
    target_triple = get_target_triple()
    # Tauri looks for "sparc-sidecar" without target triple in resources/binaries/
    binary_name = "sparc-sidecar"
    print(f"Building sidecar binary: {binary_name}")
    print(f"  Target: {target_triple}")
    print(f"  Entry:  {ENTRY_SCRIPT}")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", binary_name,
        "--clean",
        "--noconfirm",
    ]

    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])

    for src, dst in DATA_DIRS:
        src_path = Path(__file__).parent / src
        if src_path.exists():
            cmd.extend(["--add-data", f"{src_path}{':' if sys.platform != 'win32' else ';'}{dst}"])

    cmd.append(str(ENTRY_SCRIPT))

    print(f"\nRunning PyInstaller...")
    subprocess.run(cmd, check=True)

    # Copy to target directory
    dist_binary = Path("dist") / binary_name
    if sys.platform == "win32":
        dist_binary = dist_binary.with_suffix(".exe")

    if target_dir:
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        output = dest / dist_binary.name
        shutil.copy2(dist_binary, output)
        print(f"\nBinary copied to: {output}")
    else:
        print(f"\nBinary at: {dist_binary}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SPARC sidecar binary")
    parser.add_argument("--target-dir", help="Copy binary to this directory")
    args = parser.parse_args()
    build(args.target_dir)

#!/usr/bin/env python3
"""Phase 01: read-only inspection of raw IM3/HyperFACETS TGW NetCDF files.

Run this before any transformation. The script does not repair or rewrite the
raw files. It follows the inspection-first pattern used by the existing
CONUS404 wind repository.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from im3_wind.io import (
    decode_times,
    find_time_variable,
    find_variable,
    find_wind_name,
    infer_scenario_from_path,
    list_netcdf_files,
    read_2d_lat_lon,
    sha256_file,
    summarize_file,
    require_netcdf4,
)


def inspect(path: Path) -> dict:
    require_netcdf4()
    summary = summarize_file(path)
    if summary.wind_variable is None:
        raise ValueError(f"{path}: WSPD is missing")
    if summary.wind_units and summary.wind_units.replace(" ", "").lower() not in {
        "m/s", "ms-1", "ms^-1", "m.s-1"
    }:
        raise ValueError(f"{path}: unexpected WSPD units {summary.wind_units!r}")

    from netCDF4 import Dataset

    with Dataset(path, "r") as ds:
        time_name = find_time_variable(ds)
        wind_name = find_wind_name(ds)
        lat, lon, lat_name, lon_name = read_2d_lat_lon(ds)
        if time_name:
            times = decode_times(ds, time_name)
        else:
            times = []
        wind = ds.variables[wind_name]
        time_dim = len(times)
        wind_shape = tuple(int(v) for v in wind.shape)
        # Read full min/max only for the supplied file. Phase 01 is intentionally
        # read-only; Phase 02 is the memory-bounded production transformation.
        raw_values = wind[:]
        values = np.ma.filled(raw_values, np.nan).astype(np.float32) if np.ma.isMaskedArray(raw_values) else np.asarray(raw_values, dtype=np.float32)
        finite = np.isfinite(values)
        return {
            "source_file": str(path),
            "sha256": sha256_file(path),
            "scenario": infer_scenario_from_path(path),
            "wind_variable": wind_name,
            "wind_units": str(getattr(wind, "units", "")),
            "time_variable": time_name,
            "time_records": time_dim,
            "time_start": times[0].isoformat() if times else None,
            "time_end": times[-1].isoformat() if times else None,
            "latitude_variable": lat_name,
            "longitude_variable": lon_name,
            "ny": int(lat.shape[0]),
            "nx": int(lat.shape[1]),
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
            "wind_min": float(np.nanmin(values[finite])) if np.any(finite) else None,
            "wind_max": float(np.nanmax(values[finite])) if np.any(finite) else None,
            "wind_missing_count": int(np.count_nonzero(~finite)),
            "wind_shape": json.dumps(wind_shape),
            "dimensions": json.dumps({name: len(dim) for name, dim in ds.dimensions.items()}, sort_keys=True),
            "grid_mapping": str(getattr(wind, "grid_mapping", "")),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("Data/metadata/im3_file_inventory.csv"))
    args = parser.parse_args()

    files = list_netcdf_files(args.input_dir)
    if not files:
        raise SystemExit(f"No NetCDF files found under {args.input_dir}")

    records = [inspect(path) for path in files]
    records.sort(key=lambda row: (row["scenario"], row["time_start"] or "", row["source_file"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"Inspected {len(records)} files")
    print(f"Inventory: {args.output}")


if __name__ == "__main__":
    main()

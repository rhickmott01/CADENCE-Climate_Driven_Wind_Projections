#!/usr/bin/env python3
"""Phase 05: flatten IM3 period return levels to Parquet.

The output is one row per IM3 grid cell, climate period, and return period.
It is an intermediate analysis table used by the historical comparison and
bias/delta stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from im3_wind.parquet_io import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as h5:
        periods = [item.decode() for item in h5["periods"][:]]
        rps = np.asarray(h5["return_periods"][:], dtype=int)
        lat = np.asarray(h5["latitude"][:], dtype=np.float32)
        lon = np.asarray(h5["longitude"][:], dtype=np.float32)
        records = []
        ny, nx = lat.shape
        y_index = np.repeat(np.arange(ny, dtype=np.int32), nx)
        x_index = np.tile(np.arange(nx, dtype=np.int32), ny)
        base = pd.DataFrame({
            "im3_y": y_index,
            "im3_x": x_index,
            "latitude": lat.ravel(),
            "longitude": lon.ravel(),
        })
        for period in periods:
            values = np.asarray(h5[f"return_levels_{period}"][:], dtype=np.float32)
            for rp_index, rp in enumerate(rps):
                frame = base.copy()
                frame["climate_period"] = period
                frame["return_period"] = int(rp)
                frame["return_level"] = values[rp_index].ravel()
                records.append(frame)

    out = pd.concat(records, ignore_index=True)
    out["dataset"] = "IM3/HyperFACETS TGW"
    out["wind_variable"] = "WSPD"
    out["units"] = "m/s"
    write_parquet(out, args.output)
    print(f"Wrote {len(out):,} rows: {args.output}")


if __name__ == "__main__":
    main()

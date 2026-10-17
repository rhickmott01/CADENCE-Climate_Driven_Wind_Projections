#!/usr/bin/env python3
"""Phase 06: build a 1980-2019 CONUS404 historical reference for bias calibration.

The current CADENCE CONUS404 return-period product uses the full 1980-2024
record. IM3's historical simulation is only 1980-2019. We therefore create a
matched 1980-2019 CONUS404 reference for the historical comparison.

This reference is used ONLY to estimate the historical CONUS404-vs-IM3 bias.
The final CADENCE climate delta remains a multiplier and can be applied to the
existing/current CONUS404 return-period baseline used by CADENCE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from im3_wind.gev import fit_gev_grid, return_levels
from im3_wind.config import DEFAULT_RETURN_PERIODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing CONUS404 validated annual-max HDF5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--chunk-cells", type=int, default=2000)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as h5:
        cube = np.asarray(h5["spduv10max"][:], dtype=np.float32)
        years = np.asarray(h5["years"][:], dtype=np.int16)
        lat = np.asarray(h5["lat_2d"][:], dtype=np.float32)
        lon = np.asarray(h5["lon_2d"][:], dtype=np.float32)

    mask = (years >= args.start_year) & (years <= args.end_year)
    hist = cube[mask]
    selected_years = years[mask]
    expected = np.arange(args.start_year, args.end_year + 1, dtype=np.int16)
    if not np.array_equal(selected_years, expected):
        raise ValueError(f"Expected contiguous CONUS404 years {args.start_year}-{args.end_year}")

    fit = fit_gev_grid(hist, processes=args.processes, chunk_cells=args.chunk_cells, min_years=20)
    levels = return_levels(fit, DEFAULT_RETURN_PERIODS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        h5.create_dataset("return_periods", data=np.asarray(DEFAULT_RETURN_PERIODS, dtype=np.int16))
        h5.create_dataset("return_levels", data=levels, compression="gzip", compression_opts=4)
        h5.create_dataset("location", data=fit.location, compression="gzip", compression_opts=4)
        h5.create_dataset("scale", data=fit.scale, compression="gzip", compression_opts=4)
        h5.create_dataset("shape_c", data=fit.shape_c, compression="gzip", compression_opts=4)
        h5.create_dataset("converged", data=fit.converged, compression="gzip", compression_opts=4)
        h5.create_dataset("n_years", data=fit.n_years, compression="gzip", compression_opts=4)
        h5.create_dataset("lat_2d", data=lat, compression="gzip", compression_opts=4)
        h5.create_dataset("lon_2d", data=lon, compression="gzip", compression_opts=4)
        h5.attrs["dataset"] = "CONUS404"
        h5.attrs["purpose"] = "Historical matched-period reference for IM3 bias calibration only"
        h5.attrs["historical_period"] = f"{args.start_year}-{args.end_year}"
        h5.attrs["fit_method"] = "scipy.stats.genextreme MLE"
        h5.attrs["shape_parameter"] = "SciPy genextreme c; conventional xi = -c"
        h5.attrs["return_periods"] = ",".join(str(v) for v in DEFAULT_RETURN_PERIODS)
        h5.attrs["native_units"] = "m/s"
    tmp.replace(args.output)
    print(f"Wrote matched CONUS404 historical reference: {args.output}")


if __name__ == "__main__":
    main()

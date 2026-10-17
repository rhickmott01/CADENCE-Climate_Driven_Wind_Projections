#!/usr/bin/env python3
"""Phase 04: fit GEV parameters to IM3 historical/near/far climate periods.

For each IM3 scenario, the same historical 1980-2019 cube is combined with
that scenario's future 2020-2099 cube. GEVs are fitted independently to the
three 40-year climate periods used by the TGW experiment.

We do not fit a separate GEV for each calendar year. One year supplies only one
annual maximum, which is not enough data to estimate a return-period tail.
Instead, the 40-year climate periods provide stable samples. Phase 09 converts
those period-level return levels into annual 2025-2050 deltas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from im3_wind.config import PERIODS, validate_scenario
from im3_wind.gev import fit_gev_grid, return_levels


def read_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return (
            np.asarray(h5["wspd_annual_max"][:], dtype=np.float32),
            np.asarray(h5["years"][:], dtype=np.int16),
            np.asarray(h5["latitude"][:], dtype=np.float32),
            np.asarray(h5["longitude"][:], dtype=np.float32),
        )


def select_period(cube: np.ndarray, years: np.ndarray, start: int, end: int) -> np.ndarray:
    mask = (years >= start) & (years <= end)
    subset = cube[mask]
    if subset.shape[0] != end - start + 1:
        raise ValueError(f"Expected {end - start + 1} years for {start}-{end}, got {subset.shape[0]}")
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--future", required=True, type=Path)
    parser.add_argument("--scenario", required=True, choices=["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--chunk-cells", type=int, default=2000)
    parser.add_argument("--min-years", type=int, default=20)
    args = parser.parse_args()

    scenario = validate_scenario(args.scenario)
    hist_cube, hist_years, lat, lon = read_cube(args.historical)
    future_cube, future_years, future_lat, future_lon = read_cube(args.future)
    if lat.shape != future_lat.shape or not np.allclose(lat, future_lat, atol=1e-5) or not np.allclose(lon, future_lon, atol=1e-5):
        raise ValueError("Historical and future IM3 coordinate grids do not match")

    cube = np.concatenate([hist_cube, future_cube], axis=0)
    years = np.concatenate([hist_years, future_years], axis=0)
    if np.unique(years).size != years.size:
        raise ValueError("Historical and future cubes contain duplicate calendar years")
    order = np.argsort(years)
    cube = cube[order]
    years = years[order]

    fit_results = {}
    rp_results = {}
    for period in PERIODS:
        subset = select_period(cube, years, period.start_year, period.end_year)
        fit = fit_gev_grid(subset, processes=args.processes, chunk_cells=args.chunk_cells, min_years=args.min_years)
        fit_results[period.name] = fit
        rp_results[period.name] = return_levels(fit, (10, 25, 50, 100, 200, 500))
        valid_count = int(np.count_nonzero(fit.converged))
        print(f"Fitted {period.name}: {period.start_year}-{period.end_year}; converged={valid_count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        period_names = [period.name for period in PERIODS]
        rps = np.asarray([10, 25, 50, 100, 200, 500], dtype=np.int16)
        h5.create_dataset("periods", data=np.asarray([p.encode() for p in period_names]))
        h5.create_dataset("return_periods", data=rps)
        h5.create_dataset("latitude", data=lat, compression="gzip", compression_opts=4)
        h5.create_dataset("longitude", data=lon, compression="gzip", compression_opts=4)
        for name in period_names:
            fit = fit_results[name]
            h5.create_dataset(f"location_{name}", data=fit.location, compression="gzip", compression_opts=4)
            h5.create_dataset(f"scale_{name}", data=fit.scale, compression="gzip", compression_opts=4)
            h5.create_dataset(f"shape_c_{name}", data=fit.shape_c, compression="gzip", compression_opts=4)
            h5.create_dataset(f"converged_{name}", data=fit.converged, compression="gzip", compression_opts=4)
            h5.create_dataset(f"n_years_{name}", data=fit.n_years, compression="gzip", compression_opts=4)
            h5.create_dataset(f"return_levels_{name}", data=rp_results[name], compression="gzip", compression_opts=4)
        h5.attrs["dataset"] = "IM3/HyperFACETS TGW"
        h5.attrs["scenario"] = scenario
        h5.attrs["fit_method"] = "scipy.stats.genextreme MLE"
        h5.attrs["shape_parameter"] = "SciPy genextreme c; conventional xi = -c"
        h5.attrs["historical_period"] = "1980-2019"
        h5.attrs["near_future_period"] = "2020-2059"
        h5.attrs["far_future_period"] = "2060-2099"
        h5.attrs["return_periods"] = "10,25,50,100,200,500"
        h5.attrs["tail_note"] = "Long return periods remain uncertain with 40 annual maxima; retain convergence and stability diagnostics."
    tmp.replace(args.output)
    print(f"Wrote GEV parameters and period return levels: {args.output}")


if __name__ == "__main__":
    main()

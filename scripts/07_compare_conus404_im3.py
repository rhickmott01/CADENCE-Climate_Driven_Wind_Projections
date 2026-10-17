#!/usr/bin/env python3
"""Phase 07: compare matched-period historical CONUS404 and IM3 return levels.

The CONUS404 reference is first reduced to the same 1980-2019 historical period
as IM3. Its 4-km return-level fields are aggregated to the IM3 12-km grid so
that the historical bias is calculated at a common spatial scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from im3_wind.config import DEFAULT_RETURN_PERIODS
from im3_wind.parquet_io import write_parquet


def unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lat.ravel())
    lon_r = np.deg2rad(lon.ravel())
    return np.column_stack((np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)))


def aggregate_to_im3(values: np.ndarray, conus_lat: np.ndarray, conus_lon: np.ndarray, im3_lat: np.ndarray, im3_lon: np.ndarray, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(unit_xyz(im3_lat, im3_lon))
    _, target = tree.query(unit_xyz(conus_lat, conus_lon), k=1)
    target = target.astype(np.int64)
    out = np.full(im3_lat.size, np.nan, dtype=np.float32)
    counts = np.zeros(im3_lat.size, dtype=np.int32)
    source = values.ravel()
    order = np.argsort(target)
    sorted_target = target[order]
    sorted_values = source[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_target)) + 1]
    ends = np.r_[starts[1:], sorted_target.size]
    for start, end in zip(starts, ends):
        group = sorted_values[start:end]
        valid = np.isfinite(group)
        if not np.any(valid):
            continue
        idx = int(sorted_target[start])
        data = group[valid]
        out[idx] = np.float32(np.median(data) if statistic == "median" else np.mean(data))
        counts[idx] = int(data.size)
    return out.reshape(im3_lat.shape), counts.reshape(im3_lat.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conus404-historical-reference", required=True, type=Path)
    parser.add_argument("--im3-historical-return-levels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crosswalk-output", required=True, type=Path)
    parser.add_argument("--statistic", choices=("median", "mean"), default="median")
    args = parser.parse_args()

    with h5py.File(args.conus404_historical_reference, "r") as h5:
        conus_lat = np.asarray(h5["lat_2d"][:], dtype=np.float64)
        conus_lon = np.asarray(h5["lon_2d"][:], dtype=np.float64)
        conus_return_periods = np.asarray(h5["return_periods"][:], dtype=int)
        conus_levels = np.asarray(h5["return_levels"][:], dtype=np.float32)
    if tuple(conus_return_periods.tolist()) != DEFAULT_RETURN_PERIODS:
        raise ValueError("CONUS404 historical reference return periods do not match CADENCE defaults")

    im3 = pd.read_parquet(args.im3_historical_return_levels)
    im3 = im3.loc[im3["climate_period"] == "historical"].copy()
    im3_y = np.sort(im3["im3_y"].unique())
    im3_x = np.sort(im3["im3_x"].unique())
    im3_shape = (len(im3_y), len(im3_x))
    im3_lat = np.full(im3_shape, np.nan, dtype=np.float64)
    im3_lon = np.full(im3_shape, np.nan, dtype=np.float64)
    for row in im3[["im3_y", "im3_x", "latitude", "longitude"]].drop_duplicates().itertuples(index=False):
        im3_lat[int(row.im3_y), int(row.im3_x)] = float(row.latitude)
        im3_lon[int(row.im3_y), int(row.im3_x)] = float(row.longitude)

    # Precompute the grid bridge once. Simulation-time CADENCE should use this
    # ID mapping rather than performing spatial work again.
    tree = cKDTree(unit_xyz(im3_lat, im3_lon))
    _, target = tree.query(unit_xyz(conus_lat, conus_lon), k=1)
    target = target.astype(np.int64)
    crosswalk = pd.DataFrame({
        "lat_idx": np.repeat(np.arange(conus_lat.shape[0], dtype=np.int32), conus_lat.shape[1]),
        "lon_idx": np.tile(np.arange(conus_lat.shape[1], dtype=np.int32), conus_lat.shape[0]),
        "latitude": conus_lat.ravel(),
        "longitude": conus_lon.ravel(),
        "im3_y": (target // im3_shape[1]).ravel(),
        "im3_x": (target % im3_shape[1]).ravel(),
    })
    crosswalk["spatial_assignment"] = "nearest_IM3_center"
    crosswalk["note"] = "Research-phase grid bridge; future area-weighted overlay can replace this method."
    write_parquet(crosswalk, args.crosswalk_output)

    records = []
    for idx, rp in enumerate(DEFAULT_RETURN_PERIODS):
        aggregate, counts = aggregate_to_im3(conus_levels[idx], conus_lat, conus_lon, im3_lat, im3_lon, args.statistic)
        records.append(pd.DataFrame({
            "im3_y": np.repeat(np.arange(im3_shape[0]), im3_shape[1]),
            "im3_x": np.tile(np.arange(im3_shape[1]), im3_shape[0]),
            "latitude": im3_lat.ravel(),
            "longitude": im3_lon.ravel(),
            "return_period": int(rp),
            "conus404_historical_return_level": aggregate.ravel(),
            "conus404_source_cell_count": counts.ravel(),
        }))

    out = pd.concat(records, ignore_index=True)
    out["historical_period"] = "1980-2019"
    out["spatial_aggregation"] = args.statistic
    out["spatial_assignment"] = "nearest_IM3_center"
    write_parquet(out, args.output)
    print(f"Wrote historical comparison: {args.output}")
    print(f"Wrote crosswalk: {args.crosswalk_output}")


if __name__ == "__main__":
    main()

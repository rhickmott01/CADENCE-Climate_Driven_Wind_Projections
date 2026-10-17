from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import h5py
import numpy as np

from .io import Dataset, decode_times, find_time_variable, find_wind_name, read_2d_lat_lon, normalize_wind_block, require_netcdf4


def _iter_days(path: Path):
    """Yield (calendar_date, daily_max) while reading one raw NetCDF at a time."""
    require_netcdf4()
    with Dataset(path, "r") as ds:
        time_name = find_time_variable(ds)
        if not time_name:
            raise ValueError(f"No time coordinate in {path}")
        wind_name = find_wind_name(ds)
        times = decode_times(ds, time_name)
        lat, lon, _, _ = read_2d_lat_lon(ds)
        grid_shape = lat.shape

        # This code deliberately reads one day at a time to keep memory bounded.
        # V1 uses 3-hourly -> daily maximum -> annual maximum.
        # Future versioning option A: calculate the annual maximum directly from
        # the 3-hourly values. Keep that alternative separate so the production
        # temporal definition remains explicit and testable.
        start = 0
        current_day = times[0].date() if times else None
        for idx in range(1, len(times) + 1):
            next_day = times[idx].date() if idx < len(times) else None
            if next_day != current_day:
                block = ds.variables[wind_name][start:idx]
                values = normalize_wind_block(block, idx - start, grid_shape, path)
                daily_max = np.nanmax(values, axis=0).astype(np.float32)
                yield current_day, daily_max, lat, lon
                start = idx
                current_day = next_day


def build_calendar_year_maxima(input_dir: Path, scenario: str) -> dict:
    """Build calendar-year maxima from all raw three-hourly IM3 WSPD files."""
    files = sorted(input_dir.rglob("*.nc")) + sorted(input_dir.rglob("*.nc4"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {input_dir}")

    annual_max: dict[int, np.ndarray] = {}
    lat_ref = lon_ref = None
    expected_shape = None
    day_counts = defaultdict(int)

    for path in files:
        for current_day, daily_max, lat, lon in _iter_days(path):
            if expected_shape is None:
                expected_shape = daily_max.shape
                lat_ref = lat
                lon_ref = lon
            elif daily_max.shape != expected_shape:
                raise ValueError(f"Grid shape changed in {path}: {daily_max.shape} != {expected_shape}")
            elif not np.allclose(lat, lat_ref, atol=1e-5) or not np.allclose(lon, lon_ref, atol=1e-5):
                raise ValueError(f"Coordinate grid changed in {path}")

            year = current_day.year
            if year not in annual_max:
                annual_max[year] = daily_max.copy()
            else:
                annual_max[year] = np.fmax(annual_max[year], daily_max)
            day_counts[year] += 1

    years = np.asarray(sorted(annual_max), dtype=np.int16)
    if years.size == 0:
        raise ValueError("No calendar-year maxima were produced")

    return {
        "scenario": scenario,
        "years": years,
        "annual_max": np.stack([annual_max[int(year)] for year in years]).astype(np.float32),
        "lat": np.asarray(lat_ref, dtype=np.float32),
        "lon": np.asarray(lon_ref, dtype=np.float32),
        "day_counts": dict(day_counts),
    }


def write_annual_h5(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5:
        h5.create_dataset("wspd_annual_max", data=result["annual_max"], compression="gzip", compression_opts=4)
        h5.create_dataset("latitude", data=result["lat"], compression="gzip", compression_opts=4)
        h5.create_dataset("longitude", data=result["lon"], compression="gzip", compression_opts=4)
        h5.create_dataset("years", data=result["years"])
        h5.attrs["dataset"] = "IM3/HyperFACETS TGW"
        h5.attrs["scenario"] = result["scenario"]
        h5.attrs["source_variable"] = "WSPD"
        h5.attrs["units"] = "m s-1"
        h5.attrs["source_temporal_resolution"] = "3-hourly"
        h5.attrs["calendar_definition"] = "January 1 through December 31"
        h5.attrs["aggregation_method"] = "3-hourly WSPD -> daily maximum -> calendar-year maximum"
        h5.attrs["future_direct_3hourly_option"] = "Not used in V1; preserve as future sensitivity method."
        h5.attrs["grid_resolution_km"] = 12
        h5.attrs["scenario_periods"] = "historical 1980-2019; near_future 2020-2059; far_future 2060-2099"
    tmp.replace(output_path)

#!/usr/bin/env python3
"""Phase 03: validate a consolidated IM3 annual-maximum HDF5 cube."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from im3_wind.config import (
    FAR_FUTURE_END,
    FAR_FUTURE_START,
    HISTORICAL_END,
    HISTORICAL_START,
    NEAR_FUTURE_END,
    NEAR_FUTURE_START,
    validate_scenario,
)


EXPECTED_RANGES = {
    "historical": (HISTORICAL_START, HISTORICAL_END),
    "rcp45cooler": (NEAR_FUTURE_START, NEAR_FUTURE_END),
    "rcp45hotter": (NEAR_FUTURE_START, NEAR_FUTURE_END),
    "rcp85cooler": (NEAR_FUTURE_START, NEAR_FUTURE_END),
    "rcp85hotter": (NEAR_FUTURE_START, NEAR_FUTURE_END),
}


def validate(path: Path, scenario: str, start_year: int, end_year: int) -> None:
    with h5py.File(path, "r") as h5:
        required = {"wspd_annual_max", "latitude", "longitude", "years"}
        missing = required - set(h5)
        if missing:
            raise ValueError(f"{path}: missing datasets {sorted(missing)}")

        values = np.asarray(h5["wspd_annual_max"][:], dtype=np.float32)
        years = np.asarray(h5["years"][:], dtype=np.int16)
        lat = np.asarray(h5["latitude"][:], dtype=np.float32)
        lon = np.asarray(h5["longitude"][:], dtype=np.float32)

        expected_years = np.arange(start_year, end_year + 1, dtype=np.int16)
        if not np.array_equal(years, expected_years):
            raise ValueError(f"{path}: expected years {start_year}-{end_year}, got {years.tolist()}")
        if values.ndim != 3:
            raise ValueError(f"{path}: annual-max cube must be 3-D")
        if values.shape[1:] != lat.shape or lat.shape != lon.shape:
            raise ValueError(f"{path}: annual-max and coordinate shapes do not match")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError(f"{path}: non-finite coordinates detected")
        if np.nanmin(lat) < -90 or np.nanmax(lat) > 90 or np.nanmin(lon) < -180 or np.nanmax(lon) > 180:
            raise ValueError(f"{path}: coordinates outside valid WGS84 ranges")

        finite = np.isfinite(values)
        if np.any(values[finite] < 0):
            raise ValueError(f"{path}: negative wind values detected")
        if np.any(values[finite] > 100):
            print(f"WARNING: {path} contains wind values above 100 m/s")
        if not np.all(np.diff(years) == 1):
            raise ValueError(f"{path}: years are not contiguous")

        expected_attrs = {
            "scenario": scenario,
            "source_variable": "WSPD",
            "source_temporal_resolution": "3-hourly",
            "calendar_definition": "January 1 through December 31",
            "aggregation_method": "3-hourly WSPD -> daily maximum -> calendar-year maximum",
        }
        for key, expected in expected_attrs.items():
            actual = str(h5.attrs.get(key, ""))
            if actual != expected:
                raise ValueError(f"{path}: attribute {key}={actual!r}, expected {expected!r}")

        missing_count = int(np.count_nonzero(~finite))
        if missing_count:
            print(f"WARNING: {path} contains {missing_count} non-finite annual maxima")

    print(f"PASS: {path} ({scenario}, {start_year}-{end_year})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    scenario = validate_scenario(args.scenario)
    default_start, default_end = EXPECTED_RANGES[scenario]
    validate(args.input, scenario, args.start_year or default_start, args.end_year or default_end)


if __name__ == "__main__":
    main()

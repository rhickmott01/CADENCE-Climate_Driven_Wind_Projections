#!/usr/bin/env python3
"""Phase 10: publish the compact IM3 climate-delta lookup for CADENCE.

The master climate delta stays at the native IM3 ~12-km grid. CADENCE should
combine it with the historical CONUS404 return level through the precomputed
CONUS404-to-IM3 crosswalk.

We intentionally do NOT expand the delta to every 4-km CONUS404 cell here. A
full 4-km x 2050-year x scenario x return-period expansion would duplicate the
same IM3 climate signal many times and create an unnecessarily large table.
The existing CADENCE architecture favors precomputed ID joins, so the smaller
crosswalk + IM3 delta product is the preferred representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from im3_wind.config import CADENCE_END, CADENCE_START, DEFAULT_RETURN_PERIODS, FUTURE_SCENARIOS
from im3_wind.parquet_io import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, type=Path)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()

    delta = pd.read_parquet(args.delta)
    crosswalk = pd.read_parquet(args.crosswalk)
    required_delta = {"im3_y", "im3_x", "scenario", "year", "return_period", "delta"}
    required_crosswalk = {"lat_idx", "lon_idx", "im3_y", "im3_x"}
    if not required_delta.issubset(delta.columns):
        raise ValueError(f"Delta table missing {sorted(required_delta - set(delta.columns))}")
    if not required_crosswalk.issubset(crosswalk.columns):
        raise ValueError(f"Crosswalk missing {sorted(required_crosswalk - set(crosswalk.columns))}")

    if sorted(delta["scenario"].dropna().unique()) != sorted(FUTURE_SCENARIOS):
        print("WARNING: delta table does not contain exactly the four expected IM3 future scenarios")

    expected_years = set(range(CADENCE_START, CADENCE_END + 1))
    actual_years = set(delta["year"].dropna().astype(int).unique())
    missing_years = sorted(expected_years - actual_years)
    if missing_years:
        raise ValueError(f"Delta table is missing CADENCE years: {missing_years}")

    actual_rps = set(delta["return_period"].dropna().astype(int).unique())
    if actual_rps != set(DEFAULT_RETURN_PERIODS):
        raise ValueError(f"Delta table return periods {sorted(actual_rps)} do not match {DEFAULT_RETURN_PERIODS}")

    # Publish only the compact IM3-grid delta. The crosswalk remains the separate
    # mapping from CONUS404 cell -> IM3 cell. CADENCE can join on IDs at runtime.
    write_parquet(delta, args.output)

    metadata = {
        "product": "CADENCE wind_delta",
        "grid": "IM3/HyperFACETS native 12-km grid",
        "baseline": "historical CONUS404 return-period hazard",
        "climate_signal": "IM3/HyperFACETS return-period-specific delta",
        "conus404_bridge": str(args.crosswalk),
        "years": [CADENCE_START, CADENCE_END],
        "return_periods": list(DEFAULT_RETURN_PERIODS),
        "scenarios": list(FUTURE_SCENARIOS),
        "gust_conversion": "performed downstream after applying the delta",
        "runtime_join": "CONUS404 cell -> precomputed IM3 cell -> scenario/year/return-period delta",
        "note": "Do not expand the master delta to all 4-km cells unless a downstream use specifically requires a materialized wide table."
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Published CADENCE delta: {args.output}")
    print(f"Metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()

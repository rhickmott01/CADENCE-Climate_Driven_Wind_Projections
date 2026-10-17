#!/usr/bin/env python3
"""Phase 02: build annual block maxima from IM3 three-hourly WSPD.

V1 temporal definition:
    3-hourly WSPD -> daily maximum -> calendar-year maximum.

The existing CONUS404 product is also an annual maximum of daily maxima, but
CONUS404 labels those years as water years. IM3 is kept on calendar years so
its published 1980-2019 / 2020-2059 / 2060-2099 experiment structure is not
altered.

Future versioning note:
    A direct 3-hourly -> annual maximum path can be added later as a sensitivity
    test. It is intentionally not used for the V1 production delta.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from im3_wind.annual_maxima import build_calendar_year_maxima, write_annual_h5
from im3_wind.config import SCENARIOS, validate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scenario = validate_scenario(args.scenario)
    result = build_calendar_year_maxima(args.input_dir, scenario)
    write_annual_h5(result, args.output)

    years = result["years"]
    print(f"Scenario: {scenario}")
    print(f"Calendar years: {int(years[0])}-{int(years[-1])}")
    print(f"Grid: {result['annual_max'].shape[1]} x {result['annual_max'].shape[2]}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

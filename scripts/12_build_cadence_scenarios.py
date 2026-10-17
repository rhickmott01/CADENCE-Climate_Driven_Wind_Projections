#!/usr/bin/env python3
"""Phase 10: build CADENCE design-storm and Projected wind scenarios.

The design-storm rows are 10-, 25-, 50-, 100-, 200-, and 500-year future winds.
The Projected row is a user-configured weighted average of those design-storm
winds. It is a synthetic decision-support scenario, not a new return period.

This stage is downstream of the scientific delta product. It deliberately does
not change how the climate delta itself is estimated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from im3_wind.config import DEFAULT_RETURN_PERIODS
from im3_wind.parquet_io import write_parquet


def load_weights(path: Path) -> dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    weights = {int(k): float(v) for k, v in raw["projected_weights"].items()}
    missing = set(DEFAULT_RETURN_PERIODS) - set(weights)
    extra = set(weights) - set(DEFAULT_RETURN_PERIODS)
    if missing or extra:
        raise ValueError(f"Projected weights must match {DEFAULT_RETURN_PERIODS}; missing={missing}, extra={extra}")
    if any((not np.isfinite(v)) or v < 0 for v in weights.values()):
        raise ValueError("Projected weights must be finite and non-negative")
    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"Projected weights must sum to 1.0; got {total}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args()

    deltas = pd.read_parquet(args.delta)
    required = {
        "im3_y", "im3_x", "latitude", "longitude", "scenario", "year",
        "return_period", "conus404_historical_return_level", "delta",
    }
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Missing delta columns: {sorted(missing)}")

    weights = load_weights(args.weights)
    deltas = deltas.loc[deltas["return_period"].isin(DEFAULT_RETURN_PERIODS)].copy()
    deltas["wind_speed"] = deltas["conus404_historical_return_level"] * deltas["delta"]
    group_cols = ["im3_y", "im3_x", "latitude", "longitude", "scenario", "year"]

    design = deltas[group_cols + ["return_period", "wind_speed"]].copy()
    design["scenario_name"] = design["return_period"].map(lambda rp: f"{int(rp)}-Year Storm")
    design["calculation_method"] = "CONUS404 historical return level multiplied by IM3 climate delta"

    weighted = design.assign(weight=design["return_period"].map(weights))
    projected = (
        weighted.assign(weighted_wind=weighted["wind_speed"] * weighted["weight"])
        .groupby(group_cols, as_index=False)["weighted_wind"]
        .sum(min_count=1)
        .rename(columns={"weighted_wind": "wind_speed"})
    )
    projected["scenario_name"] = "Projected"
    projected["return_period"] = pd.NA
    projected["calculation_method"] = "weighted average of design-storm winds"

    out = pd.concat([design, projected], ignore_index=True, sort=False)
    out["design_storm_weights_file"] = str(args.weights)
    out["projected_scenario_note"] = "Synthetic decision-support scenario; not a statistically defined return period."
    write_parquet(out, args.output)
    print(f"Wrote CADENCE scenarios: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 08: apply historical tail-bias ratios to future IM3 return levels.

For each future IM3 climate period:
    adjusted future return level = historical bias ratio * future IM3 return level

The same historical transformation is used for each future scenario so the
scenario differences come from the climate simulation itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from im3_wind.bias import apply_multiplicative_bias
from im3_wind.parquet_io import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias", required=True, type=Path)
    parser.add_argument("--future-return-levels", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bias = pd.read_parquet(args.bias)
    bias = bias.loc[bias["valid_bias_ratio"]].copy()
    bias = bias[["im3_y", "im3_x", "return_period", "bias_ratio", "conus404_historical_return_level"]]

    future = pd.read_parquet(args.future_return_levels)
    future = future.loc[future["climate_period"].isin(["near_future", "far_future"])].copy()
    future = future.rename(columns={"return_level": "im3_future_return_level"})
    merged = future.merge(bias, on=["im3_y", "im3_x", "return_period"], how="inner", validate="many_to_one")
    merged["bias_adjusted_im3_future_return_level"] = apply_multiplicative_bias(
        merged["im3_future_return_level"].to_numpy(),
        merged["bias_ratio"].to_numpy(),
    )
    merged["scenario"] = args.scenario
    merged["bias_adjustment_method"] = "historical_return_level_multiplicative_ratio"
    merged["equation_set"] = "B_R * RCMFuture_R_t"

    write_parquet(merged, args.output)
    print(f"Wrote bias-adjusted future return levels: {args.output}")


if __name__ == "__main__":
    main()

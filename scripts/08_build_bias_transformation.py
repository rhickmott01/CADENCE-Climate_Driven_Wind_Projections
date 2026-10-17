#!/usr/bin/env python3
"""Phase 07: build return-period-specific CONUS404/IM3 historical bias ratios.

Equation implemented for V1:
    B_R = historical CONUS404 return level / historical IM3 return level

This is intentionally a multiplicative transformation. The output retains the
raw historical levels and validity flags so a future extreme-quantile or QDM
method can replace the transformation without changing the downstream schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from im3_wind.bias import multiplicative_bias_ratio
from im3_wind.parquet_io import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-comparison", required=True, type=Path)
    parser.add_argument("--im3-historical-return-levels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    compare = pd.read_parquet(args.historical_comparison)
    im3 = pd.read_parquet(args.im3_historical_return_levels)
    im3 = im3.loc[im3["climate_period"] == "historical"].copy()
    im3 = im3.rename(columns={"return_level": "im3_historical_return_level"})
    keys = ["im3_y", "im3_x", "return_period"]
    merged = compare.merge(im3[keys + ["im3_historical_return_level"]], on=keys, how="inner", validate="one_to_one")
    bias, valid = multiplicative_bias_ratio(
        merged["conus404_historical_return_level"].to_numpy(),
        merged["im3_historical_return_level"].to_numpy(),
    )
    merged["bias_ratio"] = bias
    merged["valid_bias_ratio"] = valid
    merged["bias_method"] = "historical_return_level_ratio"
    merged["bias_formula_description"] = "CONUS404 historical return level divided by IM3 historical return level"
    merged["note_for_future_qdm"] = "Replace the transformation module, not the downstream delta schema, for a future QDM/extreme-quantile method."

    write_parquet(merged, args.output)
    print(f"Wrote {len(merged):,} historical bias rows: {args.output}")


if __name__ == "__main__":
    main()

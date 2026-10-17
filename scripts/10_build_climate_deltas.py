#!/usr/bin/env python3
"""Phase 09: build annual 2025-2050 IM3 climate deltas for CADENCE.

V1 uses the TGW climate-period GEVs as statistical anchors:
    historical 1980-2019 -> anchor at 1999.5, delta = 1.0
    near future 2020-2059 -> anchor at 2039.5
    far future 2060-2099 -> anchor at 2079.5

CADENCE only publishes 2025-2050. For this V1 output we interpolate/extrapolate
from the historical anchor toward the near-future anchor. The far-future return
levels are retained in the upstream results for QA and future versioning, but
they are not required to create the 2025-2050 production delta.

This is intentionally not a year-by-year GEV fit: one year contains one annual
maximum, not enough observations to estimate a return-period distribution.
A future nonstationary GEV can replace this annualization function without
changing the final delta table schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from im3_wind.config import CADENCE_END, CADENCE_START
from im3_wind.parquet_io import write_parquet

HISTORICAL_ANCHOR = (1980 + 2019) / 2.0
NEAR_FUTURE_ANCHOR = (2020 + 2059) / 2.0
FAR_FUTURE_ANCHOR = (2060 + 2099) / 2.0


def annualize_delta(near_delta: float, years: np.ndarray) -> np.ndarray:
    if not np.isfinite(near_delta):
        return np.full(years.shape, np.nan, dtype=np.float32)
    slope = (near_delta - 1.0) / (NEAR_FUTURE_ANCHOR - HISTORICAL_ANCHOR)
    return (1.0 + slope * (years.astype(float) - HISTORICAL_ANCHOR)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-adjusted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    future = pd.read_parquet(args.future_adjusted)
    required = {
        "im3_y", "im3_x", "latitude", "longitude", "scenario", "climate_period",
        "return_period", "bias_adjusted_im3_future_return_level", "conus404_historical_return_level",
    }
    missing = required - set(future.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    years = np.arange(CADENCE_START, CADENCE_END + 1, dtype=np.int16)
    anchors = (
        future.pivot_table(
            index=["im3_y", "im3_x", "latitude", "longitude", "scenario", "return_period", "conus404_historical_return_level"],
            columns="climate_period",
            values="bias_adjusted_im3_future_return_level",
            aggfunc="first",
        )
        .reset_index()
    )
    if "near_future" not in anchors.columns or "far_future" not in anchors.columns:
        raise ValueError("Both near_future and far_future return levels are required")

    records = []
    for row in anchors.itertuples(index=False):
        baseline = float(row.conus404_historical_return_level)
        near = float(row.near_future)
        far = float(row.far_future)
        if not (np.isfinite(baseline) and baseline > 0 and np.isfinite(near) and np.isfinite(far)):
            continue
        near_delta = near / baseline
        far_delta = far / baseline
        annual = annualize_delta(near_delta, years)
        for idx, year in enumerate(years):
            records.append({
                "im3_y": int(row.im3_y),
                "im3_x": int(row.im3_x),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "scenario": str(row.scenario),
                "year": int(year),
                "return_period": int(row.return_period),
                "conus404_historical_return_level": baseline,
                "near_future_bias_adjusted_return_level": near,
                "far_future_bias_adjusted_return_level": far,
                "near_future_delta": float(near_delta),
                "far_future_delta": float(far_delta),
                "delta": float(annual[idx]),
                "delta_method": "historical_to_near_future_period_anchor_interpolation",
            })

    out = pd.DataFrame(records)
    out["historical_anchor_year"] = HISTORICAL_ANCHOR
    out["near_future_anchor_year"] = NEAR_FUTURE_ANCHOR
    out["far_future_anchor_year"] = FAR_FUTURE_ANCHOR
    out["cadence_start_year"] = CADENCE_START
    out["cadence_end_year"] = CADENCE_END
    out["future_nonstationary_gev_note"] = "Future version: replace annualization with a time-varying GEV while preserving scenario/year/return_period/delta fields."
    write_parquet(out, args.output)
    print(f"Wrote {len(out):,} annual delta rows: {args.output}")


if __name__ == "__main__":
    main()

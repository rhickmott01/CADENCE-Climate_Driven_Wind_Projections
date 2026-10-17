#!/usr/bin/env python3
"""Phase 11: create QA plots for IM3 bias ratios and annual climate deltas."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, type=Path)
    parser.add_argument("--bias", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    delta = pd.read_parquet(args.delta)
    bias = pd.read_parquet(args.bias)

    bias_summary = (
        bias.loc[bias["valid_bias_ratio"]]
        .groupby("return_period")["bias_ratio"]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
    )
    bias_summary.to_csv(args.output_dir / "historical_bias_summary.csv", index=False)

    for scenario, subset in delta.groupby("scenario"):
        plt.figure(figsize=(10, 6))
        for rp, rp_subset in subset.groupby("return_period"):
            annual = rp_subset.groupby("year", as_index=False)["delta"].median()
            plt.plot(annual["year"], annual["delta"], label=f"{int(rp)}-year")
        plt.axhline(1.0, linewidth=1)
        plt.xlabel("Calendar year")
        plt.ylabel("Climate delta")
        plt.title(f"IM3 annual climate deltas: {scenario}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / f"delta_timeseries_{scenario}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(9, 5))
    medians = bias_summary.set_index("return_period")["median"]
    plt.plot(medians.index, medians.values, marker="o")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Return period (years)")
    plt.ylabel("Historical CONUS404 / IM3 return-level ratio")
    plt.title("Historical return-level bias by return period")
    plt.tight_layout()
    plt.savefig(args.output_dir / "historical_bias_by_return_period.png", dpi=180)
    plt.close()

    delta_summary = (
        delta.groupby(["scenario", "year", "return_period"], as_index=False)["delta"]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
    )
    delta_summary.to_csv(args.output_dir / "annual_delta_summary.csv", index=False)
    print(f"Wrote QA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

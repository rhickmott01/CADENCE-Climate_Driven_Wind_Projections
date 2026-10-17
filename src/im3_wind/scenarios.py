from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: dict[int, float], return_periods: tuple[int, ...]) -> dict[int, float]:
    missing = set(return_periods) - set(weights)
    extra = set(weights) - set(return_periods)
    if missing or extra:
        raise ValueError(f"Weights must match return periods; missing={sorted(missing)}, extra={sorted(extra)}")
    values = {int(k): float(weights[k]) for k in return_periods}
    if any((not np.isfinite(v)) or v < 0 for v in values.values()):
        raise ValueError("Scenario weights must be finite and non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Scenario weights must have a positive total")
    return {k: v / total for k, v in values.items()}


def weighted_projected_wind(design_storms: pd.DataFrame, weights: dict[int, float]) -> pd.DataFrame:
    required = {"lat_idx", "lon_idx", "latitude", "longitude", "climate_scenario", "year", "return_period", "wind_speed"}
    missing = required - set(design_storms.columns)
    if missing:
        raise ValueError(f"Missing design-storm columns: {sorted(missing)}")

    weighted = design_storms.assign(weight=design_storms["return_period"].map(weights))
    weighted["weighted_wind"] = weighted["wind_speed"] * weighted["weight"]
    grouped = (
        weighted.groupby(
            ["lat_idx", "lon_idx", "latitude", "longitude", "climate_scenario", "year"],
            as_index=False,
        )["weighted_wind"]
        .sum(min_count=1)
        .rename(columns={"weighted_wind": "wind_speed"})
    )
    grouped["scenario_name"] = "Projected"
    grouped["return_period"] = pd.NA
    grouped["calculation_method"] = "weighted_average_of_design_storm_winds"
    return grouped

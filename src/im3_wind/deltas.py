from __future__ import annotations

import numpy as np

from .config import CADENCE_END, CADENCE_START


def annual_deltas_from_historical_and_near(
    near_adjusted: float,
    historical_baseline: float,
    historical_anchor_year: float,
    near_anchor_year: float,
    years: np.ndarray | None = None,
) -> dict[int, float]:
    """
    Build annual deltas for CADENCE 2025-2050 using historical -> near-future anchors.

    The historical anchor has delta=1.0. The near-future anchor is the midpoint of
    the 2020-2059 climate period. Values between and just beyond the anchors are
    linearly interpolated/extrapolated. The IM3 far-future period is calculated and
    stored elsewhere for diagnostics/future versioning, but is not needed to publish
    CADENCE's 2025-2050 annual deltas.
    """
    if not np.isfinite(near_adjusted) or not np.isfinite(historical_baseline) or historical_baseline <= 0:
        return {}
    year_values = years if years is not None else np.arange(CADENCE_START, CADENCE_END + 1)
    near_delta = float(near_adjusted / historical_baseline)
    slope = (near_delta - 1.0) / (near_anchor_year - historical_anchor_year)
    return {int(year): float(1.0 + slope * (float(year) - historical_anchor_year)) for year in year_values}

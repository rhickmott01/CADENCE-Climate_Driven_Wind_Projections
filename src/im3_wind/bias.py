from __future__ import annotations

import numpy as np


def multiplicative_bias_ratio(conus_return_level: np.ndarray, im3_return_level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute B_R = CONUS404 historical return level / IM3 historical return level."""
    conus = np.asarray(conus_return_level, dtype=np.float64)
    im3 = np.asarray(im3_return_level, dtype=np.float64)
    valid = np.isfinite(conus) & np.isfinite(im3) & (conus > 0) & (im3 > 0)
    ratio = np.full(conus.shape, np.nan, dtype=np.float32)
    ratio[valid] = (conus[valid] / im3[valid]).astype(np.float32)
    return ratio, valid


def apply_multiplicative_bias(im3_future: np.ndarray, bias_ratio: np.ndarray) -> np.ndarray:
    future = np.asarray(im3_future, dtype=np.float64)
    bias = np.asarray(bias_ratio, dtype=np.float64)
    if future.shape != bias.shape:
        raise ValueError("Future IM3 return levels and bias ratios must have the same shape")
    adjusted = future * bias
    adjusted[~np.isfinite(adjusted)] = np.nan
    return adjusted.astype(np.float32)

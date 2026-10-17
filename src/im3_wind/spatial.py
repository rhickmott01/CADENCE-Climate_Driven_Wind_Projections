from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def lonlat_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64).ravel())
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64).ravel())
    return np.column_stack((
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ))


def nearest_source_index(source_lat: np.ndarray, source_lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Return nearest source-cell index for every target point."""
    source_xyz = lonlat_to_unit_xyz(source_lat, source_lon)
    target_xyz = lonlat_to_unit_xyz(target_lat, target_lon)
    tree = cKDTree(source_xyz)
    _, indices = tree.query(target_xyz, k=1)
    return indices.astype(np.int64)


def aggregate_values_to_target(
    source_values: np.ndarray,
    source_to_target_flat: np.ndarray,
    target_shape: tuple[int, int],
    statistic: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate source values to target cells using the supplied mapping."""
    values = np.asarray(source_values, dtype=np.float64).ravel()
    mapping = np.asarray(source_to_target_flat, dtype=np.int64).ravel()
    if values.size != mapping.size:
        raise ValueError("source_values and source_to_target_flat must have the same length")
    target_n = int(np.prod(target_shape))
    groups = [[] for _ in range(target_n)]
    for value, target_idx in zip(values, mapping):
        if np.isfinite(value) and 0 <= target_idx < target_n:
            groups[int(target_idx)].append(float(value))
    output = np.full(target_n, np.nan, dtype=np.float32)
    counts = np.zeros(target_n, dtype=np.int32)
    for target_idx, group in enumerate(groups):
        if not group:
            continue
        array = np.asarray(group, dtype=np.float64)
        if statistic == "median":
            output[target_idx] = np.float32(np.median(array))
        elif statistic == "mean":
            output[target_idx] = np.float32(np.mean(array))
        else:
            raise ValueError("statistic must be 'median' or 'mean'")
        counts[target_idx] = len(group)
    return output.reshape(target_shape), counts.reshape(target_shape)

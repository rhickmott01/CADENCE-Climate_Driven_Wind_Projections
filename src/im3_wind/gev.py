from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Iterable

import numpy as np
from scipy.stats import genextreme


@dataclass(frozen=True)
class GEVGrid:
    location: np.ndarray
    scale: np.ndarray
    shape_c: np.ndarray
    converged: np.ndarray
    n_years: np.ndarray


def fit_gev_one(series: np.ndarray) -> tuple[float, float, float, int, int]:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        return np.nan, np.nan, np.nan, 0, n
    try:
        c, location, scale = genextreme.fit(values)
    except Exception:
        return np.nan, np.nan, np.nan, 0, n
    valid = np.isfinite(c) and np.isfinite(location) and np.isfinite(scale) and scale > 0
    if not valid:
        return np.nan, np.nan, np.nan, 0, n
    return float(location), float(scale), float(c), 1, n


def _fit_chunk(args: tuple[np.ndarray, int]) -> np.ndarray:
    block, min_years = args
    results = np.full((block.shape[1], 5), np.nan, dtype=np.float64)
    for idx in range(block.shape[1]):
        series = block[:, idx]
        values = series[np.isfinite(series)]
        if values.size < min_years:
            results[idx, 3] = 0
            results[idx, 4] = values.size
            continue
        results[idx] = fit_gev_one(values)
    return results


def fit_gev_grid(cube: np.ndarray, processes: int | None = None, chunk_cells: int = 2000, min_years: int = 20) -> GEVGrid:
    if cube.ndim != 3:
        raise ValueError("cube must have shape (year, y, x)")
    n_years, ny, nx = cube.shape
    flat = cube.reshape(n_years, ny * nx)
    chunks = []
    for start in range(0, flat.shape[1], chunk_cells):
        stop = min(start + chunk_cells, flat.shape[1])
        chunks.append((flat[:, start:stop], min_years))

    worker_count = processes if processes is not None else max(1, cpu_count() - 1)
    with Pool(processes=worker_count) as pool:
        result_chunks = pool.map(_fit_chunk, chunks)
    result = np.concatenate(result_chunks, axis=0)

    return GEVGrid(
        location=result[:, 0].reshape(ny, nx).astype(np.float32),
        scale=result[:, 1].reshape(ny, nx).astype(np.float32),
        shape_c=result[:, 2].reshape(ny, nx).astype(np.float32),
        converged=result[:, 3].reshape(ny, nx).astype(np.int8),
        n_years=result[:, 4].reshape(ny, nx).astype(np.int16),
    )


def return_levels(fit: GEVGrid, return_periods: Iterable[int]) -> np.ndarray:
    periods = np.asarray(tuple(return_periods), dtype=float)
    out = np.full((periods.size,) + fit.location.shape, np.nan, dtype=np.float32)
    valid = (
        (fit.converged == 1)
        & np.isfinite(fit.location)
        & np.isfinite(fit.scale)
        & np.isfinite(fit.shape_c)
        & (fit.scale > 0)
    )
    location = fit.location.astype(np.float64)
    scale = fit.scale.astype(np.float64)
    shape = fit.shape_c.astype(np.float64)
    for idx, rp in enumerate(periods):
        probability = 1.0 - 1.0 / rp
        vals = genextreme.ppf(probability, shape, location, scale)
        vals[~valid] = np.nan
        out[idx] = vals.astype(np.float32)
    return out


def interpolate_linear(year: int, anchor_years: np.ndarray, anchor_values: np.ndarray) -> float:
    """Linear interpolation/extrapolation across ordered scalar anchors."""
    years = np.asarray(anchor_years, dtype=float)
    values = np.asarray(anchor_values, dtype=float)
    if years.ndim != 1 or values.ndim != 1 or years.size != values.size or years.size < 2:
        raise ValueError("anchor_years and anchor_values must be 1-D arrays with at least two points")
    if not np.all(np.diff(years) > 0):
        raise ValueError("anchor_years must be strictly increasing")
    return float(np.interp(year, years, values)) if years[0] <= year <= years[-1] else float(
        values[0] + (year - years[0]) * (values[1] - values[0]) / (years[1] - years[0])
        if year < years[0]
        else values[-2] + (year - years[-2]) * (values[-1] - values[-2]) / (years[-1] - years[-2])
    )

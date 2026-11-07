"""Annual scenario transitions and bounded-memory CADENCE Parquet exports."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ANCHORS = (1999.5, 2039.5, 2079.5)


def annual_transition(near, far, year):
    """Log-linear positive multipliers through climate-period midpoint anchors."""
    if not ANCHORS[0] <= year <= ANCHORS[-1]:
        raise ValueError('Year outside transition anchor range')
    near, far = np.asarray(near), np.asarray(far)
    if near.shape != far.shape:
        raise ValueError('Near/far shapes differ')
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        if year <= ANCHORS[1]:
            weight = (year-ANCHORS[0])/(ANCHORS[1]-ANCHORS[0])
            value = np.exp(weight*np.log(near))
            valid = np.isfinite(near) & (near > 0)
        else:
            weight = (year-ANCHORS[1])/(ANCHORS[2]-ANCHORS[1])
            value = np.exp((1-weight)*np.log(near)+weight*np.log(far))
            valid = np.isfinite(near) & (near > 0) & np.isfinite(far) & (far > 0)
    return np.where(valid, value, np.nan)


def check_slices(near, far):
    for key in ('wind_grid_id', 'lat_idx', 'lon_idx', 'latitude', 'longitude', 'return_periods', 'baseline'):
        if not np.array_equal(near[key], far[key], equal_nan=True):
            raise ValueError(f'Near/far grid/baseline mismatch: {key}')


def annual_frames(near, far, scenario, meta):
    check_slices(near, far)
    n, nrp = near['factor'].shape
    base = {k: np.repeat(near[k], nrp) for k in ('wind_grid_id', 'lat_idx', 'lon_idx', 'latitude', 'longitude', 'source_cell_id')}
    base.update(return_period=np.tile(near['return_periods'], n), scenario=scenario,
                transition_method='log_linear_period_midpoints', provenance=json.dumps(meta, sort_keys=True))
    for year in range(2026, 2061):
        factor = annual_transition(near['factor'], far['factor'], year)
        lo = annual_transition(near['lower'], far['lower'], year)
        hi = annual_transition(near['upper'], far['upper'], year)
        flags = near['flags'].copy()
        if year > ANCHORS[1]:
            flags |= far['flags']
        flags[~np.isfinite(factor) | (factor <= 0)] |= 256
        from .baseline import POINT_UNUSABLE
        bflags = near.get('baseline_quality_flags', np.zeros_like(flags))
        future = near['baseline']*factor
        future[((bflags & POINT_UNUSABLE) != 0) | ~np.isfinite(future) | (future <= 0)] = np.nan
        yield pd.DataFrame(dict(base, baseline_quality_flags=bflags.ravel(), year=year, change_factor=factor.ravel(),
                                factor_lower=lo.ravel(), factor_upper=hi.ravel(), flags=flags.ravel(),
                                historical_wind=near['baseline'].ravel(),
                                future_wind=future.ravel(), units=meta['units'],
                                wind_definition=meta['wind_definition']))


def compatibility_frame(near, far, rp, allow_flags=0):
    """Explicit scenario/RP selection; fail rather than drop unreliable grid cells."""
    check_slices(near, far)
    idx = np.flatnonzero(near['return_periods'] == rp)
    if len(idx) != 1:
        raise ValueError('Return period not found')
    j = idx[0]
    flags = near['flags'][:, j] | far['flags'][:, j]
    if np.any(flags.astype(np.uint32) & (~allow_flags & 0xffff)):
        raise ValueError('Unaccepted quality flags; review scientific output and explicitly set --allow-flags')
    out = pd.DataFrame({k: near[k] for k in ('lat_idx', 'lon_idx')})
    # 2025 is an additional compatibility year for the current CADENCE consumer.
    for year in range(2025, 2061):
        v = annual_transition(near['factor'][:, j], far['factor'][:, j], year)
        if not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError('Compatibility export requires finite positive factors for every cell')
        out[f'scale_{year}'] = v
    return out


def write_frames(path, frames, meta):
    """Atomic exclusive Parquet with provenance in schema metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.parquet')
    os.close(fd)
    writer = None
    try:
        for frame in frames:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            table = table.replace_schema_metadata({b'cadence_provenance': json.dumps(meta, sort_keys=True).encode()})
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
            writer.write_table(table)
        if writer is None:
            raise ValueError('No frames supplied')
        writer.close()
        writer = None
        os.link(tmp, path)
    finally:
        if writer is not None:
            writer.close()
        os.unlink(tmp)


def parquet_resume(path, meta):
    if not Path(path).exists():
        return False
    old = json.loads(pq.read_schema(path).metadata[b'cadence_provenance'])
    if old != meta:
        raise ValueError(f'Stale Parquet output {path}; choose fresh results directory')
    return True

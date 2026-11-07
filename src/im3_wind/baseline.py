"""Adapt a real, already gust-converted CONUS404 table without repairing it."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

POINT_INVALID = 1
CURVE_DECREASING = 2
NOT_CONVERGED = 4
INTERVAL_INVALID = 8
PERIOD_UNCONFIRMED = 16
POINT_UNUSABLE = POINT_INVALID | CURVE_DECREASING | NOT_CONVERGED


def inspect_table(path, cfg):
    """Only explicitly configured wide/long schemas are accepted."""
    from pathlib import Path
    path = Path(path)
    columns = cfg.get('conus_columns', {})
    layout = cfg.get('conus_layout', 'wide')
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    units = cfg.get('conus_units')
    definition = cfg.get('conus_wind_definition')
    if units not in ('m s-1', 'mph') or definition != '3-second gust':
        raise ValueError('Explicit CONUS units and already-converted 3-second gust definition required')
    names = {key: columns.get(key, key) for key in ('lat_idx', 'lon_idx', 'latitude', 'longitude', 'converged')}
    base_keys = ['lat_idx', 'lon_idx', 'latitude', 'longitude']
    required = [names[k] for k in base_keys]
    periods = np.asarray(cfg['return_periods'])
    gust = columns.get('gust', 'rp_{period}_3sec_gust')
    lower = columns.get('lower', 'rp_{period}_3sec_gust_lower')
    upper = columns.get('upper', 'rp_{period}_3sec_gust_upper')
    if layout == 'wide':
        required += [gust.format(period=int(rp)) for rp in periods]
    elif layout == 'long':
        required += [columns.get('return_period', 'return_period'), gust]
    else:
        raise ValueError('conus_layout must be wide or long')
    missing = set(required)-set(frame.columns)
    if missing:
        raise ValueError(f'Unsupported baseline schema; missing {sorted(missing)}; available columns: {list(frame.columns)}')
    frame = frame.rename(columns={v:k for k,v in names.items()})
    for key in ('lat_idx', 'lon_idx'):
        values = frame[key].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError(f'Invalid grid indices: {key}')
        frame[key] = frame[key].astype(np.int64)
    if (frame.lon_idx >= 65536).any():
        raise ValueError('Grid exceeds packed CADENCE ID capacity')
    if layout == 'long':
        rp_col = columns.get('return_period', 'return_period')
        frame = frame.loc[frame[rp_col].isin(periods)].copy()
        if frame.duplicated(['lat_idx', 'lon_idx', rp_col]).any():
            raise ValueError('Duplicate grid/return-period keys')
        if frame.groupby(['lat_idx', 'lon_idx'])[['latitude', 'longitude']].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Coordinates disagree across return periods')
        coords = frame.drop_duplicates(['lat_idx', 'lon_idx']).sort_values(['lat_idx', 'lon_idx'])
        index = pd.MultiIndex.from_frame(coords[['lat_idx', 'lon_idx']])
        def field(column):
            if column not in frame:
                return None
            return frame.pivot(index=['lat_idx', 'lon_idx'], columns=rp_col, values=column).reindex(index=index, columns=periods).to_numpy(dtype=float)
        values, lo, hi = field(gust), field(lower), field(upper)
        convergence = field('converged') if 'converged' in frame else None
    else:
        if frame.duplicated(['lat_idx', 'lon_idx']).any():
            raise ValueError('Duplicate grid indices')
        coords = frame.sort_values(['lat_idx', 'lon_idx'])
        values = coords[[gust.format(period=int(rp)) for rp in periods]].to_numpy(dtype=float)
        def bound(template):
            fields = [template.format(period=int(rp)) for rp in periods]
            if not all(c in coords for c in fields):
                return None
            return coords[fields].to_numpy(dtype=float)
        lo, hi = bound(lower), bound(upper)
        convergence = np.broadcast_to(coords['converged'].to_numpy()[:, None], values.shape) if 'converged' in coords else None
    if coords.empty:
        raise ValueError('Empty baseline')
    ny, nx = int(coords.lat_idx.max())+1, int(coords.lon_idx.max())+1
    ids = coords.lat_idx.to_numpy()*nx+coords.lon_idx.to_numpy()
    if len(coords) != ny*nx or not np.array_equal(ids, np.arange(ny*nx)):
        raise ValueError('CONUS table must be a complete row-major grid')
    lat, lon = coords.latitude.to_numpy(), coords.longitude.to_numpy()
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or (abs(lat)>90).any() or (abs(lon)>180).any():
        raise ValueError('Invalid CONUS geographic coordinates')
    if coords.duplicated(['latitude', 'longitude']).any():
        raise ValueError('Duplicate geographic centers')
    flags = np.zeros(values.shape, dtype=np.uint16)
    flags[~np.isfinite(values) | (values <= 0)] |= POINT_INVALID
    decreasing = np.any(np.diff(values, axis=1) < 0, axis=1)
    flags[decreasing] |= CURVE_DECREASING
    if convergence is None:
        flags |= NOT_CONVERGED  # unknown is not treated as proven convergence
    else:
        flags[~np.isfinite(convergence) | (convergence != 1)] |= NOT_CONVERGED
    bounds_ok = np.zeros(values.shape, dtype=bool)
    if lo is not None and hi is not None:
        bounds_ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi >= lo)
    flags[~bounds_ok] |= INTERVAL_INVALID
    if not cfg.get('conus_historical_period'):
        flags |= PERIOD_UNCONFIRMED
    upstream = {}
    quality_columns = [c for c in frame.columns if 'flag' in c.lower() or 'quality' in c.lower()]
    for column in quality_columns:
        if layout == 'wide':
            upstream[column] = np.broadcast_to(coords[column].astype(str).to_numpy()[:,None],values.shape).copy()
        else:
            upstream[column] = frame.pivot(index=['lat_idx','lon_idx'],columns=rp_col,values=column).reindex(index=index,columns=periods).astype(str).to_numpy()
    result = dict(source_quality_columns=upstream, latitude=lat.reshape(ny,nx), longitude=lon.reshape(ny,nx),
                  baseline=values, baseline_quality_flags=flags,
                  units=units, wind_definition=definition)
    if lo is not None and hi is not None:
        result.update(baseline_lower=lo, baseline_upper=hi)
    report = dict(shape=[ny,nx], cells=len(coords), return_periods=periods.tolist(),
                  source_units=units, units_evidence='explicit configuration assertion; verify inspection report',
                  wind_definition=definition, historical_period=cfg.get('conus_historical_period'),
                  decreasing_cells=int(decreasing.sum()), invalid_point_estimates=int(((flags & POINT_INVALID)>0).sum()),
                  unusable_intervals=int((~bounds_ok).sum()),
                  upstream_quality_columns=quality_columns,
                  note='Source table preserved; flags recomputed from raw estimates/bounds. Unknown upstream flag meanings remain in source and must be reviewed.')
    return result, report

"""Read-only validation of the completed DCC NPZ aggregation schema."""
from __future__ import annotations
import calendar
import hashlib
import json
from pathlib import Path
import numpy as np

SCENARIO_YEARS = {'historical': (1980, 2019)}
for _forcing in ('rcp45cooler', 'rcp45hotter', 'rcp85cooler', 'rcp85hotter'):
    for _slice, _years in (('near', (2020, 2059)), ('far', (2060, 2099))):
        SCENARIO_YEARS[f'{_forcing}_{_slice}'] = _years


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def settings(path):
    cfg = json.loads(Path(path).read_text())
    cfg['settings_sha256'] = fingerprint(path)
    return cfg


def coordinates(path, shape):
    with np.load(path, allow_pickle=False) as d:
        lat, lon = d['latitude'], d['longitude']
    if lat.shape != tuple(shape) or lon.shape != lat.shape:
        raise ValueError('Coordinate shape does not match grid')
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError('Coordinates must be finite')
    if (abs(lat) > 90).any() or (abs(lon) > 180).any():
        raise ValueError('Coordinates must use degrees north/east in [-180,180]')
    return lat, lon


def validate_annual(path, scenario, cfg):
    start, end = SCENARIO_YEARS[scenario]
    with np.load(path, allow_pickle=False) as d:
        years, maxima, counts = d['years'], d['maxima'], d['valid_counts']
        totals = d['timestep_contributions']
        if str(d['scenario'].item()) != scenario:
            raise ValueError('Scenario metadata mismatch')
        if not np.array_equal(years, np.arange(start, end + 1)):
            raise ValueError('Years must exactly match scenario coverage')
        shape = (40, *cfg['grid_shape'])
        if maxima.shape != shape or counts.shape != shape or totals.shape != (40,):
            raise ValueError('Unexpected annual maxima/count shape')
        if counts.dtype.kind not in 'iu' or totals.dtype.kind not in 'iu':
            raise ValueError('Observation counts must be integers')
        expected = np.array([(366 if calendar.isleap(int(y)) else 365) * 8 for y in years])
        if (totals <= 0).any() or (totals > expected).any():
            raise ValueError('Invalid total observations; possible duplicate checkpoints')
        if (counts < 0).any() or (counts > totals[:, None, None]).any():
            raise ValueError('Invalid per-cell observation counts')
        finite = np.isfinite(maxima)
        if np.isinf(maxima).any() or (maxima[finite] < 0).any():
            raise ValueError('Wind maxima must be nonnegative or NaN')
        if np.any(finite != (counts > 0)):
            raise ValueError('Finite maxima must correspond to positive observation counts')
        metadata = dict(cfg.get('input_metadata', {}))
        for key in ('units', 'source_variable', 'temporal_resolution'):
            if key in d:
                metadata[key] = str(d[key].item())
        if metadata != {'units': 'm s-1', 'source_variable': 'WSPD', 'temporal_resolution': '3-hourly'}:
            raise ValueError('Require verified WSPD m s-1, 3-hourly metadata')
        eligible = finite & (counts / expected[:, None, None] >= cfg['min_coverage'])
        cube = np.where(eligible, maxima, np.nan)
    report = dict(scenario=scenario, shape=list(shape), years=[start, end],
                  source_sha256=fingerprint(path), metadata=metadata,
                  metadata_source='NPZ fields with configured legacy fallback',
                  missing_values=int((~finite).sum()), excluded_low_coverage=int((finite & ~eligible).sum()),
                  eligible_years_min=int(eligible.sum(axis=0).min()),
                  timestep_contributions=totals.tolist())
    return cube, report

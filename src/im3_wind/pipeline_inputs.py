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
    for key in ('chunk_cells', 'target_chunk_cells'):
        value = cfg.get(key, 4096)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < cfg['min_coverage'] <= 1 or not 3 <= cfg['min_years'] <= 40:
        raise ValueError('Invalid coverage/min_years')
    if not isinstance(cfg['bootstrap'], int) or cfg['bootstrap'] < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    if len(cfg['grid_shape']) != 2 or any(not isinstance(v, int) or v < 1 for v in cfg['grid_shape']):
        raise ValueError('grid_shape must contain two positive integers')
    periods = np.asarray(cfg['return_periods'], dtype=float)
    if periods.ndim != 1 or not len(periods) or not np.all(np.isfinite(periods) & (periods > 1)) or not np.all(np.diff(periods) > 0):
        raise ValueError('return_periods must increase and exceed 1')
    cfg['settings_sha256'] = fingerprint(path)
    return cfg


def grid_path(cfg, scenario):
    return cfg.get('scenario_grid_files', {}).get(scenario) or cfg['grid_file']


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
        coverage_path = cfg.get('coverage_sidecars', {}).get(scenario)
        coverage_report = {'method': 'raw_contribution_counts', 'duplicate_equality_verified': False}
        duplicates = np.zeros(40, dtype=np.int64)
        if coverage_path:
            from .products import read_product
            evidence, provenance = read_product(coverage_path)
            if provenance.get('evidence_type') != 'checkpoint_timestamp_union_v1' or provenance.get('annual_sha256') != fingerprint(path):
                raise ValueError('Coverage evidence is stale or unsupported')
            if not np.array_equal(evidence['years'], years) or not np.array_equal(evidence['timestep_contributions'], totals):
                raise ValueError('Coverage evidence years/counts mismatch')
            unique, duplicates = evidence['unique_timestep_counts'], evidence['duplicate_contributions']
            if unique.shape != (40,) or duplicates.shape != (40,) or unique.dtype.kind not in 'iu' or duplicates.dtype.kind not in 'iu':
                raise ValueError('Invalid coverage evidence shape/type')
            if (duplicates < 0).any() or (unique <= 0).any() or (unique > expected).any() or not np.array_equal(unique+duplicates, totals):
                raise ValueError('Invalid unique-observation coverage evidence')
            coverage_report.update(method='conservative_unique_finite_lower_bound',
                                   sidecar_sha256=fingerprint(coverage_path),
                                   unique_timestep_counts=unique.tolist(),
                                   duplicate_contributions=duplicates.tolist())
        if (totals <= 0).any() or (not coverage_path and (totals > expected).any()):
            raise ValueError('Invalid total observations; possible duplicate checkpoints; require audited coverage sidecar')
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
        if metadata.get('units') != 'm s-1' or metadata.get('temporal_resolution') != '3-hourly' or metadata.get('source_variable') not in ('WSPD', 'hypot(U10,V10)'):
            raise ValueError('Require verified wind metadata: WSPD or hypot(U10,V10), m s-1, 3-hourly')
        finite_lower = np.maximum(counts.astype(np.int64) - duplicates[:, None, None], 0)
        eligible = finite & (finite_lower / expected[:, None, None] >= cfg['min_coverage'])
        cube = np.where(eligible, maxima, np.nan)
    report = dict(scenario=scenario, shape=list(shape), years=[start, end],
                  source_sha256=fingerprint(path), metadata=metadata,
                  metadata_source='NPZ fields with configured legacy fallback',
                  missing_values=int((~finite).sum()), excluded_low_coverage=int((finite & ~eligible).sum()),
                  eligible_years_min=int(eligible.sum(axis=0).min()),
                  timestep_contributions=totals.tolist(), coverage=coverage_report)
    return cube, report

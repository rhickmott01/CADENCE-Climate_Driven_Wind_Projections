"""Geographic delta transfer; no conversion between wind averaging times."""
import numpy as np
from scipy.spatial import cKDTree
from .spatial import lonlat_to_unit_xyz

OUTSIDE = 512
BASELINE_INVALID = 1024


def geographic_map(source_lat, source_lon, target_lat, target_lon, max_distance_km):
    if not np.isfinite(max_distance_km) or max_distance_km <= 0:
        raise ValueError('max_distance_km must be positive')
    for lat, lon in ((source_lat, source_lon), (target_lat, target_lon)):
        if np.shape(lat) != np.shape(lon) or not np.isfinite(lat).all() or not np.isfinite(lon).all():
            raise ValueError('Invalid geographic coordinates')
        if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
            raise ValueError('Geographic coordinates out of range')
    xyz = lonlat_to_unit_xyz(source_lat, source_lon)
    if not len(xyz) or len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValueError('Source grid must contain unique geographic cells')
    tree = cKDTree(xyz)
    chord, index = tree.query(lonlat_to_unit_xyz(target_lat, target_lon), k=min(4, len(xyz)))
    if chord.ndim == 1:
        chord, index = chord[:, None], index[:, None]
    distance = 6371.0088 * 2 * np.arcsin(np.clip(chord / 2, 0, 1))
    weights = 1 / np.maximum(distance, 1e-9)**2
    weights[distance > max_distance_km] = 0
    exact = distance[:, 0] < 1e-8
    weights[exact] = 0
    weights[exact, 0] = 1
    np.divide(weights, weights.sum(axis=1, keepdims=True), out=weights,
              where=weights.sum(axis=1, keepdims=True) > 0)
    return dict(source_index=index, distance_km=distance, weights=weights,
                inside=distance[:, 0] <= max_distance_km)


def transfer(factors, mapping, baseline):
    """Nearest-neighbor primary, 4-neighbor IDW sensitivity; no NaN renormalizing."""
    idx = mapping['source_index']
    nearest = idx[:, 0]
    factor = factors['factor'][nearest].copy()
    flags = factors['flags'][nearest].copy()
    outside = ~mapping['inside']
    factor[outside] = np.nan
    flags[outside] |= OUTSIDE
    weights = mapping['weights'][:, :, None]
    neighbor = factors['factor'][idx]
    used = weights > 0
    idw = np.sum(np.where(used, neighbor, 0) * weights, axis=1)
    idw[outside] = np.nan
    # Any contributing invalid factor leaves IDW sensitivity missing.
    idw[~np.all(np.isfinite(neighbor) | ~used, axis=1)] = np.nan
    idw_flags = np.bitwise_or.reduce(np.where(used, factors['flags'][idx], 0), axis=1)
    idw_flags[outside] |= OUTSIDE
    baseline = np.asarray(baseline)
    if baseline.shape != factor.shape:
        raise ValueError('Baseline must have shape (target cell, return period)')
    bad = ~np.isfinite(baseline) | (baseline <= 0)
    flags[bad] |= BASELINE_INVALID
    future = baseline * factor
    future[bad] = np.nan
    lower = baseline * factors['lower'][nearest]
    upper = baseline * factors['upper'][nearest]
    lower[bad | outside[:, None]] = np.nan
    upper[bad | outside[:, None]] = np.nan
    factor_lower, factor_upper = factors['lower'][nearest].copy(), factors['upper'][nearest].copy()
    factor_lower[outside] = np.nan
    factor_upper[outside] = np.nan
    return dict(factor=factor, lower=factor_lower,
                upper=factor_upper, flags=flags,
                baseline=baseline, future_wind=future, future_lower=lower, future_upper=upper,
                idw_factor=idw, idw_flags=idw_flags, mapping_sensitivity=idw-factor,
                source_cell_id=factors['cell_id'][nearest], distance_km=mapping['distance_km'][:, 0])


def load_conus(path, periods):
    from .pipeline_inputs import coordinates
    with np.load(path, allow_pickle=False) as d:
        shape = d['latitude'].shape
        if len(shape) != 2:
            raise ValueError('CONUS coordinates must be 2-D')
        lat, lon = coordinates(path, shape)
        if not np.array_equal(d['return_periods'], periods):
            raise ValueError('CONUS return periods do not match')
        baseline = d['return_levels']
        if baseline.shape != (lat.size, len(periods)):
            raise ValueError('CONUS return_levels must be (flattened cell, return period)')
        units, definition = str(d['units'].item()), str(d['wind_definition'].item())
        if units not in ('m s-1', 'mph') or not definition.strip():
            raise ValueError('CONUS requires explicit wind definition and supported units')
        bounds = {}
        for key in ('baseline_lower', 'baseline_upper'):
            if key in d:
                if d[key].shape != baseline.shape:
                    raise ValueError('Baseline uncertainty shape mismatch')
                bounds[key] = d[key]
        if len(bounds) == 1:
            raise ValueError('Both baseline uncertainty bounds are required')
    return dict(latitude=lat, longitude=lon, baseline=baseline,
                units=units, wind_definition=definition, **bounds)

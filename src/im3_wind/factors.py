"""Return-period-specific IM3 change factors; preserve every cell and flag."""
import numpy as np

INVALID_RATIO = 256


def climate_factors(historical, future):
    for key in ('cell_id', 'return_periods', 'latitude', 'longitude'):
        if historical[key].shape != future[key].shape or not np.array_equal(historical[key], future[key]):
            raise ValueError(f'Historical/future spatial or return-period mismatch: {key}')
    h, f = historical['return_levels'], future['return_levels']
    if h.shape != f.shape:
        raise ValueError('Return-level shapes differ')
    valid = np.isfinite(h) & np.isfinite(f) & (h > 0) & (f > 0)
    ratio = np.full(h.shape, np.nan)
    np.divide(f, h, out=ratio, where=valid)
    flags = historical['flags'].astype(np.uint16) | future['flags'].astype(np.uint16)
    flags[~valid] |= INVALID_RATIO
    # Conservative marginal-interval envelope, NOT a calibrated 95% ratio CI.
    lo, hi = np.full(h.shape, np.nan), np.full(h.shape, np.nan)
    bounds_ok = valid.copy()
    for d in (historical, future):
        bounds_ok &= np.isfinite(d['lower']) & np.isfinite(d['upper']) & (d['lower'] > 0) & (d['upper'] >= d['lower'])
    np.divide(future['lower'], historical['upper'], out=lo, where=bounds_ok)
    np.divide(future['upper'], historical['lower'], out=hi, where=bounds_ok)
    return dict(factor=ratio, lower=lo, upper=hi, flags=flags,
                **{k: historical[k] for k in ('cell_id', 'return_periods', 'latitude', 'longitude')})

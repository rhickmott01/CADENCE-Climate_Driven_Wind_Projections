"""Stationary GEV MLE with explicit optimization status and tail diagnostics.

SciPy c = -xi. Bounds are numerical safeguards, not physical prior knowledge.
Intervals are iid year-resampling percentile bootstrap intervals, conditional
on the selected model; they omit climate-model and structural uncertainty.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme

# Per-return-level bitmask; multiple conditions can coexist.
FAILED = 1
EXTRAPOLATED = 2
BOOTSTRAP_POOR = 4
WIDE_INTERVAL = 8
TAIL_MISFIT = 16
BOUNDARY = 32
TOP_SENSITIVE = 64
NONPOSITIVE = 128


def _mle(values, starts=(0., -.2, .2)):
    center, spread = np.mean(values), np.std(values)
    if spread <= 1e-8 or not np.isfinite(spread):
        return None
    z = (values - center) / spread

    def objective(p):
        ll = genextreme.logpdf(z, p[0], loc=p[1], scale=np.exp(p[2]))
        return -ll.sum() if np.isfinite(ll).all() else 1e100

    best = None
    for c in starts:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fit = minimize(objective, [c, -.4, -.25], method='Nelder-Mead',
                           bounds=[(-1., 1.), (-10., 10.), (-7., 5.)],
                           options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7})
        if fit.success and np.isfinite(fit.fun) and fit.fun < 1e99:
            if best is None or fit.fun < best.fun:
                best = fit
    if best is None:
        return None
    c, loc, logscale = best.x
    p = np.array([c, center + spread * loc, spread * np.exp(logscale)])
    if not np.isfinite(genextreme.logpdf(values, *p)).all():
        return None
    boundary = abs(c) > .98 or abs(loc) > 9.8 or logscale < -6.8 or logscale > 4.8
    return p, bool(boundary)


def fit_cell(series, periods, min_years=30, bootstrap=200, seed=1729):
    periods = np.asarray(periods, dtype=float)
    if periods.ndim != 1 or not np.all(np.isfinite(periods) & (periods > 1)) or not np.all(np.diff(periods) > 0):
        raise ValueError('Return periods must be finite, increasing and > 1')
    if bootstrap < 0 or min_years < 3:
        raise ValueError('Invalid bootstrap/min_years')
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    m = len(periods)
    result = dict(parameters=np.full(3, np.nan), return_levels=np.full(m, np.nan),
                  lower=np.full(m, np.nan), upper=np.full(m, np.nan),
                  flags=np.full(m, FAILED, dtype=np.uint16), n_years=len(v),
                  converged=False, bootstrap_success=0, tail_rmse=np.nan,
                  anderson_darling=np.nan, top_sensitivity=np.full(m, np.nan))
    if len(v) < min_years:
        return result
    fit = _mle(v)
    if fit is None:
        return result
    p, boundary = fit
    levels = genextreme.isf(1 / periods, *p)
    result.update(parameters=p, return_levels=levels, converged=True)
    flags = np.where(periods > len(v), EXTRAPOLATED, 0).astype(np.uint16)
    if boundary or p[0] >= .5:  # xi <= -.5: nonregular bounded-tail MLE
        flags |= BOUNDARY
    flags[~np.isfinite(levels) | (levels <= 0)] |= NONPOSITIVE
    ordered = np.sort(v)
    plotting = (np.arange(1, len(v) + 1) - .5) / len(v)
    predicted = genextreme.ppf(plotting, *p)
    tail = plotting >= .8
    tail_rmse = np.sqrt(np.mean((ordered[tail] - predicted[tail])**2)) / p[2]
    u = np.clip(genextreme.cdf(ordered, *p), 1e-12, 1 - 1e-12)
    ad = -len(v) - np.mean((2*np.arange(1, len(v)+1)-1)*(np.log(u)+np.log1p(-u[::-1])))
    result.update(tail_rmse=float(tail_rmse), anderson_darling=float(ad))
    if tail_rmse > .5:
        flags |= TAIL_MISFIT
    without_top = _mle(ordered[:-1])
    if without_top is not None:
        sensitivity = np.abs(genextreme.isf(1 / periods, *without_top[0]) / levels - 1)
        result['top_sensitivity'] = sensitivity
        flags[sensitivity > .25] |= TOP_SENSITIVE
    else:
        flags |= TOP_SENSITIVE
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        boot = _mle(rng.choice(v, len(v), replace=True), starts=(0.,))
        if boot is not None and not boot[1]:
            q = genextreme.isf(1 / periods, *boot[0])
            if np.all(np.isfinite(q) & (q > 0)):
                draws.append(q)
    result['bootstrap_success'] = len(draws)
    if len(draws) >= max(20, int(np.ceil(.8 * bootstrap))):
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        result.update(lower=lo, upper=hi)
        flags[(hi - lo) / np.maximum(levels, 1e-12) > .5] |= WIDE_INTERVAL
    else:
        flags |= BOOTSTRAP_POOR
    result['flags'] = flags
    return result


def fit_block(cube, periods, start_cell=0, **kwargs):
    """One bounded-size block, deterministic independent seeds per global cell."""
    seed = kwargs.pop('seed', 1729)
    rows = [fit_cell(cube[:, i], periods, seed=np.random.SeedSequence([seed, start_cell+i]), **kwargs)
            for i in range(cube.shape[1])]
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}

import calendar
import numpy as np
import pytest
from im3_wind.pipeline_inputs import validate_annual


def annual(tmp_path, **changes):
    years = np.arange(1980, 2020)
    totals = np.array([(366 if calendar.isleap(int(y)) else 365)*8 for y in years])
    data = dict(years=years, maxima=np.full((40, 1, 2), 20.),
                valid_counts=np.broadcast_to(totals[:, None, None], (40, 1, 2)).copy(),
                timestep_contributions=totals, scenario=np.array('historical'))
    data.update(changes)
    p = tmp_path / 'annual.npz'
    np.savez(p, **data)
    return p


CFG = dict(grid_shape=[1, 2], min_coverage=.95, input_metadata=dict(
    units='m s-1', source_variable='WSPD', temporal_resolution='3-hourly'))


def test_validation(tmp_path):
    cube, report = validate_annual(annual(tmp_path), 'historical', CFG)
    assert cube.shape == (40, 1, 2)
    assert report['eligible_years_min'] == 40
    with pytest.raises(ValueError, match='Years'):
        validate_annual(annual(tmp_path, years=np.arange(1981, 2021)), 'historical', CFG)
    with pytest.raises(ValueError, match='metadata'):
        validate_annual(annual(tmp_path, units=np.array('mph')), 'historical', CFG)


def test_counts_and_missing(tmp_path):
    maxima = np.full((40, 1, 2), 20.)
    maxima[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        validate_annual(annual(tmp_path, maxima=maxima), 'historical', CFG)
    with pytest.raises(ValueError, match='duplicate'):
        validate_annual(annual(tmp_path, timestep_contributions=np.full(40, 4000)), 'historical', CFG)

from scipy.stats import genextreme
from im3_wind.extremes import fit_cell, fit_block, FAILED, EXTRAPOLATED, BOOTSTRAP_POOR


def test_gev_known_gumbel_and_failures():
    # Deterministic distribution quantiles avoid a flaky random accuracy test.
    values = genextreme.ppf((np.arange(40)+.5)/40, 0, loc=25, scale=3)
    r = fit_cell(values, [10, 100, 500], bootstrap=0)
    assert r['converged']
    assert np.all(np.diff(r['return_levels']) > 0)
    assert abs(r['return_levels'][0] / genextreme.isf(.1, 0, 25, 3) - 1) < .1
    assert r['flags'][1] & EXTRAPOLATED
    assert r['flags'][0] & BOOTSTRAP_POOR
    assert fit_cell(np.ones(40), [10], bootstrap=0)['flags'][0] & FAILED
    assert not fit_cell([1, 2, np.nan], [10], bootstrap=0)['converged']
    with pytest.raises(ValueError):
        fit_cell(values, [1])


def test_bootstrap_reproducible():
    values = genextreme.rvs(0, loc=25, scale=3, size=40, random_state=2)
    a = fit_block(values[:, None], [10, 100], bootstrap=25)
    b = fit_block(values[:, None], [10, 100], bootstrap=25)
    np.testing.assert_equal(a, b)
    assert a['bootstrap_success'][0] >= 20
    assert np.all(a['lower'] <= a['upper'])

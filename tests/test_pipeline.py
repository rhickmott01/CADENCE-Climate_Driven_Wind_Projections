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

from im3_wind.factors import climate_factors, INVALID_RATIO


def hazard(values):
    v = np.array([values], dtype=float)
    return dict(return_levels=v, lower=v*.9, upper=v*1.1,
                flags=np.zeros(v.shape, dtype=np.uint16), cell_id=np.array([0]),
                latitude=np.array([35.]), longitude=np.array([-80.]),
                return_periods=np.array([10, 100]))


def test_factors_preserve_bad_cells_and_coordinates():
    h, f = hazard([20, 0]), hazard([24, 30])
    h['flags'][0, 0] = EXTRAPOLATED
    r = climate_factors(h, f)
    assert r['factor'][0, 0] == 1.2
    assert r['lower'][0, 0] < 1.2 < r['upper'][0, 0]
    assert r['flags'][0, 0] & EXTRAPOLATED
    assert np.isnan(r['factor'][0, 1])
    assert r['flags'][0, 1] & INVALID_RATIO
    f['longitude'][0] += .1
    with pytest.raises(ValueError, match='mismatch'):
        climate_factors(h, f)

from im3_wind.downscale import geographic_map, transfer, OUTSIDE


def test_geographic_downscaling_and_sensitivity():
    f = climate_factors(hazard([20, 30]), hazard([24, 36]))
    # Second source has a larger signal; test exact and midpoint behavior.
    f = {k: (np.concatenate([v, v], axis=0) if k != 'return_periods' else v) for k, v in f.items()}
    f['factor'][1] = 1.4
    mapping = geographic_map(np.array([35., 35.]), np.array([-80., -79.8]),
                             np.array([35., 35., 0.]), np.array([-80., -79.9, 0.]), 30)
    r = transfer(f, mapping, np.full((3, 2), 50.))
    np.testing.assert_allclose(r['future_wind'][0], 60)
    np.testing.assert_allclose(r['idw_factor'][0], 1.2)
    np.testing.assert_allclose(r['idw_factor'][1], 1.3, atol=1e-6)
    assert np.isnan(r['future_wind'][2]).all()
    assert np.all(r['flags'][2] & OUTSIDE)
    # A constant field must survive either spatial mapping.
    f['factor'][:] = 1.1
    r = transfer(f, mapping, np.full((3, 2), 50.))
    np.testing.assert_allclose(r['mapping_sensitivity'][:2], 0, atol=1e-12)

from im3_wind.exports import annual_transition, annual_frames, compatibility_frame, write_frames, parquet_resume


def test_annual_transition_and_export(tmp_path):
    n, f = np.array([1.2]), np.array([1.5])
    np.testing.assert_allclose(annual_transition(n, f, 2039.5), n)
    np.testing.assert_allclose(annual_transition(n, f, 2079.5), f)
    assert 1.2 < annual_transition(n, f, 2060)[0] < 1.5
    assert np.isnan(annual_transition([-1.], [1.], 2026)[0])
    near = dict(factor=np.array([[1.2, 1.3]]), lower=np.array([[1.1, 1.2]]),
                upper=np.array([[1.3, 1.4]]), baseline=np.array([[50., 60.]]),
                flags=np.zeros((1, 2), dtype=np.uint16), return_periods=np.array([10, 100]),
                wind_grid_id=np.array([65538]), lat_idx=np.array([1]), lon_idx=np.array([2]),
                latitude=np.array([35.]), longitude=np.array([-80.]), source_cell_id=np.array([0]))
    far = dict(near, factor=near['factor']*1.2)
    meta = dict(units='mph', wind_definition='3-second gust')
    frames = list(annual_frames(near, far, 'rcp45cooler', meta))
    assert len(frames) == 35
    assert frames[0].year.iloc[0] == 2026 and frames[-1].year.iloc[0] == 2060
    assert len(frames[0]) == 2
    path = tmp_path / 'export.parquet'
    write_frames(path, iter(frames), meta)
    assert parquet_resume(path, meta)
    with pytest.raises(FileExistsError):
        write_frames(path, iter(frames), meta)
    compat = compatibility_frame(near, far, 100)
    assert 'scale_2025' in compat and 'scale_2060' in compat
    far['flags'] = np.full((1, 2), EXTRAPOLATED, dtype=np.uint16)
    with pytest.raises(ValueError, match='Unaccepted'):
        compatibility_frame(near, far, 100)
    assert len(compatibility_frame(near, far, 100, allow_flags=EXTRAPOLATED)) == 1

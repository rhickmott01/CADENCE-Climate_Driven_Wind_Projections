import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset, stringtochar

from im3_wind import evidence
from im3_wind.baseline import INTERVAL_INVALID, inspect_table
from im3_wind.pipeline_inputs import fingerprint


def test_select_sources_uses_inventory_recorded_filename_and_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, 'SCENARIO_YEARS', {'historical': (2000, 2000), 'rcp45cooler': (2040, 2050)})
    coverage = tmp_path / 'coverage'
    coverage.mkdir()
    inventory = {'checkpoints': [
        {'first': '2040-01-01T00:00:00', 'last': '2040-01-07T21:00:00', 'source_file': 'week-a.nc'},
        {'first': '2040-01-08T00:00:00', 'last': '2040-01-14T21:00:00', 'source_file': 'week-b.nc'},
    ]}
    inv_path = coverage / 'rcp45cooler.inventory.json'
    inv_path.write_text(json.dumps(inventory))
    result = evidence.select_sources({}, tmp_path)
    assert result == [dict(scenario='rcp45cooler',
                           source_path='/rcp45cooler_2040_2050/three_hourly/week-a.nc',
                           filename='week-a.nc', first='2040-01-01T00:00:00',
                           last='2040-01-07T21:00:00', inventory_sha256=fingerprint(inv_path))]
    inventory['checkpoints'][0]['source_file'] = '../outside.nc'
    inv_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='Unsafe'):
        evidence.select_sources({}, tmp_path)


def _grid(path):
    lat = np.array([[35., 35.1], [35.2, 35.3]])
    lon = np.array([[-80., -79.9], [-80.1, -80.]])
    np.savez(path, latitude=lat, longitude=lon)
    return lat, lon


def _source(path, lat, lon, times=('2040-01-01_00:00:00', '2040-01-01_03:00:00')):
    with Dataset(path, 'w') as ds:
        ds.createDimension('Time', len(times)); ds.createDimension('DateStrLen', 19)
        ds.createDimension('y', 2); ds.createDimension('x', 2)
        ds.createVariable('Times', 'S1', ('Time', 'DateStrLen'))[:] = stringtochar(np.array(times, dtype='S19'))
        for name, data in (('XLAT', lat), ('XLONG', lon)):
            ds.createVariable(name, 'f8', ('Time', 'y', 'x'))[:] = np.broadcast_to(data, (len(times), 2, 2))
        for name in ('U10', 'V10'):
            var = ds.createVariable(name, 'f4', ('Time', 'y', 'x'))
            var.units = 'm s-1'; var[:] = 5


def test_inspect_grid_checks_times_grid_and_writes_metadata(tmp_path):
    grid = tmp_path / 'grid.npz'
    lat, lon = _grid(grid)
    source = tmp_path / 'source.nc'
    _source(source, lat, lon)
    request = dict(scenario='rcp45cooler', source_path='/remote/week.nc',
                   first='2040-01-01T00:00:00', last='2040-01-01T03:00:00', inventory_sha256='inventory-hash')
    cfg = dict(grid_file=str(grid), grid_shape=[2, 2])
    out = tmp_path / 'grid-evidence.npz'
    meta = evidence.inspect_grid(source, request, cfg, out)
    assert meta['coordinate_time_slices_checked'] == 2
    assert meta['inventory_sha256'] == 'inventory-hash'
    assert meta['duplicate_wind_equality_verified'] is False
    with np.load(out) as saved:
        np.testing.assert_array_equal(saved['latitude'], lat)
        np.testing.assert_array_equal(saved['longitude'], lon)

    bad = tmp_path / 'bad.nc'
    _source(bad, lat + 1, lon)
    with pytest.raises(ValueError, match='XLAT differs'):
        evidence.inspect_grid(bad, request, cfg, tmp_path / 'bad-out.npz')
    bad_time = tmp_path / 'bad-time.nc'
    _source(bad_time, lat, lon, ('2040-01-01_00:00:00', '2040-01-01_06:00:00'))
    with pytest.raises(ValueError, match='differ|contiguous'):
        evidence.inspect_grid(bad_time, request, cfg, tmp_path / 'bad-time-out.npz')


def test_collect_plan_only_is_bounded_and_never_submits_transfer(tmp_path, monkeypatch):
    prior = tmp_path / 'prior'; (prior / 'coverage').mkdir(parents=True)
    (prior / 'report.json').write_text(json.dumps({'run_dir': str(prior), 'scenarios': {'historical': {'validation': {'source_sha256': 'valid'}}, 'rcp45cooler': {'validation': {'source_sha256': 'valid'}}}}))
    annual = tmp_path / 'annual'; annual.mkdir(); (annual / 'rcp45cooler.npz').write_bytes(b'x')
    transfer_cfg = tmp_path / 'transfer.json'; transfer_cfg.write_text(json.dumps({'source_collection': 'source-id'}))
    cfg = {'annual_root': str(annual), 'future_evidence': {'source_pipeline_config': str(transfer_cfg), 'max_files': 1,
           'max_bytes': 1000, 'staging_root': str(tmp_path / 'staging')}, 'results_root': str(tmp_path / 'results')}
    monkeypatch.setattr(evidence, 'settings', lambda _: cfg)
    monkeypatch.setattr(evidence, 'fingerprint', lambda _: 'valid')
    monkeypatch.setattr(evidence, 'SCENARIO_YEARS', {'historical': (2000, 2000), 'rcp45cooler': (2040, 2050)})
    monkeypatch.setattr(evidence, 'select_sources', lambda *_: [dict(scenario='rcp45cooler', source_path='/remote/week.nc', filename='week.nc')])
    calls = []
    def fake_cli(*args, **kwargs):
        calls.append(args)
        return {'DATA': [{'name': 'week.nc', 'type': 'file', 'size': '123'}]}
    monkeypatch.setattr(evidence, 'cli', fake_cli)
    report, root = evidence.collect('ignored', prior, output=tmp_path / 'out')
    assert calls == [('ls', 'source-id:/remote/', '--format', 'json')]
    assert report['requested_bytes'] == 123
    assert any('Plan only' in item for item in report['issues'])
    assert report['production_submitted'] is False
    assert not (tmp_path / 'staging').exists()
    assert json.loads((root / 'transfer_plan.json').read_text())['requested_bytes'] == 123

    # Exceeding the reviewed file cap stops before even catalog discovery.
    calls.clear()
    cfg['future_evidence']['max_files'] = 0
    with pytest.raises(ValueError, match='transfer limit'):
        evidence.collect('ignored', prior, output=tmp_path / 'out-over-cap')
    assert calls == []


def test_baseline_recovers_bounds_from_immutable_csv_and_checks_qa(tmp_path):
    frame = pd.DataFrame(dict(lat_idx=[0, 0, 1, 1], lon_idx=[0, 1, 0, 1],
                              latitude=[35., 35., 35.1, 35.1], longitude=[-80., -79.9, -80., -79.9], converged=1))
    for rp, gust in ((10, 30.), (100, 45.)):
        frame[f'rp_{rp}_3sec_gust'] = gust
        frame[f'rp_{rp}_3sec_gust_lower'] = gust * .8
        frame[f'rp_{rp}_3sec_gust_upper'] = gust * 1.2
    frame['qa_nonmonotone'] = False
    frame['qa_nonpositive'] = False
    frame['qa_nonfinite'] = False
    frame['qa_invalid_coordinates'] = False
    frame['qa_negative_lower_bound'] = False
    frame['qa_point_estimates_usable'] = True
    raw = tmp_path / 'immutable.csv'; frame.to_csv(raw, index=False)
    derived = frame.drop(columns=[c for c in frame if c.endswith('_lower') or c.endswith('_upper')])
    parquet = tmp_path / 'derived.parquet'; derived.to_parquet(parquet, index=False)
    cfg = dict(return_periods=[10, 100], conus_units='m s-1', conus_wind_definition='3-second gust',
               conus_bounds_csv=str(raw), conus_raw_sha256=fingerprint(raw))
    result, report = inspect_table(parquet, cfg)
    assert result['baseline_lower'].shape == (4, 2)
    np.testing.assert_allclose(result['baseline_lower'][0], [24, 36])
    assert report['bounds_source']['sha256'] == fingerprint(raw)
    assert report['missing_intervals'] == 0

    mismatched = derived.copy(); mismatched.loc[0, 'qa_point_estimates_usable'] = False
    mismatched.to_parquet(parquet, index=False)
    with pytest.raises(ValueError, match='qa_point_estimates_usable disagrees'):
        inspect_table(parquet, cfg)

    bad_hash_cfg = dict(cfg, conus_raw_sha256='incorrect')
    with pytest.raises(ValueError, match='checksum mismatch'):
        inspect_table(derived.drop(columns=[c for c in derived if c.startswith('qa_')]).pipe(lambda df: df.to_parquet(tmp_path / 'noqa.parquet', index=False)) or tmp_path / 'noqa.parquet', bad_hash_cfg)

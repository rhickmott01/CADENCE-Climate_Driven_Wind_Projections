import calendar
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset, stringtochar
from im3_wind.coverage import checkpoint_coverage, write_coverage
from im3_wind.pipeline_inputs import settings, validate_annual, fingerprint, SCENARIO_YEARS
from im3_wind.baseline import inspect_table, CURVE_DECREASING, POINT_INVALID, INTERVAL_INVALID, PERIOD_UNCONFIRMED
from im3_wind.pilot import run_pilot
from im3_wind.manual import plan


def checkpoints(root, scenario, years, duplicate=False):
    folder=root/scenario
    folder.mkdir(parents=True)
    start=np.datetime64(f'{years[0]}-01-01T00:00:00','s')
    end=np.datetime64(f'{years[-1]+1}-01-01T00:00:00','s')-np.timedelta64(3,'h')
    counts=np.array([(366 if calendar.isleap(int(y)) else 365)*8 for y in years])
    n=int(counts.sum())
    np.savez(folder/'whole.npz',first_time=str(start),last_time=str(end),n_timesteps=n,years=years,counts=counts,scenario=scenario)
    if duplicate:
        np.savez(folder/'duplicate.npz',first_time=str(start),last_time=str(start),n_timesteps=1,years=years[:1],counts=np.array([1]),scenario=scenario)
    return folder


def test_coverage_uses_timestamp_union_and_binds_to_input(tmp_path):
    years=np.arange(1980,2020)
    folder=checkpoints(tmp_path/'checkpoints','historical',years,True)
    arrays,meta=checkpoint_coverage(folder,years)
    assert arrays['duplicate_contributions'][0]==1
    assert arrays['unique_timestep_counts'][0]==2928
    totals=arrays['timestep_contributions']
    annual=tmp_path/'annual.npz'
    np.savez(annual,years=years,maxima=np.full((40,1,1),20.),
             valid_counts=totals[:,None,None],timestep_contributions=totals,scenario='historical')
    side=tmp_path/'coverage.npz'
    write_coverage(annual,folder,side)
    cfg=dict(grid_shape=[1,1],min_coverage=.95,input_metadata=dict(units='m s-1',source_variable='hypot(U10,V10)',temporal_resolution='3-hourly'))
    with pytest.raises(ValueError,match='duplicate'):
        validate_annual(annual,'historical',cfg)
    cube,report=validate_annual(annual,'historical',dict(cfg,coverage_sidecars={'historical':str(side)}))
    assert np.isfinite(cube).all()
    assert report['coverage']['method']=='conservative_unique_finite_lower_bound'
    with np.load(annual) as d:
        changed={k:d[k] for k in d.files}
    changed['maxima'][:]=21
    np.savez(annual,**changed)
    with pytest.raises(ValueError,match='stale'):
        validate_annual(annual,'historical',dict(cfg,coverage_sidecars={'historical':str(side)}))


def test_malformed_checkpoint_and_mismatched_aggregation_rejected(tmp_path):
    folder=checkpoints(tmp_path/'checkpoints','historical',np.arange(1980,2020))
    with np.load(folder/'whole.npz') as d:
        fields={k:d[k] for k in d.files}
    fields['n_timesteps']=1
    np.savez(folder/'whole.npz',**fields)
    with pytest.raises(ValueError,match='range'):
        checkpoint_coverage(folder,np.arange(1980,2020))


def wide_baseline(periods=(10,25,50,100,200,500)):
    frame=pd.DataFrame(dict(lat_idx=[0,0,1,1],lon_idx=[0,1,0,1],
                            latitude=[35.,35.,35.1,35.1],longitude=[-80.,-79.9,-80.,-79.9],converged=1))
    for i,rp in enumerate(periods):
        v=30.+i*5
        frame[f'rp_{rp}_3sec_gust']=v
        frame[f'rp_{rp}_3sec_gust_lower']=v*.8
        frame[f'rp_{rp}_3sec_gust_upper']=v*1.2
    return frame


def test_baseline_quarantines_bad_curves_without_changing_raw(tmp_path):
    frame=wide_baseline()
    frame.loc[0,'rp_500_3sec_gust']=-2
    frame.loc[1,'rp_100_3sec_gust_lower']=-100
    path=tmp_path/'baseline.parquet'
    frame.to_parquet(path,index=False)
    before=fingerprint(path)
    cfg=dict(return_periods=[10,25,50,100,200,500],conus_units='m s-1',conus_wind_definition='3-second gust')
    baseline,report=inspect_table(path,cfg)
    assert report['decreasing_cells']==1
    assert baseline['baseline'][0,-1]==-2
    assert baseline['baseline_quality_flags'][0,-1] & POINT_INVALID
    assert np.all(baseline['baseline_quality_flags'][0] & CURVE_DECREASING)
    assert baseline['baseline_quality_flags'][1,3] & INTERVAL_INVALID
    assert np.all(baseline['baseline_quality_flags'] & PERIOD_UNCONFIRMED)
    assert fingerprint(path)==before
    from im3_wind.downscale import transfer, geographic_map
    factors=dict(factor=np.full((4,6),1.1),lower=np.ones((4,6)),upper=np.full((4,6),1.2),
                 flags=np.zeros((4,6),dtype=np.uint16),cell_id=np.arange(4))
    mapping=geographic_map(baseline['latitude'],baseline['longitude'],baseline['latitude'],baseline['longitude'],30)
    mapped=transfer(factors,mapping,baseline['baseline'],baseline['baseline_quality_flags'])
    assert np.isnan(mapped['future_wind'][0]).all()
    assert np.isfinite(mapped['factor'][0]).all()
    assert np.isfinite(mapped['future_wind'][1]).all() # Bad interval does not erase valid point.


def test_baseline_long_schema_and_duplicate_keys(tmp_path):
    rows=[]
    for _,row in wide_baseline().iterrows():
        for rp in (10,100):
            rows.append(dict(lat_idx=row.lat_idx,lon_idx=row.lon_idx,latitude=row.latitude,longitude=row.longitude,
                             converged=1,return_period=rp,gust=row[f'rp_{rp}_3sec_gust']))
    path=tmp_path/'long.parquet'
    pd.DataFrame(rows).to_parquet(path,index=False)
    cfg=dict(return_periods=[10,100],conus_units='m s-1',conus_wind_definition='3-second gust',conus_layout='long',
             conus_columns=dict(gust='gust',lower='lower',upper='upper'))
    result,_=inspect_table(path,cfg)
    assert result['baseline'].shape==(4,2)
    pd.DataFrame(rows+[rows[0]]).to_parquet(path,index=False)
    with pytest.raises(ValueError,match='Duplicate'):
        inspect_table(path,cfg)


def make_pilot_fixture(root):
    spec=importlib.util.spec_from_file_location('synthetic',Path(__file__).parents[1]/'scripts/make_synthetic_pipeline.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=module.make_example(root,bootstrap=0)
    cfg=json.loads(config.read_text())
    for sc,(start,end) in SCENARIO_YEARS.items():
        checkpoints(root/'checkpoints',sc,np.arange(start,end+1))
    sample=root/'source.nc'
    with Dataset(sample,'w') as ds:
        ds.createDimension('Time',2);ds.createDimension('DateStrLen',19);ds.createDimension('y',2);ds.createDimension('x',2)
        var=ds.createVariable('Times','S1',('Time','DateStrLen'))
        var[:]=stringtochar(np.array(['1980-01-01_00:00:00','1980-01-01_03:00:00'],dtype='S19'))
        with np.load(root/'grid.npz') as grid:
            for key,name in (('latitude','XLAT'),('longitude','XLONG')):
                var=ds.createVariable(name,'f8',('Time','y','x'));var[:]=np.broadcast_to(grid[key],(2,2,2))
        for name in ('U10','V10'):
            var=ds.createVariable(name,'f4',('Time','y','x'));var.units='m s-1';var[:]=3
    frame=wide_baseline()
    raw=root/'baseline.csv';frame.to_csv(raw,index=False)
    pq=root/'baseline.parquet';frame.to_parquet(pq,index=False)
    inspection=root/'inspection.json';inspection.write_text('{}')
    cfg.update(checkpoint_root=str(root/'checkpoints'),pilot_cells=2,pilot_bootstrap=0,
               source_samples={'historical':[str(sample)]},grid_reviews={s:'Synthetic test provenance only' for s in SCENARIO_YEARS},
               overlap_policy_review='Synthetic complete nonoverlapping ranges; test only',conus_file=str(pq),
               conus_raw_csv=str(raw),conus_raw_sha256=fingerprint(raw),conus_raw_bytes=raw.stat().st_size,
               conus_inspection=str(inspection),conus_units='m s-1',conus_wind_definition='3-second gust')
    config.write_text(json.dumps(cfg))
    return config


def test_real_pilot_entrypoint_and_gated_manual_plan(tmp_path):
    config=make_pilot_fixture(tmp_path/'fixtures')
    report=run_pilot(config,tmp_path/'pilot')
    assert report['production_ready']
    assert report['benchmark']['cells']==18
    assert not report['production_submitted']
    assert len(report['scenarios'])==9
    with pytest.raises(ValueError,match='Review'):
        plan(config,tmp_path/'pilot/report.json',tmp_path/'plan')
    # Explicit synthetic clean-checkout report for plan-generation tests; no jobs run.
    report['git_status']=''
    (tmp_path/'pilot/report.json').write_text(json.dumps(report))
    output=plan(config,tmp_path/'pilot/report.json',tmp_path/'plan',reviewed=True)
    script=(output/'submit_reviewed.sh').read_text()
    assert script.count('%8')==2
    assert 'afterok:${FIT_JOB}' in script and 'afterok:${FACTOR_JOB}' in script
    assert 'export' not in script
    subprocess.run(['bash','-n',str(output/'submit_reviewed.sh')],check=True)
    assert not (tmp_path/'fixtures/results/production').exists()
    from im3_wind.manual import task
    effective=output/'effective_config.json'
    altered=json.loads(effective.read_text());altered['seed']+=1
    effective.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='Effective settings changed'):
        task(effective,'fit',0)
    # Changed config must invalidate the reviewed pilot.
    cfg=json.loads(config.read_text());cfg['seed']+=1;config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='changed'):
        plan(config,tmp_path/'pilot/report.json',tmp_path/'other-plan',reviewed=True)


def test_pilot_missing_inputs_writes_reports_and_blocks_production(tmp_path):
    config=make_pilot_fixture(tmp_path/'fixtures')
    cfg=json.loads(config.read_text());cfg['checkpoint_root']=str(tmp_path/'missing');config.write_text(json.dumps(cfg))
    report=run_pilot(config,tmp_path/'blocked-pilot')
    assert not report['production_ready']
    assert len(report['scenarios'])==9
    assert (tmp_path/'blocked-pilot/SUMMARY.txt').exists()
    assert not list((tmp_path/'blocked-pilot/fits').glob('*.npz'))
    assert any('No checkpoints' in text for text in report['blockers'])


def test_compact_lookup_matches_factors_and_quarantines_baseline(tmp_path):
    from im3_wind.production import fit_scenario, factor_scenario
    from im3_wind.compact import create_crosswalk, lookup
    config=make_pilot_fixture(tmp_path/'fixtures')
    # No overlap in fixture; direct counts are exact without sidecars.
    cfg=settings(config)
    for scenario in SCENARIO_YEARS:
        fit_scenario(cfg,scenario)
    for scenario in SCENARIO_YEARS:
        factor_scenario(cfg,scenario)
    create_crosswalk(cfg)
    rows=lookup(cfg,'rcp45cooler',2050,100,np.array([0,65537],dtype=np.uint64))
    assert rows.change_factor.notna().all()
    np.testing.assert_allclose(rows.future_wind,rows.historical_wind*rows.change_factor)
    baseline_path=Path(cfg['conus_file'])
    frame=pd.read_parquet(baseline_path)
    frame.loc[0,'rp_500_3sec_gust']=-2
    frame.to_parquet(baseline_path,index=False)
    with pytest.raises(ValueError,match='Stale'):
        lookup(cfg,'rcp45cooler',2050,100,[0])


def test_existing_quality_fields_are_preserved_not_silently_interpreted(tmp_path):
    frame=wide_baseline()
    frame['quality_flags']=[64,0,8,0]
    path=tmp_path/'baseline.parquet';frame.to_parquet(path,index=False)
    cfg=dict(return_periods=[10,25,50,100,200,500],conus_units='m s-1',conus_wind_definition='3-second gust')
    result,report=inspect_table(path,cfg)
    assert report['upstream_quality_columns']==['quality_flags']
    assert result['source_quality_columns']['quality_flags'][0,0]=='64'


def test_slurm_wrapper_runs_python_and_reports_blockers(tmp_path):
    import os
    import sys
    config=make_pilot_fixture(tmp_path/'fixtures')
    cfg=json.loads(config.read_text())
    cfg.update(conda_env=sys.prefix,checkpoint_root=str(tmp_path/'missing-checkpoints'))
    config.write_text(json.dumps(cfg))
    repo=Path(__file__).parents[1]
    env=dict(os.environ,SLURM_JOB_ID='synthetic-wrapper-test',SLURM_CPUS_PER_TASK='2')
    result=subprocess.run(['bash',str(repo/'hpc/run_real_pilot.sbatch'),str(config)],
                          cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode==2
    assert 'blocked_requires_review' in result.stdout
    assert 'Production submitted: no' in result.stdout
    assert (tmp_path/'fixtures/scratch/jobs/pilot-synthetic-wrapper-test').is_dir()

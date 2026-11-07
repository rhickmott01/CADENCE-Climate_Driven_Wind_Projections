"""Bounded, human-submitted future grid sampling. No annual wind reprocessing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from netCDF4 import Dataset, chartostring
from .pipeline_inputs import SCENARIO_YEARS, settings, fingerprint, coordinates
from .products import save_product
from .pilot import write_json


def cli(*args, timeout=300):
    result=subprocess.run(['globus',*map(str,args)],capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Globus {args[0]} failed (exit {result.returncode}); check authentication/network interactively. No credentials copied to reports.')
    return json.loads(result.stdout) if args[-1]=='json' else result.stdout


def select_sources(cfg, prior_run):
    """Use recorded source filenames, never infer weekly timestamps from names."""
    requests=[]
    root=Path(prior_run)
    for sc,(start,end) in SCENARIO_YEARS.items():
        if sc=='historical':
            continue
        inventory=json.loads((root/'coverage'/f'{sc}.inventory.json').read_text())['checkpoints']
        first=min(inventory,key=lambda item:item['first'])
        name=first.get('source_file')
        if not name or Path(name).name!=name or not name.endswith('.nc'):
            raise ValueError(f'Unsafe/missing recorded source filename for {sc}')
        forcing=sc.rsplit('_',1)[0]
        requests.append(dict(scenario=sc,source_path=f'/{forcing}_{start}_{end}/three_hourly/{name}',
                             filename=name,first=first['first'],last=first['last'],
                             inventory_sha256=fingerprint(root/'coverage'/f'{sc}.inventory.json')))
    return requests


def inspect_grid(path, request, cfg, destination):
    reference_lat,reference_lon=coordinates(cfg['grid_file'],cfg['grid_shape'])
    with Dataset(path) as ds:
        times=np.asarray([str(s).replace('_','T') for s in chartostring(ds['Times'][:])],dtype='datetime64[s]')
        if not len(times) or times[0]!=np.datetime64(request['first']) or times[-1]!=np.datetime64(request['last']):
            raise ValueError('Source sample timestamps differ from recorded checkpoint')
        if np.any(np.diff(times)!=np.timedelta64(3,'h')):
            raise ValueError('Source sample does not have contiguous 3-hourly times')
        for variable in ('U10','V10'):
            if ds[variable].units!='m s-1':
                raise ValueError(f'Unexpected source units for {variable}')
        lat=np.ma.filled(ds['XLAT'][0],np.nan)
        lon=np.ma.filled(ds['XLONG'][0],np.nan)
        if lat.shape!=tuple(cfg['grid_shape']) or lon.shape!=lat.shape:
            raise ValueError('Source sample grid shape mismatch')
        for i in range(len(times)):
            for variable,reference in (('XLAT',reference_lat),('XLONG',reference_lon)):
                value=np.ma.filled(ds[variable][i],np.nan)
                if not np.allclose(value,reference,rtol=0,atol=1e-6):
                    raise ValueError(f'{variable} differs from the historical grid at sample time {i}')
        projection={key:ds.getncattr(key) for key in ('MAP_PROJ','TRUELAT1','TRUELAT2','STAND_LON','DX','DY','CEN_LAT','CEN_LON') if key in ds.ncattrs()}
        projection={k:(v.item() if hasattr(v,'item') else v) for k,v in projection.items()}
    meta=dict(scenario=request['scenario'],source_file=str(path),source_path=request['source_path'],
              source_sha256=fingerprint(path),inventory_sha256=request['inventory_sha256'],
              first=str(times[0]),last=str(times[-1]),coordinate_time_slices_checked=len(times),projection=projection,
              scope='One early source week; every coordinate slice checked. Not full-period grid verification.',
              residual_assumption='Model geographic grid remains fixed for other source weeks in the scenario',
              duplicate_wind_equality_verified=False)
    save_product(destination,dict(latitude=lat,longitude=lon),meta)
    return meta


def stage_one(request, transfer_cfg, staging_root, wait_seconds):
    """Reuse a recorded transfer task after interruption; never delete staged sources."""
    folder=Path(staging_root)/request['scenario']
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/request['filename']
    pending=folder/(request['filename']+'.partial')
    state=folder/(request['filename']+'.transfer.json')
    if final.exists():
        if final.stat().st_size!=request['bytes']:
            raise ValueError('Existing source sample size mismatch; preserved for inspection')
        return final
    source=f"{transfer_cfg['source_collection']}:{request['source_path']}"
    target=f"{transfer_cfg['destination_collection']}:{pending}"
    if state.exists():
        saved=json.loads(state.read_text())
        if saved.get('source')!=source or saved.get('target')!=target:
            raise ValueError('Existing transfer state belongs to another source/destination')
        task_id=saved.get('task_id')
        if not task_id:
            raise ValueError('Prior transfer submission outcome unknown; inspect Globus tasks before retrying')
    else:
        # Persist intent before submission; do not auto-resubmit after an ambiguous failure.
        write_json(state,dict(source=source,target=target,task_id=None))
        task=cli('transfer',source,target,'--verify-checksum','--label',f"CADENCE grid evidence {request['scenario']}",'--format','json')
        task_id=task['task_id']
        temporary=state.with_suffix('.updated')
        temporary.write_text(json.dumps(dict(source=source,target=target,task_id=task_id)))
        os.replace(temporary,state)
    waited=subprocess.run(['globus','task','wait',task_id,'--timeout',str(wait_seconds)],capture_output=True,text=True,timeout=wait_seconds+60)
    if waited.returncode:
        raise RuntimeError(f'Globus task {task_id} not completed; state retained for restart')
    task=cli('task','show',task_id,'--format','json')
    if task.get('status')!='SUCCEEDED' or not pending.exists() or pending.stat().st_size!=request['bytes']:
        raise ValueError(f'Transfer task {task_id} did not produce the expected complete sample')
    os.link(pending,final)  # preserve partial as well; no source cleanup in evidence jobs
    return final


def collect(config, prior_run, allow_transfer=False, run_pilot_after=False, output=None):
    cfg=settings(config)
    prior=Path(prior_run).resolve()
    original=json.loads((prior/'report.json').read_text())
    if Path(original['run_dir']).resolve()!=prior:
        raise ValueError('Prior run/report directory mismatch')
    for sc in SCENARIO_YEARS:
        expected=original['scenarios'].get(sc,{}).get('validation',{}).get('source_sha256')
        if not expected or fingerprint(Path(cfg['annual_root'])/f'{sc}.npz')!=expected:
            raise ValueError(f'Prior annual evidence missing/stale for {sc}')
    options=cfg['future_evidence']
    transfer_cfg=json.loads(Path(options['source_pipeline_config']).read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=Path(output) if output else Path(cfg['results_root'])/'evidence_runs'/f'{stamp}-{os.environ.get("SLURM_JOB_ID","local")}'
    root.mkdir(parents=True,exist_ok=False)
    report=dict(prior_run=str(prior),requests=[],issues=[],samples={},allow_transfer=allow_transfer,
                production_submitted=False,scope='Eight early future source weeks only; no full-record overlap equality claim')
    write_json(root/'started.json',report)
    requests=select_sources(cfg,prior)
    # Bound the complete request before initiating even one transfer.
    max_files=options.get('max_files',8)
    max_bytes=options.get('max_bytes',100*1024**3)
    if not 1<=max_files<=8 or not 0<max_bytes<=128*1024**3 or len(requests)>max_files:
        raise ValueError('Evidence transfer limit invalid/exceeded (maximum eight files and 128 GiB)')
    catalogs={}
    for request in requests:
        folder=str(Path(request['source_path']).parent)+'/'
        try:
            if folder not in catalogs:
                listing=cli('ls',f"{transfer_cfg['source_collection']}:{folder}",'--format','json')
                catalogs[folder]={item['name']:item for item in listing['DATA'] if item.get('type')=='file'}
            entry=catalogs[folder][request['filename']]
            request['bytes']=int(entry['size'])
            if request['bytes']<=0:
                raise ValueError('Invalid source size')
        except Exception as exc:
            report['issues'].append(f"{request['scenario']}: source discovery failed: {exc}")
    report['requests']=requests
    total=sum(r.get('bytes',0) for r in requests)
    report.update(requested_bytes=total,max_bytes=max_bytes)
    write_json(root/'transfer_plan.json',report)
    if total>max_bytes:
        report['issues'].append('Total source bytes exceed reviewed transfer cap; nothing transferred')
    if report['issues']:
        write_json(root/'report.json',report)
        return report,root
    if not allow_transfer:
        report['issues'].append('Plan only: human must explicitly enable --allow-transfer')
        write_json(root/'report.json',report)
        return report,root
    staging=Path(options['staging_root'])
    staging.mkdir(parents=True,exist_ok=True)
    remaining=sum(r['bytes'] for r in requests if not (staging/r['scenario']/r['filename']).exists())
    if shutil.disk_usage(staging).free<remaining*1.1:
        raise ValueError('Insufficient filesystem free space for bounded samples (user quota must also be checked)')
    # Prevent duplicate collection jobs from submitting overlapping transfers.
    import fcntl
    with open(staging/'.collection.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for request in requests:
            sc=request['scenario']
            try:
                path=stage_one(request,transfer_cfg,staging,options.get('wait_seconds',7200))
                destination=root/'grids'/f'{sc}.npz'
                report['samples'][sc]=inspect_grid(path,request,cfg,destination)
                report['samples'][sc]['grid_file']=str(destination.resolve())
                logging.info('%s: sampled grid matches historical',sc)
            except Exception as exc:
                report['issues'].append(f'{sc}: {exc}')
            write_json(root/'samples'/f'{sc}.json',dict(sample=report['samples'].get(sc),issues=[s for s in report['issues'] if s.startswith(sc)]))
    if not report['issues']:
        candidate=dict(cfg)
        candidate.pop('settings_sha256')
        candidate['coverage_sidecars']=json.loads((prior/'coverage_paths.json').read_text())
        candidate['scenario_grid_files']={sc:entry['grid_file'] for sc,entry in report['samples'].items()}
        candidate['grid_reviews']={sc:dict(status='sample_verified',source_sha256=entry['source_sha256'],
                                         scope=entry['scope'],residual_assumption=entry['residual_assumption']) for sc,entry in report['samples'].items()}
        candidate['evidence_report']=str(root/'report.json')
        write_json(root/'candidate_config.json',candidate)
        report['candidate_config']=str(root/'candidate_config.json')
    write_json(root/'report.json',report)
    if run_pilot_after and not report['issues']:
        from .pilot import run_pilot
        # The overlap-policy gate intentionally remains unresolved pending human review.
        pilot=run_pilot(root/'candidate_config.json',root/'pilot')
        write_json(root/'pilot_summary.json',compact_report(pilot))
    return report,root


def compact_report(report):
    scenarios={}
    for sc,row in report.get('scenarios',{}).items():
        v=row.get('validation',{})
        coverage=v.get('coverage',{})
        unique=coverage.get('unique_timestep_counts',[])
        scenarios[sc]=dict(status=row.get('status'),issues=row.get('issues'),shape=row.get('shape'),
                           missing=v.get('missing_values'),low_coverage=v.get('excluded_low_coverage'),
                           eligible_years_min=v.get('eligible_years_min'),
                           unique_timestep_min=min(unique) if unique else None,unique_timestep_max=max(unique) if unique else None,
                           converged=row.get('converged'),fit_seconds=row.get('fit_seconds'),grid_review=row.get('grid_review'))
    return {**{key:report.get(key) for key in ('status','run_dir','blockers','raw_samples','conus','benchmark','wall_seconds','peak_rss_bytes','pilot_transfers')},'scenarios':scenarios}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--prior-run',required=True)
    p.add_argument('--allow-transfer',action='store_true')
    p.add_argument('--run-pilot',action='store_true')
    p.add_argument('--output')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    report,root=collect(args.config,args.prior_run,args.allow_transfer,args.run_pilot,args.output)
    print(f'Evidence reports: {root}',flush=True)
    print(json.dumps(dict(issues=report['issues'],samples=len(report['samples']),requested_bytes=report.get('requested_bytes'),production_submitted=False),indent=2))
    if report['issues']:
        raise SystemExit(2)


if __name__=='__main__':
    main()

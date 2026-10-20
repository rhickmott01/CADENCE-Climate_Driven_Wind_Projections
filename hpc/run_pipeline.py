#!/usr/bin/env python3
"""Resumable Globus -> Slurm allocation -> checkpoints pipeline.

Run INSIDE one 24-hour Slurm allocation. No raw data is deleted from Globus.
The controller must be the sole writer to its SQLite database and staging area.
"""
import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from process_wind_batch import extract

LOG=logging.getLogger('im3')
SCENARIOS={'historical':'historical_1980_2019',
 'rcp45cooler_near':'rcp45cooler_2020_2059', 'rcp45cooler_far':'rcp45cooler_2060_2099',
 'rcp45hotter_near':'rcp45hotter_2020_2059', 'rcp45hotter_far':'rcp45hotter_2060_2099',
 'rcp85cooler_near':'rcp85cooler_2020_2059', 'rcp85cooler_far':'rcp85cooler_2060_2099',
 'rcp85hotter_near':'rcp85hotter_2020_2059', 'rcp85hotter_far':'rcp85hotter_2060_2099'}
PATTERN=re.compile(r'^tgw_wrf_.*_three_hourly_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.nc$')


def cli(*args, timeout=120):
    p=subprocess.run(['globus',*map(str,args)],capture_output=True,text=True,timeout=timeout)
    if p.returncode: raise RuntimeError(f'Globus failed: {p.stderr.strip()} {p.stdout.strip()}')
    return p.stdout


def init_db(db):
    db.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(db,timeout=30)
    c.execute('PRAGMA journal_mode=DELETE')
    c.execute('CREATE TABLE IF NOT EXISTS files (scenario TEXT NOT NULL, filename TEXT NOT NULL, status TEXT NOT NULL, checkpoint TEXT, task_id TEXT, updated TEXT, PRIMARY KEY(scenario,filename))')
    c.commit()
    return c


def mark(c, scenario, filename, status, checkpoint=None, task_id=None):
    c.execute('INSERT INTO files VALUES (?,?,?,?,?,?) ON CONFLICT(scenario,filename) DO UPDATE SET status=excluded.status, checkpoint=COALESCE(excluded.checkpoint,files.checkpoint), task_id=COALESCE(excluded.task_id,files.task_id), updated=excluded.updated',
              (scenario,filename,status,checkpoint,task_id,datetime.now(timezone.utc).isoformat()))
    c.commit()


def inventory(config):
    result=[]
    for scenario,folder in SCENARIOS.items():
        src=f'/{folder}/three_hourly/'
        lines=cli('ls',f'{config["source_collection"]}:{src}',timeout=300).splitlines()
        files=sorted(set(s.strip() for s in lines if PATTERN.fullmatch(s.strip())))
        if not files: raise RuntimeError(f'No matching files found in {src}; verify source naming')
        LOG.info('%s: %s files',scenario,len(files))
        result.extend((scenario,src,f) for f in files)
    return result


def checkpoint_valid(path, filename, scenario):
    if not path.is_file(): return False
    try:
        import numpy as np
        with np.load(path,allow_pickle=False) as z:
            a=z['maxima']; yrs=z['years']; n=int(z['n_timesteps'])
            return (a.ndim==3 and a.shape[1:]==(299,424) and len(yrs)==a.shape[0]
                    and n>0 and str(z['source_file'])==filename and str(z['scenario'])==scenario
                    and bool(np.isfinite(a).any()))
    except Exception: return False


def globus_task_status(task_id):
    data=json.loads(cli('task','show',task_id,'--format','json'))
    return data.get('status','UNKNOWN')


def transfer(config, scenario, srcdir, filename, local, c, deadline):
    # Globus itself verifies transfers. Never trust a mere existing local file as completed.
    row=c.execute('SELECT status,task_id FROM files WHERE scenario=? AND filename=?',(scenario,filename)).fetchone()
    task_id=row[1] if row and row[0]=='transferring' else None
    if task_id:
        LOG.info('Resuming Globus task %s',task_id)
    else:
        # Isolate source scenarios in separate staging folders.
        source=f'{config["source_collection"]}:{srcdir}{filename}'
        dest=f'{config["destination_collection"]}:{local}'
        out=json.loads(cli('transfer',source,dest,'--label',f'CADENCE {scenario} {filename}','--format','json'))
        task_id=out['task_id']
        mark(c,scenario,filename,'transferring',task_id=task_id)
        LOG.info('Submitted %s %s',filename,task_id)
    while time.monotonic()<deadline:
        status=globus_task_status(task_id)
        if status=='SUCCEEDED':
            if not local.is_file() or local.stat().st_size<=0:
                raise RuntimeError(f'Transfer task succeeded but file missing/empty: {local}')
            mark(c,scenario,filename,'staged',task_id=task_id)
            return
        if status in ('FAILED','INACTIVE','CANCELED'):
            raise RuntimeError(f'Globus task {task_id} ended with {status}')
        time.sleep(config.get('poll_seconds',30))
    raise TimeoutError('Controller nearing deadline while transfer pending; task recorded for resume')


def process_one(args):
    source,output,scenario=args
    return extract(Path(source),Path(output),scenario)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='hpc/pipeline_config.json')
    p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-runtime-hours',type=float,default=23)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-batches',type=int,default=0,help='0 means unlimited')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not 1<=args.batch_size<=5 or not 1<=args.workers<=4: p.error('batch-size 1..5; workers 1..4')
    config=json.loads(Path(args.config).read_text())
    deadline=time.monotonic()+args.max_runtime_hours*3600-300
    root=Path(config['work_root']); root.mkdir(parents=True,exist_ok=True)
    checkpoints=Path(config['checkpoint_root']); checkpoints.mkdir(parents=True,exist_ok=True)
    dbpath=Path(config['state_db'])
    c=init_db(dbpath)
    items=inventory(config)
    completed={(a,b) for a,b in c.execute("SELECT scenario,filename FROM files WHERE status='completed'")}
    remaining=[item for item in items if (item[0],item[2]) not in completed]
    LOG.info('Pending files: %d',len(remaining))
    if args.dry_run:
        for scenario,src,f in remaining[:args.batch_size]: LOG.info('WOULD PROCESS %s/%s',scenario,f)
        return
    for offset in range(0,len(remaining),args.batch_size):
        if time.monotonic()>deadline: break
        if args.max_batches and offset//args.batch_size>=args.max_batches: break
        batch=remaining[offset:offset+args.batch_size]
        staged=[]
        for scenario,srcdir,filename in batch:
            out=checkpoints/scenario/(Path(filename).stem+'_max.npz')
            if checkpoint_valid(out,filename,scenario):
                mark(c,scenario,filename,'validated',checkpoint=str(out))
                LOG.info('Reusing validated checkpoint %s',out)
                continue
            local=root/'raw'/scenario/filename
            local.parent.mkdir(parents=True,exist_ok=True)
            # An existing unverified file must be re-transferred, not silently trusted.
            transfer(config,scenario,srcdir,filename,local,c,deadline)
            staged.append((str(local),str(out),scenario))
        if staged:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(process_one,a):a for a in staged}
                for fut in concurrent.futures.as_completed(futures):
                    a=futures[fut]
                    try:
                        result=fut.result()
                        mark(c,a[2],Path(a[0]).name,'validated',checkpoint=a[1])
                        LOG.info('Validated %s %s',a[2],result['source'])
                    except Exception:
                        LOG.exception('Processing failed for %s; raw file retained',a[0])
                        raise
        # Validate the ENTIRE batch before any deletion.
        for scenario,srcdir,filename in batch:
            out=checkpoints/scenario/(Path(filename).stem+'_max.npz')
            if not checkpoint_valid(out,filename,scenario):
                raise RuntimeError(f'Checkpoint validation failed: {out}; raw files retained')
        for scenario,srcdir,filename in batch:
            out=checkpoints/scenario/(Path(filename).stem+'_max.npz')
            local=root/'raw'/scenario/filename
            # Only delete inside the configured staging directory, never remote Globus files.
            rawroot=(root/'raw').resolve()
            if local.resolve().parent != (rawroot/scenario).resolve():
                raise RuntimeError('Unsafe deletion path')
            if local.exists(): local.unlink()
            mark(c,scenario,filename,'completed',checkpoint=str(out))
        LOG.info('Batch %d completed (%d files)',offset//args.batch_size+1,len(batch))
    LOG.info('Run ended. Resume by submitting the same job again.')

if __name__=='__main__': main()

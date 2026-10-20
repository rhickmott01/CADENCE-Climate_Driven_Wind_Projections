#!/usr/bin/env python3
"""IM3 WRF weekly file extraction; independently callable by the controller."""
import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def extract(source: Path, output: Path, scenario: str) -> dict:
    """Read only U10/V10; write an atomic, validated per-file checkpoint."""
    source, output = Path(source), Path(output)
    with Dataset(source) as ds:
        u, v = ds.variables['U10'], ds.variables['V10']
        if u.shape != v.shape or len(u.shape) != 3:
            raise ValueError('Expected matching U10/V10 [time,y,x]')
        raw = ds.variables['Times'][:]
        dates = [datetime.strptime(b''.join(row).decode('ascii'), '%Y-%m-%d_%H:%M:%S') for row in raw]
        if not dates or any(b <= a for a, b in zip(dates, dates[1:])):
            raise ValueError('Timestamps missing, duplicate or out of order')
        if any((b-a).total_seconds() != 10800 for a,b in zip(dates, dates[1:])):
            raise ValueError('Non-3-hourly timestamps within file')
        maxima = {}
        counts = {}
        valid_counts = {}
        for i, dt in enumerate(dates):
            a = np.ma.filled(u[i], np.nan).astype(np.float32)
            b = np.ma.filled(v[i], np.nan).astype(np.float32)
            wind = np.hypot(a, b)
            wind[~np.isfinite(wind)] = np.nan
            yr = dt.year
            if yr not in maxima:
                maxima[yr] = wind.copy()
                valid_counts[yr] = np.isfinite(wind).astype(np.uint16)
                counts[yr] = 0
            else:
                maxima[yr] = np.fmax(maxima[yr], wind)
                valid_counts[yr] += np.isfinite(wind).astype(np.uint16)
            counts[yr] += 1
    years = sorted(maxima)
    arr = np.stack([maxima[y] for y in years])
    if arr.shape[1:] != (299,424) or not np.isfinite(arr).any() or np.nanmin(arr) < 0:
        raise ValueError('Invalid wind grid or values')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.pending_', suffix='.npz', dir=output.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp, maxima=arr, years=np.asarray(years, dtype=np.int32),
                            counts=np.asarray([counts[y] for y in years], dtype=np.int32),
                            valid_counts=np.stack([valid_counts[y] for y in years]),
                            source_file=source.name, scenario=scenario,
                            first_time=dates[0].isoformat(), last_time=dates[-1].isoformat(),
                            n_timesteps=len(dates))
        with np.load(temp, allow_pickle=False) as check:
            if check['maxima'].shape != arr.shape or str(check['source_file']) != source.name:
                raise ValueError('Checkpoint verification failed')
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return dict(source=source.name, scenario=scenario, checkpoint=str(output),
                first_time=dates[0].isoformat(), last_time=dates[-1].isoformat(),
                timesteps=len(dates), years=years, status='validated')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--manifest',required=True); p.add_argument('--max-files',type=int,default=5)
    p.add_argument('--scenario',default='historical')
    args=p.parse_args()
    files=sorted(Path(args.input_dir).glob('*.nc'))[:args.max_files]
    if not files: raise SystemExit('No NetCDF files')
    results=[]
    for f in files:
        out=Path(args.output_dir)/args.scenario/(f.stem+'_max.npz')
        results.append(extract(f,out,args.scenario))
    manifest=Path(args.manifest); manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(results,indent=2))
    print(f'Processed {len(results)} files')

if __name__=='__main__': main()

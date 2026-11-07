#!/usr/bin/env python3
"""Print or submit a dependency-ordered Slurm workflow from repository root."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import numpy as np
from im3_wind.pipeline_inputs import SCENARIO_YEARS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/pipeline.json')
    p.add_argument('--submit', action='store_true', help='Otherwise print commands only')
    p.add_argument('--partition')
    p.add_argument('--account')
    p.add_argument('--max-parallel', type=int, default=8)
    p.add_argument('--fit-hours', type=int, default=12)
    args = p.parse_args()
    if args.max_parallel < 1 or args.fit_hours < 1:
        p.error('max-parallel and fit-hours must be positive')
    config = Path(args.config).resolve()
    cfg = json.loads(config.read_text())
    with np.load(cfg['conus_file'], allow_pickle=False) as d:
        target_n = d['latitude'].size
    source_chunks = math.ceil(np.prod(cfg['grid_shape']) / cfg['chunk_cells'])
    target_chunks = math.ceil(target_n / cfg.get('target_chunk_cells', 4096))
    logs = Path(cfg['results_root']).resolve() / 'logs'
    if args.submit:
        logs.mkdir(parents=True, exist_ok=True)
    previous = []
    for stage in ('validate', 'fit', 'factor', 'downscale', 'export'):
        current = []
        for sc in SCENARIO_YEARS:
            if stage in ('factor', 'downscale') and sc == 'historical':
                continue
            if stage == 'export' and not sc.endswith('_near'):
                continue
            count = 1 if stage == 'validate' else source_chunks if stage in ('fit', 'factor') else target_chunks
            command = ['sbatch', '--parsable', '--chdir', str(Path.cwd()),
                       '--output', str(logs / f'{stage}-{sc}-%A_%a.log'),
                       '--array', f'0-{count-1}%{args.max_parallel}',
                       '--time', f'{args.fit_hours if stage == "fit" else 2}:00:00']
            if args.partition:
                command += ['--partition', args.partition]
            if args.account:
                command += ['--account', args.account]
            if previous:
                command += ['--dependency', 'afterok:' + ':'.join(previous)]
            command += ['hpc/wind_stage.sbatch', str(config), stage, sc]
            if args.submit:
                result = subprocess.check_output(command, text=True).strip().split(';')[0]
                if not result.isdigit():
                    raise ValueError(f'Unexpected sbatch job ID: {result}')
                print(f'{stage} {sc}: {result}', flush=True)
            else:
                result = f'JOB_{stage}_{sc}'
                print(shlex.join(command))
            current.append(result)
        previous = current


if __name__ == '__main__':
    main()

"""Prepare reviewed batch plans and run bounded tasks. Never invokes sbatch."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
from .pipeline_inputs import SCENARIO_YEARS, settings, fingerprint
from .pilot import write_json


def plan(config, pilot_report, output, reviewed=False, concurrency=8, partition='interactive', account=None):
    if not reviewed:
        raise ValueError('Review the pilot report first; --reviewed-pilot records that human acknowledgement')
    if not 1 <= concurrency <= 8:
        raise ValueError('Concurrency must be 1–8 globally')
    cfg=settings(config)
    report_path=Path(pilot_report).resolve()
    report=json.loads(report_path.read_text())
    if report['settings_sha256'] != cfg['settings_sha256'] or not report['production_ready'] or report['blockers']:
        raise ValueError('Pilot blocked or settings changed; resolve evidence, rerun pilot, then review')
    if report.get('git_status'):
        raise ValueError('Pilot ran from a dirty checkout; commit changes and rerun')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if commit != report['git_commit']:
        raise ValueError('Code revision changed since pilot')
    if any(report['scenarios'].get(sc,{}).get('status') != 'pilot_completed' for sc in SCENARIO_YEARS):
        raise ValueError('Not all scenarios completed pilot')
    out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    sidecars=json.loads((report_path.parent/'coverage_paths.json').read_text())
    cfg.pop('settings_sha256')
    cfg.update(coverage_sidecars=sidecars, results_root=str(Path(cfg['results_root'])/'production'),
               human_review=dict(pilot_report=str(report_path),pilot_sha256=fingerprint(report_path),
                                 git_commit=commit,scope='fit/factor/crosswalk only; no annual full-grid expansion'))
    effective=out/'effective_config.json'
    write_json(effective,cfg)
    chunks=math.ceil(math.prod(cfg['grid_shape'])/cfg['chunk_cells'])
    lines=['#!/bin/bash','set -euo pipefail',f'cd {shlex.quote(str(Path.cwd()))}',f'mkdir -p {shlex.quote(str(out/"logs"))}']
    for stage,count in (('fit',9*chunks),('factor',8*chunks),('crosswalk',1)):
        command=['sbatch','--parsable','--partition',partition,'--mem=8G',
                 '--output',str(out/'logs'/f'{stage}-%A_%a.log')]
        if stage=='fit':
            command+=['--array',f'0-{count-1}%{concurrency}','--time=12:00:00']
        elif stage=='factor':
            command+=['--array',f'0-{count-1}%{concurrency}','--time=02:00:00']
        else:
            command+=['--time=02:00:00']
        if account:
            command+=['--account',account]
        # Runtime shell expansion is limited to the sbatch job ID, never user-supplied code.
        dependency='' if stage=='fit' else ' --dependency="afterok:${FIT_JOB}"' if stage=='factor' else ' --dependency="afterok:${FACTOR_JOB}"'
        text=shlex.join(command)+dependency+' '+shlex.join(['hpc/run_native_production.sbatch',str(effective),stage])
        var=stage.upper()+'_JOB'
        lines.append(f'{var}=$({text})')
        lines.append(f'{var}=${{{var}%%;*}}')
        lines.append(f'printf \'{stage}: %s\\n\' "${{{var}}}"')
    (out/'submit_reviewed.sh').write_text('\n'.join(lines)+'\n')
    write_json(out/'plan.json',dict(config=str(effective),settings_sha256=fingerprint(effective),
                                  pilot_sha256=fingerprint(report_path),max_concurrent_tasks=concurrency,
                                  arrays={'fit':9*chunks,'factor':8*chunks,'crosswalk':1},
                                  resource_note='Fit walltime/memory are initial allocations; inspect pilot and override before submitting',
                                  benchmark=report.get('benchmark'),submitted=False))
    return out


def task(config, stage, index):
    cfg=settings(config)
    manifest_path=Path(config).resolve().parent/'plan.json'
    if not manifest_path.exists() or json.loads(manifest_path.read_text()).get('settings_sha256') != cfg['settings_sha256']:
        raise ValueError('Effective settings changed or reviewed plan manifest is missing')
    review=cfg.get('human_review')
    if not review:
        raise ValueError('Use an effective configuration from a human-reviewed pilot plan')
    if fingerprint(review['pilot_report']) != review['pilot_sha256']:
        raise ValueError('Reviewed pilot report changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if commit != review['git_commit']:
        raise ValueError('Code changed after reviewed plan')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Tracked code/config changes after reviewed plan')
    pilot=json.loads(Path(review['pilot_report']).read_text())
    from .production import fit_scenario, factor_scenario
    chunks=math.ceil(math.prod(cfg['grid_shape'])/cfg['chunk_cells'])
    scenarios=list(SCENARIO_YEARS) if stage=='fit' else [s for s in SCENARIO_YEARS if s!='historical']
    if stage=='crosswalk':
        if index != 0:
            raise ValueError('crosswalk is a single task')
        if fingerprint(cfg['conus_file']) != pilot['conus']['baseline_sha256']:
            raise ValueError('Baseline changed since reviewed pilot')
        from .compact import create_crosswalk
        create_crosswalk(cfg)
        return
    if not 0 <= index < len(scenarios)*chunks:
        raise ValueError('Array task index outside plan')
    from .pipeline_inputs import grid_path
    scenario=scenarios[index//chunks]
    if fingerprint(grid_path(cfg,scenario)) != pilot['scenarios'][scenario]['grid_sha256']:
        raise ValueError('Grid changed since reviewed pilot')
    operation=fit_scenario if stage=='fit' else factor_scenario
    operation(cfg,scenario,index%chunks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    pp=sub.add_parser('plan')
    pp.add_argument('--config',required=True)
    pp.add_argument('--pilot-report',required=True)
    pp.add_argument('--output',required=True)
    pp.add_argument('--reviewed-pilot',action='store_true')
    pp.add_argument('--concurrency',type=int,default=8)
    pp.add_argument('--partition',default='interactive')
    pp.add_argument('--account')
    pt=sub.add_parser('task')
    pt.add_argument('--config',required=True)
    pt.add_argument('--stage',choices=['fit','factor','crosswalk'],required=True)
    pt.add_argument('--index',type=int,required=True)
    args=p.parse_args()
    if args.command=='plan':
        print(plan(args.config,args.pilot_report,args.output,args.reviewed_pilot,args.concurrency,args.partition,args.account))
    else:
        task(args.config,args.stage,args.index)


if __name__=='__main__':
    main()

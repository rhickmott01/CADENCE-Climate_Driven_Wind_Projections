"""Human-submitted real-input audit and bounded pilot; never submits jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time
import numpy as np
from .pipeline_inputs import SCENARIO_YEARS, settings, validate_annual, coordinates, fingerprint, grid_path
from .products import save_product
from .coverage import write_coverage
from .extremes import fit_cell


def write_json(path, value):
    from tempfile import NamedTemporaryFile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temp = f.name
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
        except BaseException:
            os.unlink(temp)
            raise
    try:
        os.link(temp, path)
    finally:
        os.unlink(temp)


def raw_sample_audit(cfg):
    """Sample only: inspect coordinates/times and compare available overlap winds."""
    from netCDF4 import Dataset, chartostring
    expected_lat, expected_lon = coordinates(cfg['grid_file'], cfg['grid_shape'])
    records, pairs, issues = [], [], []
    paths = cfg.get('source_samples', {}).get('historical', [])
    previous = None
    for filename in paths:
        path = Path(filename)
        try:
            with Dataset(path) as ds:
                for name in ('U10', 'V10'):
                    if ds[name].units != 'm s-1':
                        raise ValueError(f'{name} has unexpected units')
                times = np.asarray([str(s).replace('_','T') for s in chartostring(ds['Times'][:])], dtype='datetime64[s]')
                if not len(times) or np.any(np.diff(times) != np.timedelta64(3, 'h')):
                    raise ValueError('Noncontiguous sample timestamps')
                for name, ref in (('XLAT',expected_lat), ('XLONG',expected_lon)):
                    for index in (0,-1):
                        if not np.allclose(np.ma.filled(ds[name][index],np.nan), ref, rtol=0, atol=1e-6):
                            raise ValueError('Sample coordinates mismatch')
                row = dict(path=str(path), size=path.stat().st_size, first=str(times[0]), last=str(times[-1]),
                           timestamps=len(times), source_variable='hypot(U10,V10)', units='m s-1')
                if previous is not None:
                    shared = np.intersect1d(previous[1], times)
                    with Dataset(previous[0]) as old:
                        for timestamp in shared:
                            i, j = np.flatnonzero(previous[1] == timestamp)[0], np.flatnonzero(times == timestamp)[0]
                            a = np.hypot(np.ma.filled(old['U10'][i],np.nan),np.ma.filled(old['V10'][i],np.nan))
                            b = np.hypot(np.ma.filled(ds['U10'][j],np.nan),np.ma.filled(ds['V10'][j],np.nan))
                            same = bool(np.array_equal(a,b,equal_nan=True))
                            pairs.append(dict(timestamp=str(timestamp), files=[str(previous[0]),str(path)], identical_wind=same))
                            if not same:
                                issues.append(f'Overlapping wind values differ at {timestamp}')
                records.append(row)
                previous = path, times
        except Exception as exc:
            issues.append(f'{path}: {exc}')
    return dict(files=records, overlap_comparisons=pairs, issues=issues,
                scope='Historical samples only; not a full-record or future-grid verification',
                source_definition_confirmed=bool(records) and not issues)


def baseline_audit(cfg):
    from .baseline import inspect_table
    import pyarrow.parquet as pq
    raw = Path(cfg['conus_raw_csv'])
    baseline = Path(cfg['conus_file'])
    report = dict(raw_csv=str(raw), baseline=str(baseline), errors=[])
    if raw.exists():
        sha = fingerprint(raw)
        report.update(raw_bytes=raw.stat().st_size, raw_sha256=sha)
        if cfg.get('conus_raw_sha256') and sha != cfg['conus_raw_sha256']:
            report['errors'].append('CONUS raw CSV checksum mismatch')
        if cfg.get('conus_raw_bytes') and raw.stat().st_size != cfg['conus_raw_bytes']:
            report['errors'].append('CONUS raw CSV size mismatch')
        with open(raw,'rb') as f:
            if f.read(100).startswith(b'version https://git-lfs'):
                report['errors'].append('Raw baseline is an LFS pointer')
    else:
        report['errors'].append('Original CONUS CSV missing; provenance cannot be checked')
    inspection = Path(cfg['conus_inspection'])
    if inspection.exists():
        report['inspection_sha256'] = fingerprint(inspection)
        with open(inspection) as f:
            json.load(f)  # retain report, do not assume an unverified report schema
    else:
        report['errors'].append('Existing CONUS inspection report missing')
    if not baseline.exists():
        report['errors'].append('Quality-flagged baseline missing')
    else:
        report['baseline_sha256'] = fingerprint(baseline)
        if baseline.suffix == '.parquet':
            report['available_columns'] = pq.read_schema(baseline).names
        try:
            _, facts = inspect_table(baseline,cfg)
            report.update(facts)
            if facts['upstream_quality_columns'] and not cfg.get('conus_upstream_quality_review'):
                report['errors'].append('Upstream baseline quality-column meanings require conus_upstream_quality_review; original values are preserved')
        except Exception as exc:
            report['errors'].append(str(exc))
    return report


def select_cells(cube, count, minimum):
    flat = cube.reshape(cube.shape[0],-1)
    eligible = np.flatnonzero(np.isfinite(flat).sum(axis=0) >= minimum)
    if not len(eligible):
        return np.array([],dtype=int)
    # Uniform row-major coverage plus examples with fewest years/largest observed max.
    selected = list(eligible[np.linspace(0,len(eligible)-1,min(count,len(eligible)),dtype=int)])
    diagnostics = [eligible[np.argmin(np.isfinite(flat[:,eligible]).sum(axis=0))],
                   eligible[np.nanargmax(np.nanmax(flat[:,eligible],axis=0))]]
    return np.asarray(list(dict.fromkeys(diagnostics+selected))[:count],dtype=int)


def pilot_transfer_checks(cfg, root):
    """Compare available fitted scenario pairs at actual nearby CONUS cells only."""
    from scipy.spatial import cKDTree
    from .spatial import lonlat_to_unit_xyz
    from .products import read_product
    from .factors import climate_factors
    from .downscale import load_conus, geographic_map, transfer
    reports = {}
    hist_path = root/'fits/historical.npz'
    if not hist_path.exists():
        return {'status': 'not_run', 'reason': 'No eligible historical pilot fits'}
    hist, _ = read_product(hist_path)
    baseline = load_conus(cfg['conus_file'], cfg['return_periods'], cfg)
    lat,lon = coordinates(cfg['grid_file'],cfg['grid_shape'])
    tree = cKDTree(lonlat_to_unit_xyz(baseline['latitude'],baseline['longitude']))
    n = lat.size
    for scenario in SCENARIO_YEARS:
        path = root/'fits'/f'{scenario}.npz'
        if scenario == 'historical' or not path.exists():
            continue
        future, _ = read_product(path)
        common = np.intersect1d(hist['cell_id'],future['cell_id'])
        if not len(common):
            reports[scenario] = {'status':'not_run','reason':'No common sampled source cells'}
            continue
        def select(data):
            idx=np.searchsorted(data['cell_id'],common) if np.all(np.diff(data['cell_id'])>0) else np.array([np.flatnonzero(data['cell_id']==i)[0] for i in common])
            return {k:(v if k=='return_periods' else v[idx]) for k,v in data.items()}
        factors=climate_factors(select(hist),select(future))
        full={}
        for key in ('factor','lower','upper','flags'):
            full[key]=np.full((n,len(cfg['return_periods'])),1 if key=='flags' else np.nan,dtype=np.uint16 if key=='flags' else float)
            full[key][common]=factors[key]
        full['cell_id']=np.arange(n)
        _, targets=tree.query(lonlat_to_unit_xyz(lat.ravel()[common],lon.ravel()[common]))
        targets=np.unique(targets)
        mapping=geographic_map(lat,lon,baseline['latitude'].ravel()[targets],baseline['longitude'].ravel()[targets],cfg['max_distance_km'])
        bflags=baseline.get('baseline_quality_flags',np.zeros_like(baseline['baseline'],dtype=np.uint16))[targets]
        result=transfer(full,mapping,baseline['baseline'][targets],bflags)
        save_product(root/'transfers'/f'{scenario}.npz',dict(result,target_cell_id=targets,return_periods=np.asarray(cfg['return_periods'])),
                     dict(purpose='Sparse real pilot only; unsampled neighbor factors are missing',scenario=scenario,settings_sha256=cfg['settings_sha256']))
        reports[scenario]=dict(status='completed',common_source_cells=len(common),target_cells=len(targets),
                               finite_factors=int(np.isfinite(result['factor']).sum()),
                               finite_future_winds=int(np.isfinite(result['future_wind']).sum()),
                               idw_sensitivity_values=int(np.isfinite(result['mapping_sensitivity']).sum()),
                               note='IDW missing where required neighbors were not fitted; no silent neighbor renormalization')
    return reports


def run_pilot(config_path, run_dir=None):
    cfg = settings(config_path)
    count, bootstrap = cfg.get('pilot_cells',16), cfg.get('pilot_bootstrap',25)
    if not isinstance(count,int) or count < 1 or count > 128 or not isinstance(bootstrap,int) or not 0 <= bootstrap <= 200:
        raise ValueError('Pilot requires 1–128 cells/scenario and 0–200 bootstrap replicates')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path(run_dir) if run_dir else Path(cfg['results_root'])/'pilot_runs'/f'{stamp}-{os.environ.get("SLURM_JOB_ID","local")}'
    root.mkdir(parents=True,exist_ok=False)
    def git(*args):
        p = subprocess.run(['git',*args],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode == 0 else 'unavailable'
    started = time.perf_counter()
    report = dict(status='running', started_utc=stamp, host=socket.gethostname(), python=platform.python_version(),
                  settings_sha256=cfg['settings_sha256'], git_commit=git('rev-parse','HEAD'),
                  git_status=git('status','--short'), job_id=os.environ.get('SLURM_JOB_ID'),
                  cpus=os.environ.get('SLURM_CPUS_PER_TASK'), packages={}, scenarios={}, blockers=[],
                  pilot_cells=count, pilot_bootstrap=bootstrap, run_dir=str(root),
                  production_submitted=False, production_ready=False)
    for package in ('numpy','scipy','pandas','pyarrow','netCDF4'):
        report['packages'][package] = importlib.metadata.version(package)
    write_json(root/'started.json',report)
    logging.info('Pilot reports: %s', root)
    try:
        report['raw_samples'] = raw_sample_audit(cfg)
    except Exception as exc:
        report['raw_samples'] = dict(issues=[str(exc)], source_definition_confirmed=False)
    try:
        report['conus'] = baseline_audit(cfg)
        report['blockers'].extend(report['conus']['errors'])
    except Exception as exc:
        report['conus'] = dict(errors=[str(exc)])
        report['blockers'].append(f'CONUS audit: {exc}')
    if not report['raw_samples']['source_definition_confirmed']:
        report['blockers'].append('Historical source variable/grid verification incomplete')
    report['blockers'].extend(report['raw_samples'].get('issues',[]))
    # Audits proceed independently so one broken scenario does not hide the other eight.
    all_fit_seconds, all_fit_cells = 0., 0
    sidecars = dict(cfg.get('coverage_sidecars',{}))
    for scenario in SCENARIO_YEARS:
        tick = time.perf_counter()
        row = dict(status='blocked', issues=[])
        annual = Path(cfg['annual_root'])/f'{scenario}.npz'
        try:
            qa = annual.with_suffix('.qa.json')
            if qa.exists():
                row['qa_report'] = json.loads(qa.read_text())
            else:
                row['issues'].append('Upstream QA JSON absent')
            with np.load(annual,allow_pickle=False) as d:
                row.update(keys=d.files, shape=list(d['maxima'].shape), years=d['years'].tolist(),
                           contributions=d['timestep_contributions'].tolist())
            sidecar = root/'coverage'/f'{scenario}.npz'
            if scenario not in sidecars:
                evidence, inventory = write_coverage(annual,Path(cfg['checkpoint_root'])/scenario,sidecar)
                write_json(root/'coverage'/f'{scenario}.inventory.json',dict(provenance=evidence,checkpoints=inventory))
                sidecars[scenario] = str(sidecar.resolve())
            local_cfg = dict(cfg,coverage_sidecars=sidecars)
            cube, validation = validate_annual(annual,scenario,local_cfg)
            row['validation'] = validation
            grid = grid_path(cfg,scenario)
            lat,lon = coordinates(grid,cfg['grid_shape'])
            row['grid_sha256'] = fingerprint(grid)
            # Each future grid needs its own supplied evidence, never inferred from shape.
            grid_review = cfg.get('grid_reviews',{}).get(scenario)
            row['grid_review'] = grid_review
            if scenario != 'historical' and not grid_review:
                raise ValueError('Future grid provenance requires a documented grid_reviews entry with source evidence')
            if scenario == 'historical' and not report['raw_samples']['source_definition_confirmed']:
                raise ValueError('Historical source definition not verified from available samples')
            sample = select_cells(cube,count,cfg['min_years'])
            row['eligible_cells'] = int((np.isfinite(cube).sum(axis=0)>=cfg['min_years']).sum())
            row['ineligible_cells'] = int(np.prod(cfg['grid_shape']))-row['eligible_cells']
            if not len(sample):
                raise ValueError('No cells meet conservative coverage requirements')
            fits=[]
            fit_start=time.perf_counter()
            for cell in sample:
                fits.append(fit_cell(cube.reshape(40,-1)[:,cell],cfg['return_periods'],
                                     min_years=cfg['min_years'],bootstrap=bootstrap,
                                     seed=np.random.SeedSequence([cfg['seed'],int(cell)])))
            elapsed=time.perf_counter()-fit_start
            arrays={key:np.stack([f[key] for f in fits]) for key in fits[0]}
            arrays.update(cell_id=sample,latitude=lat.ravel()[sample],longitude=lon.ravel()[sample],
                          return_periods=np.asarray(cfg['return_periods']))
            meta=dict(scenario=scenario,source_sha256=validation['source_sha256'],
                      settings_sha256=cfg['settings_sha256'],purpose='exploratory real-data pilot, not production approval',
                      bootstrap=bootstrap,git_commit=report['git_commit'])
            save_product(root/'fits'/f'{scenario}.npz',arrays,meta)
            row.update(status='pilot_completed',sample_cell_ids=sample.tolist(),fit_seconds=elapsed,
                       converged=int(arrays['converged'].sum()),fit_seconds_per_cell=elapsed/len(sample))
            if not row['converged']:
                raise ValueError('Every sampled fit failed; investigate before production')
            all_fit_seconds+=elapsed
            all_fit_cells+=len(sample)
        except Exception as exc:
            row['issues'].append(str(exc))
            report['blockers'].append(f'{scenario}: {exc}')
        row['audit_and_pilot_seconds']=time.perf_counter()-tick
        report['scenarios'][scenario]=row
        write_json(root/'scenario_reports'/f'{scenario}.json',row)
        logging.info('%s: %s (%s)',scenario,row['status'], '; '.join(row['issues']))
    try:
        report['pilot_transfers']=pilot_transfer_checks(cfg,root)
    except Exception as exc:
        report['pilot_transfers']={'status':'blocked','reason':str(exc)}
        report['blockers'].append(f'Pilot geographic transfer: {exc}')
    write_json(root/'coverage_paths.json',sidecars)
    if not cfg.get('overlap_policy_review'):
        report['blockers'].append('Full-record overlapping-wind equality/policy remains unreviewed; timestamp unions alone do not settle it')
    report['review_requirements'] = [
        'Human review of source/grid/overlap evidence and configuration before production',
        'CONUS invalid curves/points must remain quarantined; negative bounds are not physical intervals',
        'CONUS fitting period unconfirmed unless supported provenance supplied',
        'Pilot bootstrap intervals are not final uncertainty estimates']
    report['production_ready'] = not report['blockers']
    report['status'] = 'awaiting_human_review' if report['production_ready'] else 'blocked_requires_review'
    report['wall_seconds'] = time.perf_counter()-started
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_bytes'] = int(rss if platform.system()=='Darwin' else rss*1024)
    if all_fit_cells:
        report['benchmark'] = dict(cells=all_fit_cells,bootstrap=bootstrap,fit_seconds=all_fit_seconds,
                                  seconds_per_cell=all_fit_seconds/all_fit_cells,
                                  note='Measured pilot fit time only; different bootstrap count/scenario/storage may change production cost')
    write_json(root/'report.json',report)
    summary = f"Status: {report['status']}\nReport: {root/'report.json'}\nProduction submitted: no\n"
    summary += '\nBlockers:\n' + '\n'.join('- '+s for s in report['blockers'])+'\n'
    with open(root/'SUMMARY.txt','x') as f:
        f.write(summary)
    print(summary,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--run-dir')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    report=run_pilot(args.config,args.run_dir)
    if not report['production_ready']:
        raise SystemExit(2)


if __name__=='__main__':
    main()

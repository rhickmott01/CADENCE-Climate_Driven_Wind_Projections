"""CLI for completed annual maxima; see docs/PIPELINE.md."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import subprocess
import numpy as np
import scipy
from .pipeline_inputs import SCENARIO_YEARS, settings, validate_annual, coordinates, fingerprint
from .products import save_product, read_product, resume_matches
from .extremes import fit_block
from .factors import climate_factors


def provenance(cfg, stage, **extra):
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    return dict(stage=stage, settings_sha256=cfg['settings_sha256'], git_commit=code,
                numpy=np.__version__, scipy=scipy.__version__, **extra)


def fit_scenario(cfg, scenario, chunk_index=None):
    source = Path(cfg['annual_root']) / f'{scenario}.npz'
    cube, report = validate_annual(source, scenario, cfg)
    lat, lon = coordinates(cfg['grid_file'], cfg['grid_shape'])
    flat = cube.reshape(40, -1)
    root = Path(cfg['results_root'])
    for index, start in enumerate(range(0, flat.shape[1], cfg['chunk_cells'])):
        if chunk_index is not None and index != chunk_index:
            continue
        stop = min(start + cfg['chunk_cells'], flat.shape[1])
        meta = provenance(cfg, 'fit', scenario=scenario, source=report,
                          grid_sha256=fingerprint(cfg['grid_file']), start=start, stop=stop)
        out = root / 'fits' / scenario / f'{index:05d}.npz'
        if resume_matches(out, meta):
            logging.info('Resume %s', out)
            continue
        block = fit_block(flat[:, start:stop], cfg['return_periods'], start_cell=start,
                          min_years=cfg['min_years'], bootstrap=cfg['bootstrap'], seed=cfg['seed'])
        block.update(latitude=lat.ravel()[start:stop], longitude=lon.ravel()[start:stop],
                     cell_id=np.arange(start, stop), return_periods=np.array(cfg['return_periods']))
        save_product(out, block, meta)
        logging.info('Wrote %s (%s converged)', out, block['converged'].sum())


def factor_scenario(cfg, scenario, chunk_index=None):
    if scenario == 'historical':
        return
    root = Path(cfg['results_root'])
    n = int(np.prod(cfg['grid_shape']))
    for index, start in enumerate(range(0, n, cfg['chunk_cells'])):
        if chunk_index is not None and index != chunk_index:
            continue
        paths = [root / 'fits' / sc / f'{index:05d}.npz' for sc in ('historical', scenario)]
        hist, hp = read_product(paths[0])
        fut, fp = read_product(paths[1])
        for data, meta, expected in ((hist, hp, 'historical'), (fut, fp, scenario)):
            if meta['scenario'] != expected or meta['settings_sha256'] != cfg['settings_sha256']:
                raise ValueError('Fit provenance/config mismatch')
            if not np.array_equal(data['cell_id'], np.arange(start, min(start+cfg['chunk_cells'], n))):
                raise ValueError('Fit chunk cell coverage mismatch')
        if hp['grid_sha256'] != fp['grid_sha256']:
            raise ValueError('Historical/future grid provenance mismatch')
        meta = provenance(cfg, 'factor', scenario=scenario,
                          inputs={str(p): fingerprint(p) for p in paths},
                          uncertainty='marginal percentile envelope, not calibrated ratio CI')
        out = root / 'factors' / scenario / f'{index:05d}.npz'
        if not resume_matches(out, meta):
            save_product(out, climate_factors(hist, fut), meta)
        logging.info('Factor %s', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['validate', 'fit', 'factor'])
    p.add_argument('--config', default='config/pipeline.json')
    p.add_argument('--scenario', choices=list(SCENARIO_YEARS))
    p.add_argument('--chunk', type=int)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = settings(args.config)
    for scenario in ([args.scenario] if args.scenario else SCENARIO_YEARS):
        if args.stage == 'validate':
            _, report = validate_annual(Path(cfg['annual_root']) / f'{scenario}.npz', scenario, cfg)
            coordinates(cfg['grid_file'], cfg['grid_shape'])
            out = Path(cfg['results_root']) / 'validation' / f'{scenario}.npz'
            meta = provenance(cfg, 'validate', report=report, grid_sha256=fingerprint(cfg['grid_file']))
            if not resume_matches(out, meta):
                save_product(out, {'report': np.array(json.dumps(report))}, meta)
            logging.info('%s', json.dumps(report))
        elif args.stage == 'fit':
            fit_scenario(cfg, scenario, args.chunk)
        else:
            factor_scenario(cfg, scenario, args.chunk)


if __name__ == '__main__':
    main()

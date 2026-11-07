#!/usr/bin/env python3
"""Summarize completeness and quality flags without approving scientific results."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from im3_wind.pipeline_inputs import SCENARIO_YEARS, settings
from im3_wind.products import read_product


def audit(cfg):
    root = Path(cfg['results_root'])
    with np.load(cfg['conus_file'], allow_pickle=False) as d:
        target = d['latitude'].size
    native = int(np.prod(cfg['grid_shape']))
    report = dict(settings_sha256=cfg['settings_sha256'], complete=True, products={},
                  note='Descriptive QA only; quality flags require scientific review')
    for stage in ('validation', 'fits', 'factors', 'downscaled', 'annual'):
        scenarios = list(SCENARIO_YEARS)
        if stage in ('factors', 'downscaled'):
            scenarios = [s for s in scenarios if s != 'historical']
        elif stage == 'annual':
            scenarios = [s.removesuffix('_near') for s in scenarios if s.endswith('_near')]
        for sc in scenarios:
            if stage == 'validation':
                expected, pattern = 1, root/stage/f'{sc}.npz'
                files = [pattern] if pattern.exists() else []
            else:
                cells = native if stage in ('fits', 'factors') else target
                size = cfg['chunk_cells'] if stage in ('fits', 'factors') else cfg.get('target_chunk_cells', 4096)
                expected = math.ceil(cells/size)
                files = sorted((root/stage/sc).glob('*.parquet' if stage == 'annual' else '*.npz'))
            summary = dict(files=len(files), expected_files=expected, flag_counts={}, rows=0)
            report['complete'] &= len(files) == expected
            for path in files:
                if stage == 'annual':
                    summary['rows'] += pq.ParquetFile(path).metadata.num_rows
                    continue
                data, meta = read_product(path)
                if meta['settings_sha256'] != cfg['settings_sha256']:
                    raise ValueError(f'Settings mismatch in {path}')
                if 'flags' in data:
                    flags = data['flags']
                    summary['rows'] += flags.shape[0]
                    for j, rp in enumerate(data['return_periods']):
                        counts = summary['flag_counts'].setdefault(str(int(rp)), {})
                        for bit in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                            counts[str(bit)] = counts.get(str(bit), 0) + int(np.count_nonzero(flags[:, j] & bit))
            report['products'][f'{stage}/{sc}'] = summary
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/pipeline.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(settings(args.config))
    with open(args.output, 'x') as f:
        json.dump(report, f, indent=2)
    print(f'Complete={report["complete"]}; report: {args.output}')
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

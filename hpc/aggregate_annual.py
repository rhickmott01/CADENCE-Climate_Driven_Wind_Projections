#!/usr/bin/env python3
"""Aggregate existing IM3 weekly per-calendar-year maxima to annual maxima.

Never interprets the filename's year as the data year. Read-only on input.
"""
import argparse
import json
from pathlib import Path
import numpy as np

SCENARIOS = {
    'historical': (1980, 2019),
    'rcp45cooler_near': (2020, 2059),
    'rcp45hotter_near': (2020, 2059),
    'rcp85cooler_near': (2020, 2059),
    'rcp85hotter_near': (2020, 2059),
    'rcp45cooler_far': (2060, 2099),
    'rcp45hotter_far': (2060, 2099),
    'rcp85cooler_far': (2060, 2099),
    'rcp85hotter_far': (2060, 2099),
}

def aggregate_scenario(folder: Path, output: Path, start: int, end: int) -> None:
    files = sorted(folder.rglob('*.npz'))
    if not files:
        raise RuntimeError(f'No NPZ files in {folder}')
    years = np.arange(start, end + 1, dtype=np.int32)
    annual = np.full((len(years), 299, 424), -np.inf, dtype=np.float32)
    valid_obs = np.zeros((len(years), 299, 424), dtype=np.uint32)
    week_contributions = np.zeros(len(years), dtype=np.int32)
    total_count = np.zeros(len(years), dtype=np.int64)
    for file_number, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as d:
            y = np.asarray(d['years'], dtype=np.int32)
            maxima = np.asarray(d['maxima'], dtype=np.float32)
            valid = np.asarray(d['valid_counts'], dtype=np.uint32)
            counts = np.asarray(d['counts'], dtype=np.int64)
            if maxima.shape != (len(y), 299, 424) or valid.shape != maxima.shape or counts.shape != (len(y),):
                raise ValueError(f'Unexpected shape in {path.name}: {y.shape}, {maxima.shape}, {valid.shape}, {counts.shape}')
            if len(np.unique(y)) != len(y):
                raise ValueError(f'Duplicate years in {path.name}')
            if (counts < 0).any() or (valid > counts[:, None, None]).any():
                raise ValueError(f'Invalid sample counts in {path.name}')
            for i, yr in enumerate(y):
                if yr < start or yr > end:
                    # Weeks that cross the period boundary can legitimately contribute
                    # to an adjacent calendar year not modeled in this scenario.
                    if yr not in (start - 1, end + 1):
                        raise ValueError(f'Unexpected out-of-period year {yr} in {path.name}')
                    continue
                idx = int(yr - start)
                week_contributions[idx] += 1
                total_count[idx] += int(counts[i])
                ok = (valid[i] > 0) & np.isfinite(maxima[i])
                annual[idx] = np.fmax(annual[idx], np.where(ok, maxima[i], -np.inf))
                valid_obs[idx] += valid[i]
        if file_number % 250 == 0:
            print(f'{folder.name}: {file_number}/{len(files)} weekly checkpoints', flush=True)
    if np.any(week_contributions == 0) or np.any(total_count == 0):
        raise ValueError(f'Missing years in {folder.name}: {years[(week_contributions == 0) | (total_count == 0)]}')
    annual[~np.isfinite(annual)] = np.nan
    missing = [int(years[i]) for i in range(len(years)) if not np.any(np.isfinite(annual[i]))]
    if missing:
        raise ValueError(f'No finite wind maxima in years {missing}')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, years=years, maxima=annual, valid_counts=valid_obs,
                        weekly_file_contributions=week_contributions,
                        timestep_contributions=total_count, scenario=np.array(folder.name))
    tmp.replace(output)
    qa = {
        'scenario': folder.name, 'files': len(files), 'years': [start, end],
        'shape': list(annual.shape), 'per_year_file_contributions': week_contributions.tolist(),
        'per_year_timestep_contributions': total_count.tolist(),
        'per_year_fraction_valid_cells': np.isfinite(annual).mean(axis=(1, 2)).round(6).tolist(),
        'output': str(output),
        'notes': 'Per-year maxima from checkpoint years, not filenames. Source is 3-hourly IM3 wind, not 3-second gust.'
    }
    output.with_suffix('.qa.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(f'DONE: {folder.name}: {len(files)} files -> {output}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('/work/rjh92/im3_tgw/checkpoints/files'))
    p.add_argument('--output', type=Path, default=Path('/work/rjh92/im3_tgw/annual_maxima'))
    args = p.parse_args()
    for sc, (first, last) in SCENARIOS.items():
        aggregate_scenario(args.input / sc, args.output / f'{sc}.npz', first, last)

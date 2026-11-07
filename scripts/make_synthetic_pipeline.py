#!/usr/bin/env python3
"""Create tiny, reproducible nine-scenario fixtures outside the repository."""
import argparse
import calendar
import json
from pathlib import Path
import numpy as np
from scipy.stats import genextreme
from im3_wind.pipeline_inputs import SCENARIO_YEARS


def make_example(root, bootstrap=25):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    annual = root / 'annual_inputs'
    annual.mkdir(exist_ok=True)
    lat = np.array([[35., 35.], [35.1, 35.1]])
    lon = np.array([[-80., -79.9], [-80., -79.9]])
    def save_new(path, **arrays):
        with open(path, 'xb') as f:
            np.savez_compressed(f, **arrays)
    save_new(root / 'grid.npz', latitude=lat, longitude=lon)
    periods = np.array([10, 25, 50, 100, 200, 500])
    baseline = np.broadcast_to(np.linspace(40, 65, 6), (4, 6)).copy()
    save_new(root / 'conus.npz', latitude=lat, longitude=lon, return_periods=periods,
             return_levels=baseline, units=np.array('mph'), wind_definition=np.array('3-second gust'))
    for i, (sc, (first, last)) in enumerate(SCENARIO_YEARS.items()):
        years = np.arange(first, last+1)
        totals = np.array([(366 if calendar.isleap(int(y)) else 365)*8 for y in years])
        # Matched distribution shape, scenario-dependent multiplier, independent years/cells.
        factor = 1 if sc == 'historical' else 1.1 + .01*i
        maxima = genextreme.rvs(0, loc=25, scale=3, size=(40, 2, 2), random_state=1729+i)*factor
        save_new(annual / f'{sc}.npz', years=years, maxima=maxima,
                 valid_counts=np.broadcast_to(totals[:, None, None], maxima.shape),
                 timestep_contributions=totals, scenario=np.array(sc))
    cfg = json.loads(Path('config/pipeline.json').read_text())
    cfg.update(annual_root=str(annual), results_root=str(root/'results'), scratch_root=str(root/'scratch'),
               grid_file=str(root/'grid.npz'), conus_file=str(root/'conus.npz'), grid_shape=[2, 2],
               chunk_cells=2, target_chunk_cells=2, bootstrap=bootstrap)
    with open(root/'config.json', 'x') as f:
        json.dump(cfg, f, indent=2)
    return root/'config.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--bootstrap', type=int, default=25)
    args = p.parse_args()
    print(make_example(args.output, args.bootstrap))

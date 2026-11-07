"""Coverage evidence from retained, internally 3-hourly weekly checkpoint metadata.

No raw wind reprocessing. The conservative finite-observation lower bound
subtracts *all proven duplicate occurrences*, not an assumed number of weeks.
This does not establish equality of wind values at overlapping timestamps.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .pipeline_inputs import fingerprint
from .products import save_product


def checkpoint_coverage(folder, years, scenario=None):
    paths = sorted(Path(folder).glob('*.npz'))
    if not paths:
        raise ValueError(f'No checkpoints in {folder}')
    timeline, manifest = [], []
    source_names = set()
    contributions = np.zeros(len(years), dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            if scenario is not None and ('scenario' not in d or str(d['scenario'].item()) != scenario):
                raise ValueError(f'Checkpoint scenario mismatch: {path}')
            source_name = str(d['source_file'].item()) if 'source_file' in d else path.name
            if source_name in source_names:
                raise ValueError(f'Duplicate source checkpoint: {source_name}')
            source_names.add(source_name)
            first = np.datetime64(str(d['first_time'].item()), 's')
            last = np.datetime64(str(d['last_time'].item()), 's')
            n = int(d['n_timesteps'].item())
            cy, counts = d['years'], d['counts']
            if n < 1 or np.isnat(first) or np.isnat(last) or last-first != np.timedelta64(10800*(n-1), 's'):
                raise ValueError(f'Invalid checkpoint time range: {path}')
            if int(first.astype('int64')) % 10800:
                raise ValueError(f'Off-calendar 3-hourly timestamp: {path}')
            ts = first + np.arange(n)*np.timedelta64(10800, 's')
            ty = ts.astype('datetime64[Y]').astype(int) + 1970
            uy, uc = np.unique(ty, return_counts=True)
            if not np.array_equal(cy, uy) or not np.array_equal(counts, uc):
                raise ValueError(f'Checkpoint year counts disagree with timestamp range: {path}')
            for i, year in enumerate(years):
                contributions[i] += int(uc[uy == year].sum())
            timeline.append(ts)
            stat = path.stat()
            manifest.append(dict(path=str(path), source_file=source_name, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                 first=str(first), last=str(last), n=n, years=cy.tolist(), counts=counts.tolist()))
    unique = np.unique(np.concatenate(timeline))
    unique_years = unique.astype('datetime64[Y]').astype(int) + 1970
    unique_counts = np.array([np.count_nonzero(unique_years == y) for y in years])
    return dict(unique_timestep_counts=unique_counts,
                duplicate_contributions=contributions-unique_counts,
                timestep_contributions=contributions), dict(
                    evidence_type='checkpoint_timestamp_union_v1', checkpoint_files=len(paths),
                    metadata_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                    assumptions='Each checkpoint range was verified contiguous by the retained extractor; wind equality at duplicate timestamps is not established',
                    checkpoint_metadata=manifest)


def write_coverage(annual_path, checkpoint_folder, output):
    with np.load(annual_path, allow_pickle=False) as d:
        years = d['years']
        totals = d['timestep_contributions']
        scenario = str(d['scenario'].item())
    arrays, provenance = checkpoint_coverage(checkpoint_folder, years, scenario)
    if not np.array_equal(arrays['timestep_contributions'], totals):
        raise ValueError('Annual contributions do not match the complete checkpoint inventory')
    provenance.update(annual_sha256=fingerprint(annual_path), annual_path=str(annual_path))
    arrays['years'] = years
    # Keep the potentially large checkpoint inventory in an evidence JSON, not in every fit.
    inventory = provenance.pop('checkpoint_metadata')
    save_product(output, arrays, provenance)
    return provenance, inventory

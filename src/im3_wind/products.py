"""Versioned, exclusive-write scientific products and provenance."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .pipeline_inputs import fingerprint


def save_product(path, arrays, provenance):
    """Publish atomically without replacing an existing file, even in a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays, provenance=np.array(json.dumps(provenance, sort_keys=True)))
        os.link(name, path)  # atomic exclusive creation on same filesystem
    finally:
        os.unlink(name)


def read_product(path):
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files if k != 'provenance'}
        provenance = json.loads(str(d['provenance'].item()))
    return arrays, provenance


def resume_matches(path, provenance):
    if not Path(path).exists():
        return False
    _, old = read_product(path)
    if old != provenance:
        raise ValueError(f'Stale output {path}; choose a fresh results directory')
    return True

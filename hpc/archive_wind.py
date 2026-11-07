#!/usr/bin/env python3
"""Archive persistent products with checksums; never touch annual inputs."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from im3_wind.pipeline_inputs import settings, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/pipeline.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = settings(args.config)
    root = Path(cfg['results_root']).resolve()
    target = args.output.resolve()
    if target.is_relative_to(root):
        p.error('Archive must be outside results_root')
    files = sorted(f for f in root.rglob('*') if f.is_file())
    if not files:
        p.error('No results to archive')
    manifest = {str(f.relative_to(root)): fingerprint(f) for f in files}
    # Run only after all jobs finish; exclusive mode refuses existing archives.
    with tarfile.open(target, 'x:gz') as tar:
        for f in files:
            tar.add(f, arcname=str(Path('results') / f.relative_to(root)), recursive=False)
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo('sha256_manifest.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    print(f'Archived {len(files)} files to {target}')


if __name__ == '__main__':
    main()

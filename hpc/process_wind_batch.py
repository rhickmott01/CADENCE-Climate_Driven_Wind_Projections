
#!/usr/bin/env python3
"""Process staged IM3 WRF files into validated partial wind maxima."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def process_file(source, output_dir):
    # Read only the wind variables needed for CADENCE.
    # Raw IM3 files contain U10 and V10, not WSPD.
    with Dataset(source) as ds:
        u = ds.variables["U10"]
        v = ds.variables["V10"]
        times = ds.variables["Times"][:]

        if u.shape != v.shape:
            raise ValueError("U10/V10 dimensions do not match")

        # Decode WRF timestamps and check chronological order.
        dates = [
            datetime.strptime(
                b"".join(row).decode("ascii"),
                "%Y-%m-%d_%H:%M:%S"
            )
            for row in times
        ]

        if not dates or dates != sorted(set(dates)):
            raise ValueError("Missing, duplicate, or unsorted timestamps")

        # Calculate one maximum per calendar year and grid cell.
        # We use daily maxima conceptually; taking the maximum
        # of daily maxima gives the same annual maximum.
        yearly = {}
        counts = {}

        for i, date in enumerate(dates):
            a = np.ma.filled(u[i], np.nan)
            b = np.ma.filled(v[i], np.nan)
            wind = np.hypot(a, b).astype(np.float32)

            year = date.year
            if year not in yearly:
                yearly[year] = wind.copy()
                counts[year] = 0
            else:
                yearly[year] = np.fmax(yearly[year], wind)

            counts[year] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save one checkpoint per source file. A file may cross
    # a calendar-year boundary, so store the year explicitly.
    output = output_dir / f"{source.stem}_max.npz"
    np.savez_compressed(
        output,
        years=np.array(sorted(yearly)),
        maxima=np.stack([yearly[y] for y in sorted(yearly)]),
        counts=np.array([counts[y] for y in sorted(yearly)]),
        source_file=source.name,
        first_time=dates[0].isoformat(),
        last_time=dates[-1].isoformat(),
    )

    # Verify that the checkpoint can be read before recording success.
    with np.load(output) as check:
        assert check["maxima"].shape[0] == len(yearly)
        assert np.isfinite(check["maxima"]).any()

    return {
        "source": source.name,
        "checkpoint": str(output),
        "first_time": dates[0].isoformat(),
        "last_time": dates[-1].isoformat(),
        "years": sorted(yearly),
        "timesteps": len(dates),
        "status": "validated",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    sources = sorted(Path(args.input_dir).glob("*.nc"))
    if not sources:
        raise SystemExit("No NetCDF files found")

    results = []
    for source in sources[:args.max_files]:
        print(f"Processing {source.name}", flush=True)
        results.append(
            process_file(source, Path(args.output_dir))
        )

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(results, indent=2))

    print(f"Processed {len(results)} files")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

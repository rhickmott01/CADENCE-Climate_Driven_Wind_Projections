
#!/usr/bin/env python3
"""
CADENCE IM3/HyperFACETS Globus Transfer Controller

Purpose:
    Discover and transfer historical IM3/HyperFACETS
    three-hourly NetCDF files to Duke DCC.

Modes:
    --dry-run   List files without transferring.
    --transfer  Transfer files and wait for completion.

This version does NOT:
    - Delete raw NetCDF files
    - Submit Slurm processing jobs
    - Modify annual wind maxima

Future versions will connect transfers to Slurm processing,
checkpoint validation, and automatic cleanup.
"""

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE_ID = "9392aa20-799e-4630-acb6-7e6a4ed5b795"
DEST_ID = "1ad66c7c-4f60-11e8-900c-0a6d4e044368"

SOURCE_DIR = "/historical_1980_2019/three_hourly/"
DEST_DIR = Path("/work/rjh92/im3_tgw/raw/")

MANIFEST_DIR = Path("/work/rjh92/im3_tgw/manifests/")

FILE_PREFIX = "tgw_wrf_historical_three_hourly_"

# For this pilot, only select the first five
# historical files. Later versions will track
# completed files and advance automatically.


# --------------------------------------------------
# GLOBUS COMMAND HELPER
# --------------------------------------------------

def run_globus(command):
    """Run a Globus CLI command and return its output."""

    result = subprocess.run(
        ["globus"] + command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Globus command failed:\n"
            + result.stderr
            + result.stdout
        )

    return result.stdout


# --------------------------------------------------
# DISCOVER SOURCE FILES
# --------------------------------------------------

def list_source_files():
    """List historical three-hourly NetCDF files."""

    output = run_globus([
        "ls",
        f"{SOURCE_ID}:{SOURCE_DIR}",
    ])

    files = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(FILE_PREFIX)
        and line.strip().endswith(".nc")
    ]

    return sorted(set(files))


# --------------------------------------------------
# CREATE MANIFEST
# --------------------------------------------------

def save_manifest(data, filename):
    """Save a JSON record of the batch."""

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = MANIFEST_DIR / filename
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(data, indent=2)
    )
    temporary.replace(path)

    print(f"Manifest saved: {path}")


# --------------------------------------------------
# TRANSFER ONE FILE
# --------------------------------------------------

def transfer_file(filename):
    """
    Transfer one NetCDF file using Globus.

    A file already present locally is skipped.
    This is only a pilot convenience, not proof
    of file integrity. Production automation
    will require stronger validation.
    """

    destination = DEST_DIR / filename

    if destination.exists():
        print(f"Already staged: {filename}")
        return {
            "filename": filename,
            "status": "already_staged_unverified",
            "task_id": None,
        }

    source_path = SOURCE_DIR + filename
    destination_path = str(destination)

    print(f"Submitting transfer: {filename}")

    output = run_globus([
        "transfer",
        f"{SOURCE_ID}:{source_path}",
        f"{DEST_ID}:{destination_path}",
        "--label",
        f"CADENCE IM3 {filename}",
        "--format",
        "json",
    ])

    result = json.loads(output)
    task_id = result.get("task_id")

    if not task_id:
        raise RuntimeError(
            f"No Globus task ID returned for {filename}"
        )

    print(f"Globus task submitted: {task_id}")

    # Wait until Globus reports task completion.
    # globus task wait returns success for a
    # completed task; inspect status afterward.
    wait_result = subprocess.run([
        "globus", "task", "wait", task_id,
        "--timeout", "7200",
    ])

    if wait_result.returncode != 0:
        raise RuntimeError(
            f"Transfer did not finish within timeout: {task_id}"
        )

    task_output = run_globus([
        "task", "show", task_id,
        "--format", "json",
    ])

    task = json.loads(task_output)

    if task.get("status") != "SUCCEEDED":
        raise RuntimeError(
            f"Transfer task {task_id} status: "
            f"{task.get('status')}"
        )

    if not destination.is_file():
        raise RuntimeError(
            f"Transferred file not found: {destination}"
        )

    print(f"Transfer complete: {filename}")

    return {
        "filename": filename,
        "status": "transferred",
        "task_id": task_id,
        "local_size_bytes": destination.stat().st_size,
    }


# --------------------------------------------------
# MAIN CONTROLLER
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="CADENCE IM3 Globus transfer controller"
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files without transferring",
    )

    mode.add_argument(
        "--transfer",
        action="store_true",
        help="Transfer selected files to DCC",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    files = list_source_files()
    batch = files[:args.batch_size]

    print(f"\nTotal historical files: {len(files)}")
    print(f"Selected files: {len(batch)}\n")

    for index, filename in enumerate(batch, 1):
        print(f"{index}. {filename}")

    manifest = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_collection": SOURCE_ID,
        "destination_collection": DEST_ID,
        "source_directory": SOURCE_DIR,
        "destination_directory": str(DEST_DIR),
        "batch_size": len(batch),
        "mode": "dry_run" if args.dry_run else "transfer",
        "files": [],
    }

    if args.dry_run:

        manifest["files"] = [
            {
                "filename": filename,
                "status": "selected",
            }
            for filename in batch
        ]

        save_manifest(
            manifest,
            "globus_dry_run.json",
        )

        print("\nDRY RUN COMPLETE")
        print("No files transferred or deleted.")
        return

    # Real transfers require an existing DCC
    # staging directory. Never create or delete
    # source files on the remote collection.
    if not DEST_DIR.is_dir():
        raise RuntimeError(
            f"Destination directory missing: {DEST_DIR}"
        )

    # Save a unique manifest so previous runs
    # are not overwritten.
    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    manifest_name = f"globus_transfer_{run_id}.json"

    for filename in batch:

        try:
            result = transfer_file(filename)
            manifest["files"].append(result)

        except Exception as error:
            manifest["files"].append({
                "filename": filename,
                "status": "failed",
                "error": str(error),
            })

            manifest["status"] = "failed"

            save_manifest(
                manifest,
                manifest_name,
            )

            raise

        # Save progress after every completed
        # transfer to support troubleshooting.
        save_manifest(
            manifest,
            manifest_name,
        )

    manifest["status"] = "completed"

    save_manifest(
        manifest,
        manifest_name,
    )

    print("\nTRANSFER BATCH FINISHED")
    print("No source or destination files deleted.")
    print("Slurm processing was not started.")


if __name__ == "__main__":
    main()

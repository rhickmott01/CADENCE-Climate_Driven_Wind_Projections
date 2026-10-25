#!/usr/bin/env python3
"""
CADENCE IM3/HyperFACETS HPC Automation Controller

Runs inside a DCC Slurm allocation.

Workflow:
  1. Discover historical and future IM3 NetCDF files.
  2. Select up to five unprocessed files.
  3. Transfer files from Globus to DCC /work.
  4. Calculate wind maxima using up to four workers.
  5. Validate and save per-file checkpoints.
  6. Record progress in SQLite.
  7. Delete validated temporary raw files.
  8. Repeat until the time limit approaches.

The Globus source is NEVER deleted.

The controller is resumable but requires a new Slurm
submission after its allocation ends.

Important:
  - File checkpoints are partial-year products.
  - Complete calendar-year maxima are assembled later.
  - /work checkpoints require persistent backup.
"""

import argparse
import concurrent.futures
import fcntl
import json
import logging
import os
import re
import sqlite3
import subprocess
import time

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from process_wind_batch import extract


LOG = logging.getLogger("cadence_im3")

SCENARIOS = {
    "historical": "historical_1980_2019",
    "rcp45cooler_near": "rcp45cooler_2020_2059",
    "rcp45cooler_far": "rcp45cooler_2060_2099",
    "rcp45hotter_near": "rcp45hotter_2020_2059",
    "rcp45hotter_far": "rcp45hotter_2060_2099",
    "rcp85cooler_near": "rcp85cooler_2020_2059",
    "rcp85cooler_far": "rcp85cooler_2060_2099",
    "rcp85hotter_near": "rcp85hotter_2020_2059",
    "rcp85hotter_far": "rcp85hotter_2060_2099",
}

PATTERN = re.compile(
    r"^tgw_wrf_.*_three_hourly_"
    r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.nc$"
)



def cli(*args, timeout=120):
    """
    Run Globus CLI commands with retries for transient failures.

    Do not automatically retry transfer submissions because
    the remote service may have accepted a transfer even if
    the response was lost.
    """
    command = ["globus", *map(str, args)]
    is_submission = bool(args) and args[0] == "transfer"
    max_attempts = 1 if is_submission else 6

    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return result.stdout

            error = result.stderr + "\n" + result.stdout

        except subprocess.TimeoutExpired as exc:
            error = str(exc)

        transient_patterns = (
            "operation timed out",
            "connection timed out",
            "connection reset",
            "temporarily unavailable",
            "externalerror",
            "service unavailable",
        )

        transient = any(
            pattern in error.lower()
            for pattern in transient_patterns
        )

        if not transient or attempt == max_attempts - 1:
            raise RuntimeError(
                f"Globus command failed:\n{error}"
            )

        delay = min(60 * (2 ** attempt), 900)

        LOG.warning(
            "Temporary Globus failure. Retrying in %d seconds "
            "(attempt %d of %d). Error: %s",
            delay,
            attempt + 1,
            max_attempts,
            error[:300],
        )

        time.sleep(delay)


@contextmanager
def pipeline_lock(db_path):
    """Prevent concurrent controllers using the same state."""

    lock_path = Path(str(db_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another CADENCE controller is already running."
            )

        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def init_db(path):
    """Initialize persistent processing history."""

    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=DELETE")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            scenario TEXT NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            checkpoint TEXT,
            task_id TEXT,
            updated TEXT,
            PRIMARY KEY (scenario, filename)
        )
    """)

    conn.commit()
    return conn


def mark(conn, scenario, filename, status,
         checkpoint=None, task_id=None):
    """Update one file's persistent processing state."""

    conn.execute("""
        INSERT INTO files
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(scenario, filename)
        DO UPDATE SET
            status=excluded.status,
            checkpoint=COALESCE(
                excluded.checkpoint, files.checkpoint
            ),
            task_id=COALESCE(
                excluded.task_id, files.task_id
            ),
            updated=excluded.updated
    """, (
        scenario,
        filename,
        status,
        checkpoint,
        task_id,
        datetime.now(timezone.utc).isoformat(),
    ))

    conn.commit()


def inventory(config):
    """Discover all nine IM3 scenario-period directories."""

    items = []

    for scenario, folder in SCENARIOS.items():
        source_dir = f"/{folder}/three_hourly/"

        output = cli(
            "ls",
            f"{config['source_collection']}:{source_dir}",
            timeout=300,
        )

        files = sorted(set(
            line.strip()
            for line in output.splitlines()
            if PATTERN.fullmatch(line.strip())
        ))

        if not files:
            raise RuntimeError(
                f"No matching NetCDF files in {source_dir}"
            )

        LOG.info("%s: %d files", scenario, len(files))

        for filename in files:
            items.append((scenario, source_dir, filename))

    return items


def checkpoint_valid(path, filename, scenario):
    """
    Validate a processed file-level wind checkpoint.

    This checks internal metadata and sampled-time coverage.
    It does not establish complete calendar-year coverage.
    """

    path = Path(path)

    if not path.is_file():
        return False

    try:
        with np.load(path, allow_pickle=False) as z:
            maxima = z["maxima"]
            years = z["years"]
            counts = z["counts"]
            n = int(z["n_timesteps"])

            first = datetime.fromisoformat(
                str(z["first_time"])
            )
            last = datetime.fromisoformat(
                str(z["last_time"])
            )

            if first > last:
                return False

            seconds = (last - first).total_seconds()

            if seconds % 10800 != 0:
                return False

            expected_steps = int(seconds / 10800) + 1

            if maxima.ndim != 3:
                return False

            if maxima.shape[1:] != (299, 424):
                return False

            if len(years) != maxima.shape[0]:
                return False

            if len(counts) != len(years):
                return False

            if n <= 0 or expected_steps != n:
                return False

            if int(counts.sum()) != n:
                return False

            if str(z["source_file"]) != filename:
                return False

            if str(z["scenario"]) != scenario:
                return False

            if not np.isfinite(maxima).any():
                return False

            if np.isfinite(maxima).any() and (
                np.nanmin(maxima) < 0
            ):
                return False

            return True

    except Exception as error:
        LOG.warning(
            "Checkpoint validation failed for %s: %s",
            path, error
        )
        return False


def globus_task_status(task_id):
    """Retrieve the status of a Globus transfer."""

    output = cli(
        "task", "show", task_id,
        "--format", "json"
    )

    return json.loads(output).get("status", "UNKNOWN")


def transfer(config, scenario, source_dir,
             filename, local, conn, deadline):
    """
    Transfer one file, or resume a previous transfer.

    Reuse a successful task only if the local file
    still exists and is nonempty.
    """

    row = conn.execute(
        """
        SELECT status, task_id
        FROM files
        WHERE scenario=? AND filename=?
        """,
        (scenario, filename)
    ).fetchone()

    task_id = None

    if row and row[0] in ("transferring", "staged"):
        task_id = row[1]

    if task_id:
        status = globus_task_status(task_id)

        if status == "SUCCEEDED":
            if local.is_file() and local.stat().st_size > 0:
                LOG.info(
                    "Reusing completed transfer: %s",
                    filename
                )
                mark(
                    conn, scenario, filename,
                    "staged", task_id=task_id
                )
                return

            LOG.warning(
                "Previous transfer completed but local file "
                "is missing. Re-transferring %s",
                filename
            )
            task_id = None

        elif status in ("FAILED", "INACTIVE", "CANCELED"):
            LOG.warning(
                "Previous task %s ended with %s; retrying",
                task_id, status
            )
            task_id = None

        else:
            LOG.info(
                "Resuming Globus task %s for %s",
                task_id, filename
            )

    if not task_id:
        source = (
            f"{config['source_collection']}:"
            f"{source_dir}{filename}"
        )

        destination = (
            f"{config['destination_collection']}:"
            f"{local}"
        )

        output = cli(
            "transfer",
            source,
            destination,
            "--label",
            f"CADENCE {scenario} {filename}",
            "--format", "json",
        )

        task_id = json.loads(output)["task_id"]

        mark(
            conn, scenario, filename,
            "transferring", task_id=task_id
        )

        LOG.info(
            "Submitted transfer %s: %s",
            filename, task_id
        )

    while time.monotonic() < deadline:
        status = globus_task_status(task_id)

        if status == "SUCCEEDED":
            if not local.is_file():
                raise RuntimeError(
                    f"Transferred file missing: {local}"
                )

            if local.stat().st_size <= 0:
                raise RuntimeError(
                    f"Transferred file empty: {local}"
                )

            mark(
                conn, scenario, filename,
                "staged", task_id=task_id
            )
            return

        if status in ("FAILED", "INACTIVE", "CANCELED"):
            raise RuntimeError(
                f"Globus task {task_id} ended with {status}"
            )

        time.sleep(config.get("poll_seconds", 30))

    raise TimeoutError(
        "Controller deadline approaching. "
        "Globus task remains recorded for resumption."
    )


def process_one(args):
    """Worker entry point for parallel processing."""

    source, output, scenario = args

    return extract(
        Path(source),
        Path(output),
        scenario
    )


def run_pipeline(args):
    """Execute the automated transfer-processing loop."""

    config = json.loads(
        Path(args.config).read_text()
    )

    # Reserve five minutes inside the configured runtime,
    # plus 30 minutes before beginning another batch.
    deadline = (
        time.monotonic()
        + args.max_runtime_hours * 3600
        - 300
    )

    root = Path(config["work_root"]).resolve()
    rawroot = root / "raw"

    checkpoints = Path(
        config["checkpoint_root"]
    ).resolve()

    dbpath = Path(config["state_db"]).resolve()

    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)

    with pipeline_lock(dbpath):
        conn = init_db(dbpath)

        try:
            items = inventory(config)

            # Only trust completed SQLite records if their
            # corresponding checkpoints still validate.
            completed = set()

            rows = conn.execute("""
                SELECT scenario, filename, checkpoint
                FROM files
                WHERE status='completed'
            """).fetchall()

            for scenario, filename, checkpoint in rows:
                if checkpoint and checkpoint_valid(
                    Path(checkpoint), filename, scenario
                ):
                    completed.add((scenario, filename))
                else:
                    LOG.warning(
                        "Invalid completed checkpoint: %s/%s",
                        scenario, filename
                    )
                    if not args.dry_run:
                        mark(
                            conn, scenario, filename,
                            "needs_reprocessing"
                        )

            remaining = [
                item for item in items
                if (item[0], item[2]) not in completed
            ]

            LOG.info(
                "Total files: %d; completed: %d; pending: %d",
                len(items), len(completed), len(remaining)
            )

            if args.dry_run:
                for scenario, source_dir, filename in (
                    remaining[:args.batch_size]
                ):
                    LOG.info(
                        "WOULD PROCESS %s/%s",
                        scenario, filename
                    )
                return

            for offset in range(
                0, len(remaining), args.batch_size
            ):
                # Do not start another batch with fewer than
                # 30 minutes of the controller budget left.
                if time.monotonic() > deadline - 1800:
                    LOG.info(
                        "Approaching runtime limit; "
                        "stopping before next batch."
                    )
                    break

                batch_number = (
                    offset // args.batch_size + 1
                )

                if (
                    args.max_batches
                    and batch_number > args.max_batches
                ):
                    break

                batch = remaining[
                    offset:offset + args.batch_size
                ]

                staged = []

                # Phase 1: Obtain required source files.
                for scenario, source_dir, filename in batch:
                    output = (
                        checkpoints / scenario /
                        (Path(filename).stem + "_max.npz")
                    )

                    # Existing validated checkpoints can be
                    # reused without downloading raw data.
                    if checkpoint_valid(
                        output, filename, scenario
                    ):
                        mark(
                            conn, scenario, filename,
                            "validated",
                            checkpoint=str(output)
                        )
                        LOG.info(
                            "Reusing checkpoint %s", output
                        )
                        continue

                    local = rawroot / scenario / filename
                    local.parent.mkdir(
                        parents=True, exist_ok=True
                    )

                    transfer(
                        config, scenario, source_dir,
                        filename, local, conn, deadline
                    )

                    staged.append((
                        str(local),
                        str(output),
                        scenario
                    ))

                # Phase 2: Process independent files
                # concurrently using four CPU workers.
                if staged:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=args.workers
                    ) as pool:

                        futures = {
                            pool.submit(process_one, item): item
                            for item in staged
                        }

                        for future in (
                            concurrent.futures.as_completed(
                                futures
                            )
                        ):
                            item = futures[future]

                            try:
                                result = future.result()

                                scenario = item[2]
                                filename = Path(
                                    item[0]
                                ).name

                                if not checkpoint_valid(
                                    Path(item[1]),
                                    filename,
                                    scenario
                                ):
                                    raise RuntimeError(
                                        "Worker output failed "
                                        "checkpoint validation"
                                    )

                                mark(
                                    conn, scenario, filename,
                                    "validated",
                                    checkpoint=item[1]
                                )

                                LOG.info(
                                    "Validated %s/%s",
                                    scenario, result["source"]
                                )

                            except Exception:
                                LOG.exception(
                                    "Processing failed: %s",
                                    item[0]
                                )
                                raise

                # Phase 3: Validate every checkpoint
                # before deleting any staged raw files.
                for scenario, source_dir, filename in batch:
                    output = (
                        checkpoints / scenario /
                        (Path(filename).stem + "_max.npz")
                    )

                    if not checkpoint_valid(
                        output, filename, scenario
                    ):
                        raise RuntimeError(
                            f"Checkpoint invalid: {output}. "
                            "Raw files retained."
                        )

                # Phase 4: Safe local cleanup.
                for scenario, source_dir, filename in batch:
                    output = (
                        checkpoints / scenario /
                        (Path(filename).stem + "_max.npz")
                    )

                    local = rawroot / scenario / filename

                    # Never delete outside the scenario
                    # staging directory.
                    expected_parent = (
                        rawroot / scenario
                    ).resolve()

                    if local.resolve().parent != expected_parent:
                        raise RuntimeError(
                            "Unsafe deletion path"
                        )

                    if local.is_symlink():
                        raise RuntimeError(
                            "Refusing to delete symlink"
                        )

                    if local.exists():
                        local.unlink()
                        LOG.info(
                            "Deleted staged raw file: %s",
                            local
                        )

                    mark(
                        conn, scenario, filename,
                        "completed",
                        checkpoint=str(output)
                    )

                LOG.info(
                    "Batch %d completed (%d files)",
                    batch_number, len(batch)
                )

            LOG.info(
                "Controller stopped normally. "
                "Resubmit the Slurm job to resume."
            )

        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="CADENCE IM3 HPC automation"
    )

    parser.add_argument(
        "--config",
        default="hpc/pipeline_config.json"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=23
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means unlimited"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if not 1 <= args.batch_size <= 5:
        parser.error("batch-size must be between 1 and 5")

    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")

    if args.max_runtime_hours <= 0:
        parser.error("max-runtime-hours must be positive")

    if args.max_batches < 0:
        parser.error("max-batches cannot be negative")

    run_pipeline(args)


if __name__ == "__main__":
    main()

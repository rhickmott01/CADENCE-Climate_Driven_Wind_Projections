# Human-submitted DCC audit, pilot, and compact production

This workflow runs ordinary Python in Slurm. Codex need not run on DCC, and no
pilot or Python entry point submits jobs. Develop/test locally, synchronize code,
review the settings/scripts, and submit from an ordinary DCC login terminal.
This is an execution guide, not institutional compliance certification.

## What exists and what must be checked

The local checkout contains code and synthetic tests, not the DCC scientific
inputs. The nine annual NPZs and saved historical grid were located during the
previous investigation. The CONUS baseline retrieval and quality inspection
were reported by the user; their artifacts are not in this Mac checkout.

`config/dcc.production.json` supplies all paths. Its CONUS paths assume artifacts
are under the DCC repository's `Data/` directory. Verify or change these paths:

- `conus_raw_csv`: original CSV (551189043 bytes, SHA-256
  `d18d08af8487277b1912c2fd888d56a16705313e0abc13e7adc98d33b456fe58`).
- `conus_file`: existing quality-flagged Parquet, not a replacement download.
- `conus_inspection`: existing JSON inspection report.
- `grid_file`, `checkpoint_root`, `annual_root`, `source_samples`, `conda_env`.

The adapter accepts explicit wide or long schemas. Default wide columns are
`lat_idx`, `lon_idx`, `latitude`, `longitude`, `converged`, and
`rp_T_3sec_gust`/`rp_T_3sec_gust_lower`/`rp_T_3sec_gust_upper`. Set `conus_layout`
and `conus_columns` if the existing derived Parquet differs. Long layout needs
`return_period`, one gust field, optional bound fields, and grid keys/coordinates
on each row. Unsupported schemas are reported with available columns; they are
never guessed. Units and gust definition are explicit configuration assertions,
consistent with the reported inspection: m/s, already converted to 3-second gust.
No additional gust or mph conversion occurs.

The original table remains unchanged. Baseline quality bits (separate from IM3
flags) are: 1 invalid/nonpositive point; 2 decreasing return-level curve;
4 convergence unconfirmed/failed; 8 missing/nonphysical interval;
16 historical period unconfirmed. All points of a decreasing curve are quarantined
from consumer winds; factors and raw future products remain available for audit.
Negative uncertainty bounds remain raw evidence and are not physical intervals.
Unknown upstream quality columns remain in the original source and are named
in the audit; their semantics must be reviewed, not assumed equivalent to these
new bits. The pilot blocks publication planning until `conus_upstream_quality_review`
records a review of those meanings; compact lookups retain their original string
values in `source_quality_*` columns. Missing convergence is treated as unconfirmed, not silently accepted.

An unconfirmed CONUS fitting period is a provenance limitation, not grounds to
invent dates or to substitute this product for matched 1980–2019 calibration.
IM3 future/historical ratios are calculated within IM3 and can be applied to
this existing baseline without matched-period CONUS refitting. This does not
establish scientific approval of either model or gust conversion.

## Submit exactly one audit/pilot job

On DCC, preserve any local modifications before updating. Do not reset the checkout.

```bash
cd /hpc/home/rjh92/CADENCE-Climate_Driven_Wind_Projections
git status --short
# Only after confirming a clean checkout:
git pull --ff-only origin feature/im3-hpc-automation
mkdir -p logs
```

Review/edit `config/dcc.production.json`. For site-only settings, copy it outside
Git and pass that path instead. Choose a new results_root for an independent run.
The Python environment must already contain the declared dependencies. Job scripts
use `PYTHONPATH` to run this checkout, without installing or changing environments.
A bootstrap `python` command must be available to parse the configured interpreter
path. Slurm must start in the repository root (the directory from which sbatch is
called); logs/ must exist before submission. No scheduler cluster option is used.

```bash
sbatch --partition=interactive \
  --output=logs/im3-pilot-%j.log \
  hpc/run_real_pilot.sbatch config/dcc.production.json
```

The human can override partition/account/resources with sbatch options. Defaults
are one node/task, 2 CPUs, 8 GB, 2 hours. These are starting allocations, not DCC
benchmarks. The pilot limits numerical library threads and fits cells serially.

```bash
squeue -u rjh92
sacct -j JOBID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
tail -f logs/im3-pilot-JOBID.log
# If necessary, cancel only the identified project job:
scancel JOBID
```

Replace JOBID with Slurm's returned number. Closing the SSH connection does not
cancel a submitted batch job. No AI agent is embedded in that job.

## What the pilot actually does

Reports go under `results_root/pilot_runs/UTCSTAMP-JOBID/`:

- `started.json`: settings hash, code revision/status, packages, host/allocation.
- `coverage/SCENARIO.npz`: timestamp-union coverage evidence tied to annual SHA-256.
- `coverage/SCENARIO.inventory.json`: paths, file stats, time ranges, counts and
  digest of every audited checkpoint's metadata.
- `scenario_reports/SCENARIO.json`: independent audit result for each of nine inputs.
- `fits/SCENARIO.npz`: up to 16 deterministic eligible-cell fits with 25 bootstrap
  replicates (exploratory intervals, not final production uncertainty).
- `coverage_paths.json`: reusable coverage evidence locations.
- `transfers/SCENARIO.npz`: sparse pilot factor/transfer comparisons where
  common historical/future sample cells exist; unsampled IDW neighbors stay missing.
- `report.json` and `SUMMARY.txt`: blockers, fit timing and peak memory, review needs.

It inspects all nine annual schemas/QA reports, the raw baseline hash and derived
table, and configured historical source samples. Sample inspection reads only
coordinates, timestamps, and available overlapping wind slices; it does not
reprocess raw annual maxima. Source wind is `hypot(U10,V10)` in m s-1; embedded
conflicting metadata fails. Legacy WSPD inputs remain supported explicitly in
other workflows, but the DCC settings describe the actual HPC extractor.

Coverage is reconstructed from each checkpoint's first_time, last_time,
n_timesteps, years and counts. These must imply a contiguous calendar-aligned
three-hourly sequence and agree with annual summed contributions. The extractor
originally checked internal continuity; this audit relies on that metadata claim.
It does not rediscover timestamps from filenames or prove source file integrity.

For each cell/year, a lower bound on unique finite samples is:
`max(valid_contributions - proven_duplicate_occurrences, 0)`.
Subtracting every duplicate occurrence is conservative even if some duplicated
records are missing at a cell. Eligibility requires this lower bound to meet the
configured calendar coverage. No guessed week count, clipping, or disabling of
validation is used. Wind equality at duplicate timestamps is a separate question.
Original counts and maxima are never rewritten.

Missing/stale/malformed evidence blocks that scenario but does not hide the
other scenario audits. Existing sidecars can be reused through `coverage_sidecars`
after interruption; annual hashes/counts are revalidated. Partial reports persist
per scenario. An interrupted run can therefore be restarted with reviewed sidecars
from completed scenarios; choose a new pilot run directory, not an overwritten run.

Future scenario grids require actual coordinate files plus `grid_reviews` entries
recording their supporting evidence. The default historical file alone does not
verify future geography. The historical pilot also requires source sample checks.
`overlap_policy_review` must document reviewed full-record equality evidence or an
explicit scientific treatment of overlapping records. Do not populate these fields
with an unsupported blanket assertion just to pass a gate.

**With the currently known inputs, the first pilot is expected to report blockers**
for future-grid provenance and full-record overlap policy. It may still produce
eligible historical fits and valuable coverage evidence. Exit code 2 means the
audit finished with blockers, not that all outputs were lost. Exit 1 indicates
an unexpected runtime/configuration failure. A zero exit means ready for human
review, not scientific approval. No production is launched in any case.

## Review and production gate

Review all nine scenario statuses, coverage evidence, raw overlap comparisons,
baseline adapter/flags, and fit failures/uncertainty. Resolve missing inputs and
record evidence, rerun the pilot with the final settings, and review again.
A clean committed checkout is required before production planning.

The pilot benchmark reports measured seconds/cell at its configured bootstrap
count. Do not linearly promise production time from that number: bootstrap count,
scenario behavior, filesystem traffic, and parallelism affect it. Inspect sacct
memory/time and adjust resources before full submission. Default production uses
200 bootstrap replicates; a 25-replicate pilot is not a replacement.

On a compute allocation with the project Python (planning reads only small
reports/configs and produces scripts; it never invokes sbatch):

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/work/rjh92/conda_envs/cadence-wind-rebuilt/bin/python -m im3_wind.manual plan \
  --config config/dcc.production.json \
  --pilot-report /ABSOLUTE/PILOT/RUN/report.json \
  --output /hpc/home/rjh92/CADENCE_Data/IM3/reviewed_plan_v1 \
  --partition interactive --concurrency 8 --reviewed-pilot
```

The acknowledgement flag records human review. Planning refuses blocked/changed
settings, incomplete pilot scenarios, dirty pilot checkouts or a different code
revision. It produces an effective configuration with the verified coverage
sidecars, `plan.json`, and `submit_reviewed.sh`. Review all three, including
resources, allocation/account and destination paths. Add --account if needed.
The script uses one array across all scenarios per stage, with afterok barriers
and at most eight active tasks globally, avoiding per-scenario multiplication.
It submits only native fits, native factors, and a compact geographic crosswalk.
There is no billion-row annual export. Scripts contain ordinary Python only.

Only after this separate human review, execute from the login terminal:

```bash
bash /hpc/home/rjh92/CADENCE_Data/IM3/reviewed_plan_v1/submit_reviewed.sh
```

The script prints exact job IDs; logs are under the reviewed plan's logs/.
Pending dependencies may remain after failures; inspect those jobs and cancel
only obsolete project jobs before resubmission. Identical valid fit/crosswalk
chunks are skipped. Stale products fail; choose a new versioned destination
rather than overwrite. Keep the code/config stable throughout an approved run.

## Compact CADENCE consumption

Keep the immutable CONUS baseline, eight native factor grids, and crosswalk
partitions. Calculate factors/winds for selected keys and years at runtime:

```python
from im3_wind.pipeline_inputs import settings
from im3_wind.compact import lookup
cfg = settings('/path/to/reviewed_plan_v1/effective_config.json')
rows = lookup(cfg, 'rcp45cooler', 2050, 100, [65538])
# Review flags; future_wind is NaN when the baseline point/curve is unusable.
# change_factor can remain valid even when its baseline is quarantined.
```

The existing documented log-linear midpoint method supplies 2026–2060 annual
trajectories on demand; this is an assumed scenario transition, not an observed
annual trend. All six return periods have independent factors. Bad baseline
curves are never repaired by positive scaling. Consumers must reject/quarantine
invalid requested winds rather than silently dropping keys.

The legacy `submit_wind.py --submit` workflow is not this reviewed compact route.
Do not use it for this run: it materializes annual tables and has different
concurrency semantics. Existing full-grid exports remain available for deliberate
special uses. This change does not deploy/update the live CADENCE application.

## Local verification versus DCC execution

Local tests exercise timestamp overlaps, stale evidence, malformed metadata,
baseline curve/bound quarantine, wide/long adapters, pilot reports, review gates,
array command construction and compact lookups using synthetic files. No DCC
connection, data transfer, scheduler submission, or real scientific computation
is performed from this local development session. Real results require the
human-submitted pilot and subsequent reviewed production jobs described above.

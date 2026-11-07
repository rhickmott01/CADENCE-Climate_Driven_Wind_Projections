# NPZ production workflow

This workflow consumes completed annual NPZ maxima; it never reprocesses raw
three-hourly data. Legacy numbered scripts remain available but are not the
production entry point described here.

## Input contract

`config/pipeline.json` holds all DCC locations and scientific settings. Nine
scenario filenames match `hpc/aggregate_annual.py`: historical and four forcings
with `_near` / `_far`. Each contains years, maxima, valid_counts,
timestep_contributions, and scalar scenario. Inputs are read-only. Require
40 ordered calendar years and the configured grid (production 299 x 424).
A cell-year is excluded from fitting if fewer than 95% of expected Gregorian
3-hourly observations are valid. Counts above calendar expectations fail, as
they may indicate duplicated checkpoints. Missing/low-coverage records are
reported, never converted to zero.

The existing aggregator omits units and geographic coordinates. The configured
legacy metadata assertion MUST be checked against source NetCDF metadata on
DCC. Embedded metadata takes precedence and conflicts fail. Supply `grid_file`
as an NPZ containing 2-D `latitude`, `longitude` in degrees north/east, extracted
from the same IM3 source grid. Shape alone cannot establish spatial identity.
No real scientific validation has been completed locally.

## GEV model and diagnostic contract

`extremes.py` fits a stationary MLE independently to each cell/slice. It uses
three standardized Nelder-Mead starts, explicit success, finite likelihood and
support checks, and c in [-1,1] numerical safeguards. SciPy c is minus conventional
xi. Shape-boundary fits and nonregular c>=0.5 are flagged. Compare alternatives
before accepting production extremes. This is not a nonstationary model.
See [SciPy fit documentation](https://scipy.github.io/devdocs/reference/generated/scipy.stats.rv_continuous.fit.html)
for optimizer and likelihood conventions.

95% percentile iid-year bootstrap intervals (default 200 replicates) describe
sampling uncertainty only. Require >=80% and at least 20 successful replicates.
Tail diagnostics: upper-20% quantile RMSE divided by scale, descriptive
Anderson-Darling statistic (not a calibrated p-value), and relative return-level
change after removing the largest observation. Flags are bitwise OR:
1 failed, 2 return period exceeds sample length, 4 insufficient bootstrap,
8 interval width >50% of estimate, 16 tail RMSE >0.5 scale, 32 boundary/nonregular
shape, 64 top-observation sensitivity >25%, 128 nonpositive/nonfinite estimate.
These thresholds are screening assumptions, not proof of validity. In particular
all 50-500 year levels from 40 years are explicitly marked extrapolated.

## Climate factors

`factor` divides future by historical return levels for matching cell IDs,
coordinates, and return periods; shape/grid fingerprint mismatches fail.
Outputs retain all cells. NaN/zero/negative levels produce NaN factors and
bit 256. Upstream flags propagate even when a numerical ratio exists. Bounds
are future lower / historical upper and future upper / historical lower:
a conservative marginal-interval envelope, **not** a calibrated 95% ratio CI.
Historical/future dependence and climate-model uncertainty are not estimated.
Arrays use `(cell, return_period)`; coordinates and cell IDs are 1-D native IM3
row-major IDs. NPZ products contain JSON provenance with source hashes, settings
hash, code commit and NumPy/SciPy versions. No pickle loading is needed.

## CONUS404 geographic transfer

Inspected the existing `scripts/06*`, `scripts/07*`, and the adjacent main CADENCE
`src/cadence/reference_data/climate_delta.py` and README. The former expect
`spduv10max`, `lat_2d`, `lon_2d`, `years` in historical HDF5; the latter uses
`lat_idx`, `lon_idx` and wide `scale_YEAR` Parquet. No standalone CONUS404 or
Climate_Delta_Creation repository was available locally or among repositories
exposed by the authenticated GitHub account. The real grid is unavailable.

The configurable `conus_file` interface is NPZ with 2-D latitude/longitude,
return_periods, return_levels shaped `(flattened cell, return period)`, scalar
units (`m s-1` or `mph`), and explicit scalar wind_definition (e.g. `3-second gust`).
Optional baseline_lower/baseline_upper use the same shape. The caller must adapt
its real baseline schema, verify averaging time/units, grid index orientation,
and original uncertainty definition. Never infer geography from row indices.

Primary mapping uses nearest native IM3 center on the sphere, with a configurable
30-km maximum distance; out-of-range cells remain present with NaNs and bit 512.
This is a distance-limited domain test, not a polygon land mask. Supply masked
CONUS baseline values where needed (bit 1024 invalid baseline). Secondary inverse
squared distance mapping over up to four neighbors reports method sensitivity,
not a probabilistic interval. Missing contributing factors are not renormalized
away. IDW flags retain all contributing source flags.

`future_wind(T) = baseline(T) * factor(T)` preserves the original high-resolution
baseline. Transferring a dimensionless three-hourly climate signal to gusts is
an explicit assumption of invariant relative change across averaging times;
it does not establish their equivalence. Validate this with observations or
appropriate gust simulations. Conditional future bounds multiply factor bounds
by a fixed baseline; optional combined bounds multiply both marginal envelopes
and are not calibrated joint intervals. No baseline uncertainty is invented.

## Annualization and CADENCE integration

Scientific exports contain every cell/RP/year for 2026-2060 for each of four
forcings. Log-linear interpolation joins multiplier 1 at 1999.5, near factor at
2039.5, and far factor at 2079.5. This is an assumed smooth scenario transition,
not an inferred annual trend or prediction of annual maximum wind. Midpoints
summarize climate slices, not measured annual conditions. Factors stay positive;
no extrapolation is needed. Bounds undergo the same transition and represent
interpolated marginal envelopes, not annual confidence intervals. Near flags
propagate throughout; far flags also propagate after 2039.5. Missing values
remain rows with NaNs/flags.

Partitioned Parquet under `annual/SCENARIO/` includes wind_grid_id, lat_idx,
lon_idx, latitude/longitude, source_cell_id, scenario, year, return_period,
change_factor, factor_lower/upper, flags, historical_wind, future_wind, units,
wind_definition, transition_method and provenance. Packed IDs match main
CADENCE: `lat_idx * 65536 + lon_idx`.

The `export --compatibility-rp T` adapter explicitly selects one forcing and
return period, generating `compatibility/SCENARIO/rp_T/` with lat_idx, lon_idx
and scale_2025 through scale_2060. 2025 is an additional compatibility-only
transition year. Flags are rejected by default; `--allow-flags MASK` requires an
explicit reviewed bitmask (2 accepts extrapolation alone). NaN/invalid factors
always fail compatibility publication, even if their flags are allowed.

The main CADENCE consumer inspected here applies a single multiplier to all six
return periods and hardcodes source years 2025-2050, extrapolating thereafter.
It can read a chosen compatibility partition for existing 2025-2050 runs.
Full scientific integration requires accepting scenario and return-period keys,
using the exported factors through 2060, and updating its beyond-range policy.
Do not replace a return-period-dependent product with an unreviewed average.
The separate Climate_Delta_Creation interface remains unverified; the long table
provides a documented adapter boundary, not a claim of drop-in compatibility.

Example downstream consumption (no changes made to the adjacent repository):

```python
import pandas as pd
rows = pd.read_parquet('results_v2/annual/rcp45cooler')
selected = rows.query('year == 2050 and return_period == 100')
# Preserve quality flags and join selected cells, never silently drop missing keys.
assets = pd.DataFrame({'wind_grid_id': [65538]})
lookup = assets.merge(selected, on='wind_grid_id', how='left', validate='many_to_one')
assert lookup.change_factor.notna().all()
# future_wind already equals the high-resolution historical baseline * factor.
```

## DCC setup and exact execution commands

Run from the repository root. Confirm source units, observation calendar,
coordinate identity, baseline wind definition, and grid orientation before a
production submission. A shared grid_file asserts all nine inputs share that
verified grid; optional scenario_grid_files maps each scenario to its own grid
NPZ, allowing coordinate comparisons to reject mismatches. Never use synthetic
coordinates for real inputs. Annual counts cannot prove timestamp uniqueness;
retain upstream aggregation QA as evidence.

```bash
# On DCC, clone once or fast-forward the existing clean checkout.
git clone --branch feature/im3-hpc-automation https://github.com/rhickmott01/CADENCE-Climate_Driven_Wind_Projections.git
cd CADENCE-Climate_Driven_Wind_Projections
# For an existing checkout instead: git pull --ff-only origin feature/im3-hpc-automation

# Activate the rebuilt environment specified in the settings file.
conda activate "$(python -c 'import json; print(json.load(open("config/pipeline.json"))["conda_env"])')"
python -m pip install -e .
python -m pip install pytest
python -m pytest -q

# Copy config/pipeline.json to a site-specific settings file outside the repository.
# Set grid_file and conus_file, confirm metadata, and choose a fresh results_root.
export WIND_CONFIG="$HOME/CADENCE_Data/IM3/pipeline.production.json"
cp -n config/pipeline.json "$WIND_CONFIG"
# Edit WIND_CONFIG before proceeding; its default coordinate paths are null.

# Stage 1: nine read-only validations (reports + JSON in logs).
python -m im3_wind.production validate --config "$WIND_CONFIG"
# Stage 2: GEV fitting; run through Slurm arrays for the full grid.
python -m im3_wind.production fit --config "$WIND_CONFIG"
# Stage 3: eight future/historical factor grids.
python -m im3_wind.production factor --config "$WIND_CONFIG"
# Stage 4: eight near/far high-resolution transfers.
python -m im3_wind.production downscale --config "$WIND_CONFIG"
# Stage 5: four 2026-2060 scientific scenario exports.
python -m im3_wind.production export --config "$WIND_CONFIG"
```

The commands above specify each stage exactly; full-grid fitting belongs on
compute nodes. Add `--scenario rcp45cooler_near --chunk 0` to process a single
chunk (export uses the `_near` selector to load both slices). `validate` takes
no chunk. After reviewing quality flags, an optional compatibility export is:

```bash
python -m im3_wind.production export --config "$WIND_CONFIG" \
  --scenario rcp45cooler_near --compatibility-rp 100 --allow-flags 2
```

This example accepts extrapolation alone; it will correctly fail if other
quality problems exist. Review those problems rather than blindly allowing all
flags. Compatibility export is optional and is not automatically submitted.

Stage 6 provides the dependency-ordered Slurm execution instead of running the
five processing commands on a login node:

```bash
# Preview commands (no jobs submitted).
python hpc/submit_wind.py --config "$WIND_CONFIG" --max-parallel 8
# Supply --partition and --account if required by your DCC allocation.
python hpc/submit_wind.py --config "$WIND_CONFIG" --max-parallel 8 --submit
```

The submitter creates per-scenario arrays and afterok dependencies between
stages. Fit chunks are 256 cells; target chunks default to 4096 cells. Each fit
task reads one ~20-MB maxima cube plus counts (compressed NPZ cannot be memory
mapped); only the chunk's fits/bootstrap draws are held. Downscaling loads the
native factor grid and baseline and processes bounded target chunks. Parquet
writes one year per row group. One CPU/task and BLAS thread limits avoid nested
parallelism; start with a small pilot and tune memory, chunk size, fit walltime
and array concurrency for real DCC measurements. Default 12-hour fit walltime
is a starting allocation, not a measured guarantee. The default concurrency is
per scenario, so nine fit arrays can run up to 72 tasks. Check DCC limits first.

Logs go to persistent results_root/logs. Job TMPDIR uses scratch_root/jobs.
Results and chunk checkpoints are published directly on persistent storage via
atomic exclusive creation. A matching complete chunk is skipped; partial temp
files are ignored. Source/settings/code/software changes invalidate matching
outputs. Use a new versioned results_root rather than overwrite a stale run.
Resubmitting the same workflow skips completed chunks. Failed dependencies can
leave pending jobs: inspect/cancel those jobs before submitting a replacement
workflow. No script deletes annual inputs or weekly checkpoints.

After all jobs finish and quality review is recorded, archive persistent results:

```bash
python hpc/archive_wind.py --config "$WIND_CONFIG" \
  --output "$HOME/CADENCE_Data/IM3/wind_results_reviewed.tar.gz"
```

Archive creation is exclusive and includes a SHA-256 manifest. It does not move
or remove source results. Choose a new archive filename for each reviewed run.

## Expected outputs and validation limits

Under results_root:

| Path | Contents |
| --- | --- |
| validation/SCENARIO.npz | JSON validation report and provenance for each of nine inputs |
| fits/SCENARIO/CHUNK.npz | Parameters (c, location, scale), n_years, convergence, bootstrap successes, return levels, bounds, flags, AD/tail/top diagnostics |
| factors/FUTURE_SLICE/CHUNK.npz | Eight native-grid factor products with bounds and propagated flags |
| downscaled/FUTURE_SLICE/CHUNK.npz | CONUS baseline, future levels, conditional bounds, mapping distance, IDs, IDW sensitivity and flags |
| annual/FORCING/CHUNK.parquet | Four full scientific products, 2026-2060, all six return periods |
| compatibility/FORCING/rp_T/CHUNK.parquet | Optional explicitly reviewed wide CADENCE adapter |
| logs/ | Slurm stdout/stderr, including validation reports and progress |

Additional bit 2048 marks invalid optional baseline uncertainty bounds. Combined
bounds are emitted only when both baseline bounds were supplied. NPZs retain
all cells, even failures. Do not interpret a numerical factor as quality approval.

Locally tested: synthetic schema errors, missing/count/unit handling, known
Gumbel quantiles, failed/constant samples, bootstrap repeatability, ratio edge
cases, coordinate mismatch, constant-field preservation, IDW sensitivity,
transition anchors, Parquet round trips, exclusive publication, stale restart
rejection, and a nine-scenario end-to-end run with restart invariance. Synthetic
test fixtures use tiny grids and never enter Git. The automated end-to-end test
disables bootstrap for speed; a separate CLI smoke run uses 25 replicates.

Not executed locally: real 299x424 input validation, full-grid GEV fits,
uncertainty calibration, temporal independence/stationarity checks, real CONUS
mapping validation, DCC resource benchmarks, Slurm submission, or downstream
CADENCE production integration. Tail diagnostics are descriptive screening;
production review must inspect spatial flag maps, top-event sensitivity and
return-level uncertainty before publication. No claim of scientific validation
is made from synthetic checks.

To reproduce the local synthetic CLI run in a new external directory:

```bash
python scripts/make_synthetic_pipeline.py /tmp/cadence-wind-example --bootstrap 25
for stage in validate fit factor downscale export; do
  python -m im3_wind.production "$stage" --config /tmp/cadence-wind-example/config.json
done
```

The fixture maker refuses to replace an existing fixture. For reliable
provenance run production from a clean committed checkout and retain the exact
settings file, code revision, environment package list and upstream QA beside
the reviewed archive. Only NumPy/SciPy versions are embedded automatically.

A completeness and per-return-period bitmask report can be generated before
archiving (output filename must be new):

```bash
python scripts/audit_wind_results.py --config "$WIND_CONFIG" \
  --output "$HOME/CADENCE_Data/IM3/wind_quality_report.json"
```

Local execution record (2026-11-07): 12 tests passed; all five CLI stages ran
for nine 2x2-grid synthetic inputs with 25 bootstrap replicates, producing four
annual scenario products with 840 rows each (3,360 total). Completeness audit
passed, Slurm command generation and shell syntax checks passed, and an archive
of 68 synthetic output files was verified against every SHA-256 manifest entry.
This used Python 3.14, NumPy 2.5.4 and SciPy 1.18.1 in an isolated temporary
environment. DCC's rebuilt environment must run the same tests before deployment.

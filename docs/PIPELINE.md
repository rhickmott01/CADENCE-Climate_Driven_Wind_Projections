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

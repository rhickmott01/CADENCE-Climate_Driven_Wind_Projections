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

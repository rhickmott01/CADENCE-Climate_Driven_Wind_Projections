# IM3 / HyperFACETS Extreme-Wind Processing for CADENCE

This repository is the proposed processing pipeline for bringing the raw
IM3/HyperFACETS Thermodynamic Global Warming (TGW) simulations into CADENCE.
It follows the existing CONUS404 repository pattern: numbered preprocessing
scripts, explicit validation, GEV fitting, return-period products, QA outputs,
and a compact Parquet lookup for downstream simulation.

## Scientific objective

The goal is not to replace the high-resolution CONUS404 baseline. Instead:

- **CONUS404** supplies the historical, high-resolution return-period wind.
- **IM3/HyperFACETS** supplies the climate-model change signal.
- The resulting `wind_delta` is applied to the historical CONUS404 return level.

For return period `R` and year `t`, the production concept is:

1. Fit historical and future IM3 annual-extreme distributions.
2. Compare historical IM3 return levels with historical CONUS404 return levels.
3. Use the historical return-level ratio as the V1 multiplicative bias factor.
4. Apply that factor to future IM3 return levels.
5. Express the bias-adjusted future IM3 result relative to the historical CONUS404 return level to create the climate delta.
6. Apply the delta to the CONUS404 baseline when CADENCE evaluates future wind.

The V1 delta is produced for 2025-2050 and for six CADENCE return periods:
10, 25, 50, 100, 200, and 500 years.

## Source and data acquisition

Use the **raw 12-km TGW grid data**, not the derived county-level product. The
official TGW download page provides historical and four future scenarios in
three-hourly files through Globus. The published TGW data record describes the
same 1980-2019 historical period, 2020-2059 and 2060-2099 future periods, 12-km
resolution, and three-hourly variables. Keep the raw files exactly as
provided by the source and place them under:

```text
Data/raw/im3_tgw/historical/
Data/raw/im3_tgw/rcp45cooler/
Data/raw/im3_tgw/rcp45hotter/
Data/raw/im3_tgw/rcp85cooler/
Data/raw/im3_tgw/rcp85hotter/
```

Source links:

- TGW downloads: https://tgw-data.msdlive.org/downloads
- TGW data record / DOI: https://doi.org/10.57931/1885756
- County-level derived product (provenance only; do not use for this pipeline): https://data.msdlive.org/records/g4kct-fhy96
- Scientific Data description: https://doi.org/10.1038/s41597-023-02485-5

## Temporal choice

V1 uses:

```text
3-hourly WSPD
    -> daily maximum
    -> calendar-year maximum
```

We use **calendar years**, not the CONUS404 water-year convention, because the
TGW experiment is explicitly organized as 1980-2019 historical, 2020-2059 near
future, and 2060-2099 far future. This keeps the published experiment boundaries
intact.

The code contains comments marking a future alternative:

```text
3-hourly WSPD -> direct calendar-year maximum
```

That alternative is useful as a temporal-sampling sensitivity test but is not
the V1 production path.

## GEV method and annual deltas

A separate GEV is **not** fitted for each calendar year. One year provides one
annual maximum, which is not enough information to estimate a 10- to 500-year
tail.

Instead, the repository fits GEVs to three 40-year climate states:

```text
1980-2019  historical
2020-2059  near future
2060-2099  far future
```

The period return levels provide climate-state anchors. For the current CADENCE
2025-2050 output, V1 uses the historical and near-future anchors to produce a
smooth annual change signal. The far-future return levels are retained for QA
and future versioning but do not drive the published 2025-2050 delta.

The annual delta is therefore a **modeled climate signal**, not the observed
maximum in that individual year and not a new return period.

A future nonstationary GEV implementation can replace the annualization step
without changing the final `scenario + year + return_period + delta` schema.

## V1 bias transformation

V1 implements the transformation discussed in the CADENCE proposal:

```text
historical bias ratio = CONUS404 historical return level / IM3 historical return level

bias-adjusted future IM3 return level = historical bias ratio * future IM3 return level

climate delta = bias-adjusted future IM3 return level / historical CONUS404 return level
```

The code keeps both the raw values and the ratio so the methodology can later
be replaced by a tail quantile / QDM transformation without redesigning the
rest of the repository.

One mathematical property is worth documenting: when the historical bias is a
single multiplicative ratio and the same ratio is applied to future IM3 values,
the ratio cancels when the final climate delta is formed. That means the
multiplicative bias factor is important for producing the **absolute adjusted
future IM3 return level**, but the relative climate delta reduces to the modeled
future-to-historical IM3 change. A future tail-dependent correction (for example
QDM) can change that behavior and is therefore kept as a deliberate extension
point.

## Spatial treatment

IM3 is approximately 12 km and CONUS404 is approximately 4 km. V1 does not
pretend that IM3 predicts independent 4-km future winds.

Instead the workflow is:

```text
CONUS404 4-km return levels
        -> aggregate historical CONUS404 to IM3 cells for comparison

IM3 12-km climate delta
        -> store once at IM3 resolution

CONUS404 cell
        -> precomputed CONUS404-to-IM3 crosswalk
        -> retrieve its IM3 climate delta
```

The research-phase crosswalk uses nearest IM3 grid center. The code stores that
method explicitly so it can later be replaced by an area-weighted grid overlay
if exact cell footprints are added.

## Scripts

### `01_inspect_im3_data.py`

Read-only source inspection. Confirms `WSPD`, units, time coordinate, grid
coordinates, dates, dimensions, and basic data ranges. Writes a CSV inventory
with a SHA-256 hash for every source file.

### `02_build_annual_maxima.py`

Streams raw three-hourly WSPD files and builds daily maxima, then calendar-year
maxima. It writes a consolidated HDF5 cube for one scenario.

### `03_validate_annual_maxima.py`

Checks the expected calendar-year sequence, coordinates, units, dimensions,
non-negative wind, and aggregation metadata.

### `04_fit_gev_parameters.py`

Fits independent GEVs to the historical, near-future, and far-future 40-year
periods. Uses multiprocessing in spatial chunks and stores fit diagnostics,
parameters, convergence, and return levels.

### `05_calculate_return_periods.py`

Flattens the six return periods for each IM3 cell and climate period to Parquet.

### `06_build_conus404_historical_reference.py`

Re-fits the CONUS404 annual-max record using the matched 1980-2019 historical
period. This reference is used for historical bias calibration only. CADENCE
can continue to use its existing current CONUS404 return-period baseline when
future winds are generated.

### `07_compare_conus404_im3.py`

Aggregates the matched 1980-2019 CONUS404 return-level field to the IM3 grid,
compares it to historical IM3 return levels, and writes the geometry-free
CONUS404-to-IM3 crosswalk used downstream.

### `08_build_bias_transformation.py`

Creates the V1 return-period-specific historical bias ratios.

### `09_build_future_return_levels.py`

Applies the historical bias ratio to each future IM3 scenario's near- and
far-future return levels.

### `10_build_climate_deltas.py`

Creates annual 2025-2050 climate deltas for each scenario, IM3 cell, and return
period using the historical-to-near-future period anchors.

### `11_build_cadence_delta_lookup.py`

Creates a CADENCE-facing bridge from each CONUS404 cell to its IM3 delta. The
master delta remains 12-km to avoid unnecessary replication of the same climate
signal across every 4-km cell.

### `12_build_cadence_scenarios.py`

Builds 10-, 25-, 50-, 100-, 200-, and 500-year design-storm scenarios and a
synthetic `Projected` scenario using configurable weights. `Projected` is a
user-facing decision-support scenario, not a statistically defined return
period.

### `13_visualize_deltas.py`

Creates QA summaries and delta time-series plots by scenario and return period.

## Example command sequence

Run from the repository root after downloading the raw NetCDF files:

```bash
python -m pip install -r requirements.txt

python scripts/01_inspect_im3_data.py \
  --input-dir Data/raw/im3_tgw \
  --output Data/metadata/im3_file_inventory.csv

python scripts/02_build_annual_maxima.py \
  --input-dir Data/raw/im3_tgw/historical \
  --scenario historical \
  --output Data/annual_maxima/historical.h5

python scripts/02_build_annual_maxima.py \
  --input-dir Data/raw/im3_tgw/rcp45cooler \
  --scenario rcp45cooler \
  --output Data/annual_maxima/rcp45cooler.h5
```

Repeat Phase 02 for the other three future scenarios.

Validate:

```bash
python scripts/03_validate_annual_maxima.py \
  --input Data/annual_maxima/historical.h5 \
  --scenario historical

python scripts/03_validate_annual_maxima.py \
  --input Data/annual_maxima/rcp45cooler.h5 \
  --scenario rcp45cooler
```

Fit GEVs for each future scenario:

```bash
python scripts/04_fit_gev_parameters.py \
  --historical Data/annual_maxima/historical.h5 \
  --future Data/annual_maxima/rcp45cooler.h5 \
  --scenario rcp45cooler \
  --output Data/return_periods/gev_rcp45cooler.h5
```

Then flatten:

```bash
python scripts/05_calculate_return_periods.py \
  --input Data/return_periods/gev_rcp45cooler.h5 \
  --output Data/return_periods/return_levels_rcp45cooler.parquet
```

For the historical comparison, use the existing CADENCE/CONUS404 return-level
NetCDF (`gev_return_periods.nc` from the current CONUS404 pipeline):

```bash
python scripts/06_build_conus404_historical_reference.py \
  --input /path/to/CONUS404/output/validated_annual_max_data.h5 \
  --output Data/return_periods/conus404_historical_1980_2019.h5

python scripts/07_compare_conus404_im3.py \
  --conus404-historical-reference Data/return_periods/conus404_historical_1980_2019.h5 \
  --im3-historical-return-levels Data/return_periods/return_levels_rcp45cooler.parquet \
  --output Data/return_periods/historical_comparison.parquet \
  --crosswalk-output Data/metadata/conus404_to_im3_crosswalk.parquet
```

Build bias ratios:

```bash
python scripts/08_build_bias_transformation.py \
  --historical-comparison Data/return_periods/historical_comparison.parquet \
  --im3-historical-return-levels Data/return_periods/return_levels_rcp45cooler.parquet \
  --output Data/metadata/bias_transformation.parquet
```

The historical return levels inside the future scenario file are the same
historical GEV, so this is intentional. In production, it may be cleaner to
store a standalone historical return-level Parquet once and reuse it across the
four scenarios.

Apply the bias ratio:

```bash
python scripts/09_build_future_return_levels.py \
  --bias Data/metadata/bias_transformation.parquet \
  --future-return-levels Data/return_periods/return_levels_rcp45cooler.parquet \
  --scenario rcp45cooler \
  --output Data/return_periods/bias_adjusted_rcp45cooler.parquet
```

Build the annual delta:

```bash
python scripts/10_build_climate_deltas.py \
  --future-adjusted Data/return_periods/bias_adjusted_rcp45cooler.parquet \
  --output Data/metadata/wind_delta_rcp45cooler.parquet
```

Build the optional CADENCE bridge:

```bash
python scripts/11_build_cadence_delta_lookup.py \
  --delta Data/metadata/wind_delta_rcp45cooler.parquet \
  --crosswalk Data/metadata/conus404_to_im3_crosswalk.parquet \
  --output Data/metadata/cadence_wind_delta_bridge_rcp45cooler.parquet
```

## CADENCE scenario layer

A starting `Projected` weighting file is supplied under `config/projected_weights.json`.
These weights are illustrative and should be reviewed and made user-configurable.

The six design-storm scenarios are the current CADENCE return periods. The
`Projected` scenario is a weighted average of those design-storm wind values.
It should not be interpreted as a new return period or a probability model.

## Important limitations

1. IM3/HyperFACETS is a thermodynamic-modification experiment. The published
   dataset is intended to explore the effects of changing thermodynamics on
   historical weather and is not a complete projection of changes in event
   frequency caused by future atmospheric circulation changes.
2. The multiplicative return-level bias ratio is a V1 method, not a final tail
   correction. A future QDM/extreme-quantile method should be evaluated.
3. Long return periods such as 200 and 500 years are estimated from 40 annual
   maxima within each IM3 climate state and therefore have substantial tail
   uncertainty.
4. The nearest-center crosswalk is a research-phase spatial approximation.
5. The V1 annual delta uses period-anchor interpolation rather than a fully
   nonstationary GEV.
6. CADENCE currently has a single-climate-trajectory V1 architecture. The four
   IM3 scenarios are prepared here so the future UI can expose scenario choice.
7. The existing CONUS404 return-period confidence bounds are documented as
   heuristic and should not automatically be propagated as formal tail-risk
   intervals.

## Data provenance note

The MSD-LIVE county record provided during project planning is a **derived
county-level dataset**. The official TGW download page separately provides the
raw three-hourly grid files that are required here. The repository intentionally
keeps those raw files outside version control.

## Final CADENCE calculation

The production concept is deliberately split into a high-resolution baseline
and a climate signal:

```text
Historical CONUS404 return level (4 km)
                x
IM3 climate delta (12 km, linked by precomputed crosswalk)
                =
Future CADENCE return-level wind
```

For example, a CADENCE simulation asset first resolves to its existing
`wind_grid_id` (CONUS404 cell) and separately to the precomputed IM3 cell in
`conus404_to_im3_crosswalk.parquet`. The simulation then retrieves the matching
scenario/year/return-period delta. This keeps the production engine geometry-free
and consistent with the existing ID-based CADENCE architecture.

The existing CONUS404 3-second gust conversion should occur **after** this
climate adjustment. The climate delta is therefore applied in the native
return-level wind units first.

## Relationship to the existing Historical Trend scenario

The existing CADENCE Historical Trend scenario remains the current Theil-Sen
CONUS404 historical extrapolation and is not overwritten by this repository.
This repository supplies the replacement climate-model-informed delta pathway.
The final CADENCE application can expose both methods as separate scenario
choices:

```text
Historical Trend  -> existing CONUS404 historical trend pipeline
Projected         -> IM3/HyperFACETS climate delta + weighted design storms
10-Year Storm     -> IM3 delta applied to CONUS404 10-year baseline
25-Year Storm     -> IM3 delta applied to CONUS404 25-year baseline
50-Year Storm     -> IM3 delta applied to CONUS404 50-year baseline
100-Year Storm    -> IM3 delta applied to CONUS404 100-year baseline
200-Year Storm    -> IM3 delta applied to CONUS404 200-year baseline
500-Year Storm    -> IM3 delta applied to CONUS404 500-year baseline
```

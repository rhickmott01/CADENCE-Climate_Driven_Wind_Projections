# DCC real-input handoff

The local Mac checkout now implements a human-submitted audit/pilot and compact
production route. No DCC commands, scientific input transfers, Slurm submissions,
or real-data numerical calculations were run in this development session.

Implemented:
- U10/V10-derived wind metadata support, preserving explicit legacy WSPD support.
- Conservative unique-finite observation bounds from checkpoint timestamp unions,
  bound to annual hashes, without changing annual maxima or original counts.
- Explicit CONUS wide/long schema adapters; bad-curve/point quarantine, raw bound
  preservation, unknown upstream quality semantics review, and separate flags.
- Nine-scenario audit with persistent per-scenario reports and exploratory fits;
  sparse paired factor/spatial checks where eligible paired samples exist.
- Slurm wrappers running only Python and a separately human-reviewed production
  plan generator. One stage-wide array caps total active tasks at eight.
- Compact crosswalk and selected-key/year/period lookup, preserving baseline flags.

First human action: read docs/DCC_MANUAL_EXECUTION.md, verify all configured DCC
paths and the actual derived CONUS schema, then submit one pilot job. The first
run is expected to report future-grid and overlap-review evidence blockers.
Do not manufacture review assertions to pass them. Use the report to determine
what real evidence is still needed; retain usable historical pilot results.

Actual CONUS files were reported as Data/raw/conus404/gev_return_periods.csv,
Data/baseline/conus404_baseline.parquet and Data/metadata/conus404/inspection.json,
but are absent in this Mac checkout. Do not repeat retrieval unnecessarily.
Historical fitting period is still unconfirmed and stays a provenance limitation.

Remaining work requires real-data reports: resolve all-scenario grid provenance,
full-record overlap value treatment, any schema adaptation, empirical resource
sizing, reviewed full-grid fitting/factors/crosswalk execution, scientific review,
and downstream application changes/deployment (not performed here).

Verification: 21 local tests passed, including an ordinary Bash invocation of the
Slurm pilot wrapper with synthetic Slurm environment variables and a blocked-input
case, plus shell syntax checks. These are local simulations, not DCC runs.

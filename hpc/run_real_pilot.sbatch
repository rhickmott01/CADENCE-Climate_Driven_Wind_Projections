#!/bin/bash
#SBATCH --job-name=im3-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/im3-pilot-%j.log
set -euo pipefail
# Run sbatch from the repository root; partition/account are supplied by the human.
CONFIG=${1:?Usage: sbatch hpc/run_real_pilot.sbatch CONFIG}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Bootstrap Python only parses JSON; the configured environment does the science.
WIND_PY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_env"]+"/bin/python")' "$CONFIG")
SCRATCH=$("$WIND_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scratch_root"])' "$CONFIG")
export TMPDIR="$SCRATCH/jobs/pilot-${SLURM_JOB_ID:?Submit with sbatch}"
mkdir -p "$TMPDIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'Host: %s\nJob: %s\nConfig: %s\n' "$(hostname)" "$SLURM_JOB_ID" "$CONFIG"
"$WIND_PY" -m im3_wind.pilot --config "$CONFIG"
# Exit 2 means audit completed but input/review blockers remain; do not run production.

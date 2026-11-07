#!/bin/bash
#SBATCH --job-name=cadence-wind
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
set -euo pipefail
# Invoke from repository root through submit_wind.py; configure partition/account there.
CONFIG=$1
STAGE=$2
SCENARIO=$3
ENV_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_env"])' "$CONFIG")
SCRATCH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scratch_root"])' "$CONFIG")
export TMPDIR="$SCRATCH/jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ARGS=("$STAGE" --config "$CONFIG" --scenario "$SCENARIO")
if [[ "$STAGE" != validate ]]; then
    ARGS+=(--chunk "${SLURM_ARRAY_TASK_ID:?Array task ID required}")
fi
"$ENV_PATH/bin/python" -m im3_wind.production "${ARGS[@]}"
# Persistent results are published atomically by Python; failed chunks can be rerun.
# Leave scratch diagnostics intact for inspection. No input/checkpoint cleanup.

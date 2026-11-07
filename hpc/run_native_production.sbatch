#!/bin/bash
#SBATCH --job-name=im3-production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
CONFIG=${1:?Production settings from reviewed plan required}
STAGE=${2:?Stage required: fit, factor, crosswalk}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
WIND_PY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_env"]+"/bin/python")' "$CONFIG")
SCRATCH=$("$WIND_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scratch_root"])' "$CONFIG")
export TMPDIR="$SCRATCH/jobs/production-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$WIND_PY" -m im3_wind.manual task --config "$CONFIG" --stage "$STAGE" --index "${SLURM_ARRAY_TASK_ID:-0}"

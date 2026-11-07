#!/bin/bash
#SBATCH --job-name=im3-evidence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/im3-evidence-%j.log
set -euo pipefail
CONFIG=${1:?Settings path required}
PRIOR_RUN=${2:?Prior pilot run directory required}
shift 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
WIND_PY=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_env"]+"/bin/python")' "$CONFIG")
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'Host: %s\nJob: %s\n' "$(hostname)" "${SLURM_JOB_ID:?Submit with sbatch}"
"$WIND_PY" -m im3_wind.evidence --config "$CONFIG" --prior-run "$PRIOR_RUN" "$@"

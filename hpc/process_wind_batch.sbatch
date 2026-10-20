#!/bin/bash
#SBATCH --job-name=im3-wind-test
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/rjh92/im3_tgw/logs/im3_%j.out
#SBATCH --error=/work/rjh92/im3_tgw/logs/im3_%j.err

set -euo pipefail

# Use the exact Python environment previously tested on DCC.
PYTHON=/work/rjh92/conda_envs/cadence-wind/bin/python

# Navigate to the CADENCE repository.
cd /hpc/home/rjh92/CADENCE-Climate_Driven_Wind_Projections

echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "Started: $(date)"
echo "Python executable: $PYTHON"

# Verify the required NetCDF library before processing.
"$PYTHON" -c "import netCDF4; print('netCDF4 available')"

# Process the five staged historical NetCDF files.
# No raw files are deleted.
"$PYTHON" hpc/process_wind_batch.py \
    --input-dir /work/rjh92/im3_tgw/raw \
    --output-dir /work/rjh92/im3_tgw/checkpoints/files \
    --manifest /work/rjh92/im3_tgw/manifests/slurm_test_${SLURM_JOB_ID}.json \
    --max-files 5

echo "Finished: $(date)"

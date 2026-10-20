#!/bin/bash
#SBATCH --job-name=cadence-im3
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/work/rjh92/im3_tgw/logs/pipeline_%j.out
#SBATCH --error=/work/rjh92/im3_tgw/logs/pipeline_%j.err
set -euo pipefail
cd /hpc/home/rjh92/CADENCE-Climate_Driven_Wind_Projections
export PATH="/work/rjh92/conda_envs/cadence-wind/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python hpc/run_pipeline.py --config hpc/pipeline_config.json --batch-size 5 --workers 4 --max-runtime-hours 23

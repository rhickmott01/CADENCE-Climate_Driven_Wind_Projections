#!/bin/bash
#SBATCH --job-name=im3-annual
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/home/rjh92/im3_annual_%j.out
#SBATCH --error=/hpc/home/rjh92/im3_annual_%j.err
set -euo pipefail
module purge
module load Anaconda3/2024.02
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/rjh92/conda_envs/cadence-wind-rebuilt
python /hpc/home/rjh92/CADENCE-Climate_Driven_Wind_Projections/hpc/aggregate_annual.py

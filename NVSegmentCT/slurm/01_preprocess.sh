#!/bin/bash
#SBATCH --job-name=nv_prep_test
#SBATCH --account=jhu159
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/01_prep_%j.out
#SBATCH --error=logs/01_prep_%j.err

set -euo pipefail
mkdir -p logs

module purge
source $HOME/bdd_env/bin/activate

echo "Running Preprocessing on $(hostname) | Job ID: ${SLURM_JOB_ID}"
python -u -m src.dataset_prep --max-cases 5 --train-budget 5 --seed 42
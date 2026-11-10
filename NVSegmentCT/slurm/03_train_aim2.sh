#!/bin/bash
#SBATCH --job-name=nv_aim2_test
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/03_aim2_%A_%a.out
#SBATCH --error=logs/03_aim2_%A_%a.err

set -euo pipefail
mkdir -p logs

module purge
source $HOME/bdd_env/bin/activate

LUSTRE_ROOT="/expanse/lustre/scratch/${USER}/temp_project/segrap_nv_project"
NVME_STAGE="/scratch/${USER}/job_${SLURM_JOB_ID}/preprocessed_2ch"
mkdir -p "${NVME_STAGE}"
cp -r "${LUSTRE_ROOT}/preprocessed_2ch/"* "${NVME_STAGE}/"

echo "Running Aim II Sparse OAR Prompting Smoke Test on $(hostname)"
python -m src.train_aim2 \
    --manifest "aim2_sparse_N4_seed42.json" \
    --epochs 2 \
    --smoke-test
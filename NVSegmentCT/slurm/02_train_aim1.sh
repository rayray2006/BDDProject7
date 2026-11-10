#!/bin/bash
#SBATCH --job-name=nv_aim1_50ep
#SBATCH --account=jhu159
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --array=0-0
#SBATCH --output=logs/02_aim1_%A_%a.out
#SBATCH --error=logs/02_aim1_%A_%a.err

set -euo pipefail
mkdir -p logs

module purge
source $HOME/bdd_env/bin/activate

LUSTRE_ROOT="/expanse/lustre/scratch/${USER}/temp_project/segrap_nv_project"
NVME_STAGE="/scratch/${USER}/job_${SLURM_JOB_ID}/preprocessed_2ch"
mkdir -p "${NVME_STAGE}"

echo "Staging preprocessed 2-channel tensors to node-local NVMe: ${NVME_STAGE}"
cp -r "${LUSTRE_ROOT}/preprocessed_2ch/"* "${NVME_STAGE}/"

# 1. Full 50-Epoch Aim I Pretrained Fine-Tuning (N=5)
python -u -m src.train_aim1 \
    --manifest "aim1_N5_seed42.json" \
    --init-mode pretrained \
    --epochs 50

# 2. Full 50-Epoch Aim III Random-Init Control (N=5)
python -u -m src.train_aim1 \
    --manifest "aim1_N5_seed42.json" \
    --init-mode scratch \
    --epochs 50
#!/bin/bash
#SBATCH --job-name=nv_eval_aim3
#SBATCH --account=jhu159
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/04_eval_%j.out
#SBATCH --error=logs/04_eval_%j.err

set -euo pipefail
mkdir -p logs

module purge
source $HOME/bdd_env/bin/activate

LUSTRE_ROOT="/expanse/lustre/scratch/${USER}/temp_project/segrap_nv_project"
NVME_STAGE="/scratch/${USER}/job_${SLURM_JOB_ID}/preprocessed_2ch"
mkdir -p "${NVME_STAGE}"
cp -r "${LUSTRE_ROOT}/preprocessed_2ch/"* "${NVME_STAGE}/"

python -u -m src.evaluate_aim3 \
    --manifest "aim1_N5_seed42.json" \
    --checkpoints "nv_aim1_pretrained_N5_seed42.pt" "nv_aim1_scratch_N5_seed42.pt" \
    --output-csv "aim3_N5_smoke_metrics.csv"
#!/bin/bash
#SBATCH --job-name=nv_aim2
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/03_aim2_%j.out
#SBATCH --error=logs/03_aim2_%j.err
set -euo pipefail
# Usage:
# sbatch slurm/03_train_aim2.sh <manifest.json> [smoke|full] [epochs]
#
# Examples:
# sbatch slurm/03_train_aim2.sh aim2_dense_B5_seed42.json smoke
# sbatch slurm/03_train_aim2.sh aim2_sparse_B5_seed42.json full 50

MANIFEST="${1:?Please provide an Aim II manifest filename}"
MODE="${2:-smoke}"
EPOCHS="${3:-50}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "ERROR: Mode must be smoke or full"
    exit 1
fi

if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Epochs must be a positive integer"
    exit 1
fi

# Run from the repository root.
cd "${SLURM_SUBMIT_DIR}"

# Activate the existing team environment.
module purge
source "${HOME}/bdd_env/bin/activate"

# Preserve the team's existing Lustre and NVMe staging paths.
LUSTRE_ROOT="/expanse/lustre/scratch/${USER}/temp_project/segrap_nv_project"
NVME_STAGE="/scratch/${USER}/job_${SLURM_JOB_ID}/preprocessed_2ch"

mkdir -p "${NVME_STAGE}"

# Copy preprocessed CT data to local scratch.
cp -a "${LUSTRE_ROOT}/preprocessed_2ch/." "${NVME_STAGE}/"

echo "======================================"
echo "Aim II: Distributed Annotation Budget"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Manifest: ${MANIFEST}"
echo "Mode: ${MODE}"
echo "Epochs: ${EPOCHS}"
echo "Staged data: ${NVME_STAGE}"
echo "======================================"

# The Python trainer reads the strategy, budget, and seed
# directly from the selected manifest.

if [[ "${MODE}" == "smoke" ]]; then

    echo "Running Aim II smoke test..."

    python -m src.train_aim2 \
        --manifest "${MANIFEST}" \
        --epochs 1 \
        --smoke-test

else

    echo "Running full Aim II training..."

    python -m src.train_aim2 \
        --manifest "${MANIFEST}" \
        --epochs "${EPOCHS}"

fi

echo "Aim II training job completed."

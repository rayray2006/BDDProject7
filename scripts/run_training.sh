#!/usr/bin/env bash
# Train nnU-Net folds as a batch job. Works with SLURM (sbatch) or plain bash.
#
#   sbatch --array=0-4 scripts/run_training.sh 101          # one fold per array task
#   sbatch scripts/run_training.sh 101 0                     # single fold
#   bash scripts/run_training.sh 101 "0 1 2 3 4"             # sequentially, no scheduler
#   PREPROCESS=1 bash scripts/run_training.sh 101 0          # preprocess first
#   TRAINER=nnUNetTrainer_100epochs sbatch scripts/run_training.sh 101 0
#
#SBATCH --job-name=nnunet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

DATASET_ID="${1:?usage: run_training.sh DATASET_ID [FOLDS]}"
FOLDS="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
DEVICE="${DEVICE:-cuda}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate
# Fewer augmentation workers if the node has few CPUs (nnU-Net default is 12).
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-${SLURM_CPUS_PER_TASK:-8}}"

python -m src.config
if [ "${PREPROCESS:-0}" = "1" ]; then
  python -m src.preprocess "$DATASET_ID" -c "$CONFIGURATION"
fi
# shellcheck disable=SC2086  # FOLDS is intentionally word-split
python -m src.train "$DATASET_ID" $FOLDS -c "$CONFIGURATION" -tr "$TRAINER" -device "$DEVICE" --npz

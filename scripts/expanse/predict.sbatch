#!/usr/bin/env bash
# GPU job: predict the 20 shared test cases with an already-trained model and write Dice + HD95.
# This is the second half of train.sbatch on its own, for when training finished but prediction did not
# (e.g. job 55852587, killed for memory while exporting the first test case). Submit from the repo root:
#
#   sbatch -A $EXPANSE_PROJECT --time=04:00:00 scripts/expanse/predict.sbatch 110 nnUNetTrainerNoVal_5epochs
#
# Test cases already in the prediction folder are skipped, so a resubmission continues where it stopped.
#
#SBATCH --job-name=nnunet-predict
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=92G
#SBATCH --time=04:00:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DATASET_ID="${1:?usage: predict.sbatch DATASET_ID [TRAINER]}"
TRAINER="${2:-nnUNetTrainerNoVal_250epochs}"
source scripts/expanse/env.sh
activate_env

python -m src.config
nvidia-smi --query-gpu=name,memory.total --format=csv
start=$(date +%s)
python -m src.evaluate "$DATASET_ID" -f all -tr "$TRAINER" -device cuda -np 8 --low-memory
echo "test prediction + metrics wall time: $(( ($(date +%s) - start) / 60 )) min"

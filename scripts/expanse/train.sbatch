#!/usr/bin/env bash
# GPU job: train one nnU-Net model on one subset (fold "all" = every case in the subset), then
# predict the 20 shared test cases and write Dice + HD95. One V100 on the gpu-shared partition.
# Submit from the repo root:
#
#   sbatch -A $EXPANSE_PROJECT scripts/expanse/train.sbatch 110                            # 250 epochs
#   sbatch -A $EXPANSE_PROJECT --time=01:00:00 scripts/expanse/train.sbatch 110 nnUNetTrainer_5epochs   # timing test
#
# Cost: GPU hours = wall-clock hours (1 GPU). The preprocessed data is copied to the node's local
# NVMe first (faster than reading Lustre every iteration); checkpoints and results stay on Lustre.
# If the job hits the time limit, submit the same command again: training resumes from
# checkpoint_latest.pth (written every 50 epochs).
#
#SBATCH --job-name=nnunet-train
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=92G
#SBATCH --time=24:00:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DATASET_ID="${1:?usage: train.sbatch DATASET_ID [TRAINER]}"
TRAINER="${2:-nnUNetTrainer_250epochs}"
source scripts/expanse/env.sh
activate_env

# Data-augmentation workers: one per CPU we asked for, minus one for the main process.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-$(( ${SLURM_CPUS_PER_TASK:-10} - 1 ))}"
export nnUNet_compile="${nnUNet_compile:-f}"   # torch.compile gains little on V100 and slows start-up

name=$(python -c "from src import config; print(config.dataset_name($DATASET_ID))")
LOCAL="/scratch/$USER/job_$SLURM_JOB_ID"
if [ "${STAGE_LOCAL:-1}" = "1" ] && [ -d "$LOCAL" ]; then
  echo "copying preprocessed $name to local NVMe $LOCAL"
  mkdir -p "$LOCAL/nnUNet_preprocessed"
  cp -r "$NNUNET_ROOT/nnUNet_preprocessed/$name" "$LOCAL/nnUNet_preprocessed/"
  export nnUNet_preprocessed="$LOCAL/nnUNet_preprocessed"
fi
python -m src.config
nvidia-smi --query-gpu=name,memory.total --format=csv

start=$(date +%s)
python -m src.train "$DATASET_ID" all -tr "$TRAINER" -device cuda
echo "training wall time: $(( ($(date +%s) - start) / 60 )) min"
python -m src.evaluate "$DATASET_ID" -f all -tr "$TRAINER" -device cuda -np 8
nvidia-smi --query-gpu=memory.used --format=csv || true

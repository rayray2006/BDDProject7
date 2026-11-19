#!/usr/bin/env bash
# CPU job: copy the 120 raw SegRap cases from the shared Google Drive to $RAW_DATA_DIR with rclone.
# Needs an rclone remote called "gdrive" (read-only Drive access), set up once with `rclone config`.
# Re-running only copies what is missing. Submit from the repo root:
#
#   sbatch -A $EXPANSE_PROJECT scripts/expanse/fetch_data.sbatch
#
#SBATCH --job-name=segrap-fetch
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/expanse/env.sh
activate_env

SRC="${SRC:-gdrive:Biomedical Data Design/SegRap2023_Training_Set_120cases}"
mkdir -p "$RAW_DATA_DIR"
# Drive keeps duplicate uploads as "Name (1).nii.gz"; prepare_dataset.py ignores them, so skip them here.
rclone copy "$SRC" "$RAW_DATA_DIR" --drive-shared-with-me --exclude "*(*)*" --fast-list \
  --transfers 8 --checkers 16 --stats 60s --stats-one-line -v 2>&1 | grep --line-buffered -v NOTICE
echo "cases: $(ls -d "$RAW_DATA_DIR"/segrap_* | wc -l)"
du -sh "$RAW_DATA_DIR"

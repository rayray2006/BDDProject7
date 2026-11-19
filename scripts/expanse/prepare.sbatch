#!/usr/bin/env bash
# CPU job: make the shared split file (first time only), convert the chosen subsets to nnU-Net
# datasets, and run nnU-Net planning + preprocessing for them. Uses CPU hours, not GPU hours.
# Submit from the repo root:
#
#   sbatch -A $EXPANSE_PROJECT scripts/expanse/prepare.sbatch                     # n005_r0 n010_r0
#   sbatch -A $EXPANSE_PROJECT scripts/expanse/prepare.sbatch "n005_r0"           # just one subset
#
# The first run writes splits/segrap_exp1.json. Commit that file so everyone uses the same split.
#
#SBATCH --job-name=nnunet-prepare
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/expanse/env.sh
activate_env

SUBSETS="${1:-n005_r0 n010_r0}"
SPLIT=splits/segrap_exp1.json
WORKERS="${WORKERS:-4}"   # each full SegRap case needs a few GB of RAM while converting/preprocessing

python -m src.config
if [ ! -f "$SPLIT" ]; then
  python -m src.make_splits -o "$SPLIT"
  echo ">>> New split written to $SPLIT. Commit it to git so the test set never changes."
fi
# shellcheck disable=SC2086  # SUBSETS is intentionally word-split
python -m src.prepare_dataset --split-file "$SPLIT" --subset $SUBSETS --workers "$WORKERS"

# shellcheck disable=SC2086
for id in $(python - "$SPLIT" $SUBSETS <<'PY'
import json, sys
split = json.load(open(sys.argv[1]))
ids = {s["name"]: s["dataset_id"] for s in split["subsets"]}
print(" ".join(str(ids[n]) for n in sys.argv[2:]))
PY
); do
  python -m src.preprocess "$id" -c 3d_fullres -np "$WORKERS"
done
echo "done. Next: sbatch -A \$EXPANSE_PROJECT scripts/expanse/train.sbatch <dataset_id>"

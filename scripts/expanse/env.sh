# Shared settings for every Expanse job. Sourced by the other scripts in this folder; edit the two
# lines marked EDIT once you know your project (run `expanse-client user -r expanse` on Expanse).
#
# Layout on Expanse (Lustre projects storage, 5 TB on this allocation, not backed up):
#   $WORK/BDDProject7/                  this repo (git clone)
#   $WORK/SegRap2023_Training_Set_120cases/  raw case folders (segrap_0000/ ...)
#   $WORK/nnunet/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}/
#   $WORK/conda/                        Python environment made by setup_env.sh

EXPANSE_PROJECT="${EXPANSE_PROJECT:-CHANGE_ME}"                       # EDIT: allocation/project name, e.g. abc123
WORK="${WORK:-/expanse/lustre/projects/${EXPANSE_PROJECT}/${USER}}"   # EDIT only if your project space differs

export WORK
export NNUNET_ROOT="${NNUNET_ROOT:-$WORK/nnunet}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$WORK/SegRap2023_Training_Set_120cases}"
CONDA_DIR="$WORK/conda"
ENV_NAME=nnunet

if [ "$EXPANSE_PROJECT" = "CHANGE_ME" ] && [ -z "${SKIP_PROJECT_CHECK:-}" ]; then
  echo "Set EXPANSE_PROJECT in scripts/expanse/env.sh (see: expanse-client user -r expanse)" >&2
  return 1 2>/dev/null || exit 1
fi

# Everything (Python, PyTorch with its own CUDA libraries) comes from this conda env, so no Expanse
# modules are loaded; the GPU driver is part of the node's OS.
activate_env() {
  set +u  # conda's activation scripts read unset variables
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  set -u
}

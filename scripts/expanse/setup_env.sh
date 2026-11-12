#!/usr/bin/env bash
# One-time setup on an Expanse login node: Python environment + folder layout.
#
#   cd $WORK/BDDProject7 && bash scripts/expanse/setup_env.sh
#
# What it downloads: Miniforge (conda-forge's installer, from github.com/conda-forge/miniforge),
# PyTorch from download.pytorch.org, and requirements.txt from PyPI. Nothing else.
#
# Why not Expanse's own anaconda module: it ships an old Python, and this code needs Python 3.10+.
# Why a special PyTorch index: Expanse GPUs are V100s (compute capability sm_70). Recent default PyTorch
# wheels may leave sm_70 out, so we install a CUDA 12.6 build and check_gpu.sbatch confirms it works.
# Override with e.g.  TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/expanse/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/expanse/env.sh

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
TORCH_SPEC="${TORCH_SPEC:-torch}"

mkdir -p "$WORK" "$NNUNET_ROOT"/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results} logs

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  installer="$(mktemp -d)/miniforge.sh"
  curl -fsSL -o "$installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$installer" -b -p "$CONDA_DIR"
  rm -f "$installer"
fi
set +u
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.12
set -u
activate_env

pip install --upgrade pip
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
pip install -r requirements.txt          # nnunetv2 keeps the torch installed above

python - <<'EOF'
import torch, nnunetv2
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("GPU architectures in this build:", torch.cuda.get_arch_list() or "(none listed on a login node)")
EOF
python -m src.config
echo
echo "Environment ready. Next: sbatch -A \$EXPANSE_PROJECT scripts/expanse/check_gpu.sbatch"

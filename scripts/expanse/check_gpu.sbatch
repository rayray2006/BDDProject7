#!/usr/bin/env bash
# 10-minute check on the gpu-debug partition (costs ~0.2 GPU hours): does PyTorch see the V100 and
# can it run a 3D convolution with mixed precision, the way nnU-Net trains? Submit from the repo root:
#
#   sbatch -A $EXPANSE_PROJECT scripts/expanse/check_gpu.sbatch
#
#SBATCH --job-name=nnunet-gpucheck
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --constraint=lustre
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/expanse/env.sh
activate_env

nvidia-smi
python - <<'PY'
import time, torch
print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch cannot see a GPU"
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("device:", name, arch, "| build supports:", torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f"this PyTorch build has no kernels for {arch}; reinstall with another TORCH_INDEX"
x = torch.randn(2, 2, 96, 160, 160, device="cuda")
conv = torch.nn.Conv3d(2, 32, 3, padding=1).cuda()
with torch.autocast("cuda", dtype=torch.float16):
    for _ in range(3):
        y = conv(x)
torch.cuda.synchronize(); t = time.time()
with torch.autocast("cuda", dtype=torch.float16):
    for _ in range(20):
        y = conv(x)
torch.cuda.synchronize()
print(f"OK: 3D conv fp16 works, {(time.time() - t) / 20 * 1000:.1f} ms per call")
PY
python -m src.config

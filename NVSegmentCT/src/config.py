from __future__ import annotations
import os
from pathlib import Path

USER = os.getenv("USER", "test_user")
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "local_test")

# 1. Resolve SDSC Expanse Lustre persistent scratch vs. local fallback
EXPANSE_LUSTRE = Path(f"/expanse/lustre/scratch/{USER}/temp_project")
BASE_REPO_DIR = Path(__file__).resolve().parent.parent

if EXPANSE_LUSTRE.exists() or "SLURM_JOB_ID" in os.environ:
    PERSISTENT_ROOT = EXPANSE_LUSTRE / "segrap_nv_project"
else:
    PERSISTENT_ROOT = BASE_REPO_DIR / "local_workspace"

# 2. Resolve Node-Local NVMe Scratch (/scratch/$USER/job_$SLURM_JOB_ID on Expanse)
EXPANSE_NVME = Path(f"/scratch/{USER}/job_{SLURM_JOB_ID}")
if EXPANSE_NVME.exists():
    SCRATCH_ROOT = EXPANSE_NVME
else:
    SCRATCH_ROOT = PERSISTENT_ROOT / "tmp_scratch" / f"job_{SLURM_JOB_ID}"

# 3. Directory Hierarchy
RAW_DATA_DIR = Path(os.getenv("SEGRAP_RAW_DIR", BASE_REPO_DIR / "data"))
PREPROCESSED_LUSTRE_DIR = PERSISTENT_ROOT / "preprocessed_2ch"
MANIFEST_DIR = PERSISTENT_ROOT / "manifests"
CHECKPOINT_DIR = PERSISTENT_ROOT / "checkpoints"
EVAL_RESULTS_DIR = PERSISTENT_ROOT / "eval_results"
CLASS_MAP_PATH = BASE_REPO_DIR / "configs" / "class_mapping.json"

# Active training data directory (prefers staged NVMe if populated by SLURM script)
STAGED_DATA_DIR = SCRATCH_ROOT / "preprocessed_2ch"

for directory in [
    RAW_DATA_DIR,
    PREPROCESSED_LUSTRE_DIR,
    MANIFEST_DIR,
    CHECKPOINT_DIR,
    EVAL_RESULTS_DIR,
    SCRATCH_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


def get_active_data_dir() -> Path:
    """Returns fast node-local NVMe path if staged, otherwise Lustre path."""
    if STAGED_DATA_DIR.exists() and any(STAGED_DATA_DIR.iterdir()):
        return STAGED_DATA_DIR
    return PREPROCESSED_LUSTRE_DIR
# Ensure PRETRAINED_WEIGHTS_PATH is explicitly exported for model_factory.py
PRETRAINED_WEIGHTS_PATH = Path(
    f"/expanse/lustre/scratch/{os.environ.get('USER', 'mjung3')}/temp_project/segrap_nv_project/pretrained_weights/models/model.pt"
)

# Ensure EVAL_DIR is explicitly exported for evaluate_aim3.py
EVAL_DIR = Path(
    f"/expanse/lustre/scratch/{os.environ.get('USER', 'mjung3')}/temp_project/segrap_nv_project/eval_results"
)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

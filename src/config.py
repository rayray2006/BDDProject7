"""Central path resolution & environment loader.

Every script imports this module first. It reads `.env` from the project root, resolves the
nnU-Net folders, and exports nnUNet_raw / nnUNet_preprocessed / nnUNet_results into os.environ
so that nnunetv2 (and any subprocess we launch) sees the same locations.

Precedence for each path: real environment variable > .env > default derived from NNUNET_ROOT.
Relative paths are resolved against the project root, so `NNUNET_ROOT=data/nnunet` works locally.

Run `python -m src.config` to print the resolved paths and check that they exist.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=False)


def _resolve(value: str) -> Path:
    p = Path(os.path.expandvars(value)).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _get(name: str, default: str | Path | None = None) -> Path | None:
    value = os.environ.get(name) or (str(default) if default is not None else None)
    return _resolve(value) if value else None


NNUNET_ROOT = _get("NNUNET_ROOT", "data/nnunet")
RAW_DATA_DIR = _get("RAW_DATA_DIR", "data/toy_raw")  # source case folders (segrap_XXXX/) before conversion

NNUNET_RAW = _get("nnUNet_raw", NNUNET_ROOT / "nnUNet_raw")
NNUNET_PREPROCESSED = _get("nnUNet_preprocessed", NNUNET_ROOT / "nnUNet_preprocessed")
NNUNET_RESULTS = _get("nnUNet_results", NNUNET_ROOT / "nnUNet_results")

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)

DEFAULT_TRAINER = os.environ.get("NNUNET_TRAINER", "nnUNetTrainer")
DEFAULT_PLANS = os.environ.get("NNUNET_PLANS", "nnUNetPlans")
DEFAULT_CONFIG = os.environ.get("NNUNET_CONFIG", "3d_fullres")


def ensure_dirs() -> None:
    for p in (NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def dataset_name(dataset_id: int) -> str:
    """Return the full folder name (e.g. 'Dataset101_SegRapOAR') for a numeric dataset ID."""
    prefix = f"Dataset{int(dataset_id):03d}_"
    for root in (NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS):
        if root.is_dir():
            matches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
            if len(matches) > 1:
                raise RuntimeError(f"Dataset ID {dataset_id:03d} is used by several folders in {root}: {matches}. "
                                   "Check the Dataset ID Registry in README.md.")
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No folder named {prefix}* in {NNUNET_RAW}, {NNUNET_PREPROCESSED} or {NNUNET_RESULTS}. "
                            "Run src/prepare_dataset.py first.")


def model_dir(dataset_id: int, trainer: str, plans: str, configuration: str) -> Path:
    return NNUNET_RESULTS / dataset_name(dataset_id) / f"{trainer}__{plans}__{configuration}"


def run(cmd: list[str]) -> None:
    """Echo and run an nnU-Net CLI command with the resolved environment; exit on failure."""
    cmd = [str(c) for c in cmd]
    print("$", shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {cmd[0]}")


def describe() -> str:
    rows = [
        (".env file", ENV_FILE, ENV_FILE.exists()),
        ("RAW_DATA_DIR", RAW_DATA_DIR, RAW_DATA_DIR.exists()),
        ("nnUNet_raw", NNUNET_RAW, NNUNET_RAW.exists()),
        ("nnUNet_preprocessed", NNUNET_PREPROCESSED, NNUNET_PREPROCESSED.exists()),
        ("nnUNet_results", NNUNET_RESULTS, NNUNET_RESULTS.exists()),
    ]
    return "\n".join(f"{name:<20} {'ok     ' if ok else 'MISSING'} {path}" for name, path, ok in rows)


if __name__ == "__main__":
    print(describe())

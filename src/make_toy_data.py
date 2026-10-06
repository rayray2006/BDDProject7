"""Generate a tiny synthetic dataset in the SegRap folder layout for local testing.

Each case has image.nii.gz, image_contrast.nii.gz and a few organ masks, including nested ones
(Brain > TemporalLobe_L, Eye_L > Lens_L) so the overlap handling in prepare_dataset.py is exercised.
Volumes are 64x64x32, so the whole pipeline runs on a laptop CPU (one 1-epoch fold takes ~15 min).

    python -m src.make_toy_data                       # 8 cases -> data/toy_raw
    python -m src.prepare_dataset 900 --name Toy --source data/toy_raw --num-test 2
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from src import config

SHAPE = (64, 64, 32)
SPACING = (1.0, 1.0, 3.0)


def ellipsoid(center, radii) -> np.ndarray:
    grid = np.indices(SHAPE).astype(np.float32)
    return sum(((grid[i] - center[i]) / radii[i]) ** 2 for i in range(3)) <= 1.0


def make_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    j = lambda: rng.uniform(-2, 2)
    head = ellipsoid((32 + j(), 32 + j(), 16), (28, 26, 15))
    brain = ellipsoid((32 + j(), 34 + j(), 18), (20, 18, 10))
    temporal = brain & ellipsoid((42 + j(), 30 + j(), 15), (6, 8, 4))
    eye = ellipsoid((40 + j(), 14 + j(), 18), (4, 4, 2))
    lens = eye & ellipsoid((40, 11, 18), (2, 1.5, 1))
    cord = ellipsoid((32, 46 + j(), 8), (2.5, 2.5, 8))
    masks = {"Brain": brain, "TemporalLobe_L": temporal, "Eye_L": eye, "Lens_L": lens, "SpinalCord": cord}

    ct = np.full(SHAPE, -1000.0, dtype=np.float32)
    for region, hu in ((head, 20), (brain, 35), (temporal, 45), (eye, 10), (lens, 80), (cord, 50)):
        ct[region] = hu
    ct += rng.normal(0, 8, SHAPE)
    contrast = ct + (head * 25) + rng.normal(0, 8, SHAPE)
    return ct.astype(np.int16), contrast.astype(np.int16), masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "data" / "toy_raw")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    affine = np.diag([*SPACING, 1.0])
    for i in range(args.cases):
        case = args.out / f"segrap_{i:04d}"
        case.mkdir(parents=True, exist_ok=True)
        ct, contrast, masks = make_case(rng)
        nib.save(nib.Nifti1Image(ct, affine), case / "image.nii.gz")
        nib.save(nib.Nifti1Image(contrast, affine), case / "image_contrast.nii.gz")
        for organ, m in masks.items():
            nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), case / f"{organ}.nii.gz")
    print(f"wrote {args.cases} toy cases to {args.out}")


if __name__ == "__main__":
    main()

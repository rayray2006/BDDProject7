"""
Aim II: Distributed Annotation Budget Training

Compare fully annotated patients (dense) with partially annotated
patients (sparse) under the same annotation budget.

Patient-specific OAR assignments are loaded from Aim II manifests.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    get_active_data_dir,
    MANIFEST_DIR,
    CHECKPOINT_DIR,
    PRETRAINED_WEIGHTS_PATH,
)
from src.model_factory import build_model_and_optimizer
from src.train_aim1 import compute_prompted_dice_bce_loss

NUM_OARS = 45

def validate_manifest(manifest):
    """Validate patient assignments and matched annotation budgets."""

    strategy = manifest["strategy"]
    budget = int(manifest["budget_units"])
    train = manifest["train"]
    test = manifest["test"]

    if strategy not in ("dense", "sparse"):
        raise ValueError(f"Unknown annotation strategy: {strategy}")

    if budget not in (5, 10, 20):
        raise ValueError(f"Unexpected annotation budget: {budget}")

    # Dense uses fewer fully labeled patients; sparse uses four times as many.
    expected_patients = budget if strategy == "dense" else budget * 4

    if len(train) != expected_patients:
        raise ValueError("Incorrect number of training patients")

    train_ids = [item["case_id"] for item in train]
    test_ids = [item["case_id"] for item in test]

    # Ensure each patient is unique and the test cohort remains held out.
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate training patients")

    if len(test_ids) != 20 or len(set(test_ids)) != 20:
        raise ValueError("Expected 20 unique test patients")

    if set(train_ids) & set(test_ids):
        raise ValueError("Training/test patient overlap detected")

    oar_counts = {oar: 0 for oar in range(1, NUM_OARS + 1)}

    for item in train:
        active_oars = item["active_oars"]

        # Validate each patient's assigned OARs.
        if not active_oars or len(active_oars) != len(set(active_oars)):
            raise ValueError(f"Invalid OAR assignments: {item['case_id']}")

        if any(oar not in oar_counts for oar in active_oars):
            raise ValueError(f"Invalid OAR ID: {item['case_id']}")

        if strategy == "dense" and len(active_oars) != NUM_OARS:
            raise ValueError("Dense patients must have all 45 OARs")

        if strategy == "sparse" and len(active_oars) not in (11, 12):
            raise ValueError("Sparse patients must have 11-12 OARs")

        for oar in active_oars:
            oar_counts[oar] += 1

    # Both strategies must contain the same total number of annotations.
    expected_annotations = budget * NUM_OARS

    if sum(oar_counts.values()) != expected_annotations:
        raise ValueError("Annotation budget mismatch")

    # Balance supervision so each OAR is represented equally often.
    if any(count != budget for count in oar_counts.values()):
        raise ValueError("Unbalanced OAR annotation assignments")

    print(
        f"[Manifest] {strategy.upper()} | "
        f"Budget={budget} | Patients={len(train)} | "
        f"Annotations={expected_annotations}",
        flush=True,
    )

    return strategy, budget

def annotation_aware_crop(
    img: torch.Tensor,
    lbl: torch.Tensor,
    active_oars: list[int],
    crop_size=(128, 128, 96),
):
    """
    Select a training patch centered on an annotated OAR.

    The same cropping procedure is used for dense and sparse training.
    """

    _, _, D, H, W = img.shape

    cd = min(D, crop_size[0])
    ch = min(H, crop_size[1])
    cw = min(W, crop_size[2])

    # Identify annotated OARs that are present in this CT volume.
    available_oars = [
        oar for oar in active_oars
        if torch.any(lbl[0] == oar).item()
    ]

    if available_oars:
        # Sample an organ first so large structures do not dominate crops.
        selected_oar = random.choice(available_oars)
        coords = torch.nonzero(lbl[0] == selected_oar)

        center = coords[
            torch.randint(len(coords), (1,)).item()
        ]

        z0 = max(0, min(int(center[0]) - cd // 2, D - cd))
        y0 = max(0, min(int(center[1]) - ch // 2, H - ch))
        x0 = max(0, min(int(center[2]) - cw // 2, W - cw))

    else:
        # Use a random crop if none of the selected OARs are present.
        z0 = random.randint(0, D - cd)
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)

    img_crop = img[:, :, z0:z0+cd, y0:y0+ch, x0:x0+cw]
    lbl_crop = lbl[:, z0:z0+cd, y0:y0+ch, x0:x0+cw]

    # Pad odd dimensions to maintain encoder-decoder shape compatibility.
    pd, ph, pw = cd % 2, ch % 2, cw % 2

    if pd or ph or pw:
        img_crop = F.pad(
            img_crop,
            (0, pw, 0, ph, 0, pd),
            mode="replicate",
        )

        lbl_crop = F.pad(
            lbl_crop,
            (0, pw, 0, ph, 0, pd),
            value=-1,
        )

    return img_crop, lbl_crop

def prepare_supervision_labels(
    lbl: torch.Tensor,
    active_oars: list[int],
):
    """
    Exclude deliberately withheld OARs from the training loss.

    The original fully annotated mask remains unchanged. Voxels belonging
    to unselected OARs are marked -1 so the imported Aim I loss ignores
    them rather than incorrectly treating them as background.
    """

    supervision_labels = lbl.clone()

    # Identify voxels belonging to OARs annotated for this patient.
    annotated_mask = torch.zeros_like(lbl, dtype=torch.bool)

    for oar_id in active_oars:
        annotated_mask |= (lbl == oar_id)

    # Positive labels not selected for training represent withheld OARs.
    withheld_mask = (lbl > 0) & ~annotated_mask

    # The shared loss excludes voxels labeled -1.
    supervision_labels[withheld_mask] = -1

    return supervision_labels

def main():
    parser = argparse.ArgumentParser(
        description="Aim II Dense and Sparse OAR Trainer"
    )

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=3,
        default=(128, 128, 96),
    )

    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1 or any(x < 2 for x in args.crop_size):
        parser.error("Epochs and crop dimensions must be valid")

    # Load the experiment configuration generated for Aim II.
    manifest_path = MANIFEST_DIR / args.manifest

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    strategy, budget_units = validate_manifest(manifest)
    seed = int(manifest["seed"])

    # Set random seeds for reproducible training and crop selection.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    # Avoid running a pretrained experiment without checkpoint weights.
    if not PRETRAINED_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {PRETRAINED_WEIGHTS_PATH}"
        )

    # Reuse the model factory and optimizer configuration from Aim I.
    model, optimizer = build_model_and_optimizer(
        init_mode="pretrained",
        budget_N=budget_units,
        device=device,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_dir = get_active_data_dir()

    train_items = manifest["train"]
    epochs = args.epochs
    crop_size = tuple(args.crop_size)

    # A smoke test processes two patients for one epoch.
    if args.smoke_test:
        train_items = train_items[:2]
        epochs = 1
        crop_size = (32, 32, 16)

    print(
        f"[Aim II] Strategy={strategy} | Budget={budget_units} | "
        f"Seed={seed} | Device={device} | Data={data_dir}",
        flush=True,
    )

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for item in train_items:
            case_id = item["case_id"]
            active_oars = item["active_oars"]

            # Load the preprocessed CT and original segmentation mask.
            npz_path = data_dir / f"{case_id}.npz"

            with np.load(npz_path) as arr:
                img = torch.from_numpy(
                    arr["image"].copy()
                ).unsqueeze(0)

                lbl = torch.from_numpy(
                    arr["label"].copy()
                ).unsqueeze(0)

            # Select a patch around an annotated structure.
            img_crop, lbl_crop = annotation_aware_crop(
                img,
                lbl,
                active_oars,
                crop_size,
            )

            # Mask withheld OARs without modifying the original labels.
            supervision_labels = prepare_supervision_labels(
                lbl_crop,
                active_oars,
            )

            img_crop = img_crop.to(device)
            supervision_labels = supervision_labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):

                # Generate predictions only for annotated OAR classes.
                logits = model(img_crop, active_oars)

                # Reuse the Aim I loss with withheld regions ignored.
                loss = compute_prompted_dice_bce_loss(
                    logits,
                    supervision_labels,
                    active_oars,
                )

            # Stop training if the loss becomes invalid.
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss for patient {case_id}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(train_items)

        print(
            f"[Aim II {strategy.upper()}] "
            f"Epoch {epoch+1}/{epochs} | Loss={mean_loss:.4f}",
            flush=True,
        )

    # Use unique checkpoint names for each strategy, budget, and seed.
    suffix = "_smoke" if args.smoke_test else ""

    ckpt_name = (
        f"nv_aim2_{strategy}_B{budget_units}_seed{seed}{suffix}.pt"
    )

    ckpt_path = CHECKPOINT_DIR / ckpt_name

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aim": f"aim2_{strategy}",
            "strategy": strategy,
            "init_mode": "pretrained",
            "budget_N": budget_units,
            "budget_units": budget_units,
            "seed": seed,
            "epochs": epochs,
            "crop_size": crop_size,
            "manifest": args.manifest,
            "smoke_test": args.smoke_test,
        },
        ckpt_path,
    )

    print(
        f"[Aim II] Saved checkpoint: {ckpt_path}",
        flush=True,
    )

if __name__ == "__main__":
    main()

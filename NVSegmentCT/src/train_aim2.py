from __future__ import annotations
import argparse
import json
import torch
import numpy as np
from src.config import get_active_data_dir, MANIFEST_DIR, CHECKPOINT_DIR
from src.model_factory import build_model_and_optimizer
from src.train_aim1 import compute_prompted_dice_bce_loss


def annotation_aware_crop(img: torch.Tensor, lbl: torch.Tensor, active_oars: list[int], crop_size=(32, 32, 16)):
    """
    Ensures 3D training patches center on one of the 11-12 annotated OARs present
    in the partially labeled scan rather than unannotated background.
    """
    _, _, D, H, W = img.shape
    cd, ch, cw = min(D, crop_size[0]), min(H, crop_size[1]), min(W, crop_size[2])

    # Find coordinates belonging to any of the active annotated OARs
    mask = torch.zeros_like(lbl[0], dtype=torch.bool)
    for oar_id in active_oars:
        mask |= (lbl[0] == oar_id)

    coords = torch.nonzero(mask, as_tuple=False)
    if len(coords) > 0:
        center = coords[torch.randint(0, len(coords), (1,)).item()]
        z0 = int(torch.clamp(center[0] - cd // 2, 0, D - cd))
        y0 = int(torch.clamp(center[1] - ch // 2, 0, H - ch))
        x0 = int(torch.clamp(center[2] - cw // 2, 0, W - cw))
    else:
        z0, y0, x0 = 0, 0, 0

    return (
        img[:, :, z0:z0+cd, y0:y0+ch, x0:x0+cw],
        lbl[:, z0:z0+cd, y0:y0+ch, x0:x0+cw],
    )


def main():
    parser = argparse.ArgumentParser(description="Aim II Distributed Sparse OAR Trainer")
    parser.add_argument("--manifest", type=str, required=True, help="Sparse Aim II manifest in MANIFEST_DIR")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    with open(MANIFEST_DIR / args.manifest, "r") as f:
        manifest = json.load(f)

    budget_units = manifest.get("budget_units", 5)
    seed = manifest.get("seed", 42)
    torch.manual_seed(seed)

    model, optimizer = build_model_and_optimizer(
        init_mode="pretrained", budget_N=budget_units, device=device
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_dir = get_active_data_dir()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for item in manifest["train"]:
            npz_path = data_dir / f"{item['case_id']}.npz"
            arr = np.load(npz_path)
            img = torch.from_numpy(arr["image"]).unsqueeze(0).to(device)
            lbl = torch.from_numpy(arr["label"]).unsqueeze(0).to(device)

            # Prompt Filtering: only pass the 11-12 annotated OARs for this patient
            active_oars = item["active_oars"]
            img_crop, lbl_crop = annotation_aware_crop(img, lbl, active_oars)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img_crop, active_oars)
                loss = compute_prompted_dice_bce_loss(logits, lbl_crop, active_oars)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        print(f"[Aim II Sparse] Epoch [{epoch+1}/{args.epochs}] - Prompt-Filtered Loss: {epoch_loss / len(manifest['train']):.4f}")

    ckpt_path = CHECKPOINT_DIR / f"nv_aim2_sparse_B{budget_units}_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "aim": "aim2_sparse",
        "init_mode": "pretrained",
        "budget_N": budget_units,
        "seed": seed,
    }, ckpt_path)
    print(f"[Aim II Sparse] Saved verified checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
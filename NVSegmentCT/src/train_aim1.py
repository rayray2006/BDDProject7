from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from src.config import get_active_data_dir, MANIFEST_DIR, CHECKPOINT_DIR
from src.model_factory import build_model_and_optimizer


def foreground_crop(img: torch.Tensor, lbl: torch.Tensor, crop_size=(128, 128, 96)):
    """Crops a 3D patch (33%+ foreground oversampling) so full 381x381x258 volumes fit in GPU VRAM."""
    _, _, D, H, W = img.shape
    cd, ch, cw = min(D, crop_size[0]), min(H, crop_size[1]), min(W, crop_size[2])

    fg_coords = torch.nonzero(lbl[0] > 0, as_tuple=False)
    if len(fg_coords) > 0 and torch.rand(1).item() < 0.67:
        center = fg_coords[torch.randint(0, len(fg_coords), (1,)).item()]
        z0 = int(torch.clamp(center[0] - cd // 2, 0, D - cd))
        y0 = int(torch.clamp(center[1] - ch // 2, 0, H - ch))
        x0 = int(torch.clamp(center[2] - cw // 2, 0, W - cw))
    else:
        z0 = torch.randint(0, max(1, D - cd + 1), (1,)).item()
        y0 = torch.randint(0, max(1, H - ch + 1), (1,)).item()
        x0 = torch.randint(0, max(1, W - cw + 1), (1,)).item()

    return (
        img[:, :, z0:z0+cd, y0:y0+ch, x0:x0+cw],
        lbl[:, z0:z0+cd, y0:y0+ch, x0:x0+cw],
    )


def compute_prompted_dice_bce_loss(
    logits: torch.Tensor,
    label_vol: torch.Tensor,
    active_oars: list[int],
    ignore_index: int = -1,
) -> torch.Tensor:
    valid_mask = (label_vol != ignore_index).float()
    probs = torch.sigmoid(logits)
    total_loss = 0.0

    for ch_idx, oar_id in enumerate(active_oars):
        target_bin = (label_vol == oar_id).float()
        pred_ch = probs[:, ch_idx]

        bce = F.binary_cross_entropy_with_logits(
            logits[:, ch_idx], target_bin, reduction="none"
        )
        bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        inter = (pred_ch * target_bin * valid_mask).sum()
        denom = ((pred_ch + target_bin) * valid_mask).sum()
        dice_loss = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)

        total_loss += 0.5 * (bce + dice_loss)

    return total_loss / len(active_oars)


def main():
    parser = argparse.ArgumentParser(description="Aim I & Aim III Control Trainer")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--init-mode", type=str, default="pretrained", choices=["pretrained", "scratch"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--use-interactive-fallback", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    with open(MANIFEST_DIR / args.manifest, "r") as f:
        manifest = json.load(f)

    budget_N = manifest.get("budget_N", len(manifest["train"]))
    seed = manifest.get("seed", 42)
    torch.manual_seed(seed)

    model, optimizer = build_model_and_optimizer(
        init_mode=args.init_mode, budget_N=budget_N, device=device
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    data_dir = get_active_data_dir()
    print(f"[Aim I Train] Reading tensors from: {data_dir} | Device: {device} | AMP: {use_amp}", flush=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for item in manifest["train"]:
            npz_path = data_dir / f"{item['case_id']}.npz"
            arr = np.load(npz_path)
            img = torch.from_numpy(arr["image"]).unsqueeze(0).to(device)
            lbl = torch.from_numpy(arr["label"]).unsqueeze(0).to(device)
            img_crop, lbl_crop = foreground_crop(img, lbl, crop_size=(128, 128, 96))
            active_oars = item["active_oars"]

            click_coords = None
            if args.use_interactive-fallback if hasattr(args, "use_interactive-fallback") else args.use_interactive_fallback:
                click_coords = torch.zeros((len(active_oars), 3), device=device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img_crop, active_oars, click_coords=click_coords)
                loss = compute_prompted_dice_bce_loss(logits, lbl_crop, active_oars)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(manifest["train"])
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {mean_loss:.4f}", flush=True)

    ckpt_name = f"nv_aim1_{args.init_mode}_N{budget_N}_seed{seed}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "aim": "aim1",
        "init_mode": args.init_mode,
        "budget_N": budget_N,
        "seed": seed,
    }, ckpt_path)
    print(f"[Aim I Train] Saved verified checkpoint -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
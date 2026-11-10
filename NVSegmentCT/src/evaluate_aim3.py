from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from src.config import (
    get_active_data_dir,
    MANIFEST_DIR,
    CHECKPOINT_DIR,
    EVAL_DIR,
    CLASS_MAP_PATH,
)
from src.model_factory import PromptableVISTA3DAdapter


def sliding_window_logits(
    model: torch.nn.Module,
    img: torch.Tensor,
    active_oars: list[int],
    roi_size: tuple[int, int, int] = (128, 128, 96),
     overlap: float = 0.25,
) -> torch.Tensor:
    """
    Runs memory-safe 3D sliding-window inference over full resampled volumes
    (e.g., 381x381x258) and guarantees even dimensions for stride-2 convs.
    """
    _, _, D, H, W = img.shape
    rd, rh, rw = roi_size
    sd = max(1, int(rd * (1.0 - overlap)))
    sh = max(1, int(rh * (1.0 - overlap)))
    sw = max(1, int(rw * (1.0 - overlap)))

    out_logits = torch.zeros((1, len(active_oars), D, H, W), dtype=torch.float32, device="cpu")
    count_map = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device="cpu")

    z_starts = list(range(0, max(1, D - rd + 1), sd))
    if D > rd and z_starts[-1] + rd < D:
        z_starts.append(D - rd)
    y_starts = list(range(0, max(1, H - rh + 1), sh))
    if H > rh and y_starts[-1] + rh < H:
        y_starts.append(H - rh)
    x_starts = list(range(0, max(1, W - rw + 1), sw))
    if W > rw and x_starts[-1] + rw < W:
        x_starts.append(W - rw)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1, y1, x1 = min(D, z0 + rd), min(H, y0 + rh), min(W, x0 + rw)
                patch = img[:, :, z0:z1, y0:y1, x0:x1]

                # Pad any odd dimension by 1 voxel so stride-2 encoder/decoder aligns
                pd = (2 - (patch.shape[2] % 2)) % 2
                ph = (2 - (patch.shape[3] % 2)) % 2
                pw = (2 - (patch.shape[4] % 2)) % 2
                if pd > 0 or ph > 0 or pw > 0:
                    patch = F.pad(patch, (0, pw, 0, ph, 0, pd), mode="replicate")

                with torch.amp.autocast("cuda", enabled=(img.device.type == "cuda")):
                    patch_logits = model(patch, active_oars)

                patch_logits = patch_logits[:, :, : (z1 - z0), : (y1 - y0), : (x1 - x0)].float().cpu()
                out_logits[:, :, z0:z1, y0:y1, x0:x1] += patch_logits
                count_map[:, :, z0:z1, y0:y1, x0:x1] += 1.0

    return out_logits / torch.clamp(count_map, min=1.0)


def compute_dsc_and_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.5, 1.5, 1.5)):
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    dsc = (2.0 * inter) / denom if denom > 0 else (1.0 if gt_bin.sum() == 0 else 0.0)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return round(dsc, 4), 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return round(dsc, 4), 99.0

    pred_border = np.logical_xor(pred_bin, binary_erosion(pred_bin))
    gt_border = np.logical_xor(gt_bin, binary_erosion(gt_bin))
    if not pred_border.any():
        pred_border = pred_bin
    if not gt_border.any():
        gt_border = gt_bin

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dists = np.concatenate([dt_gt[pred_border], dt_pred[gt_border]])
    hd95 = float(np.percentile(dists, 95))
    return round(dsc, 4), round(hd95, 4)


def main():
    parser = argparse.ArgumentParser(description="Aim III Unified Evaluation Harness")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--output-csv", type=str, default="aim3_unified_metrics.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    with open(MANIFEST_DIR / args.manifest, "r") as f:
        manifest = json.load(f)

    data_dir = get_active_data_dir()
    rows = []

    for ckpt_rel in args.checkpoints:
        ckpt_path = CHECKPOINT_DIR / ckpt_rel
        if not ckpt_path.exists():
            print(f"[Warn] Checkpoint not found, skipping: {ckpt_path}", flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = PromptableVISTA3DAdapter(in_channels=2, num_classes=45).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        init_mode = ckpt.get("init_mode", "pretrained")
        budget_N = ckpt.get("budget_N", 5)
        seed = ckpt.get("seed", 42)
        aim_tag = ckpt.get("aim", "aim1")
        print(f"[Eval] Evaluating {ckpt_rel} on {len(manifest['test'])} test case(s)...", flush=True)

        with torch.no_grad():
            for test_item in manifest["test"]:
                cid = test_item["case_id"]
                arr = np.load(data_dir / f"{cid}.npz")
                img = torch.from_numpy(arr["image"]).unsqueeze(0).to(device)
                lbl = arr["label"]
                valid_mask = (lbl != -1)

                active_oars = list(range(1, 46))
                logits = sliding_window_logits(model, img, active_oars)
                preds = (torch.sigmoid(logits[0]) > 0.5).numpy()

                for ch_idx, oar_id in enumerate(active_oars):
                    meta = class_map[str(oar_id)]
                    oar_name = meta["name"] if isinstance(meta, dict) else str(meta)
                    vol_tier = meta.get("volume_tier", "all") if isinstance(meta, dict) else "all"
                    prior_grp = meta.get("prior_group", "all") if isinstance(meta, dict) else "all"

                    pred_bin = np.logical_and(preds[ch_idx], valid_mask)
                    gt_bin = (lbl == oar_id)

                    dsc, hd95 = compute_dsc_and_hd95(pred_bin, gt_bin)
                    rows.append({
                        "case_id": cid,
                        "model_family": f"NV-Segment-CT_{init_mode}",
                        "aim": aim_tag,
                        "budget_N": budget_N,
                        "seed": seed,
                        "oar_id": oar_id,
                        "oar_name": oar_name,
                        "volume_tier": vol_tier,
                        "prior_group": prior_grp,
                        "DSC": dsc,
                        "HD95_mm": hd95,
                    })
                print(f"  -> Completed 45-OAR DSC & HD95 for {cid}", flush=True)

    out_csv = EVAL_DIR / args.output_csv
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        mean_dsc = np.mean([r["DSC"] for r in rows])
        print(f"[Eval] Exported {len(rows)} OAR evaluations to {out_csv} (Mean DSC={mean_dsc:.4f})", flush=True)


if __name__ == "__main__":
    main()
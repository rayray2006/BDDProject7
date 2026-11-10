from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from src.config import RAW_DATA_DIR, PREPROCESSED_LUSTRE_DIR, MANIFEST_DIR, CLASS_MAP_PATH

NUM_OARS = 45
IGNORE_INDEX = -1


def load_segrap_case(case_dir: Path, class_map: dict):
    t0 = time.time()
    nc_path = case_dir / "image.nii.gz" if (case_dir / "image.nii.gz").exists() else case_dir / "ncct.nii.gz"
    ce_path = case_dir / "image_contrast.nii.gz" if (case_dir / "image_contrast.nii.gz").exists() else case_dir / "cect.nii.gz"

    if not nc_path.exists() or not ce_path.exists():
        print(f"  [Skip] Missing image.nii.gz or image_contrast.nii.gz in {case_dir}", flush=True)
        return None, None, None, None

    print(f"  -> Loading CT volumes from {case_dir.name}...", flush=True)
    nc_nii = nib.load(str(nc_path))
    ce_nii = nib.load(str(ce_path))
    orig_spacing = nc_nii.header.get_zooms()[:3]

    nc_data = np.asarray(nc_nii.dataobj, dtype=np.float32)
    ce_data = np.asarray(ce_nii.dataobj, dtype=np.float32)

    combined_label_path = case_dir / "label.nii.gz"
    if combined_label_path.exists():
        lbl_data = np.asarray(nib.load(str(combined_label_path)).dataobj, dtype=np.int16)
        lbl_data[lbl_data >= 46] = IGNORE_INDEX
    else:
        lbl_data = np.zeros(nc_data.shape, dtype=np.int16)
        found_oars = 0
        for oar_id_str, meta in class_map.items():
            oar_id = int(oar_id_str)
            oar_file = case_dir / f"{meta['name']}.nii.gz"
            if oar_file.exists():
                mask = np.asarray(nib.load(str(oar_file)).dataobj) > 0
                lbl_data[mask] = oar_id
                found_oars += 1

        for gtv_name in ["GTVp.nii.gz", "GTVnx.nii.gz", "GTVnd.nii.gz"]:
            gtv_file = case_dir / gtv_name
            if gtv_file.exists():
                gtv_mask = np.asarray(nib.load(str(gtv_file)).dataobj) > 0
                lbl_data[gtv_mask] = IGNORE_INDEX

        print(f"  -> Merged {found_oars}/45 OAR masks + GTVs in {time.time() - t0:.1f}s", flush=True)

    return nc_data, ce_data, lbl_data, orig_spacing


def preprocess_and_audit(max_cases: int = 5, target_spacing=(1.5, 1.5, 1.5)):
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)

    print(f"Scanning RAW_DATA_DIR: {RAW_DATA_DIR}", flush=True)
    case_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])[:max_cases]
    print(f"Found {len(case_dirs)} subject folders: {[d.name for d in case_dirs]}", flush=True)

    audit_records = []
    processed_ids = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        out_npz = PREPROCESSED_LUSTRE_DIR / f"{case_id}.npz"

        if out_npz.exists():
            print(f"[Resume] {out_npz.name} already exists, reusing!", flush=True)
            processed_ids.append(case_id)
            continue

        t_case = time.time()
        nc_data, ce_data, lbl_data, orig_spacing = load_segrap_case(case_dir, class_map)
        if nc_data is None:
            continue

        nc_norm = np.clip(nc_data, -1000, 1500)
        ce_norm = np.clip(ce_data, -1000, 1500)
        nc_norm = (nc_norm - nc_norm.mean()) / (nc_norm.std() + 1e-6)
        ce_norm = (ce_norm - ce_norm.mean()) / (ce_norm.std() + 1e-6)

        zoom_factors = [o / t for o, t in zip(orig_spacing, target_spacing)]
        print(f"  -> Resampling {case_id} from {orig_spacing} to {target_spacing}...", flush=True)
        nc_res = zoom(nc_norm, zoom_factors, order=1)
        ce_res = zoom(ce_norm, zoom_factors, order=1)
        lbl_res = zoom(lbl_data, zoom_factors, order=0)

        v_before = float(np.sum(lbl_data == 4) * np.prod(orig_spacing))
        v_after = float(np.sum(lbl_res == 4) * np.prod(target_spacing))
        deg_pct = abs(v_after - v_before) / (v_before + 1e-6) * 100.0
        audit_records.append({
            "case_id": case_id,
            "cochlea_vol_before_mm3": round(v_before, 2),
            "cochlea_vol_after_mm3": round(v_after, 2),
            "vol_change_pct": round(deg_pct, 2),
        })

        img_2ch = np.stack([nc_res, ce_res], axis=0).astype(np.float32)
        np.savez(out_npz, image=img_2ch, label=lbl_res.astype(np.int16))
        processed_ids.append(case_id)
        print(f"[Done] {case_id}: shape={img_2ch.shape}, Cochlea dVol={deg_pct:.1f}% ({time.time() - t_case:.1f}s)", flush=True)

    with open(MANIFEST_DIR / "boundary_fidelity_audit.json", "w") as f:
        json.dump(audit_records, f, indent=2)

    return processed_ids


def build_manifests(processed_ids: list, train_budget_N: int = 5, seed: int = 42):
    rng = random.Random(seed)
    ids = list(processed_ids)
    rng.shuffle(ids)

    train_pool = ids[:train_budget_N]
    test_ids = ids[train_budget_N:] if len(ids) > train_budget_N else ids[-1:]
    all_oars = list(range(1, NUM_OARS + 1))

    aim1_manifest = {
        "seed": seed,
        "budget_N": len(train_pool),
        "train": [{"case_id": cid, "active_oars": all_oars} for cid in train_pool],
        "test": [{"case_id": cid, "active_oars": all_oars} for cid in test_ids],
    }
    manifest_file = MANIFEST_DIR / f"aim1_N{len(train_pool)}_seed{seed}.json"
    with open(manifest_file, "w") as f:
        json.dump(aim1_manifest, f, indent=2)
    print(f"Created Aim I manifest: {manifest_file} (Train N={len(train_pool)}, Test={len(test_ids)})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--train-budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case_ids = preprocess_and_audit(max_cases=args.max_cases)
    build_manifests(case_ids, train_budget_N=args.train_budget, seed=args.seed)
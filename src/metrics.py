"""Boundary metric for predictions: HD95 (95th-percentile Hausdorff distance, in mm) per case and label.

nnUNetv2_evaluate_folder reports Dice/IoU but no boundary distance, so evaluate.py calls this afterwards.
HD95 follows the usual (MONAI / SegRap) definition: the larger of the two directed 95th-percentile
distances between the predicted and the reference surface. It is undefined (NaN) when either mask is
empty for that case; those cases are counted separately as "missed" (organ in reference, none predicted)
or "false positive" (organ predicted, absent in reference).

    python -m src.metrics PRED_DIR LABEL_DIR DATASET_JSON -o hd95.csv
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred: np.ndarray, ref: np.ndarray, spacing) -> float:
    if not pred.any() or not ref.any():
        return float("nan")
    union = pred | ref
    box = tuple(slice(max(int(i.min()) - 1, 0), int(i.max()) + 2) for i in np.nonzero(union))
    pred, ref = pred[box], ref[box]  # distances only matter inside the joint bounding box
    sp, sr = _surface(pred), _surface(ref)
    d_pr = ndimage.distance_transform_edt(~sr, sampling=spacing)[sp]
    d_rp = ndimage.distance_transform_edt(~sp, sampling=spacing)[sr]
    return float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95)))


def case_hd95(task) -> tuple[str, dict[int, tuple[float, bool, bool]]]:
    """Per label: (hd95, present in reference, present in prediction)."""
    pred_file, ref_file, labels = task
    pred_img, ref_img = nib.load(pred_file), nib.load(ref_file)
    pred, ref = np.asarray(pred_img.dataobj), np.asarray(ref_img.dataobj)
    if pred.shape != ref.shape:
        raise ValueError(f"{pred_file.name}: prediction {pred.shape} and reference {ref.shape} differ")
    spacing = ref_img.header.get_zooms()[:3]
    out = {}
    for lid in labels:
        p, r = pred == lid, ref == lid
        out[lid] = (hd95(p, r, spacing), bool(r.any()), bool(p.any()))
    return pred_file.name, out


def compute(pred_dir: Path, label_dir: Path, dataset_json: Path, num_processes: int = 2) -> dict:
    """Return {'per_case': {case: {label_name: hd95}}, 'per_label': {label_name: {...summary...}}}."""
    label_ids = {int(v): k for k, v in json.loads(dataset_json.read_text())["labels"].items() if int(v) > 0}
    refs = sorted(label_dir.glob("*.nii.gz"))
    tasks = [(pred_dir / r.name, r, list(label_ids)) for r in refs if (pred_dir / r.name).exists()]
    if len(tasks) < len(refs):
        print(f"warning: {len(refs) - len(tasks)} reference case(s) have no prediction in {pred_dir}")
    with Pool(max(1, num_processes)) as pool:
        results = dict(pool.map(case_hd95, tasks))

    per_case = {case: {label_ids[lid]: v[0] for lid, v in vals.items()} for case, vals in results.items()}
    per_label = {}
    for lid, name in label_ids.items():
        rows = [results[c][lid] for c in results]
        finite = np.array([d for d, _, _ in rows if np.isfinite(d)])
        per_label[name] = {"hd95_mean": float(finite.mean()) if finite.size else float("nan"),
                           "hd95_median": float(np.median(finite)) if finite.size else float("nan"),
                           "n_cases": int(finite.size),
                           "n_missed": sum(r and not p for _, r, p in rows),
                           "n_false_positive": sum(p and not r for _, r, p in rows)}
    return {"per_case": per_case, "per_label": per_label}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("label_dir", type=Path)
    ap.add_argument("dataset_json", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("hd95.csv"))
    ap.add_argument("-np", "--num-processes", type=int, default=2)
    args = ap.parse_args()
    res = compute(args.pred_dir, args.label_dir, args.dataset_json, args.num_processes)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "hd95_mean_mm", "hd95_median_mm", "n_cases", "n_missed", "n_false_positive"])
        for name, m in res["per_label"].items():
            w.writerow([name, m["hd95_mean"], m["hd95_median"], m["n_cases"], m["n_missed"], m["n_false_positive"]])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Inference, metrics, and post-processing for a trained nnU-Net model.

Steps (each can be skipped):
  1. predict     nnUNetv2_predict on the test images (default: nnUNet_raw/DatasetXXX/imagesTs)
  2. postprocess optional connected-component cleanup learned from cross-validation (--postprocess)
  3. metrics     nnUNetv2_evaluate_folder against the test labels -> summary.json + per-organ dice.csv

Predictions go to nnUNet_results/DatasetXXX/predictions/<trainer__plans__config>_folds_<f>/.

Examples:
    python -m src.evaluate 101 -f 0                    # fold-0 model on the held-out test set
    python -m src.evaluate 101 --postprocess           # 5-fold ensemble + postprocessing (needs train.py --npz on all folds)
    python -m src.evaluate 101 -f 0 -i /path/to/new_scans --no-metrics
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from src import config


def find_postprocessing(model: Path, dataset_id: int, args) -> Path:
    folds = "_".join(args.folds)
    pkl = model / f"crossval_results_folds_{folds}" / "postprocessing.pkl"
    if not pkl.exists():
        print(f"no {pkl.relative_to(config.NNUNET_RESULTS)} yet; determining postprocessing from cross-validation")
        config.run(["nnUNetv2_find_best_configuration", dataset_id, "-c", args.configuration, "-tr", args.trainer,
                    "-p", args.plans, "-f", *args.folds, "--disable_ensembling"])
    if not pkl.exists():
        sys.exit(f"{pkl} was not created. Train every fold in {args.folds} with --npz first.")
    return pkl


def report(summary_file: Path, dataset_json: Path, out_csv: Path) -> None:
    summary = json.loads(summary_file.read_text())
    names = {str(v): k for k, v in json.loads(dataset_json.read_text())["labels"].items()}
    rows = sorted(((names.get(k, k), m["Dice"], m.get("IoU")) for k, m in summary["mean"].items()), key=lambda r: r[1])
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "dice", "iou"])
        w.writerows(rows)
    print(f"\n{'label':<20}{'Dice':>8}")
    for name, dice, _ in rows:
        print(f"{name:<20}{dice:>8.3f}")
    print(f"{'mean (foreground)':<20}{summary['foreground_mean']['Dice']:>8.3f}")
    print(f"\nwrote {summary_file}\nwrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_id", type=int)
    ap.add_argument("-f", "--folds", nargs="+", default=["0", "1", "2", "3", "4"], help="folds to ensemble, or 'all'")
    ap.add_argument("-c", "--configuration", default=config.DEFAULT_CONFIG)
    ap.add_argument("-tr", "--trainer", default=config.DEFAULT_TRAINER)
    ap.add_argument("-p", "--plans", default=config.DEFAULT_PLANS)
    ap.add_argument("-chk", "--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("-device", "--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("-i", "--input", type=Path, help="folder of images (_0000.nii.gz ...); default imagesTs")
    ap.add_argument("-l", "--labels", type=Path, help="ground-truth folder; default labelsTs")
    ap.add_argument("-o", "--output", type=Path, help="prediction folder; default under nnUNet_results/.../predictions")
    ap.add_argument("--postprocess", action="store_true", help="apply cross-validation postprocessing")
    ap.add_argument("--skip-predict", action="store_true", help="reuse existing predictions in the output folder")
    ap.add_argument("--no-metrics", action="store_true", help="only predict (e.g. unlabeled scans)")
    args, extra = ap.parse_known_args()

    name = config.dataset_name(args.dataset_id)
    model = config.model_dir(args.dataset_id, args.trainer, args.plans, args.configuration)
    raw = config.NNUNET_RAW / name
    images = args.input or raw / "imagesTs"
    labels = args.labels or raw / "labelsTs"
    tag = f"{model.name}_folds_{'_'.join(args.folds)}"
    out = args.output or config.NNUNET_RESULTS / name / "predictions" / tag
    if not images.is_dir() or not any(images.iterdir()):
        sys.exit(f"No images in {images}. Re-run prepare_dataset.py with --num-test N, or pass -i.")

    if not args.skip_predict:
        missing = [f for f in args.folds if not (model / f"fold_{f}" / args.checkpoint).exists()]
        if missing:
            sys.exit(f"No {args.checkpoint} for fold(s) {missing} in {model}. Train them first or pass -f.")
        config.run(["nnUNetv2_predict", "-i", images, "-o", out, "-d", args.dataset_id, "-c", args.configuration,
                    "-tr", args.trainer, "-p", args.plans, "-f", *args.folds, "-chk", args.checkpoint,
                    "-device", args.device, "--continue_prediction"] + extra)

    if args.postprocess:
        pkl = find_postprocessing(model, args.dataset_id, args)
        pp_out = out.with_name(out.name + "_pp")
        config.run(["nnUNetv2_apply_postprocessing", "-i", out, "-o", pp_out, "-pp_pkl_file", pkl,
                    "-plans_json", model / "plans.json", "-dataset_json", model / "dataset.json", "-np", 2])
        out = pp_out

    if args.no_metrics:
        print(f"predictions in {out}")
        return
    if not labels.is_dir():
        sys.exit(f"No labels at {labels}; pass -l or use --no-metrics.")
    summary = out / "summary.json"
    config.run(["nnUNetv2_evaluate_folder", labels, out, "-djfile", model / "dataset.json",
                "-pfile", model / "plans.json", "-o", summary, "-np", 2])
    report(summary, model / "dataset.json", out / "dice.csv")


if __name__ == "__main__":
    main()

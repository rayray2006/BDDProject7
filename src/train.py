"""CLI runner for nnU-Net training folds.

Resumes automatically: if a fold already has checkpoint_latest.pth (e.g. Colab timed out), training
continues from it. Folds that already have checkpoint_final.pth are skipped unless --retrain is given.
nnU-Net writes checkpoint_latest.pth every 50 epochs, so at most ~50 epochs are lost on a disconnect.

Examples:
    python -m src.train 101 0                         # fold 0, 3d_fullres
    python -m src.train 101 0 1 2 3 4 --npz           # all folds; --npz needed for evaluate.py --postprocess
    python -m src.train 101 all                       # train on all cases (no validation split)
    python -m src.train 900 0 -tr nnUNetTrainer_5epochs -device cpu   # toy smoke test

Any extra flags are passed straight to nnUNetv2_train.
"""
import argparse

from src import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_id", type=int)
    ap.add_argument("folds", nargs="+", help="fold numbers 0-4, or 'all'")
    ap.add_argument("-c", "--configuration", default=config.DEFAULT_CONFIG)
    ap.add_argument("-tr", "--trainer", default=config.DEFAULT_TRAINER,
                    help="e.g. nnUNetTrainer_100epochs for shorter runs (default: %(default)s)")
    ap.add_argument("-p", "--plans", default=config.DEFAULT_PLANS)
    ap.add_argument("-device", "--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--npz", action="store_true", help="save validation softmax (needed for postprocessing search)")
    ap.add_argument("--retrain", action="store_true", help="start over even if a checkpoint exists")
    ap.add_argument("--pretrained", help="checkpoint to initialise from (only for fresh runs)")
    args, extra = ap.parse_known_args()

    model = config.model_dir(args.dataset_id, args.trainer, args.plans, args.configuration)
    for fold in args.folds:
        fold_dir = model / f"fold_{fold}"
        resume = (fold_dir / "checkpoint_latest.pth").exists()
        if (fold_dir / "checkpoint_final.pth").exists() and not args.retrain:
            print(f"fold {fold}: already finished ({fold_dir / 'checkpoint_final.pth'}), skipping. Use --retrain to redo.")
            continue
        print(f"fold {fold}: {'resuming from checkpoint_latest' if resume and not args.retrain else 'starting'} -> {fold_dir}")

        cmd = ["nnUNetv2_train", args.dataset_id, args.configuration, fold,
               "-tr", args.trainer, "-p", args.plans, "-device", args.device]
        if args.npz:
            cmd.append("--npz")
        if args.pretrained and not resume:
            cmd += ["-pretrained_weights", args.pretrained]
        elif not args.retrain:
            cmd.append("--c")  # nnU-Net starts fresh if there is no checkpoint to continue from
        config.run(cmd + extra)


if __name__ == "__main__":
    main()

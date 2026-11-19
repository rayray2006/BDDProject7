"""Wrapper around nnUNetv2_plan_and_preprocess using the paths from src/config.py.

Examples:
    python -m src.preprocess 101                     # 3d_fullres, 2 workers, integrity check
    python -m src.preprocess 101 -c 2d 3d_fullres -np 4
    python -m src.preprocess 101 -pl nnUNetPlannerResEncM   # residual-encoder preset (plans: nnUNetResEncUNetMPlans)

Any extra flags are passed straight to nnUNetv2_plan_and_preprocess.
"""
import argparse

from src import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_id", type=int)
    ap.add_argument("-c", "--configurations", nargs="+", default=[config.DEFAULT_CONFIG],
                    help="configurations to preprocess (default: %(default)s)")
    ap.add_argument("-np", "--num-processes", type=int, default=2,
                    help="worker processes; full-res SegRap CTs are large, so keep this low on Colab (default: 2)")
    ap.add_argument("-pl", "--planner", help="experiment planner class, e.g. nnUNetPlannerResEncM")
    ap.add_argument("--no-verify", action="store_true", help="skip --verify_dataset_integrity")
    ap.add_argument("--clean", action="store_true", help="recompute the fingerprint (needed after changing the data)")
    args, extra = ap.parse_known_args()

    config.ensure_dirs()
    print(f"preprocessing {config.dataset_name(args.dataset_id)} -> {config.NNUNET_PREPROCESSED}")
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", args.dataset_id, "-c", *args.configurations,
           "-np", *[args.num_processes] * len(args.configurations), "-npfp", args.num_processes]
    if not args.no_verify:
        cmd.append("--verify_dataset_integrity")
    if args.clean:
        cmd.append("--clean")
    if args.planner:
        cmd += ["-pl", args.planner]
    config.run(cmd + extra)


if __name__ == "__main__":
    main()

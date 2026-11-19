"""Write the fixed patient split and the label-budget subsets for Experiment 1.

The labelled cases are split once into
    test   fixed held-out set, used to score every model (nnU-Net and NV-Segment-CT alike)
    val    fixed validation set, kept for tuning/early stopping of the fine-tuned foundation model
    pool   everything else; the training subsets are drawn from here

For each repeat r, the pool is shuffled once and the subsets are its first N cases, so the subsets are
nested (5 ⊂ 10 ⊂ 20 ⊂ 40 ⊂ 80). Nesting means the comparison between budgets is not muddied by which
patients happened to be drawn. A budget equal to the whole pool only gets one repeat.

Each subset gets its own nnU-Net dataset ID (first_id, first_id + 1, ...), so each is planned and
preprocessed on its own cases only, as it would be for a group that really had only N labelled patients.

The split is written to a JSON file that should be committed, so everyone uses the same cases:

    python -m src.make_splits                                  # 20 test / 20 val / 80 pool, 3 repeats
    python -m src.make_splits --source data/toy_raw --test 2 --val 1 --budgets 2 5 --repeats 2 \\
        --first-id 910 -o data/toy_splits.json                 # toy version
"""
import argparse
import json
import random
from pathlib import Path

from src import config
from src.prepare_dataset import CHANNEL_PRESETS, CHANNELS, image_source, organ_files

DEFAULT_OUT = config.PROJECT_ROOT / "splits" / "segrap_exp1.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=config.RAW_DATA_DIR, help="folder with case subfolders")
    ap.add_argument("--channels", choices=CHANNEL_PRESETS, default="both",
                    help="only cases that have these CT(s) are eligible (default: %(default)s)")
    ap.add_argument("--test", type=int, default=20, help="fixed test cases (default: %(default)s)")
    ap.add_argument("--val", type=int, default=20, help="fixed validation cases (default: %(default)s)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 40, 80],
                    help="training-set sizes (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=3, help="random draws per budget (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-id", type=int, default=110, help="dataset ID of the first subset (default: %(default)s)")
    ap.add_argument("--name", default="SegRapN", help="dataset name prefix, e.g. SegRapN -> Dataset110_SegRapN005r0")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Everyone should share one split; pass --overwrite only on purpose.")
    source = args.source.resolve()
    channels = CHANNEL_PRESETS[args.channels]
    cases = sorted(p.name for p in source.iterdir()
                   if p.is_dir() and all(image_source(p, CHANNELS[c][0]) for c in channels))
    pool_size = len(cases) - args.test - args.val
    if pool_size < max(args.budgets):
        raise SystemExit(f"{len(cases)} cases leave a pool of {pool_size}, smaller than the largest budget "
                         f"{max(args.budgets)}")

    rng = random.Random(args.seed)
    shuffled = rng.sample(cases, len(cases))
    test, val, pool = sorted(shuffled[:args.test]), sorted(shuffled[args.test:args.test + args.val]), \
        sorted(shuffled[args.test + args.val:])

    subsets = []
    for r in range(args.repeats):
        order = random.Random(f"{args.seed}-repeat-{r}").sample(pool, len(pool))
        for n in sorted(args.budgets):
            if n == len(pool) and r > 0:
                continue  # the whole pool is the same set every time
            subsets.append({"name": f"n{n:03d}_r{r}", "budget": n, "repeat": r, "cases": sorted(order[:n])})
    for i, s in enumerate(subsets):
        s["dataset_id"] = args.first_id + i
        s["dataset_name"] = f"Dataset{s['dataset_id']:03d}_{args.name}{s['budget']:03d}r{s['repeat']}"

    # Label numbering must be identical in every dataset, so it is fixed here from all cases' mask names.
    organs = sorted({o for c in cases for o in organ_files(source / c)}, key=str.lower)
    split = {
        "description": __doc__.split("\n\n")[0],
        "source": source.name,
        "seed": args.seed,
        "channels": args.channels,
        "organs": organs,
        "test": test,
        "val": val,
        "pool": pool,
        "subsets": subsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2))
    print(f"{len(cases)} cases: {len(test)} test, {len(val)} val, {len(pool)} pool; {len(organs)} organs")
    for s in subsets:
        print(f"  {s['dataset_name']:<28} {s['budget']:>3} cases")
    print(f"wrote {args.output}\nnext: python -m src.prepare_dataset --split-file {args.output} --subset all")


if __name__ == "__main__":
    main()

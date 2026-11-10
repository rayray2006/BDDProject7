
"""
Generate dense and sparse annotation manifests for Aim II.

Compare fully annotating fewer patients against distributing the same
annotation budget across more patients with partially annotated OARs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.config import MANIFEST_DIR, PREPROCESSED_LUSTRE_DIR


# Experiment settings from the Aim II proposal
NUM_OARS = 45
BUDGETS = (5, 10, 20)
TEST_COUNT = 20
MAX_TRAIN = 80
ALL_OARS = list(range(1, NUM_OARS + 1))


def load_id_list(path: Path) -> list[str]:
    """Load existing patient IDs to preserve the team's test cohort."""

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())

        # Support both standalone ID lists and existing Aim I manifests.
        if isinstance(data, dict):
            data = data.get(
                "test", data.get("case_ids", data.get("ids"))
            )

        if not isinstance(data, list):
            raise ValueError("Expected a list of patient IDs")

        ids = [
            item["case_id"] if isinstance(item, dict) else str(item)
            for item in data
        ]
    else:
        ids = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs found")

    return ids


def build_sparse_oars(seed: int) -> list[list[int]]:
    """
    Assign 11-12 annotated OARs to each of 80 patients.

    Each group of 20 patients receives exactly 225 annotations,
    ensuring the sparse strategy matches the dense annotation budget.
    """

    rng = random.Random(seed + 10000)

    # Shuffle OARs reproducibly so different seeds produce new assignments.
    shuffled_oars = ALL_OARS.copy()
    rng.shuffle(shuffled_oars)

    assignments = []

    # Four blocks of 20 patients correspond to the largest sparse budget.
    for block in range(4):

        # Rotate the class order to vary organ assignments across blocks.
        shift = (block * 7) % NUM_OARS
        rotated = shuffled_oars[shift:] + shuffled_oars[:shift]

        # Repeating all 45 OARs five times gives 225 annotations per block.
        stream = rotated * 5
        offset = 0

        for i in range(20):

            # Five patients get 12 OARs and fifteen get 11.
            # Total: (5 * 12) + (15 * 11) = 225 annotations.
            count = 12 if i % 4 == 0 else 11

            oars = sorted(stream[offset:offset + count])

            # Each patient should have the expected number of unique OARs.
            assert len(oars) == count
            assert len(oars) == len(set(oars))

            assignments.append(oars)
            offset += count

        assert offset == 225

    return assignments


def create_manifests(
    patient_ids: list[str],
    test_ids: list[str],
    seed: int,
    out_dir: Path
) -> list[Path]:
    """
    Create matched dense and sparse manifests for budgets 5, 10, and 20.

    Dense: 5, 10, or 20 patients with all 45 OARs annotated.
    Sparse: 20, 40, or 80 patients with approximately 11-12 OARs each.
    """

    # Keep the same 20 fully annotated test patients across experiments.
    if len(test_ids) != TEST_COUNT:
        raise ValueError("Exactly 20 test patients are required")

    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate test patients found")

    available = set(patient_ids)

    if not set(test_ids).issubset(available):
        raise ValueError("Some test patients are missing from the dataset")

    # Exclude held-out patients to prevent training/test data leakage.
    training_candidates = sorted(available - set(test_ids))

    if len(training_candidates) < MAX_TRAIN:
        raise ValueError("At least 80 training patients are required")

    rng = random.Random(seed)
    rng.shuffle(training_candidates)

    # Use the same ordered patient pool for all budgets within a seed.
    # Smaller training subsets are therefore nested in larger subsets.
    train_ids = training_candidates[:MAX_TRAIN]

    test_items = [
        {"case_id": cid, "active_oars": ALL_OARS.copy()}
        for cid in test_ids
    ]

    # Generate the patient-specific OAR assignments for sparse training.
    sparse_oars = build_sparse_oars(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for budget in BUDGETS:

        dense_count = budget
        sparse_count = budget * 4

        # Dense strategy: every selected patient contributes all 45 OARs.
        dense_train = [
            {
                "case_id": cid,
                "active_oars": ALL_OARS.copy()
            }
            for cid in train_ids[:dense_count]
        ]

        # Sparse strategy: more patients contribute fewer OARs each.
        # The original segmentation labels remain unchanged on disk.
        sparse_train = [
            {
                "case_id": cid,
                "active_oars": oars
            }
            for cid, oars in zip(
                train_ids[:sparse_count],
                sparse_oars[:sparse_count]
            )
        ]

        conditions = {
            "dense": dense_train,
            "sparse": sparse_train
        }

        for strategy, train in conditions.items():

            # Count how often each OAR is annotated across the patients.
            counts = Counter(
                oar
                for item in train
                for oar in item["active_oars"]
            )

            # Both strategies must use exactly budget * 45 annotations.
            assert sum(counts.values()) == budget * NUM_OARS

            # Ensure each OAR receives equal supervision across patients.
            assert all(
                counts[oar] == budget for oar in ALL_OARS
            )

            # Confirm that no held-out patient appears in training.
            assert not (
                {item["case_id"] for item in train} & set(test_ids)
            )

            # Save experiment metadata and patient-specific OAR assignments.
            manifest = {
                "aim": "aim2",
                "strategy": strategy,
                "budget_units": budget,
                "budget_N": budget,
                "seed": seed,
                "annotation_count": sum(counts.values()),
                "train": train,
                "test": test_items
            }

            filename = (
                f"aim2_{strategy}_B{budget}_seed{seed}.json"
            )

            path = out_dir / filename

            path.write_text(
                json.dumps(manifest, indent=2) + "\n"
            )

            written.append(path)

            print(
                f"{filename}: "
                f"{len(train)} training patients, "
                f"{len(test_items)} test patients, "
                f"{manifest['annotation_count']} annotations"
            )

    return written


def main():

    parser = argparse.ArgumentParser(
        description="Generate Aim II annotation manifests"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-seed", type=int, default=2026)

    parser.add_argument(
        "--test-ids-file",
        type=Path,
        help="Existing fixed Aim I test cohort"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PREPROCESSED_LUSTRE_DIR
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MANIFEST_DIR
    )

    args = parser.parse_args()

    # Find all available preprocessed SegRap2023 patient volumes.
    ids = sorted(
        p.stem for p in args.data_dir.glob("*.npz")
    )

    # Require enough patients for 80 training cases and 20 test cases.
    if len(ids) < MAX_TRAIN + TEST_COUNT:
        parser.error(
            f"Found {len(ids)} patients; "
            f"at least {MAX_TRAIN + TEST_COUNT} are required."
        )

    if args.test_ids_file:

        # Prefer the existing Aim I test cohort for fair comparisons.
        test_ids = load_id_list(args.test_ids_file)

    else:

        # Generate a reproducible holdout only if one is not provided.
        rng = random.Random(args.holdout_seed)

        shuffled = ids.copy()
        rng.shuffle(shuffled)

        test_ids = sorted(shuffled[:TEST_COUNT])

        args.output_dir.mkdir(
            parents=True, exist_ok=True
        )

        holdout_file = (
            args.output_dir /
            f"aim2_fixed_test_ids_seed{args.holdout_seed}.json"
        )

        # Prevent accidentally replacing an established holdout cohort.
        if holdout_file.exists():
            existing = json.loads(holdout_file.read_text())
            if existing != test_ids:
                parser.error(
                    "Existing holdout differs from generated IDs"
                )

        holdout_file.write_text(
            json.dumps(test_ids, indent=2) + "\n"
        )

        print(
            "IMPORTANT: Verify this holdout matches "
            f"Aim I: {holdout_file}"
        )

    # Produce the six matched-budget manifests for the selected seed.
    create_manifests(
        ids,
        test_ids,
        args.seed,
        args.output_dir
    )


if __name__ == "__main__":
    main()
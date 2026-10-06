"""Convert raw SegRap-style case folders into an nnU-Net v2 raw dataset.

Expected source layout (one folder per case, as in SegRap2023_Training_Set_120cases):

    RAW_DATA_DIR/
      segrap_0000/
        image.nii.gz            non-contrast CT   (or image/ = folder of DICOM slices)
        image_contrast.nii.gz   contrast CT       (or image_contrast/ = DICOM folder)
        Brain.nii.gz            one binary mask per organ
        ...

Output: nnUNet_raw/DatasetXXX_<name>/{imagesTr,labelsTr,imagesTs,labelsTs,dataset.json}.

Overlapping masks: SegRap structures are nested (Brain ⊃ TemporalLobe ⊃ Hippocampus, Larynx ⊃ Larynx_Glottic,
MiddleEar ⊃ Cochlea, Eye ⊃ Lens, ...). nnU-Net needs one label per voxel, so where masks overlap the
SMALLER structure (fewer voxels in that case) wins. Parents therefore become "parent minus children".

Examples:
    python -m src.prepare_dataset 101 --name SegRapOAR
    python -m src.prepare_dataset 900 --name Toy --source data/toy_raw --num-test 2
"""
import argparse
import json
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

from src import config

# channel key -> (file/folder stem in each case, channel name written to dataset.json)
CHANNELS = {
    "nc": ("image", "CT"),
    "ce": ("image_contrast", "CT"),
}
CHANNEL_PRESETS = {"both": ["nc", "ce"], "nc": ["nc"], "ce": ["ce"]}
IMAGE_STEMS = {stem for stem, _ in CHANNELS.values()}


def image_source(case_dir: Path, stem: str) -> Path | None:
    for candidate in (case_dir / f"{stem}.nii.gz", case_dir / f"{stem}.nii", case_dir / stem):
        if candidate.exists():
            return candidate
    return None


def organ_files(case_dir: Path) -> dict[str, Path]:
    files = {}
    for p in case_dir.glob("*.nii*"):
        name = p.name.removesuffix(".gz").removesuffix(".nii")
        if name in IMAGE_STEMS or "(" in name:  # skip CT volumes and Drive duplicates like "Brain (1).nii.gz"
            continue
        files[name] = p
    return files


def write_image(src: Path, dst: Path) -> None:
    if src.is_dir():  # DICOM series
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(src))
        if not series:
            raise ValueError(f"No DICOM series found in {src}")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(src), series[0]))
        sitk.WriteImage(reader.Execute(), str(dst), useCompression=True)
    elif src.name.endswith(".nii.gz"):
        shutil.copyfile(src, dst)
    else:
        nib.save(nib.load(src), dst)


def build_label(organs: dict[str, Path], label_ids: dict[str, int], ref: nib.Nifti1Image) -> np.ndarray:
    """Merge binary masks into one uint8 label map; on overlap the smaller structure wins."""
    label = np.zeros(ref.shape[:3], dtype=np.uint8)
    sizes = np.full(len(label_ids) + 1, np.iinfo(np.int64).max, dtype=np.int64)  # sizes[0] = background
    for organ, path in organs.items():
        img = nib.load(path)
        if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{path.name} geometry {img.shape} does not match the CT {ref.shape}")
        mask = np.asarray(img.dataobj) > 0  # masks may be 0/1 or 0/255
        box = bounding_box(mask)
        if box is None:
            continue
        mask, sub = mask[box], label[box]  # work on the crop: full-volume boolean indexing is ~50x slower
        n = int(mask.sum())
        lid = label_ids[organ]
        sizes[lid] = n
        region = sub[mask]
        region[sizes[region] > n] = lid  # overwrite background and larger structures only
        sub[mask] = region
    return label


def bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    box = []
    for axis in range(mask.ndim):
        nz = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        if nz.size == 0:
            return None
        box.append(slice(nz[0], nz[-1] + 1))
    return tuple(box)


def convert_case(task) -> tuple[str, str]:
    case_dir, case_id, split, channels, label_ids, out = task
    try:
        img_dir, lbl_dir = out / f"images{split}", out / f"labels{split}"
        ref = None
        for idx, ch in enumerate(channels):
            src = image_source(case_dir, CHANNELS[ch][0])
            dst = img_dir / f"{case_id}_{idx:04d}.nii.gz"
            write_image(src, dst)
            if ref is None:
                ref = nib.load(dst)
        label = build_label(organ_files(case_dir), label_ids, ref)
        lbl = nib.Nifti1Image(label, ref.affine, ref.header)
        lbl.set_data_dtype(np.uint8)
        lbl.header.set_slope_inter(1, 0)
        nib.save(lbl, lbl_dir / f"{case_id}.nii.gz")
        present = len(np.unique(label)) - 1
        return case_id, f"ok ({split}, {present}/{len(label_ids)} labels present)"
    except Exception as e:
        return case_id, f"FAILED: {e!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_id", type=int, help="3-digit nnU-Net dataset ID (claim it in README.md first)")
    ap.add_argument("--name", default="SegRapOAR", help="dataset name suffix, e.g. SegRapOAR -> Dataset101_SegRapOAR")
    ap.add_argument("--source", type=Path, default=config.RAW_DATA_DIR, help="folder with case subfolders")
    ap.add_argument("--channels", choices=CHANNEL_PRESETS, default="both",
                    help="which CT(s) to use as input channels (default: both)")
    ap.add_argument("--num-test", type=int, default=0, help="hold out this many cases into imagesTs/labelsTs")
    ap.add_argument("--limit", type=int, help="only convert the first N cases (quick tests)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the test split")
    ap.add_argument("--workers", type=int, default=2, help="parallel cases (each full SegRap case needs ~2 GB RAM)")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing dataset folder with this ID first")
    args = ap.parse_args()

    if not 1 <= args.dataset_id <= 999:
        sys.exit("dataset_id must be between 1 and 999")
    source = args.source.resolve()
    channels = CHANNEL_PRESETS[args.channels]
    name = f"Dataset{args.dataset_id:03d}_{args.name}"
    config.ensure_dirs()

    existing = [p for p in config.NNUNET_RAW.glob(f"Dataset{args.dataset_id:03d}_*") if p.is_dir()]
    if existing and not args.overwrite:
        sys.exit(f"{existing[0]} already exists. Pick another ID (see README registry) or pass --overwrite.")
    for p in existing:
        shutil.rmtree(p)
    out = config.NNUNET_RAW / name

    cases = sorted(p for p in source.iterdir() if p.is_dir() and all(image_source(p, CHANNELS[c][0]) for c in channels))
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit(f"No case folders with {[CHANNELS[c][0] for c in channels]} found in {source}")

    organs = sorted({o for c in cases for o in organ_files(c)}, key=str.lower)
    label_ids = {o: i for i, o in enumerate(organs, start=1)}
    if len(label_ids) > 254:
        sys.exit("More than 254 labels do not fit in uint8")
    for c in cases:
        missing = set(organs) - set(organ_files(c))
        if missing:
            print(f"warning: {c.name} has no mask for {sorted(missing)} (treated as empty)")

    if args.num_test >= len(cases):
        sys.exit(f"--num-test {args.num_test} leaves no training cases (found {len(cases)})")
    test = set(random.Random(args.seed).sample([c.name for c in cases], args.num_test))
    for sub in ("imagesTr", "labelsTr") + (("imagesTs", "labelsTs") if test else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"{name}: {len(cases)} cases ({len(cases) - len(test)} train, {len(test)} test), "
          f"{len(organs)} labels, channels {channels}\n  from {source}\n  to   {out}")
    tasks = [(c, c.name, "Ts" if c.name in test else "Tr", channels, label_ids, out) for c in cases]
    failed = []
    with Pool(max(1, args.workers)) as pool:
        for i, (case_id, status) in enumerate(pool.imap_unordered(convert_case, tasks), 1):
            print(f"[{i}/{len(tasks)}] {case_id}: {status}", flush=True)
            if status.startswith("FAILED"):
                failed.append(case_id)
    if failed:
        sys.exit(f"{len(failed)} case(s) failed: {failed}. Fix them and re-run with --overwrite.")

    dataset_json = {
        "name": name,
        "description": f"Converted from {source.name} by src/prepare_dataset.py. "
                       "Overlapping masks resolved per case: the smaller structure wins.",
        "channel_names": {str(i): CHANNELS[c][1] for i, c in enumerate(channels)},
        "channel_sources": {str(i): CHANNELS[c][0] for i, c in enumerate(channels)},
        "labels": {"background": 0, **label_ids},
        "numTraining": len(cases) - len(test),
        "file_ending": ".nii.gz",
        "test_cases": sorted(test),
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"wrote {out / 'dataset.json'}\nnext: python -m src.preprocess {args.dataset_id}")


if __name__ == "__main__":
    main()

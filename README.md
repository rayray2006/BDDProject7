# nnU-Net project · SegRap2023 OAR segmentation

Modular nnU-Net v2 pipeline for the Biomedical Data Design group.
**Code lives in GitHub; data, preprocessed files, and checkpoints live on the shared Google Drive.**

```
BDDProject7/
├── .env.example            # Template for local path configuration
├── .gitignore              # Ignores data, checkpoints, and local .env
├── requirements.txt        # Pinned Python dependencies
├── README.md               # This file
├── WeeklyPresentations/    # Group presentation slides
├── notebooks/
│   └── colab_runner.ipynb  # Colab runner: mount Drive, clone, run the CLI steps
├── scripts/
│   └── run_training.sh     # Batch / SLURM training job
└── src/
    ├── __init__.py
    ├── config.py           # Central path resolution & environment loader
    ├── prepare_dataset.py  # Converts raw DICOM/NIfTI into nnU-Net format
    ├── preprocess.py       # Wrapper around plan_and_preprocess
    ├── train.py            # CLI runner for training folds (auto-resume)
    ├── evaluate.py         # Inference, metrics, and post-processing
    └── make_toy_data.py    # Synthetic toy dataset for local testing
```

## Dataset ID Registry

nnU-Net identifies datasets by a 3-digit ID, and everyone writes to the same shared `nnUNet_raw/`,
`nnUNet_preprocessed/` and `nnUNet_results/`. **Add a row here (in a PR) before you create a dataset**
so that nobody overwrites anyone else's data. `prepare_dataset.py` refuses to overwrite an existing ID
unless you pass `--overwrite`.

| ID  | Folder name             | Owner | Description |
|-----|-------------------------|-------|-------------|
| 101 | Dataset101_SegRapOAR    | rayray2006 | SegRap2023 Task 1, 45 OARs, 2 channels (non-contrast + contrast CT), 20 held-out test cases |
| 102 |                         |       | _free_ |
| 103 |                         |       | _free_ |
| 900 | Dataset900_Toy          | everyone (local only) | Synthetic toy data from `make_toy_data.py`; never put on Drive |

Suggested ranges: 100–199 shared SegRap experiments, 200–299 personal experiments (one block of 10 per person), 900+ local scratch.

## Workflow

### Local development (code only)

1. Clone and branch:
   ```bash
   git clone https://github.com/rayray2006/BDDProject7.git && cd BDDProject7
   git checkout -b <your-name>/<feature>
   ```
2. Set up the environment (Python 3.10+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env      # then switch to the "Local development" lines
   python -m src.config      # prints the resolved paths
   ```
3. Smoke-test the whole pipeline on the toy dataset (~15–20 min on a laptop CPU, almost all of it the one training epoch):
   ```bash
   python -m src.make_toy_data
   python -m src.prepare_dataset 900 --name Toy --source data/toy_raw --num-test 2
   python -m src.preprocess 900
   python -m src.train 900 0 -tr nnUNetTrainer_1epoch -device cpu
   python -m src.evaluate 900 -f 0 -tr nnUNetTrainer_1epoch -device cpu
   ```
   (Use `-device mps` on Apple Silicon to try the GPU.) Dice is meaningless after 1 epoch; the point is that every step runs.
4. Commit and push your branch, then open a PR.

### Execution in Colab (compute only)

1. Once: in Drive, open *Shared with me*, right-click **Biomedical Data Design**, choose **Organize → Add shortcut** and put it in *My Drive*.
   Why: `drive.mount()` in Colab only exposes *My Drive* (and Shared drives), not *Shared with me*. The folder is owned by
   a teammate, so without a shortcut it does not appear under `/content/drive/MyDrive/` and the paths in `.env` don't resolve.
   The shortcut doesn't copy anything; everyone still reads and writes the same files.
2. Open `notebooks/colab_runner.ipynb` in Colab (File → Open notebook → GitHub) and select a GPU runtime.
3. Run the cells in order. They:
   - mount Drive and clone the repo at your branch,
   - install `requirements.txt`,
   - write a `.env` pointing at the shared folder (`Biomedical Data Design/nnunet/`),
   - run the CLI steps below.

Because `nnUNet_preprocessed/` and `nnUNet_results/` are on the mounted Drive, preprocessed data and
checkpoints are saved to Drive as they are written and survive a Colab timeout. After a disconnect,
re-run the setup cells and the same `train` command: it resumes from `checkpoint_latest.pth`.
nnU-Net saves that checkpoint every 50 epochs, so a disconnect loses at most 50 epochs.

### Pipeline steps

| Step | Command | Notes |
|------|---------|-------|
| Dataset conversion | `python -m src.prepare_dataset 101 --name SegRapOAR --num-test 20` | Once per dataset ID. `--channels nc\|ce\|both`, `--limit N` for quick tests |
| Preprocessing | `python -m src.preprocess 101` | Once per dataset/configuration. Keep `-np` low on Colab (RAM) |
| Training | `python -m src.train 101 0 --npz` | One fold per session is realistic on Colab; `-tr nnUNetTrainer_100epochs` for shorter runs |
| Inference + metrics | `python -m src.evaluate 101 -f 0` | Writes `summary.json` and `dice.csv` next to the predictions |
| Ensemble + postprocessing | `python -m src.evaluate 101 --postprocess` | Needs folds 0–4 trained with `--npz` |

All wrappers forward any extra flags to the underlying `nnUNetv2_*` command, and `python -m src.<step> -h` lists the options.

On a SLURM cluster: `sbatch --array=0-4 scripts/run_training.sh 101` trains the 5 folds in parallel.

## Where things live

| What | Where |
|------|-------|
| Code | GitHub (this repo). Colab clones it fresh each session |
| Raw SegRap scans | Drive: `Biomedical Data Design/SegRap2023_Training_Set_120cases/` (120 cases) |
| nnU-Net datasets, preprocessed data, checkpoints, predictions | Drive: `Biomedical Data Design/nnunet/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}/` |
| Your `.env` | Only in your clone/Colab session (git-ignored) |

## Data notes

- **Source layout:** one folder per case (`segrap_0000/`) containing `image.nii.gz` (non-contrast CT),
  `image_contrast.nii.gz` (contrast CT) and one binary mask per organ (`Brain.nii.gz`, …).
  An image can also be a folder of DICOM slices (`image/`). Masks must be NIfTI.
- **Overlapping masks:** SegRap structures are nested (Brain ⊃ TemporalLobe ⊃ Hippocampus, Larynx ⊃ Larynx_Glottic,
  MiddleEar ⊃ Cochlea/IAC, Eye ⊃ Lens, …), but nnU-Net needs exactly one label per voxel. `prepare_dataset.py` lets
  the **smaller structure win** where masks overlap, so a parent label means "parent minus its children".
  Merge the labels back afterwards if you need the full parent volume (e.g. Brain = Brain + BrainStem + TemporalLobes + …).
- **Labels** are numbered alphabetically by organ name, and the mapping is in each dataset's `dataset.json`.
- **Test split:** `--num-test` moves a seeded random subset to `imagesTs/labelsTs`, and the chosen case IDs are listed in `dataset.json` under `test_cases`.
  Those cases are never used for training or cross-validation.

## Shared-folder etiquette

- Only the owner of a dataset ID runs `prepare_dataset` or `preprocess` for it.
- Two people must not train the **same dataset + trainer + plans + configuration + fold** at the same time, because they would overwrite each other's checkpoints.
  Use a different fold or trainer, or write it in the team chat first.
- Never commit `.env`, data, or checkpoints (the `.gitignore` already excludes them).

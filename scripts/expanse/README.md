# Running the first nnU-Net tests on SDSC Expanse

These scripts run the first Experiment 1 tests (the 5- and 10-patient models) on Expanse's V100 GPUs.
Everything is submitted from the repo root on an Expanse login node.

## Before the first login

1. Enroll in Expanse 2FA at https://passive.sdsc.edu ("Login using Globus", then pick **ACCESS CI**,
   then "Manage 2FA"). Then log in: `ssh rmundra@login.expanse.sdsc.edu` (ACCESS password, then the 6-digit code).
2. Find the project name: `expanse-client user -r expanse` (`expanse-client resource` lists the resource names, including the GPU one).

## One-time setup (login node)

```bash
cd /expanse/lustre/projects/<project>/$USER
git clone -b nnunet-pipeline https://github.com/rayray2006/BDDProject7.git && cd BDDProject7
# edit EXPANSE_PROJECT in scripts/expanse/env.sh, then:
export EXPANSE_PROJECT=<project>
bash scripts/expanse/setup_env.sh        # Miniforge + Python 3.12 + PyTorch + requirements.txt
```

Copy the 120 raw SegRap case folders to `$WORK/SegRap2023_Training_Set_120cases/`.
For a transfer this size, use Globus (collection "SDSC HPC - Projects") rather than the login node.

## First tests, in order

| Step | Command | Cost |
|------|---------|------|
| 1. GPU check | `sbatch -A $EXPANSE_PROJECT scripts/expanse/check_gpu.sbatch` | ~0.2 GPU h |
| 2. Split + convert + preprocess the 5- and 10-patient subsets | `sbatch -A $EXPANSE_PROJECT scripts/expanse/prepare.sbatch` | CPU hours only |
| 3. Timing test (5 epochs) | `sbatch -A $EXPANSE_PROJECT --time=01:00:00 scripts/expanse/train.sbatch 110 nnUNetTrainer_5epochs` | <1 GPU h |
| 4. 5-patient model, 250 epochs | `sbatch -A $EXPANSE_PROJECT scripts/expanse/train.sbatch 110` | ~10–15 GPU h (step 3 gives the real number) |
| 5. 10-patient model, 250 epochs | `sbatch -A $EXPANSE_PROJECT scripts/expanse/train.sbatch 111` | same as step 4 |

Dataset IDs come from `splits/segrap_exp1.json` (written by step 2; commit it). With the default split,
110 = 5 patients and 111 = 10 patients for repeat 0. Check the `dataset_name` lines in the file if in doubt.

Each training job ends by predicting the 20 shared test cases and writing
`nnUNet_results/<dataset>/predictions/<model>_folds_all/dice.csv` (Dice, IoU, HD95, missed organs).

## Useful commands

```bash
squeue -u $USER                          # your queued/running jobs
tail -f logs/nnunet-train-<jobid>.out    # follow a job's output
scancel <jobid>                          # stop a job
expanse-client user -r expanse           # projects and usage (see expanse-client resource for names)
```

If a training job hits its time limit, submit the same command again: it resumes from the last checkpoint
(saved every 50 epochs).

"""nnU-Net trainers that skip nnU-Net's own end-of-training validation.

With fold "all" nnU-Net's validation set is the training set, so the prediction it runs after the
last epoch scores the model on cases it was trained on. It is meaningless for our experiments, and on
the 1024 x 1024 SegRap scans with 45 organs it takes several GPU minutes per case and runs out of
memory (job 55844359). src/evaluate.py predicts the held-out test cases instead.

nnU-Net finds these classes through the nnUNet_extTrainer environment variable, which src/config.py
points at this folder, so the installed nnunetv2 package is not modified.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_1epoch, nnUNetTrainer_5epochs, nnUNetTrainer_250epochs)


class _SkipValidation:
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("Skipping nnU-Net's own validation (fold 'all' would validate on the training "
                               "cases); src/evaluate.py scores the test cases.")


class nnUNetTrainerNoVal(_SkipValidation, nnUNetTrainer):
    """1000 epochs, as nnUNetTrainer."""


class nnUNetTrainerNoVal_1epoch(_SkipValidation, nnUNetTrainer_1epoch):
    pass


class nnUNetTrainerNoVal_5epochs(_SkipValidation, nnUNetTrainer_5epochs):
    pass


class nnUNetTrainerNoVal_250epochs(_SkipValidation, nnUNetTrainer_250epochs):
    pass

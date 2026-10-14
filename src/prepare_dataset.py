import shutil
from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np
import typer
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from src.config import RAW_DATA_DIR

app = typer.Typer()

# SegRAP2023 Challenge classes: 45 Organs-at-Risk (OARs)
OAR_NAMES = [
    "Brain", "BrainStem", "Chiasma", "Cochlea_L", "Cochlea_R", "ETube_L", "ETube_R",
    "Eye_L", "Eye_R", "IAC_L", "IAC_R", "Larynx", "Larynx_Glottic", "Larynx_Supraglot",
    "Lens_L", "Lens_R", "Mandible", "Mastoid_L", "Mastoid_R", "MiddleEar_L", "MiddleEar_R",
    "OpticNerve_L", "OpticNerve_R", "OralCavity", "Parotid_L", "Parotid_R", "PharynxConst",
    "Pituitary", "SpinalCord", "Submandibular_L", "Submandibular_R", "TemporalLobe_L",
    "TemporalLobe_R", "Thyroid", "TMJ_L", "TMJ_R", "Trachea", "TympanicCavity_L",
    "TympanicCavity_R", "VestibularSys_L", "VestibularSys_R", "Bone_Hyoid",
    "Bone_Mandible", "Bone_Cervical", "Bone_Clavicle"
]

@app.command()
def build_segrap(
    raw_source_dir: Path = typer.Option(..., help="Path to raw SegRAP download containing patient folders"),
    task: str = typer.Option("oar", help="Task target: 'oar' (Dataset501) or 'gtv' (Dataset502)"),
    max_cases: Optional[int] = typer.Option(None, help="Limit number of cases for testing"),
):
    # Phase 1 Roadmap: Split tasks to prevent high-variance tumor gradients from washing out tiny OARs.
    dataset_id = 501 if task.lower() == "oar" else 502
    dataset_name = f"Dataset{dataset_id:03d}_SegRAP2023_{task.upper()}"
    target_dir = RAW_DATA_DIR / dataset_name
    images_tr = target_dir / "imagesTr"
    labels_tr = target_dir / "labelsTr"
    
    # Ensure nnU-Net's strict folder structure exists
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Gather and sort patient directories for consistent processing
    patient_dirs = [p for p in raw_source_dir.iterdir() if p.is_dir()]
    patient_dirs = sorted(patient_dirs)

    # Allow local smoke-testing on a subset of the data (e.g., 5 subjects)
    if max_cases is not None:
        patient_dirs = patient_dirs[:max_cases]
        print(f"Testing mode enabled: limiting to {len(patient_dirs)} cases.")
    
    num_cases = 0

    for patient in patient_dirs:
        case_id = patient.name
        ncct_path = patient / "image.nii.gz"
        cect_path = patient / "image_contrast.nii.gz"

        # Skip folder if either baseline scan is missing
        if not (ncct_path.exists() and cect_path.exists()):
            continue

        # 1. Pair 2-Channel Inputs
        # ncCT provides baseline bone boundaries; ceCT highlights vascular and tumor structures.
        target_ncct = images_tr / f"{case_id}_0000.nii.gz"
        target_cect = images_tr / f"{case_id}_0001.nii.gz"
        
        # Copy to imagesTr using nnU-Net's _0000 and _0001 naming convention.
        if not target_ncct.exists():
            shutil.copy2(ncct_path, target_ncct)
        if not target_cect.exists():
            shutil.copy2(cect_path, target_cect)

        # 2. Build Mutually Exclusive Ground Truth Mask
        target_label_file = labels_tr / f"{case_id}.nii.gz"
        if not target_label_file.exists():
            # Load ncCT to borrow its real-world affine coordinates and 3D matrix shape
            ref_nii = nib.load(ncct_path)
            master_mask = np.zeros(ref_nii.shape, dtype=np.uint8)
            
            # Iteratively stack individual binary masks into the single master mask.
            # This satisfies the requirement for a multi-class softmax output layer where every voxel holds exactly one integer.
            if task.lower() == "oar":
                for idx, oar_name in enumerate(OAR_NAMES):
                    oar_file = patient / f"{oar_name}.nii.gz"
                    if oar_file.exists():
                        oar_data = nib.load(oar_file).get_fdata()
                        master_mask[oar_data > 0] = idx + 1
            else:
                for idx, gtv_name in enumerate(["GTVp", "GTVnd"]):
                    gtv_file = patient / f"{gtv_name}.nii.gz"
                    if gtv_file.exists():
                        gtv_data = nib.load(gtv_file).get_fdata()
                        master_mask[gtv_data > 0] = idx + 1

            # Save the compiled integer mask using the original scan's geometry
            out_nii = nib.Nifti1Image(master_mask, affine=ref_nii.affine, header=ref_nii.header)
            nib.save(out_nii, str(target_label_file))

        num_cases += 1

    # 3. Generate dataset.json metadata configuration
    if task.lower() == "oar":
        labels_map = {"background": 0}
        labels_map.update({name: idx + 1 for idx, name in enumerate(OAR_NAMES)})
    else:
        labels_map = {"background": 0, "GTVp": 1, "GTVnd": 2}

    # Register both channel 0 and 1 as "CT"
    # This automatically triggers nnU-Net's CT preprocessing heuristic (global foreground percentile clipping and z-score standardization) across both scans.
    generate_dataset_json(
        output_folder=str(target_dir),
        channel_names={"0": "CT", "1": "CT"},
        labels=labels_map,
        file_ending=".nii.gz",
        num_training_cases=num_cases,
    )
    print(f"Generated {dataset_name} with {num_cases} subjects.")

if __name__ == "__main__":
    app()
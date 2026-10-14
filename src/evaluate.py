import subprocess
from pathlib import Path
import typer
from src.config import os

app = typer.Typer()

@app.command()
def optimize_and_postprocess(
    dataset_id: int = typer.Option(501, help="Dataset ID: 501 or 502"),
    configuration: str = typer.Option("3d_fullres", help="Model configuration"),
):
    """Executes Phase 2: Ensembling and morphological component filtering."""
    print("Finding best ensembling configuration across folds...")
    cmd_best = [
        "nnUNetv2_find_best_configuration",
        str(dataset_id),
        "-c", configuration,
    ]
    subprocess.run(cmd_best, check=True)

    print("Determining postprocessing rules (connected component filtering)")
    cmd_post = [
        "nnUNetv2_determine_postprocessing",
        "-d", str(dataset_id),
        "-c", configuration,
    ]
    subprocess.run(cmd_post, check=True)

@app.command()
def predict(
    dataset_id: int = typer.Option(501, help="Dataset ID"),
    input_folder: Path = typer.Option(..., help="Folder with raw test cases"),
    output_folder: Path = typer.Option(..., help="Folder to save final segmentations"),
    configuration: str = typer.Option("3d_fullres", help="Model configuration"),
    folds: str = typer.Option("0,1,2,3,4", help="Ensemble across all 5 folds"),
):
    output_folder.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-d", str(dataset_id),
        "-c", configuration,
        "-device", "cuda",
        "--not_on_device",
        "-f", *folds.split(","),
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    app()
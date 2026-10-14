import subprocess
import typer
from src.config import os

app = typer.Typer()

@app.command()
def train(
    dataset_id: int = typer.Option(501, help="Dataset ID: 501 or 502"),
    configuration: str = typer.Option("3d_fullres", help="Network architecture plan"),
    fold: int = typer.Option(0, help="Fold index (0-4)"),
    trainer: str = typer.Option("nnUNetTrainer", help="Trainer class"),
    device: str = typer.Option("cuda", help="Execution target"),
):
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "-device", device,
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    app()
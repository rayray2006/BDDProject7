import subprocess
import typer
from src.config import os

app = typer.Typer()


@app.command()
def run(
    dataset_id: int = typer.Option(501, help="Dataset ID (501 for OAR, 502 for GTV)"),
    configuration: str = typer.Option("3d_fullres", help="Configuration to preprocess (e.g., 3d_fullres)"),
    num_processes: int = typer.Option(4, help="Parallel CPU workers for preprocessing (-np)"),
    num_processes_fp: int = typer.Option(
        None, 
        help="Workers for fingerprint extraction (-npfp). Defaults to num_processes if not set."
    ),
    verify_dataset_integrity: bool = typer.Option(False, help="Verify data integrity before planning"),
    clean: bool = typer.Option(False, help="Overwrite existing preprocessed files if re-running"),
):
    # If a distinct fingerprint worker count isn't specified, match num_processes
    npfp = num_processes_fp if num_processes_fp is not None else num_processes

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", configuration,
        "-np", str(num_processes),
        "-npfp", str(npfp),
    ]

    if verify_dataset_integrity:
        cmd.append("--verify_dataset_integrity")

    if clean:
        cmd.append("--clean")

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Detect Colab runtime vs local machine
IS_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

if IS_COLAB:
    # Adjust this to your team's mounted Google Drive folder path
    DATA_ROOT = Path("/content/drive/MyDrive/YourSharedProjectFolder")
else:
    # Local fallback to tests/dummy_data or a path in .env
    DATA_ROOT = Path(
        os.getenv("LOCAL_DATA_DIR", Path(__file__).resolve().parent.parent / "tests" / "dummy_data")
    )

# nnU-Net v2 required directories
RAW_DATA_DIR = Path(os.getenv("nnUNet_raw", DATA_ROOT / "nnUNet_raw"))
PREPROCESSED_DIR = Path(os.getenv("nnUNet_preprocessed", DATA_ROOT / "nnUNet_preprocessed"))
RESULTS_DIR = Path(os.getenv("nnUNet_results", DATA_ROOT / "nnUNet_results"))

# Export to environment variables so nnU-Net CLI tools pick them up automatically
os.environ["nnUNet_raw"] = str(RAW_DATA_DIR)
os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)
os.environ["nnUNet_results"] = str(RESULTS_DIR)

# Ensure directories exist
for folder in [RAW_DATA_DIR, PREPROCESSED_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
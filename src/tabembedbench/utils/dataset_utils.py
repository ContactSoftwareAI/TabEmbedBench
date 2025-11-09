import zipfile
from pathlib import Path

import requests

ADBENCH_URL = "https://github.com/Minqi824/ADBench/archive/refs/heads/main.zip"


def download_adbench_tabular_datasets(
    save_path: str | Path | None = None,
) -> None:
    """Downloads tabular datasets for ADBench from the specified GitHub repository and saves them to the
    specified path. If no path is provided, it defaults to './data/adbench_tabular_datasets'. If the
    directory does not exist, it is created.

    Args:
        save_path (Optional[str]): The directory where the ADBench tabular datasets should be saved. If
            None, the default path './data/adbench_tabular_datasets' will be used.
    """
    save_path = save_path or "./data/adbench_tabular_datasets"
    save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    # Download the repository as a zip file
    print("Downloading ADBench repository...")
    response = requests.get(ADBENCH_URL, stream=True)
    response.raise_for_status()

    # Save zip file temporarily
    zip_path = save_path / "adbench_temp.zip"
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract only the Classical datasets
    print("Extracting datasets...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get all files in the Classical datasets directory
        classical_files = [
            f
            for f in zip_ref.namelist()
            if f.startswith("ADBench-main/adbench/datasets/Classical/")
        ]

        for file_path in classical_files:
            if file_path.endswith("/"):  # Skip directories
                continue

            # Extract relative path after Classical/
            relative_path = file_path.split("Classical/", 1)[1]
            target_path = save_path / relative_path

            # Create directory if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract file
            with zip_ref.open(file_path) as source, open(target_path, "wb") as target:
                target.write(source.read())

    # Clean up temporary zip file
    zip_path.unlink()
    print(f"ADBench tabular datasets downloaded to: {save_path}")

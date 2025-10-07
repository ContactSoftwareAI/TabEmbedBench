import math
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch

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


def get_data_description(
    X: np.ndarray, y: np.ndarray, dataset_name: str
) -> dict[str, str | int | float]:
    """Provides a summary of the dataset by computing statistical information
    such as the number of samples, features, anomalies, and the anomaly ratio.

    Args:
        X (np.ndarray): The input features data with shape (n_samples, n_features).
        y (np.ndarray): Corresponding target labels with shape (n_samples,) where
            anomalies are marked (e.g., 1 for anomalies, 0 otherwise).

    Returns:
        dict: A dictionary containing the data description with the following keys:
            - "Samples": Number of samples in the dataset.
            - "Features": Number of features in the dataset.
            - "Anomalies": Total count of anomalies in the dataset.
            - "Anomaly Ratio (%)": Percentage of anomalies in the dataset.
    """
    des_dict = {}
    des_dict["dataset"] = dataset_name
    des_dict["samples"] = X.shape[0]
    des_dict["features"] = X.shape[1]
    des_dict["anomalies"] = sum(y)
    des_dict["anomaly Ratio (%)"] = round(sum(y) / len(y) * 100, 2)

    return des_dict


def read_data(data_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)

    X = data["X"]
    y = data["y"]

    return X, y


def select_random_combined_datasets(datasets_dir: str) -> list[str | Path]:
    """Selects a random subset of dataset files from a given directory.

    This function identifies all files within the provided directory and randomly selects
    a subset of these files. The number of files selected is between 2 and the square root
    of the total number of available files in the directory. The selected files are returned
    as a list.

    Args:
        datasets_dir (str): The directory containing dataset files from which to select.

    Returns:
        list[Union[str, Path]]: A list of randomly selected dataset files.
    """
    files = list(datasets_dir.glob("*.npz"))
    datasets = set([dataset for dataset in files if dataset.is_file()])

    num_selected_datasets = random.randint(2, int(math.sqrt(len(datasets))))

    random_datasets = random.sample(datasets, num_selected_datasets)

    return random_datasets


def prepare_data_for_torch(X: np.ndarray, device: str = "cpu") -> torch.Tensor:
    X = torch.from_numpy(X).to(device)
    return X


def check_tabpfn_dataset_restrictions(X: np.ndarray) -> bool:
    raise NotImplementedError


def check_tabicl_dataset_restrictions(X: np.ndarray) -> bool:
    raise NotImplementedError

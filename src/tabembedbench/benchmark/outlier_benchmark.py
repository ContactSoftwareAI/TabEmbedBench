"""Simplified outlier detection benchmark using ADBench datasets."""

from datetime import datetime
from pathlib import Path
from typing import Iterator, Callable

import zipfile

import requests

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

TASK_TYPE = "Outlier Detection"

NPZ_FILE_PATTERN = "*.npz"

ADBENCH_CLASSICAL_DATASETS = [
    "10_cover.npz",
    "11_donors.npz",
    "12_fault.npz",
    "13_fraud.npz",
    "14_glass.npz",
    "15_Hepatitis.npz",
    "16_http.npz",
    "17_InternetAds.npz",
    "18_Ionosphere.npz",
    "19_landsat.npz",
    "1_ALOI.npz",
    "20_letter.npz",
    "21_Lymphography.npz",
    "22_magic.gamma.npz",
    "23_mammography.npz",
    "24_mnist.npz",
    "25_musk.npz",
    "26_optdigits.npz",
    "27_PageBlocks.npz",
    "28_pendigits.npz",
    "29_Pima.npz",
    "2_annthyroid.npz",
    "30_satellite.npz",
    "31_satimage-2.npz",
    "32_shuttle.npz",
    "33_skin.npz",
    "34_smtp.npz",
    "35_SpamBase.npz",
    "36_speech.npz",
    "37_Stamps.npz",
    "38_thyroid.npz",
    "39_vertebral.npz",
    "3_backdoor.npz",
    "40_vowels.npz",
    "41_Waveform.npz",
    "42_WBC.npz",
    "43_WDBC.npz",
    "44_Wilt.npz",
    "45_wine.npz",
    "46_WPBC.npz",
    "47_yeast.npz",
    "4_breastw.npz",
    "5_campaign.npz",
    "6_cardio.npz",
    "7_Cardiotocography.npz",
    "8_celeba.npz",
    "9_census.npz",
]

EXPECTED_DATASET_COUNT = len(ADBENCH_CLASSICAL_DATASETS)

IMAGE_CATEGORY = [
    "1_ALOI.npz",
    "8_celeba.npz",
    "17_InternetAds.npz",
    "20_letter.npz",
    "24_mnist.npz",
    "26_optdigits.npz",
    "28_pendigits.npz",
    "33_skin.npz",
]

ADBENCH_URL = "https://github.com/Minqi824/ADBench/archive/refs/heads/main.zip"


class OutlierBenchmark(AbstractBenchmark):
    """Simplified benchmark for outlier detection using ADBench tabular datasets.

    This benchmark evaluates embedding models on outlier detection tasks
    using datasets from the ADBench benchmark suite.
    """

    def __init__(
        self,
        dataset_paths: str | Path | None = None,
        exclude_datasets: list[str] | None = None,
        exclude_image_datasets: bool = True,
        result_dir: str | Path = "result_outlier",
        timestamp: str = TIMESTAMP,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 10000,
        upper_bound_num_features: int = 500,
    ):
        """Initialize the outlier detection benchmark.

        Args:
            dataset_paths: Path to the dataset directory. If None, uses default.
            exclude_datasets: List of dataset filenames to exclude.
            exclude_image_datasets: Whether to exclude image datasets.
            result_dir: Directory for saving results.
            timestamp: Timestamp string for result file naming.
            save_result_dataframe: Whether to save results to disk.
            upper_bound_num_samples: Maximum dataset size to process.
            upper_bound_num_features: Maximum number of features to process.
        """
        super().__init__(
            name="TabEmbedBench_Outlier",
            task_type=TASK_TYPE,
            result_dir=result_dir,
            timestamp=timestamp,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )

        # Handle dataset paths
        if dataset_paths is None:
            dataset_paths = Path("data/adbench_tabular_datasets")
        else:
            dataset_paths = Path(dataset_paths)

        if (
            not dataset_paths.exists()
            or len(list(dataset_paths.glob(NPZ_FILE_PATTERN))) < EXPECTED_DATASET_COUNT
        ):
            found_files = [
                file_path.stem
                for file_path in dataset_paths.glob(NPZ_FILE_PATTERN)
                if file_path.stem in ADBENCH_CLASSICAL_DATASETS
            ]
            if len(found_files) == 0:
                missing_files = ADBENCH_CLASSICAL_DATASETS
            else:
                missing_files = set(ADBENCH_CLASSICAL_DATASETS) - set(found_files)

            self.logger.warning(
                f"Dataset directory is missing or incomplete. "
                f"Downloading ADBench tabular datasets (expecting {EXPECTED_DATASET_COUNT} files)..."
            )
            self.download_adbench_tabular_datasets(
                save_path=dataset_paths,
                files_to_download=missing_files,
            )

        self.dataset_paths = dataset_paths

        # Handle exclusions
        self.exclude_datasets = exclude_datasets or []
        if exclude_image_datasets:
            self.exclude_datasets.extend(IMAGE_CATEGORY)

    def _load_datasets(self, **kwargs) -> list:
        """Load ADBench datasets from the specified directory.

        Returns:
            List of dataset file paths.
        """
        return list(self.dataset_paths.glob("*.npz"))

    def _should_skip_dataset(
        self, dataset_file: Path, **kwargs
    ) -> bool:
        """Check if a dataset should be skipped.

        Args:
            dataset_file: Path to the dataset file.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (should_skip, reason).
        """
        # Load dataset to check size constraints
        with np.load(dataset_file) as dataset:
            num_samples = dataset["X"].shape[0]
            num_features = dataset["X"].shape[1]
            dataset_name = dataset_file.stem

        skip_reasons = []

        skip_reasons.extend(self._check_dataset_size_constraints(num_samples, num_features, dataset_name))

        # Check if in exclusion list
        if dataset_file.name in self.exclude_datasets:
            skip_reasons.append(f"Excluded by user")

        if skip_reasons:
            reason = " | ".join(skip_reasons)
            self.logger.warning(f"Skipping dataset {dataset_name}: {reason}")
            return True

        self.logger.info(
            f"Starting experiments for dataset {dataset_name} "
            f"and task: Outlier Detection."
            f"Samples: {num_samples}, Features: {num_features}"
        )

        return False

    def _prepare_dataset(self, dataset_file: Path, **kwargs) -> Iterator[dict]:
        """Prepare data from an ADBench dataset file.

        For outlier detection, we yield a single dict with the full dataset.

        Args:
            dataset_file: Path to the dataset file.
            **kwargs: Additional parameters (unused).

        Yields:
            Dictionary containing prepared data in standardized format.
        """
        dataset = np.load(dataset_file)
        X = dataset["X"]
        y = dataset["y"]

        dataset_name = dataset_file.stem
        num_samples = X.shape[0]
        num_features = X.shape[1]
        outlier_ratio = y.sum() / y.shape[0]

        # Yield single split for outlier detection
        yield {
            "X_train": X,
            "y_eval": y,
            "task_type": TASK_TYPE,
            "dataset_metadata": {
                "dataset_name": dataset_name,
                "num_samples": num_samples,
                "num_features": num_features,
                "outlier_ratio": outlier_ratio,
            },
            "feature_metadata": {
                "outlier_ratio": outlier_ratio,
            },
        }

    def _get_default_metrics(self) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        """
        Retrieves a dictionary of default metrics for evaluation purposes.

        The method provides a mapping between metric names and their corresponding
        evaluation functions. Each metric function is expected to be callable and
        accept arguments appropriate for the respective metric.

        Returns:
            dict: A dictionary where keys are metric categories, and values are
            dictionaries containing metric names mapped to their evaluation
            functions.
        """
        return {
            "Outlier Detection": {
                "auc_score": roc_auc_score,
            }
        }

    def _get_evaluator_prediction(
        self,
        embeddings: tuple,
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        """Evaluate embeddings for outlier detection.

        Args:
            embeddings: Tuple of (embeddings, test_embeddings, compute_time).
            evaluator: The evaluator to use.
            dataset_configurations: Dictionary with data and metadata.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings, _, _ = embeddings

        # Get prediction from evaluator
        prediction, _ = evaluator.get_prediction(train_embeddings)

        return prediction

    def _get_benchmark_name(self) -> str:
        """Get the benchmark name for result saving.

        Returns:
            String identifier for the benchmark.
        """
        return "ADBench_Tabular"

    def download_adbench_tabular_datasets(
        self,
        save_path: str | Path | None = None,
        files_to_download: list[str] = None,
    ) -> None:
        """
        Downloads and extracts tabular datasets from the ADBench repository.

        This function downloads a zip file containing the datasets from the ADBench
        repository, extracts only the required Classical datasets, and saves them
        to the specified directory.

        Args:
            save_path (str | Path | None): The directory where the datasets will
                be saved. If not provided, defaults to "./data/adbench_tabular_datasets".
            files_to_download (list[str] | None): List of filenames to be downloaded.
                Only files specified in this list are extracted from the zip.
        """

        save_path = save_path or "./data/adbench_tabular_datasets"
        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Download the repository as a zip file
        self.logger.info("Downloading ADBench repository...")
        response = requests.get(ADBENCH_URL, stream=True)
        response.raise_for_status()

        # Save zip file temporarily
        zip_path = save_path / "adbench_temp.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract only the Classical datasets
        self.logger.info("Extracting datasets...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get all files in the Classical datasets directory
            classical_files = [
                f
                for f in zip_ref.namelist()
                if f.startswith("ADBench-main/adbench/datasets/Classical/")
                and f.endswith(".npz")
            ]
            for file_path in classical_files:
                if Path(file_path).name not in files_to_download:
                    continue

                # Extract relative path after Classical/
                relative_path = file_path.split("Classical/", 1)[1]
                target_path = save_path / relative_path

                # Create directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                with (
                    zip_ref.open(file_path) as source,
                    open(target_path, "wb") as target,
                ):
                    target.write(source.read())

        # Clean up temporary zip file
        zip_path.unlink()
        print(f"ADBench tabular datasets downloaded to: {save_path}")


def run_outlier_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    dataset_paths: str | Path | None = None,
    exclude_datasets: list[str] | None = None,
    exclude_image_datasets: bool = True,
    upper_bound_num_samples: int = 10000,
    upper_bound_num_features: int = 500,
    save_result_dataframe: bool = True,
    result_dir: str | Path = "result_outlier",
    timestamp: str = TIMESTAMP,
) -> pl.DataFrame:
    """Run outlier detection benchmark using the provided embedding models.

    This function benchmarks the effectiveness of various embedding models in
    detecting outliers using the tabular datasets from the ADBench benchmark [1].

    Args:
        embedding_models: List of embedding models to be evaluated.
        evaluators: List of evaluator algorithms.
        dataset_paths: Optional path to the dataset directory. If not specified,
            uses default directory and downloads datasets if missing.
        exclude_datasets: Optional list of dataset filenames to exclude.
        exclude_image_datasets: Whether to exclude image datasets. Defaults to True.
        upper_bound_num_samples: Maximum dataset size to include. Defaults to 10000.
        upper_bound_num_features: Maximum number of features to include. Defaults to 500.
        save_result_dataframe: Whether to save results to disk. Defaults to True.
        result_dir: Directory where results should be saved. Defaults to "result_outlier".
        timestamp: Timestamp string for saving results. Defaults to current timestamp.

    Returns:
        pl.DataFrame: Polars DataFrame containing the benchmark results.

    References:
        [1] Han, S., et al. (2022). "Adbench: Anomaly detection benchmark."
            Advances in neural information processing systems, 35, 32142-32159.
    """
    benchmark = OutlierBenchmark(
        dataset_paths=dataset_paths,
        exclude_datasets=exclude_datasets,
        exclude_image_datasets=exclude_image_datasets,
        result_dir=result_dir,
        timestamp=timestamp,
        save_result_dataframe=save_result_dataframe,
        upper_bound_num_samples=upper_bound_num_samples,
        upper_bound_num_features=upper_bound_num_features,
    )

    return benchmark.run_benchmark(embedding_models, evaluators)

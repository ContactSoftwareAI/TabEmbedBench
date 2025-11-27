"""Simplified outlier detection benchmark using ADBench datasets."""

from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.dataset_utils import download_adbench_tabular_datasets

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            logger_name="TabEmbedBench_Outlier",
            task_type="Outlier Detection",
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
        if not dataset_paths.exists():
            self.logger.warning("Downloading ADBench tabular datasets...")
            download_adbench_tabular_datasets(dataset_paths)
        self.path = dataset_paths.resolve()
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
    ) -> tuple[bool, str | None]:
        """Check if a dataset should be skipped.

        Args:
            dataset_file: Path to the dataset file.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (should_skip, reason).
        """
        # Check if in exclusion list
        if dataset_file.name in self.exclude_datasets:
            return True, f"Dataset {dataset_file.name} is in exclusion list"

        # Load dataset to check size constraints
        with np.load(dataset_file) as dataset:
            num_samples = dataset["X"].shape[0]
            num_features = dataset["X"].shape[1]
            dataset_name = dataset_file.stem

        # Check size constraints
        should_skip, reason = self._check_dataset_size_constraints(
            num_samples, num_features, dataset_name
        )

        if not should_skip:
            self.logger.info(
                f"Running experiments on {dataset_name}. "
                f"Samples: {num_samples}, Features: {num_features}"
            )

        return should_skip, reason

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
            "X": X,
            "X_train": None,
            "X_test": None,
            "y": y,
            "y_train": None,
            "y_test": None,
            "dataset_name": dataset_name,
            "dataset_size": num_samples,
            "num_features": num_features,
            "metadata": {
                "outlier_ratio": outlier_ratio,
            },
        }

    def _evaluate(
        self,
        embeddings: tuple,
        evaluator: AbstractEvaluator,
        data_split: dict,
    ) -> dict:
        """Evaluate embeddings for outlier detection.

        Args:
            embeddings: Tuple of (embeddings, test_embeddings, compute_time).
            evaluator: The evaluator to use.
            data_split: Dictionary with data and metadata.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings, test_embeddings, compute_time = embeddings
        y = data_split["y"]
        outlier_ratio = data_split["metadata"]["outlier_ratio"]

        # Get prediction from evaluator
        prediction, _ = evaluator.get_prediction(train_embeddings)

        # Compute AUC score
        score_auc = roc_auc_score(y, prediction)

        # Build result dictionary with all metadata
        result_dict = {
            "dataset_name": [data_split["dataset_name"]],
            "dataset_size": [data_split["dataset_size"]],
            "num_features": [data_split["num_features"]],
            "embed_dim": [train_embeddings.shape[-1]],
            "time_to_compute_embedding": [compute_time],
            "algorithm": [evaluator._name],
            "auc_score": [score_auc],
            "outlier_ratio": [outlier_ratio],
            "task": ["Outlier Detection"],
        }

        # Add evaluator parameters
        evaluator_params = evaluator.get_parameters()
        for key, value in evaluator_params.items():
            result_dict[f"algorithm_{key}"] = [value]

        return result_dict

    def _get_benchmark_name(self) -> str:
        """Get the benchmark name for result saving.

        Returns:
            String identifier for the benchmark.
        """
        return "ADBench_Tabular"


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

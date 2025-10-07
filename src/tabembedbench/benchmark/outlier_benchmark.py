from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
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
    """Benchmark for outlier detection using ADBench tabular datasets.

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
            result_dir=result_dir,
            timestamp=timestamp,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )

        # Handle dataset paths
        if dataset_paths is None:
            dataset_paths = Path("data/adbench_tabular_datasets")
            if not dataset_paths.exists():
                self.logger.warning("Downloading ADBench tabular datasets...")
                download_adbench_tabular_datasets(dataset_paths)
        else:
            dataset_paths = Path(dataset_paths)

        self.dataset_paths = dataset_paths

        # Handle exclusions
        self.exclude_datasets = exclude_datasets or []
        if exclude_image_datasets:
            self.exclude_datasets.extend(IMAGE_CATEGORY)

    def _load_datasets(self, **kwargs):
        """Load ADBench datasets from the specified directory.

        Returns:
            List of dataset file paths.
        """
        return list(self.dataset_paths.glob("*.npz"))

    def _should_skip_dataset(self, dataset_file, **kwargs) -> tuple[bool, str | None]:
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

    def _prepare_data(self, dataset_file, **kwargs):
        """Prepare data from an ADBench dataset file.

        Args:
            dataset_file: Path to the dataset file.
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary containing prepared data and metadata.
        """
        dataset = np.load(dataset_file)
        X = dataset["X"]
        y = dataset["y"]

        dataset_name = dataset_file.stem
        num_samples = X.shape[0]
        num_features = X.shape[1]
        outlier_ratio = y.sum() / y.shape[0]

        return {
            "data": X,
            "labels": y,
            "dataset_name": dataset_name,
            "dataset_size": num_samples,
            "num_features": num_features,
            "outlier_ratio": outlier_ratio,
            "embedding_kwargs": {"outlier": True},
            "eval_kwargs": {"y": y, "outlier_ratio": outlier_ratio},
        }

    def _evaluate_embeddings(
        self,
        embedding_results,
        evaluator: AbstractEvaluator,
        dataset_info: dict,
        **kwargs,
    ) -> dict:
        """Evaluate embeddings for outlier detection.

        Args:
            embedding_results: Tuple of (embeddings, compute_time, test_embeddings, test_compute_time).
            evaluator: The evaluator to use.
            dataset_info: Dictionary with dataset metadata.
            **kwargs: Additional parameters including 'y' (labels) and 'outlier_ratio'.

        Returns:
            Dictionary containing evaluation results.
        """
        embeddings = embedding_results[0]
        y = kwargs.get("y")
        outlier_ratio = kwargs.get("outlier_ratio")

        # Get prediction from evaluator
        prediction, _ = evaluator.get_prediction(embeddings)

        # Compute AUC score
        score_auc = roc_auc_score(y, prediction)

        # Build result dictionary
        result_dict = {
            **dataset_info,
            "auc_score": [score_auc],
            "outlier_ratio": [outlier_ratio],
            "task": ["Outlier Detection"],
        }

        return result_dict

    def _get_benchmark_name(self) -> str:
        """Get the benchmark name for result saving.

        Returns:
            String identifier for the benchmark.
        """
        return "ADBench_Tabular"

    def _is_evaluator_compatible(self, evaluator: AbstractEvaluator, **kwargs) -> bool:
        """Check if evaluator is compatible with outlier detection.

        Args:
            evaluator: The evaluator to check.
            **kwargs: Additional parameters (unused).

        Returns:
            True if evaluator supports outlier detection.
        """
        return evaluator.task_type == "Outlier Detection"


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
    """Runs an outlier detection benchmark using the provided embedding models
    and datasets. It uses the tabular datasets from the ADBench benchmark [1]
    for evaluation.

    This function benchmarks the effectiveness of various embedding models in
    detecting outliers. It supports the exclusion of specific datasets,
    exclusion of image datasets, limiting the dataset size, and optionally
    saving computed embeddings for analysis.

    Args:
        embedding_models: A list of embedding models to be evaluated. Each
            embedding model must implement methods for preprocessing data,
            computing embeddings, and resetting the model.
        evaluators: A list of algorithm.
        dataset_paths: Optional path to the dataset directory. If not specified,
            a default directory for tabular datasets will be used,
            and datasets will be downloaded if missing.
        exclude_datasets: Optional list of dataset filenames to exclude from the
            benchmark. Each filename should match a file in the dataset directory.
        exclude_image_datasets: Boolean flag that indicates whether to exclude
            image datasets from the benchmark. Defaults to False.
        upper_bound_num_samples: Integer specifying the maximum size of rows
            (in number of samples) to include in the benchmark. Datasets exceeding
            this size will be skipped. Defaults to 10000.
        upper_bound_num_features: Integer specifying the maximum number of features
            to include in the benchmark. Datasets with more features than this
            value will be skipped. Defaults to 500.
        save_result_dataframe: Boolean flag to determine whether to save the result
            dataframe to disk. Defaults to True.
        result_dir: Optional path to the directory where the result dataframe should
            be saved. Defaults to "result_outlier".
        timestamp: Optional timestamp string to use for saving the result dataframe.
            Defaults to the current timestamp.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the benchmark results, including
            dataset names, dataset sizes, embedding model names, number of neighbors
            used for outlier detection, AUC scores, computation times for embeddings,
            and the benchmark category.

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

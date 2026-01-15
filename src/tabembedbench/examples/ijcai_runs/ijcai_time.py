"""IJCAI benchmark runner for evaluating tabular embedding models.

This module provides functionality to run comprehensive benchmarks on various
tabular embedding models using multiple evaluation algorithms across different
datasets. It supports classification, regression, and outlier detection tasks.
"""

import logging

from tabembedbench.benchmark.run_benchmark import (
    BenchmarkConfig,
    DatasetConfig,
    run_benchmark,
)
from tabembedbench.embedding_models import (
    SphereBasedEmbedding,
    TableVectorizerEmbedding,
    TabPFNEmbedding,
    TabPFNEmbeddingClusterLabels,
    TabPFNEmbeddingConstantVector,
    TabPFNEmbeddingRandomVector,
    TabStarEmbedding,
)
from tabembedbench.embedding_models.tabicl_embedding import (
    TabICLEmbedding,
    TabICLWrapper,
)

logger = logging.getLogger("IJCAI_Run_Benchmark")

DEBUG = False
GOOGLE_BUCKET = "bucket_tabdata"
GCS_DIR = "ijcai"
DATA_DIR = "embedding_times"

DATASETCONFIG = DatasetConfig(
    adbench_dataset_path="data/adbench_tabular_datasets",
    exclude_adbench_datasets=[],
    upper_bound_dataset_size=15000,
    upper_bound_num_features=500,
)


BENCHMARK_CONFIG = BenchmarkConfig(
    run_outlier=True,
    run_tabarena=True,
    run_dataset_separation=False,
    run_dataset_tabpfn_separation=False,
    data_dir=DATA_DIR,
    dataset_separation_configurations_json_path="dataset_separation_tabarena.json",
    dataset_separation_configurations_tabpfn_subset_json_path="dataset_separation_tabarena_tabpfn_subset.json",
    gcs_bucket=GOOGLE_BUCKET,
    gcs_filepath=GCS_DIR,
)

NUM_ESTIMATORS = 5


def get_embedding_models(debug=False):
    """
    Gets a list of embedding models and optional embedding clustering models.

    This function initializes and returns a list of embedding models.

    Args:
        debug (bool, optional): If True, returns a test embedding model. Defaults to False.

    Returns:
        list: List of embedding models
    """
    if debug:
        return [
            # SphereBasedEmbedding(embed_dim=16),
            TabICLWrapper(),
            TabICLEmbedding(),
        ]

    sphere_model_64 = SphereBasedEmbedding(embed_dim=64)
    sphere_model_192 = SphereBasedEmbedding(embed_dim=192)
    sphere_model_256 = SphereBasedEmbedding(embed_dim=256)
    sphere_model_512 = SphereBasedEmbedding(embed_dim=512)
    tabicl_embedding = TabICLEmbedding()
    tablevectorizer = TableVectorizerEmbedding()
    tabpfn = TabPFNEmbedding(num_estimators=NUM_ESTIMATORS)
    tabpfn_random = TabPFNEmbeddingRandomVector(num_estimators=NUM_ESTIMATORS)
    tabpfn_constant = TabPFNEmbeddingConstantVector(num_estimators=NUM_ESTIMATORS)
    tabStar_embedding = TabStarEmbedding()

    embedding_models = [
        sphere_model_64,
        sphere_model_192,
        sphere_model_256,
        sphere_model_512,
        tabicl_embedding,
        tablevectorizer,
    ]

    embedding_models.extend(
        [
            tabpfn,
            tabpfn_random,
            tabpfn_constant,
            tabStar_embedding,
        ]
    )

    return embedding_models


def get_evaluators(debug=False):
    """Configure and initialize evaluation algorithms for the benchmark.

    Creates a comprehensive set of evaluators including K-Nearest Neighbors
    (KNN) and neural network evaluators (MLP) for classification and regression,
    and outlier detection algorithms (DeepSVDD, Isolation Forest, Local Outlier Factor).

    Args:
        debug (bool, optional): If True, uses a minimal set of evaluators
            for faster debugging. If False, creates a full grid of evaluator
            configurations with various hyperparameters. Defaults to False.

    Returns:
        list: A list of configured evaluator instances ready for benchmarking.
    """
    return []


def main(debug=False):
    embedding_models = get_embedding_models(debug=debug)

    evaluators = get_evaluators(debug=debug)

    logger.info(f"Using {len(embedding_models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    dataset_config = DATASETCONFIG

    benchmark_config = BENCHMARK_CONFIG

    (
        result_outlier,
        result_tabarena,
        result_dataset_separation_tabpfn,
        result_dataset_separation,
        result_dir,
    ) = run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )


if __name__ == "__main__":
    main(debug=DEBUG)

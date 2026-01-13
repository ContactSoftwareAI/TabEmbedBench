"""IJCAI benchmark runner for evaluating tabular embedding models.

This module provides functionality to run comprehensive benchmarks on various
tabular embedding models using multiple evaluation algorithms across different
datasets. It supports classification, regression, and outlier detection tasks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import click

from tabembedbench.benchmark.run_benchmark import (
    BenchmarkConfig,
    DatasetConfig,
    run_benchmark,
)
from tabembedbench.embedding_models import (
    SphereBasedEmbedding,
    TabICLEmbedding,
    TableVectorizerEmbedding,
    TabPFNEmbedding,
    TabPFNEmbeddingConstantVector,
)
from tabembedbench.evaluators import (
    KNNClassifierEvaluatorHPO,
    LogisticRegressionHPOEvaluator,
    SVMClassifierEvaluator,
)
from tabembedbench.evaluators.classification import (
    KNNClassifierEvaluator,
    MLPClassifierEvaluator,
)
from tabembedbench.evaluators.outlier import (
    DeepSVDDEvaluator,
    IsolationForestEvaluator,
    LocalOutlierFactorEvaluator,
)
from tabembedbench.evaluators.regression import (
    KNNRegressorEvaluator,
    MLPRegressorEvaluator,
)

logger = logging.getLogger("IJCAI_Run_Benchmark")

DEBUG = False
GOOGLE_BUCKET = "bucket_tabdata"
GCS_DIR = "ijcai"
DATA_DIR = "sphere_based_model"


EXCLUDE_TABARENA_DATASETS = [
    "airfoil_self_noise",
    "anneal",
    "Another-Dataset-on-used-Fiat-500",
    "Bank_Customer_Churn",
    "blood-transfusion-service-center",
    "churn",
    "coil2000_insurance_policies",
    "concrete_compressive_strength",
    "credit-g",
    "diabetes",
    "E-CommereShippingData",
    "Fitness_Club",
    "hazelnut-spread-contaminant-detection",
    "healthcare_insurance_expenses",
    "heloc",
    "in_vehicle_coupon_recommendation",
]


DATASETCONFIG = DatasetConfig(
    adbench_dataset_path="data/adbench_tabular_datasets",
    exclude_adbench_datasets=[],
    exclude_tabarena_datasets=EXCLUDE_TABARENA_DATASETS,
    upper_bound_dataset_size=15000,
    upper_bound_num_features=500,
)


BENCHMARK_CONFIG = BenchmarkConfig(
    run_outlier=False,
    run_tabarena=True,
    run_dataset_separation=False,
    run_dataset_tabpfn_separation=False,
    data_dir=DATA_DIR,
    dataset_separation_configurations_json_path="dataset_separation_tabarena.json",
    dataset_separation_configurations_tabpfn_subset_json_path="dataset_separation_tabarena_tabpfn_subset.json",
    gcs_bucket=GOOGLE_BUCKET,
    gcs_filepath=GCS_DIR,
)


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
            SphereBasedEmbedding(embed_dim=16),
        ]

    sphere_model_64 = SphereBasedEmbedding(embed_dim=64)
    sphere_model_192 = SphereBasedEmbedding(embed_dim=192)
    sphere_model_256 = SphereBasedEmbedding(embed_dim=256)
    sphere_model_512 = SphereBasedEmbedding(embed_dim=512)

    embedding_models = [
        sphere_model_64,
        sphere_model_192,
        sphere_model_256,
        sphere_model_512,
    ]

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
    evaluator_algorithms = []

    if debug:
        evaluator_algorithms.extend(
            [
                KNNRegressorEvaluator(
                    num_neighbors=5, weights="distance", metric="euclidean"
                ),
                KNNClassifierEvaluator(
                    num_neighbors=5, weights="distance", metric="euclidean"
                ),
                LocalOutlierFactorEvaluator(
                    model_params={
                        "n_neighbors": 5,
                    }
                ),
            ]
        )
        return evaluator_algorithms

    for num_neighbors in range(5, 50, 5):
        for metric in [
            "euclidean",
        ]:
            for weights in [
                "distance",
            ]:
                evaluator_algorithms.extend(
                    [
                        KNNRegressorEvaluator(
                            num_neighbors=num_neighbors, weights=weights, metric=metric
                        ),
                        KNNClassifierEvaluator(
                            num_neighbors=num_neighbors, weights=weights, metric=metric
                        ),
                    ]
                )
            evaluator_algorithms.append(
                LocalOutlierFactorEvaluator(
                    model_params={
                        "n_neighbors": num_neighbors,
                        "metric": metric,
                    }
                )
            )
    for num_estimators in range(50, 300, 50):
        evaluator_algorithms.append(
            IsolationForestEvaluator(
                model_params={
                    "n_estimators": num_estimators,
                }
            )
        )

    deep_svdd_dynamic = DeepSVDDEvaluator(dynamic_hidden_neurons=True)
    deep_svdd_dynamic._name = "DeepSVDD-dynamic"

    evaluator_algorithms.extend(
        [
            LogisticRegressionHPOEvaluator(),
            MLPRegressorEvaluator(),
            MLPClassifierEvaluator(),
            deep_svdd_dynamic,
            DeepSVDDEvaluator(),
        ]
    )

    return evaluator_algorithms


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

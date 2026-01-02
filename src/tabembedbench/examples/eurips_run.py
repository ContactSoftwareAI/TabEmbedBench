"""EuRIPS benchmark runner for evaluating tabular embedding models.

This module provides functionality to run comprehensive benchmarks on various
tabular embedding models using multiple evaluation algorithms across different
datasets. It supports classification, regression, and outlier detection tasks.
"""

import logging

import click

from tabembedbench.benchmark.run_benchmark import (
    BenchmarkConfig,
    DatasetConfig,
    run_benchmark,
)
from tabembedbench.embedding_models import (
    TabICLEmbedding,
    TableVectorizerEmbedding,
    TabPFNEmbedding,
    TabStarEmbedding,
    ConTextTabEmbedding,
)
from tabembedbench.evaluators.knn_classifier import KNNClassifierEvaluator
from tabembedbench.evaluators.knn_regressor import KNNRegressorEvaluator
from tabembedbench.evaluators.mlp_classifier import MLPClassifierEvaluator
from tabembedbench.evaluators.mlp_regressor import MLPRegressorEvaluator
from tabembedbench.evaluators.outlier import (
    DeepSVDDEvaluator,
    IsolationForestEvaluator,
    LocalOutlierFactorEvaluator,
)
from tabembedbench.utils.eda_utils import (
    create_outlier_plots,
    create_tabarena_plots,
)

logger = logging.getLogger("EuRIPS_Run_Benchmark")


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
        return [TableVectorizerEmbedding()]

    tabicl_row_embedder = TabICLEmbedding()

    tablevector = TableVectorizerEmbedding()

    # tabpfn_embedder = TabPFNEmbedding(
    #     num_estimators=5,
    # )

    tabstar_embedder = TabStarEmbedding()

    context_tab_embedder = ConTextTabEmbedding()

    embedding_models = [
        # tabicl_row_embedder,
        # tabpfn_embedder,
        # tabstar_embedder,
        # tablevector,
        context_tab_embedder,
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
            MLPRegressorEvaluator(),
            MLPClassifierEvaluator(),
            deep_svdd_dynamic,
            DeepSVDDEvaluator(),
        ]
    )

    return evaluator_algorithms


def run_main(
    debug,
    max_samples,
    max_features,
    run_outlier,
    run_supervised,
    adbench_dataset_path,
    data_dir,
    bin_edges,
):
    """
    Runs the main execution flow for the benchmark process, including data configuration,
    model evaluation, and result visualization. This function coordinates the various
    components involved in embedding model evaluation and creates plots for analysis.

    Args:
        debug (bool): Enables or disables debug mode. If True, reduces processing limits
            for faster debugging.
        max_samples (int): Maximum number of samples to process in each dataset.
        max_features (int): Maximum number of features to include for the datasets.
        run_outlier (bool): Specifies whether to run the outlier detection evaluation.
        run_supervised (bool): Specifies whether to run the supervised evaluation.
        adbench_dataset_path (str): Path to the ADBench dataset directory.
        data_dir (str): Directory where processed data and results will be stored.

    Raises:
        ValueError: If any configuration or dataset parameters are invalid.

    Returns:
        None
    """
    if debug:
        logger.info("Running in DEBUG mode")
        max_samples = 910
        max_features = 50

    embedding_models = get_embedding_models(debug=debug)
    models_to_keep = [embedding_model._name for embedding_model in embedding_models]

    evaluators = get_evaluators(debug=debug)
    order_evaluators_regression = [
        evaluator._name
        for evaluator in evaluators
        if evaluator.task_type == "Supervised Regression"
    ]
    order_evaluators_regression = list(dict.fromkeys(order_evaluators_regression))
    order_evaluators_classification = [
        evaluator._name
        for evaluator in evaluators
        if evaluator.task_type == "Supervised Classification"
    ]
    order_evaluators_classification = list(
        dict.fromkeys(order_evaluators_classification)
    )
    order_evaluators_outlier = [
        evaluator._name
        for evaluator in evaluators
        if evaluator.task_type == "Outlier Detection"
    ]
    order_evaluators_outlier = list(dict.fromkeys(order_evaluators_outlier))

    logger.info(f"Using {len(embedding_models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    dataset_config = DatasetConfig(
        adbench_dataset_path=adbench_dataset_path,
        exclude_adbench_datasets=[],
        upper_bound_dataset_size=max_samples,
        upper_bound_num_features=max_features,
    )

    benchmark_config = BenchmarkConfig(
        run_outlier=run_outlier,
        run_supervised=run_supervised,
        run_tabpfn_subset=True,
        logging_level=logging.DEBUG,
        data_dir=data_dir,
    )

    result_outlier, result_tabarena, result_dir = run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )


@click.command()
@click.option("--debug", is_flag=True, help="Run in debug mode ")
@click.option("--max-samples", default=10000, help="Upper bound for dataset size")
@click.option("--max-features", default=200, help="Upper bound for number of features")
@click.option(
    "--run-outlier/--no-run-outlier", default=True, help="Run outlier detection"
)
@click.option(
    "--run-supervised/--no-run-supervised",
    default=True,
    help="Run supervised evaluations",
)
@click.option(
    "--adbench-data",
    default="data/adbench_tabular_datasets",
    help="Run supervised evaluations",
)
@click.option("--data-dir", default="data", help="Upper bound for dataset size")
def main(
    debug,
    max_samples,
    max_features,
    run_outlier,
    run_supervised,
    adbench_data,
    data_dir,
):
    """Command-line interface for running the EurIPS benchmark.

    This CLI entry point allows configuring and executing the benchmark with
    various options for debugging, dataset filtering, and task selection.

    Args:
        debug (bool): Run in debug mode with reduced configurations.
        max_samples (int): Maximum number of samples per dataset.
        max_features (int): Maximum number of features per dataset.
        run_outlier (bool): Enable/disable outlier detection benchmarks.
        run_supervised (bool): Enable/disable supervised evaluations.
        adbench_data (str): Path to ADBoard benchmark datasets.
        data_dir (str): Directory where processed data and results will be stored.
    """
    run_main(
        debug=debug,
        max_samples=max_samples,
        max_features=max_features,
        run_outlier=run_outlier,
        run_supervised=run_supervised,
        adbench_dataset_path=adbench_data,
        data_dir=data_dir,
        bin_edges=[0.05, 0.1],
    )


if __name__ == "__main__":
    main()

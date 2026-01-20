"""IJCAI benchmark runner for evaluating tabular embedding models.

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
    SphereBasedEmbedding,
    TableVectorizerEmbedding,
    TabPFNEmbedding,
    TabPFNEmbeddingConstantVector,
    TabPFNEmbeddingRandomVector,
    TabStarEmbedding,
)
from tabembedbench.embedding_models.tabicl_embedding import (
    TabICLEmbedding,
    TabICLWrapper,
)
from tabembedbench.evaluators import (
    LogisticRegressionHPOEvaluator,
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
DATA_DIR = "ijcai_run"

DATASETCONFIG = DatasetConfig(
    adbench_dataset_path="data/adbench_tabular_datasets",
    exclude_adbench_datasets=[],
    exclude_tabarena_datasets=[],
    upper_bound_dataset_size=15000,
    upper_bound_num_features=500,
)


BENCHMARK_CONFIG = BenchmarkConfig(
    run_outlier=True,
    run_tabarena=True,
    data_dir=DATA_DIR,
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
        sphere_model_192,
        sphere_model_256,
        sphere_model_512,
        tabicl_embedding,
        tablevectorizer,
        tabpfn,
        tabpfn_random,
        tabpfn_constant,
        tabStar_embedding,
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


@click.command()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Run in debug mode with minimal models and evaluators for testing.",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="ijcai_run",
    help="Directory where results will be saved.",
)
@click.option(
    "--adbench-path",
    type=click.Path(exists=True),
    default="data/adbench_tabular_datasets",
    help="Path to ADBench tabular datasets.",
)
@click.option(
    "--dataset-size",
    type=int,
    default=15000,
    help="Upper bound on dataset size (number of samples).",
)
@click.option(
    "--num-features",
    type=int,
    default=500,
    help="Upper bound on number of features.",
)
@click.option(
    "--run-outlier",
    is_flag=True,
    default=True,
    help="Enable outlier detection benchmarking.",
)
@click.option(
    "--run-tabarena",
    is_flag=True,
    default=True,
    help="Enable TabArena benchmarking.",
)
@click.option(
    "--exclude-adbench",
    multiple=True,
    default=[],
    help="ADBench dataset names to exclude (can be used multiple times).",
)
@click.option(
    "--exclude-tabarena",
    multiple=True,
    default=[],
    help="TabArena dataset names to exclude (can be used multiple times).",
)
def main(
    debug,
    data_dir,
    adbench_path,
    dataset_size,
    num_features,
    run_outlier,
    run_tabarena,
    exclude_adbench,
    exclude_tabarena,
):
    """
    Main entry point for running the benchmarking pipeline. This function processes
    command-line options, initializes configurations, and runs the benchmarking for
    outlier detection and TabArena datasets using specified embedding models and
    evaluators.

    Args:
        debug (bool): Run in debug mode with minimal models and evaluators for testing.
        data_dir (click.Path): Directory where results will be saved.
        adbench_path (click.Path): Path to ADBench tabular datasets.
        dataset_size (int): Upper bound on dataset size (number of samples).
        num_features (int): Upper bound on number of features.
        run_outlier (bool): Enable outlier detection benchmarking.
        run_tabarena (bool): Enable TabArena benchmarking.
        exclude_adbench (Tuple[str, ...]): ADBench dataset names to exclude; can be
            specified multiple times.
        exclude_tabarena (Tuple[str, ...]): TabArena dataset names to exclude; can be
            specified multiple times.
    """
    dataset_config = DatasetConfig(
        adbench_dataset_path=adbench_path,
        exclude_adbench_datasets=list(exclude_adbench),
        exclude_tabarena_datasets=list(exclude_tabarena),
        upper_bound_dataset_size=dataset_size,
        upper_bound_num_features=num_features,
    )

    benchmark_config = BenchmarkConfig(
        run_outlier=run_outlier,
        run_tabarena=run_tabarena,
        data_dir=data_dir,
    )

    embedding_models = get_embedding_models(debug=debug)
    evaluators = get_evaluators(debug=debug)

    click.echo(f"📊 Configuration Summary:")
    click.echo(f"   Debug Mode: {click.style(str(debug), fg='yellow')}")
    click.echo(
        f"   Embedding Models: {click.style(str(len(embedding_models)), fg='green')}"
    )
    click.echo(f"   Evaluators: {click.style(str(len(evaluators)), fg='green')}")
    click.echo(f"   Max Dataset Size: {click.style(str(dataset_size), fg='green')}")
    click.echo(f"   Max Features: {click.style(str(num_features), fg='green')}")
    click.echo(f"   Run Outlier Detection: {click.style(str(run_outlier), fg='green')}")
    click.echo(f"   Run TabArena: {click.style(str(run_tabarena), fg='green')}")
    click.echo(f"   Results Directory: {click.style(data_dir, fg='cyan')}")
    click.echo("")

    logger.info(f"Using {len(embedding_models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    (
        result_outlier,
        result_tabarena,
        result_dir,
    ) = run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )


if __name__ == "__main__":
    main()

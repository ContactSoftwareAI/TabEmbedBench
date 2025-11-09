"""EuRIPS benchmark runner for evaluating tabular embedding models.

This module provides functionality to run comprehensive benchmarks on various
tabular embedding models using multiple evaluation algorithms across different
datasets. It supports classification, regression, and outlier detection tasks.
"""
import logging
import click

from tabembedbench.benchmark.run_benchmark import (
    run_benchmark,
    DatasetConfig,
    BenchmarkConfig,
)
from tabembedbench.embedding_models import (
    TabICLEmbedding,
    TabVectorizerEmbedding,
    TabPFNEmbedding,
    TabICLClusteringEmbedding
)
from tabembedbench.evaluators.classifier import KNNClassifierEvaluator
from tabembedbench.evaluators.outlier import (
    IsolationForestEvaluator,
    LocalOutlierFactorEvaluator,
    DeepSVDDEvaluator
)
from tabembedbench.evaluators.mlp_evaluator import (
    MLPClassifierEvaluator,
    MLPRegressorEvaluator
)
from tabembedbench.evaluators.regression import KNNRegressorEvaluator

logger = logging.getLogger("EuRIPS_Run_Benchmark")


def get_embedding_models():
    """Initialize and configure embedding models for benchmarking.
    
    Creates a collection of tabular embedding models including TabICL,
    TabPFN, and TableVectorizer. Also creates task-specific embedding
    models like TabICL clustering.
    
    Returns:
        tuple: A tuple containing:
            - list: General-purpose embedding models
            - list: Task-specific embedding models (e.g., clustering)
    """
    tabicl_row_embedder = TabICLEmbedding()

    tabicl_clustering = TabICLClusteringEmbedding(
        num_samples_per_center=50,
        random_state=42
    )

    tablevector = TabVectorizerEmbedding()
    tablevector.name = "TableVectorizer"

    tabpfn_embedder = TabPFNEmbedding(
        num_estimators=5,
    )

    embedding_models = [
            tabicl_row_embedder,
            tabpfn_embedder,
            tablevector,
    ]

    return embedding_models, [tabicl_clustering]

def get_evaluators(debug=False):
    """Configure and initialize evaluation algorithms for the benchmark.
    
    Creates a comprehensive set of evaluators including K-Nearest Neighbors
    (KNN) for classification and regression, outlier detection algorithms
    (DeepSVDD, Isolation Forest, Local Outlier Factor), and neural network
    evaluators (MLP).
    
    Args:
        debug (bool, optional): If True, uses a minimal set of evaluators
            for faster debugging. If False, creates a full grid of evaluator
            configurations with various hyperparameters. Defaults to False.
    
    Returns:
        list: A list of configured evaluator instances ready for benchmarking.
    """
    evaluator_algorithms = []

    deep_svdd_dynamic = DeepSVDDEvaluator(dynamic_hidden_neurons=True)
    deep_svdd_dynamic._name = "DeepSVDD-dynamic"

    if debug:
        evaluator_algorithms.extend(
            [
                KNNRegressorEvaluator(
                    num_neighbors=5,
                    weights="distance",
                    metric="euclidean"
                ),
                KNNClassifierEvaluator(
                    num_neighbors=5,
                    weights="distance",
                    metric="euclidean"
                ),
                DeepSVDDEvaluator(),
                deep_svdd_dynamic,
                LocalOutlierFactorEvaluator(
                    model_params={
                        "n_neighbors": 5,
                    }
                ),
            ]
        )
        return evaluator_algorithms

    for num_neighbors in range(0, 50, 5):
        if num_neighbors > 0:
            for metric in ["euclidean", "cosine"]:
                for weights in ["uniform", "distance"]:
                    evaluator_algorithms.extend([
                        KNNRegressorEvaluator(
                            num_neighbors=num_neighbors,
                            weights=weights,
                            metric=metric
                        ),
                        KNNClassifierEvaluator(
                            num_neighbors=num_neighbors,
                            weights=weights,
                            metric=metric
                        )
                    ])
                evaluator_algorithms.append(
                    LocalOutlierFactorEvaluator(
                        model_params={
                            "n_neighbors": num_neighbors,
                            "metric": metric,
                        }
                    )
                )
    for num_estimators in range(0, 300, 50):
        if num_estimators > 0:
            evaluator_algorithms.append(
                IsolationForestEvaluator(
                    model_params={
                        "n_estimators": num_estimators,
                    }
                )
            )

    evaluator_algorithms.extend(
        [
            MLPRegressorEvaluator(),
            MLPClassifierEvaluator(),
            deep_svdd_dynamic,
            DeepSVDDEvaluator(),
        ]
    )

    return evaluator_algorithms

def run_main(debug, max_samples, max_features, run_outlier,
             run_task_specific, adbench_dataset_path):
    """Execute the main benchmark pipeline.
    
    Orchestrates the complete benchmarking process including initializing
    embedding models, configuring evaluators, setting up dataset and
    benchmark configurations, and running the benchmark.
    
    Args:
        debug (bool): Enable debug mode with reduced dataset sizes and
            minimal evaluator configurations for faster testing.
        max_samples (int): Upper bound on the number of samples per dataset.
            In debug mode, this is overridden to 910.
        max_features (int): Upper bound on the number of features per dataset.
            In debug mode, this is overridden to 50.
        run_outlier (bool): Whether to run outlier detection benchmarks.
        run_task_specific (bool): Whether to run task-specific evaluations
            (e.g., TabArena-specific tasks).
        adbench_dataset_path (str): Path to the ADBoard benchmark datasets
            directory.
    """
    if debug:
        logger.info("Running in DEBUG mode")
        max_samples = 910
        max_features = 50


    embedding_models, task_embedding_models = get_embedding_models()

    evaluators = get_evaluators(debug)

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
        run_task_specific=run_task_specific,
        run_tabpfn_subset=True,
        logging_level=logging.DEBUG,
    )

    run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        tabarena_specific_embedding_models=task_embedding_models,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config
    )



@click.command()
@click.option('--debug', is_flag=True, help='Run in debug mode ')
@click.option('--max-samples', default=15000, help='Upper bound for dataset '
                                                  'size')
@click.option('--max-features', default=500, help='Upper bound for number of features')
@click.option('--run-outlier/--no-run-outlier', default=True, help='Run outlier detection')
@click.option('--run-task-specific/--no-run-task-specific', default=True, help='Run task-specific evaluations')
@click.option('--adbench-data', default='data/adbench_tabular_datasets',
              help='Run task-specific '
                                                 'evaluations')
def main(debug, max_samples, max_features, run_outlier, run_task_specific,
         adbench_data):
    """Command-line interface for running the EuRIPS benchmark.
    
    This CLI entry point allows configuring and executing the benchmark with
    various options for debugging, dataset filtering, and task selection.
    
    Args:
        debug (bool): Run in debug mode with reduced configurations.
        max_samples (int): Maximum number of samples per dataset.
        max_features (int): Maximum number of features per dataset.
        run_outlier (bool): Enable/disable outlier detection benchmarks.
        run_task_specific (bool): Enable/disable task-specific evaluations.
        adbench_data (str): Path to ADBoard benchmark datasets.
    """
    run_main(
        debug=debug,
        max_samples=max_samples,
        max_features=max_features,
        run_outlier=run_outlier,
        run_task_specific=run_task_specific,
        adbench_dataset_path=adbench_data
    )

if __name__ == "__main__":
    main()

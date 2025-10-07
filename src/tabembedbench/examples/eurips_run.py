import logging

import click

from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models import (
    TabICLEmbedding,
    TabVectorizerEmbedding,
    TabPFNEmbedding,
    SphereBasedEmbedding,
)
from tabembedbench.evaluators.classifier import KNNClassifierEvaluator
from tabembedbench.evaluators.outlier import (
    IsolationForestEvaluator,
    LocalOutlierFactorEvaluator,
    DeepSVDDEvaluator,
)
from tabembedbench.evaluators.mlp_evaluator import (
    MLPClassifierEvaluator,
    MLPRegressorEvaluator,
)
from tabembedbench.evaluators.regression import KNNRegressorEvaluator

logger = logging.getLogger("EuRIPS_Run_Benchmark")


def get_embedding_models(debug=False):
    embedding_models = []
    for n in range(3, 10):
        sphere_model = SphereBasedEmbedding(embed_dim=2**n)
        sphere_model.name = f"sphere-model-d{2**n}"
        embedding_models.append(sphere_model)

    tabicl_with_preproccessing = TabICLEmbedding(preprocess_tabicl_data=True)

    tabicl_with_preproccessing.name = "tabicl-classifier-v1.1-0506_preprocessed"

    tabicl_with_tabvectorizer_preprocessing = TabICLEmbedding(
        preprocess_tabicl_data=True, tabvectorizer_preprocess=True
    )

    tabicl_with_tabvectorizer_preprocessing.name = (
        "tabicl-classifier-v1.1-0506_with_tabvectorizer_preprocessed"
    )

    tablevector = TabVectorizerEmbedding()

    tabpfn_embedder = TabPFNEmbedding()

    embedding_models.extend(
        [
            tablevector,
            tabicl_with_preproccessing,
            tabicl_with_tabvectorizer_preprocessing,
            tabpfn_embedder,
        ]
    )

    if debug:
        embedding_models = [tablevector, tabicl_with_tabvectorizer_preprocessing]
        sphere_model = SphereBasedEmbedding(embed_dim=8)
        sphere_model.name = "sphere-model-d8-debug"

        embedding_models.append(sphere_model)

        return embedding_models

    return embedding_models


def get_evaluators(debug=False):
    evaluator_algorithms = []

    deep_svdd_dynamic = DeepSVDDEvaluator(dynamic_hidden_neurons=True)
    deep_svdd_dynamic._name = "DeepSVDD-dynamic"

    if debug:
        evaluator_algorithms.extend(
            [
                KNNRegressorEvaluator(
                    num_neighbors=5, weights="uniform", metric="euclidean"
                ),
                KNNClassifierEvaluator(
                    num_neighbors=5, weights="uniform", metric="euclidean"
                ),
                MLPClassifierEvaluator(n_trials=5, cv_folds=2, verbose=False),
                MLPRegressorEvaluator(n_trials=5, cv_folds=2, verbose=False),
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
                    evaluator_algorithms.extend(
                        [
                            KNNRegressorEvaluator(
                                num_neighbors=num_neighbors,
                                weights=weights,
                                metric=metric,
                            ),
                            KNNClassifierEvaluator(
                                num_neighbors=num_neighbors,
                                weights=weights,
                                metric=metric,
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
    run_task_specific,
    adbench_dataset_path,
):
    if debug:
        logger.info("Running in DEBUG mode")
        max_samples = 800
        max_features = 50

    embedding_models = get_embedding_models(debug)

    evaluators = get_evaluators(debug)

    logger.info(f"Using {len(embedding_models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        adbench_dataset_path=adbench_dataset_path,
        exclude_adbench_datasets=["3_backdoor.npz"],
        upper_bound_dataset_size=max_samples,
        upper_bound_num_features=max_features,
        run_outlier=run_outlier,
        run_task_specific=run_task_specific,
        logging_level=logging.DEBUG,
    )


@click.command()
@click.option("--debug", is_flag=True, help="Run in debug mode ")
@click.option("--max-samples", default=100001, help="Upper bound for dataset size")
@click.option("--max-features", default=500, help="Upper bound for number of features")
@click.option(
    "--run-outlier/--no-run-outlier", default=True, help="Run outlier detection"
)
@click.option(
    "--run-task-specific/--no-run-task-specific",
    default=True,
    help="Run task-specific evaluations",
)
@click.option(
    "--adbench-data",
    default="data/adbench_tabular_datasets",
    help="Run task-specific evaluations",
)
def main(
    debug, max_samples, max_features, run_outlier, run_task_specific, adbench_data
):
    run_main(
        debug=debug,
        max_samples=max_samples,
        max_features=max_features,
        run_outlier=run_outlier,
        run_task_specific=run_task_specific,
        adbench_dataset_path=adbench_data,
    )


if __name__ == "__main__":
    main()

import os
import time
from typing import Dict, Tuple

import numpy as np
import openml
import polars as pl
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.embedding_utils import check_nan
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import update_result_dict

logger = get_benchmark_logger("TabEmbedBench_TabArena")


def run_tabarena_benchmark(
    embedding_models: list[BaseEmbeddingGenerator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 100000,
    save_embeddings: bool = False,
    neighbors: int = 51,
    neighbors_step: int = 5,
    distance_metrics=None,
):
    """Run the TabArena benchmark for a set of embedding models.

    This function evaluates the performance of specified embedding models on a suite
    of tasks from the TabArena benchmark. Depending on the size of datasets and
    parameters provided, the function dynamically adjusts the number of repetitions
    and splits for the benchmark. It evaluates each embedding model on each task
    and computes the AUC/MSR scores for supervised classification and regression tasks
    with k-nearest neighbors.

    Args:
        embedding_models: List of embedding model instances that inherit from
            BaseEmbeddingGenerator. Each model will be evaluated for its performance.
        tabarena_version: The version identifier for the TabArena benchmark study.
            Defaults to "tabarena-v0.1".
        tabarena_lite: Boolean indicating whether to run in lite mode. If True, uses fewer
            splits and repetitions for quicker evaluations. Defaults to True.
        upper_bound_dataset_size: Integer representing the maximum dataset size to
            consider for benchmarking. Datasets larger than this value will be skipped.
            Defaults to 100000.
        save_embeddings: Boolean indicating whether to save computed embeddings during
            the benchmark process. Defaults to False.
        neighbors: Integer specifying the number of neighbors to use for KNN.
            Defaults to 51.
        neighbors_step: Integer specifying the step size for neighbors.
            Defaults to 5.
        distance_metrics: List of distance metrics to use for KNN algorithms.
            Defaults to None.

    Returns:
        polars.DataFrame: A dataframe summarizing the benchmark results. The columns
            include dataset information, embedding model names, number of neighbors,
            metrics such as AUC/MSR scores, embedding computation time, and benchmark
            type.
    """
    if distance_metrics is None:
        distance_metrics = ["euclidean", "cosine"]

    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks

    result_tabarena_dict = {
        "dataset_name": [],
        "dataset_size": [],
        "embedding_model": [],
        "num_neighbors": [],
        "auc_score": [],
        "msr_score": [],
        "task": [],
        "time_to_compute_embeddings": [],
        "benchmark": [],
        "distance_metric": [],
    }

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities["NumberOfInstances"] > upper_bound_dataset_size:
            logger.warning(
                f"Skipping {dataset.name} - dataset size "
                f"{dataset.qualities["NumberOfInstances"]} exceeds "
                f"limit {upper_bound_dataset_size}"
            )
            continue

        folds, tabarena_repeats = _get_task_configuration(dataset,
                                                          tabarena_lite,
                                                          task)

        for repeat in range(tabarena_repeats):
            for fold in range(folds):
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=task.target_name, dataset_format="dataframe"
                )

                train_indices, test_indices = task.get_train_test_split_indices(
                    fold=fold,
                    repeat=repeat,
                )

                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]

                numerical_transformer = TransformToNumerical()
                X_train = numerical_transformer.fit_transform(X_train)
                X_test = numerical_transformer.transform(X_test)

                if task.task_type == "Supervised Classification":
                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)

                for embedding_model in embedding_models:
                    logger.info(
                        f"Starting experiment for dataset {dataset.name} "
                        f"with model {embedding_model.name}..."
                    )

                    X_train_embed, X_test_embed, compute_embeddings_time = (
                        embedding_model.compute_embeddings(X_train, X_test)
                    )

                    if check_nan(X_train_embed):
                        logger.warning(
                            f"The train embeddings for {dataset.name} contain "
                            f"NaN "
                            f"values with embedding model {embedding_model.name}. "
                            f"Skipping."
                        )
                        continue

                    if check_nan(X_test_embed):
                        logger.warning(
                            f"The test embeddings for {dataset.name} contain "
                            f"NaN "
                            f"values with embedding model {embedding_model.name}. "
                            f"Skipping."
                        )
                        continue

                    if save_embeddings:
                        embedding_file = (
                            f"task_{embedding_model.name}"
                            f"_{dataset.name}_embeddings.npz"
                        )
                        np.savez(
                            embedding_file,
                            x_train=X_train_embed,
                            y_train=y_train,
                            x_test=X_test_embed,
                            y_test=y_test,
                        )

                        os.remove(embedding_file)

                    for num_neighbors in range(1, neighbors, neighbors_step):
                        for distance_metric in distance_metrics:
                            if task.task_type == "Supervised Classification":
                                score_auc, exception = (
                                    _evaluate_classification(
                                    X_train_embed,
                                    X_test_embed,
                                    y_train,
                                    y_test,
                                    num_neighbors,
                                    distance_metric=distance_metric,
                                ))

                                if exception is not None:
                                    logger.warning(
                                        f"Error occurred while running experiment for "
                                        f"{embedding_model.name} with "
                                        f"KNN: {exception}"
                                    )
                                    continue

                                update_result_dict(
                                    result_tabarena_dict,
                                    dataset_name=dataset.name,
                                    dataset_size=X_train.shape[0],
                                    embedding_model_name=embedding_model.name,
                                    num_neighbors=num_neighbors,
                                    compute_time=compute_embeddings_time,
                                    auc_score=score_auc,
                                    distance_metric=distance_metric,
                                    task=task.task_type
                                )

                            elif task.task_type == "Supervised Regression":
                                score_msr, _ = _evaluate_regression(
                                    X_train_embed,
                                    X_test_embed,
                                    y_train,
                                    y_test,
                                    num_neighbors,
                                    distance_metric=distance_metric,
                                )

                                if exception is not None:
                                    logger.warning(
                                        f"Error occurred while running experiment for "
                                        f"{embedding_model.name} with "
                                        f"KNN: {exception}"
                                    )
                                    continue

                                update_result_dict(
                                    result_tabarena_dict,
                                    dataset_name=dataset.name,
                                    dataset_size=X_train.shape[0],
                                    embedding_model_name=embedding_model.name,
                                    num_neighbors=num_neighbors,
                                    compute_time=compute_embeddings_time,
                                    msr_score=score_msr,
                                    distance_metric=distance_metric,
                                    task=task.task_type
                                )

                    embedding_model.reset_embedding_model()

                    if get_device() in ["cuda", "mps"]:
                        empty_gpu_cache()
    logger.info("TabArena benchmark completed.")

    result_df = pl.from_dict(
        result_tabarena_dict,
        schema={
            "dataset_name": pl.Categorical,
            "dataset_size": pl.UInt64,
            "embedding_model": pl.Categorical,
            "num_neighbors": pl.UInt64,
            "auc_score": pl.Float64,
            "msr_score": pl.Float64,
            "time_to_compute_embeddings": pl.Float64,
            "benchmark": pl.Categorical,
            "distance_metric": pl.Categorical,
            "task": pl.Categorical,
        },
    )

    return result_df


def _evaluate_classification(
    X_train_embed: np.ndarray,
    X_test_embed: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_neighbors: int,
    distance_metric: str = "euclidean",
) -> Tuple[float, Exception]:
    """Evaluate classification task with KNN Classifier from scikit-learn.."""
    try:
        knn_params = {"n_neighbors": num_neighbors, "n_jobs": -1}
        if distance_metric != "euclidean":
            knn_params["metric"] = distance_metric

        knn_classifier = KNeighborsClassifier(**knn_params)
        knn_classifier.fit(X_train_embed, y_train)
        y_pred_proba = knn_classifier.predict_proba(X_test_embed)

        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            score_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            score_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

        return score_auc, None
    except Exception as e:
        return None, e


def _evaluate_regression(
    X_train_embed: np.ndarray,
    X_test_embed: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_neighbors: int,
    distance_metric: str = "euclidean",
) -> Tuple[float, Exception]:
    """Evaluate regression task with KNN Regressor from scikit-learn."""
    try:
        knn_params = {"n_neighbors": num_neighbors, "n_jobs": -1}
        if distance_metric != "euclidean":
            knn_params["metric"] = distance_metric

        knn_regressor = KNeighborsRegressor(**knn_params)
        knn_regressor.fit(X_train_embed, y_train)
        y_pred = knn_regressor.predict(X_test_embed)
        score_msr = mean_squared_error(y_test, y_pred)

        return score_msr, None
    except Exception as e:
        return None, e


def _get_task_configuration(dataset, tabarena_lite: bool, task) -> Tuple[int, int]:
    """Get the number of folds and repeats for a task."""
    if tabarena_lite:
        return 1, 1

    _, folds, _ = task.get_split_dimensions()
    n_samples = dataset.qualities["NumberOfInstances"]

    if n_samples < 2_500:
        tabarena_repeats = 10
    elif n_samples > 250_000:
        tabarena_repeats = 1
    else:
        tabarena_repeats = 3

    return folds, tabarena_repeats
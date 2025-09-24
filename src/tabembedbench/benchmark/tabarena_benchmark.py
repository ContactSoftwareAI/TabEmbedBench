import os
from typing import Dict, Tuple
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import openml
import polars as pl
import sklearn.metrics
from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_percentage_error, log_loss
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.embedding_utils import check_nan
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import (
    get_batch_dict_result_df,
    update_batch_dict,
    update_result_df,
    save_result_df
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = get_benchmark_logger("TabEmbedBench_TabArena")


def run_tabarena_benchmark(
    embedding_models: list[BaseEmbeddingGenerator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    save_embeddings: bool = False,
    neighbors: int = 51,
    neighbors_step: int = 5,
    distance_metrics=None,
    save_result_dataframe: bool = True,
    result_dir: str | Path = "result_task_specific",
    timestamp: str = TIMESTAMP,
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
        upper_bound_num_samples: Integer representing the maximum dataset size to
            consider for benchmarking. Datasets larger than this value will be skipped.
            Defaults to 100000.
        upper_bound_num_features: Integer representing the maximum number of features
            considered for benchmarking. Datasets with more features than this value
            will be skipped. Defaults to 500.
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

    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks

    batch_dict, result_df = get_batch_dict_result_df()

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities["NumberOfInstances"] > upper_bound_num_samples:
            logger.warning(
                f"Skipping {dataset.name} - dataset size "
                f"{dataset.qualities["NumberOfInstances"]} exceeds "
                f"limit {upper_bound_num_samples}"
            )
            continue

        if dataset.qualities["NumberOfFeatures"] > upper_bound_num_features:
            logger.warning(
                f"Skipping {dataset.name} - number of features size "
                f"{dataset.qualities["NumberOfFeatures"]} exceeds "
                f"limit {upper_bound_num_features}"
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

                    (X_train_embed, X_test_embed, compute_embeddings_time,
                     compute_test_embeddings_time) = (
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

                    embedding_dim = X_train_embed.shape[-1]

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

                    for num_neighbors in range(0, neighbors, neighbors_step):
                        if num_neighbors == 0:
                            continue
                        for distance_metric in distance_metrics:
                            if task.task_type == "Supervised Classification":
                                score, predict_time, exception_message = (
                                    _evaluate_classification(
                                    X_train_embed,
                                    X_test_embed,
                                    y_train,
                                    y_test,
                                    num_neighbors,
                                    distance_metric=distance_metric,
                                ))

                                if exception_message is not None:
                                    logger.warning(
                                        f"Error occurred while running experiment for "
                                        f"{embedding_model.name} with "
                                        f"KNN: {exception_message}"
                                    )
                                    continue

                                if np.unique(y_train).shape[0] == 2:
                                    batch_dict = update_batch_dict(
                                        batch_dict,
                                        dataset_name=dataset.name,
                                        dataset_size=X_train.shape[0],
                                        embedding_model_name=embedding_model.name,
                                        num_neighbors=num_neighbors,
                                        time_to_compute_train_embeddings=compute_embeddings_time,
                                        auc_score=score,
                                        distance_metric=distance_metric,
                                        task=task.task_type,
                                        algorithm="KNNClassifier",
                                        embedding_dimension=embedding_dim,
                                        prediction_time=predict_time,
                                        time_to_compute_test_embeddings=compute_test_embeddings_time,
                                    )
                                else:
                                    batch_dict = update_batch_dict(
                                        batch_dict,
                                        dataset_name=dataset.name,
                                        dataset_size=X_train.shape[0],
                                        embedding_model_name=embedding_model.name,
                                        num_neighbors=num_neighbors,
                                        time_to_compute_train_embeddings=compute_embeddings_time,
                                        log_loss_score=score,
                                        distance_metric=distance_metric,
                                        task=task.task_type,
                                        algorithm="KNNClassifier",
                                        embedding_dimension=embedding_dim,
                                        prediction_time=predict_time,
                                        time_to_compute_test_embeddings=compute_test_embeddings_time,
                                    )

                            elif task.task_type == "Supervised Regression":
                                score_mape, predict_time, exception_message = (
                                    _evaluate_regression(
                                    X_train_embed,
                                    X_test_embed,
                                    y_train,
                                    y_test,
                                    num_neighbors,
                                    distance_metric=distance_metric,
                                ))

                                if exception_message is not None:
                                    logger.warning(
                                        f"Error occurred while running experiment for "
                                        f"{embedding_model.name} with "
                                        f"KNN: {exception_message}"
                                    )
                                    continue

                                batch_dict = update_batch_dict(
                                    batch_dict,
                                    dataset_name=dataset.name,
                                    dataset_size=X_train.shape[0],
                                    embedding_model_name=embedding_model.name,
                                    num_neighbors=num_neighbors,
                                    time_to_compute_train_embeddings=compute_embeddings_time,
                                    mape_score=score_mape,
                                    distance_metric=distance_metric,
                                    task=task.task_type,
                                    algorithm="KNNRegressor",
                                    embedding_dimension=embedding_dim,
                                    prediction_time=predict_time,
                                    time_to_compute_test_embeddings=compute_test_embeddings_time,
                                )
                    logger.debug(
                        f"Finished experiment for {embedding_model.name} and "
                        f"resetting the model."
                    )
                    embedding_model.reset_embedding_model()

                    batch_dict, result_df = update_result_df(
                        batch_dict=batch_dict, result_df=result_df, logger=logger
                    )

                    if save_result_dataframe:
                        save_result_df(result_df=result_df,
                                       output_path=result_dir,
                                       benchmark_name="TabArena",
                                       timestamp=timestamp)

                    if get_device() in ["cuda", "mps"]:
                        empty_gpu_cache()
    logger.info("TabArena benchmark completed.")

    return result_df


def _evaluate_classification(
    X_train_embed: np.ndarray,
    X_test_embed: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_neighbors: int,
    distance_metric: str = "euclidean",
):
    """Evaluate classification task with KNN Classifier from scikit-learn.."""
    try:
        knn_params = {"n_neighbors": num_neighbors, "n_jobs": -1}
        if distance_metric != "euclidean":
            knn_params["metric"] = distance_metric

        knn_classifier = KNeighborsClassifier(**knn_params)

        start_predict_time = time.time()
        knn_classifier.fit(X_train_embed, y_train)
        y_pred_proba = knn_classifier.predict_proba(X_test_embed)
        predict_time = time.time() - start_predict_time

        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            score = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            score = log_loss(y_test, y_pred_proba)
            #score = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

        return score, predict_time, None
    except Exception as e:
        return None, None, e


def _evaluate_regression(
    X_train_embed: np.ndarray,
    X_test_embed: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_neighbors: int,
    distance_metric: str = "euclidean",
):
    """Evaluate regression task with KNN Regressor from scikit-learn."""
    try:
        knn_params = {"n_neighbors": num_neighbors, "n_jobs": -1}
        if distance_metric != "euclidean":
            knn_params["metric"] = distance_metric

        knn_regressor = KNeighborsRegressor(**knn_params)

        start_predict_time = time.time()
        knn_regressor.fit(X_train_embed, y_train)
        y_pred = knn_regressor.predict(X_test_embed)
        predict_time = time.time() - start_predict_time

        #score_mse = mean_squared_error(y_test, y_pred)
        score_mape = mean_absolute_percentage_error(y_test, y_pred)

        return score_mape, predict_time, None
    except Exception as e:
        return None, None, e


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
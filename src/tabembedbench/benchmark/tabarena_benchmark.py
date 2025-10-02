import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openml
import polars as pl
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
    log_loss
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator
from tabembedbench.evaluators.abstractevaluator import AbstractEvaluator
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import save_result_df

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = get_benchmark_logger("TabEmbedBench_TabArena")


def run_tabarena_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    result_dir: str | Path = "result_tabarena",
    save_result_dataframe: bool = True,
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
        evaluators: List of evaluators
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
        result_dir:
        save_result_dataframe:
        timestamp:

    Returns:
        polars.DataFrame: A dataframe summarizing the benchmark results. The columns
            include dataset information, embedding model names, number of neighbors,
            metrics such as AUC/MSR scores, embedding computation time, and benchmark
            type.
    """
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks

    result_df = pl.DataFrame()

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
        logger.info(f"Starting experiments for dataset {dataset.name}"
                    f" and task {task.task_type}"
                    )
        for repeat in range(tabarena_repeats):
            for fold in range(folds):
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=task.target_name, dataset_format="dataframe"
                )

                categorical_indices = np.nonzero(categorical_indicator)[0]
                categorical_indices = categorical_indices.tolist()

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
                        f"Starting experiment for embedding model"
                        f" {embedding_model.name}..."
                    )
                    try:
                        (train_embeddings, test_embeddings, compute_embeddings_time,
                         compute_test_embeddings_time) = (
                            embedding_model.generate_embeddings(
                                X_train,
                                X_test,
                                categorical_indices=categorical_indices
                            )
                        )
                    except Exception as e:
                        logger.exception(
                            f"By computing embeddings, the following ValueError "
                            f"occured: {e}. Skipping"
                        )
                        new_row_dict = {
                            "dataset_name": [dataset.name],
                            "dataset_size": [X.shape[0]],
                            "embedding_model": [embedding_model.name],
                        }

                        new_row = pl.DataFrame(
                                new_row_dict
                            )

                        result_df = pl.concat(
                            [result_df, new_row],
                            how="diagonal"
                        )
                        continue

                    embed_dim = train_embeddings.shape[-1]

                    for evaluator in evaluators:
                        if task.task_type == evaluator.task_type:
                            logger.debug(f"Starting experiment for evaluator "
                                         f"{evaluator._name}...")
                            prediction_train, _ = evaluator.get_prediction(
                                train_embeddings,
                                y_train,
                                train=True,
                            )

                            test_prediction, _ = evaluator.get_prediction(
                                test_embeddings,
                                train=False,
                            )

                            parameters = evaluator.get_parameters()

                            new_row_dict = {
                                "dataset_name": [dataset.name],
                                "dataset_size": [X.shape[0]],
                                "embedding_model": [embedding_model.name],
                                "embed_dim": [embed_dim],
                                "time_to_compute_train_embedding": [
                                    compute_embeddings_time
                                ],
                                "algorithm": [evaluator._name],
                            }
                            logger.debug(f"Parameters for {evaluator._name}: "
                                         f"{new_row_dict}")

                            for key, value in parameters.items():
                                new_row_dict[f"algorithm_{key}"] = [value]

                            if task.task_type == "Supervised Regression":
                                mape_score = mean_absolute_percentage_error(
                                    y_test, test_prediction
                                )
                                new_row_dict["task"] = ["regression"]
                                new_row_dict["mape_score"] = [mape_score]
                            if task.task_type == "Supervised Classification":
                                n_classes = test_prediction.shape[1]
                                if n_classes == 2:
                                    auc_score = roc_auc_score(
                                        y_test, test_prediction[:, 1]
                                    )
                                    new_row_dict["task"] = ["classification"]
                                    new_row_dict["classification_type"] = [
                                        "binary"]
                                else:
                                    auc_score = roc_auc_score(
                                        y_test, test_prediction, multi_class="ovr"
                                    )
                                    log_loss_score = log_loss(
                                        y_test, test_prediction, multi_class="ovr"
                                    )
                                    new_row_dict["task"] = ["classification"]
                                    new_row_dict["classification_type"] = [
                                        "multiclass"]
                                    new_row_dict["log_loss_score"] = [log_loss_score]
                                new_row_dict["auc_score"] = [auc_score]


                            new_row = pl.DataFrame(
                                new_row_dict
                            )

                            result_df = pl.concat(
                                [result_df, new_row],
                                how="diagonal"
                            )

                            if save_result_dataframe:
                                save_result_df(
                                    result_df=result_df,
                                    output_path=result_dir,
                                    benchmark_name="TabArena",
                                    timestamp=timestamp,
                                )

                            evaluator.reset_evaluator()

                    logger.debug(
                        f"Finished experiment for {embedding_model.name} and "
                        f"resetting the model."
                    )
                    embedding_model.reset_embedding_model()

                    logger.debug(f"Dataframe rows: {result_df.shape[0]}")

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
            score_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            score_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

        return score_auc, predict_time, None
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

        score_mse = mean_squared_error(y_test, y_pred)

        return score_mse, predict_time, None
    except Exception as e:
        return None, None, e


def _get_task_configuration(dataset, tabarena_lite: bool, task) -> tuple[int, int]:
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

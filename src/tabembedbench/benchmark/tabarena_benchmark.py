import logging
import os
import time
from typing import List

import numpy as np
import openml
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger("TabEmbedBench_TabArena")


def run_tabarena_benchmark(
    embedding_models: List[BaseEmbeddingGenerator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 100000,
    save_embeddings: bool = False,
):
    """
    Run the TabArena benchmark for a set of embedding models.

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

    Returns:
        polars.DataFrame: A dataframe summarizing the benchmark results. The columns
            include dataset information, embedding model names, number of neighbors,
            metrics such as AUC/MSR scores, embedding computation time, and benchmark
            type.
    """
    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks

    result_tabarena_dict = {
        "dataset_name": [],
        "dataset_size": [],
        "embedding_model": [],
        "num_neighbors": [],
        "auc_score": [],
        "msr_score": [],
        "time_to_compute_embeddings": [],
        "benchmark": [],
    }

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.qualities["NumberOfInstances"] > upper_bound_dataset_size:
            logger.warning(
                f"""
                Skipping {dataset.name} - dataset size {dataset.qualities["NumberOfInstances"]} exceeds limit {upper_bound_dataset_size}
                """
            )
            continue

        if tabarena_lite:
            folds = 1
            tabarena_repeats = 1
        else:
            _, folds, _ = task.get_split_dimensions()
            n_samples = dataset.qualities["NumberOfInstances"]
            if n_samples < 2_500:
                tabarena_repeats = 10
            elif n_samples > 250_000:
                tabarena_repeats = 1
            else:
                tabarena_repeats = 3

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
                        f"Starting experiment for dataset {dataset.name} with model {embedding_model.name}..."
                    )

                    X_train = embedding_model.preprocess_data(X_train, train=True)

                    start_time = time.time()
                    X_train_embed = embedding_model.compute_embeddings(X_train)
                    compute_embeddings_time = time.time() - start_time

                    X_test = embedding_model.preprocess_data(X_test, train=False)
                    X_test_embed = embedding_model.compute_embeddings(X_test)

                    if save_embeddings:
                        embedding_file = (
                            f"task_{embedding_model.name}_{dataset.name}_embeddings.npz"
                        )
                        np.savez(
                            embedding_file,
                            x_train=X_train_embed,
                            y_train=y_train,
                            x_test=X_test_embed,
                            y_test=y_test,
                        )

                        os.remove(embedding_file)

                    for num_neighbors in range(1, 51):
                        result_tabarena_dict["dataset_name"].append(dataset.name)
                        result_tabarena_dict["dataset_size"].append(X_train.shape[0])
                        result_tabarena_dict["embedding_model"].append(
                            embedding_model.name
                        )
                        result_tabarena_dict["num_neighbors"].append(num_neighbors)
                        result_tabarena_dict["time_to_compute_embeddings"].append(
                            compute_embeddings_time
                        )
                        result_tabarena_dict["benchmark"].append("tabarena")
                        if task.task_type == "Supervised Classification":
                            knn_classifier = KNeighborsClassifier(
                                n_neighbors=num_neighbors,
                                n_jobs=-1,
                            )

                            knn_classifier.fit(X_train_embed, y_train)

                            y_pred_proba = knn_classifier.predict_proba(X_test_embed)

                            n_classes = y_pred_proba.shape[1]
                            if n_classes == 2:
                                score_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            else:
                                score_auc = roc_auc_score(
                                    y_test, y_pred_proba, multi_class="ovr"
                                )

                            result_tabarena_dict["auc_score"].append(score_auc)
                            result_tabarena_dict["msr_score"].append(np.inf)

                        elif task.task_type == "Supervised Regression":
                            knn_regressor = KNeighborsRegressor(
                                n_neighbors=num_neighbors,
                                n_jobs=-1,
                            )

                            knn_regressor.fit(X_train_embed, y_train)
                            y_pred = knn_regressor.predict(X_test_embed)
                            score_msr = mean_squared_error(y_test, y_pred)

                            result_tabarena_dict["msr_score"].append(score_msr)
                            result_tabarena_dict["auc_score"].append((-1) * np.inf)

                    embedding_model.reset_embedding_model()

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
        },
    )

    return result_df

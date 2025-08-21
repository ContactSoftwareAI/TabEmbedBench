import logging
import os
import time

import mlflow
import numpy as np
import openml
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import List

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_tabarena_benchmark(
    embedding_models: List[BaseEmbeddingGenerator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 100000,
    save_embeddings: bool = False,
):
    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks

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

                for embedding_model in embedding_models:
                    with mlflow.start_run(
                        run_name=f"task_{embedding_model.name}_{dataset.name}",
                        nested=True,
                    ):
                        mlflow.log_param("dataset_name", dataset.name)
                        mlflow.log_param("task_type", task.task_type)
                        mlflow.log_param("embedding_model", embedding_model.name)
                        logger.info(
                            f"Starting experiment for dataset {dataset.name} with model {embedding_model.name}..."
                        )

                        X_train = embedding_model.preprocess_data(X_train, train=True)

                        start_time = time.time()
                        X_train_embed = embedding_model.compute_embeddings(X_train)
                        compute_embeddings_time = time.time() - start_time

                        mlflow.log_param("embed_dim", X_train_embed.shape[-1])

                        if mlflow.active_run():
                            mlflow.log_metric(
                                "compute_embedding_time", compute_embeddings_time
                            )

                        X_test = embedding_model.preprocess_data(X_test, train=False)
                        X_test_embed = embedding_model.compute_embeddings(X_test)

                        if save_embeddings:
                            embedding_file = f"task_{embedding_model.name}_{dataset.name}_embeddings.npz"
                            np.savez(
                                embedding_file,
                                x_train=X_train_embed,
                                y_train=y_train,
                                x_test=X_test_embed,
                                y_test=y_test,
                            )

                            mlflow.log_artifact(embedding_file)

                            os.remove(embedding_file)

                        for num_neighbors in range(1, 51):
                            if task.task_type == "Supervised Classification":
                                knn_classifier = KNeighborsClassifier(
                                    n_neighbors=num_neighbors,
                                    n_jobs=-1,
                                )

                                knn_classifier.fit(X_train_embed, y_train)

                                y_pred_proba = knn_classifier.predict_proba(
                                    X_test_embed
                                )

                                score_auc = roc_auc_score(y_test, y_pred_proba)

                                if mlflow.active_run():
                                    mlflow.log_metric(
                                        "auc_score",
                                        score_auc,
                                        step=num_neighbors,
                                    )
                            elif task.task_type == "Supervised Regression":
                                knn_regressor = KNeighborsRegressor(
                                    n_neighbors=num_neighbors,
                                    n_jobs=-1,
                                )

                                knn_regressor.fit(X_train_embed, y_train)

                                y_pred = knn_regressor.predict(X_test_embed)

                                score_msr = mean_squared_error(y_test, y_pred)

                                if mlflow.active_run():
                                    mlflow.log_metric(
                                        "mean_squared_error",
                                        score_msr,
                                        step=num_neighbors,
                                    )

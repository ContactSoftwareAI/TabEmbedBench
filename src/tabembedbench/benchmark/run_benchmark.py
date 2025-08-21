from datetime import datetime
from typing import Optional, Union
from pathlib import Path

import mlflow

from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_benchmark(
    embedding_model: Optional[BaseEmbeddingGenerator],
    embedding_models: Optional[list[BaseEmbeddingGenerator]],
    tracking_uri: str = None,
    mlflow_experiment_name: Optional[str] = None,
    adbench_dataset_path: Optional[Union[str, Path]] = None,
    exclude_adbench_datasets: Optional[list[str]] = None,
    exclude_adbench_image_datasets: bool = True,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    save_embeddings: bool = False,
    run_outlier: bool = True,
    run_task_specific: bool = True,
):
    """
    Runs a benchmarking process for evaluating tabular embedding models. This function supports a range of input embedding
    models and provides the ability to benchmark both outlier detection performance and task-specific performance using
    datasets from ADBench and TabArena.

    Args:
        embedding_model (Optional[BaseEmbeddingGenerator]): A single embedding model instance for benchmarking.
        embedding_models (Optional[list[BaseEmbeddingGenerator]]): A list of embedding model instances for benchmarking. One
            of embedding_model or embedding_models must be provided, but not both.
        tracking_uri (str, optional): The MLflow tracking URI for logging and tracking experiments. If not provided, no custom
            tracking URI will be used.
        mlflow_experiment_name (Optional[str]): The name of the MLflow experiment where benchmarking results should be logged.
            Defaults to "tabular_embedding_benchmark" if not provided.
        adbench_dataset_path (Optional[Union[str, Path]]): The file path to the ADBench dataset directory for outlier
            benchmarking.
        exclude_adbench_datasets (Optional[list[str]]): A list of dataset names to exclude from the ADBench dataset suite.
        exclude_adbench_image_datasets (bool): Whether to exclude image datasets from ADBench during outlier benchmarking. Defaults
            to True.
        tabarena_version (str): Version of the TabArena dataset to use for task-specific benchmarking. Defaults to
            "tabarena-v0.1".
        tabarena_lite (bool): Whether to use the lightweight version of TabArena. Defaults to True.
        upper_bound_dataset_size (int): The maximum size (number of samples) of datasets to consider during benchmarking.
            Defaults to 100000.
        save_embeddings (bool): Whether to save the computed embeddings during benchmarking for future use. Defaults to False.
        run_outlier (bool): Whether to perform outlier detection benchmarking using the provided embedding models and ADBench
            datasets. Defaults to True.
        run_task_specific (bool): Whether to perform task-specific benchmarking using the TabArena datasets. Defaults to True.

    Raises:
        ValueError: If neither embedding_model nor embedding_models is provided.
        ValueError: If both embedding_model and embedding_models are provided.
        ValueError: If any model in embedding_models lacks the required 'compute_embeddings', 'name', or 'preprocess_data'
            attributes.

    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    if mlflow_experiment_name is None:
        mlflow_experiment_name = "tabular_embedding_benchmark"

    mlflow.set_experiment(mlflow_experiment_name)

    if embedding_model is None and embedding_models is None:
        raise ValueError("Either model or models must be provided.")
    if embedding_model is not None and embedding_models is not None:
        raise ValueError(
            "Only one of the parameters 'model' or 'models' should be provided, not both."
        )

    if embedding_model is not None:
        models_to_process = [embedding_model]
    else:
        models_to_process = embedding_models

    for model in models_to_process:
        if not hasattr(model, "compute_embeddings"):
            raise ValueError(
                "There is an element within the list of models that does not have a function 'compute_embeddings'."
            )
        elif not hasattr(model, "name"):
            raise ValueError(
                "There is an element within the list of models that does not have a property 'name'."
            )
        elif not hasattr(model, "preprocess_data"):
            raise ValueError(
                "There is an element within the list of models that does not have a property 'name'."
            )

    with mlflow.start_run(run_name=f"tabular_embedding_benchmark"):
        if run_outlier:
            outlier_embedding_models = []
            for embedding_model in models_to_process:
                if not embedding_model.task_only:
                    outlier_embedding_models.append(embedding_model)
            if len(outlier_embedding_models) > 0:
                run_outlier_benchmark(
                    embedding_models=outlier_embedding_models,
                    dataset_paths=adbench_dataset_path,
                    save_embeddings=save_embeddings,
                    exclude_datasets=exclude_adbench_datasets,
                    exclude_image_datasets=exclude_adbench_image_datasets,
                    upper_bound_dataset_size=upper_bound_dataset_size,
                )
        if run_task_specific:
            run_tabarena_benchmark(
                embedding_models=embedding_models,
                tabarena_version=tabarena_version,
                tabarena_lite=tabarena_lite,
                upper_bound_dataset_size=upper_bound_dataset_size,
                save_embeddings=save_embeddings,
            )

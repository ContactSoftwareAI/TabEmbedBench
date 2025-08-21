from datetime import datetime
from typing import Optional

import mlflow

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

IMAGE_CATEGORY = [
    "1_ALOI.npz",
    "8_celeba.npz",
    "17_InternetAds.npz",
    "20_letter.npz",
    "24_mnist.npz",
    "26_optdigits.npz",
    "28_pendigits.npz",
    "33_skin.npz",
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_benchmark(
    embedding_model: Optional[BaseEmbeddingGenerator],
    embedding_models: Optional[list[BaseEmbeddingGenerator]],
    tracking_uri: str = None,
    mlflow_experiment_name: Optional[str] = None,
    exclude_adbench_datasets: Optional[list[str]] = None,
    exclude_adbench_image_datasets: bool = True,
    upper_bound_dataset_size: int = 100000,
    run_outlier: bool = True,
    run_task_specific: bool = True,
):
    """
    Run a benchmark for evaluating embedding models on specified datasets with tracking through MLflow.

    This function handles initialization of MLflow experiment tracking, provides flexibility for defining
    single or multiple embedding models to benchmark, and configures dataset-specific exclusions or size
    limits. The function ensures that passed models have required functionalities to perform embeddings
    and pre-processing tasks.

    Args:
        embedding_model: A single embedding generator model to benchmark.
        embedding_models: A list of embedding generator models to benchmark.
        tracking_uri: The URI for the MLflow tracking server.
        mlflow_experiment_name: The name of the MLflow experiment for logging results.
        exclude_adbench_datasets: A list of dataset names to exclude from the benchmark.
        exclude_adbench_image_datasets: Specifies whether to exclude image datasets from the benchmark.
        upper_bound_dataset_size: The maximum size of datasets to use in the benchmark.
        run_outlier: A flag to run outlier detection tasks during the benchmark process.
        run_task_specific: A flag to run task-specific benchmarks during the evaluation process.

    Raises:
        ValueError: If neither `embedding_model` nor `embedding_models` is provided.
        ValueError: If both `embedding_model` and `embedding_models` are simultaneously provided.
        ValueError: If any model in the passed list lacks required properties or methods, such as
                    `compute_embeddings`, `name`, or `preprocess_data`.
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
            pass
        if run_task_specific:
            pass

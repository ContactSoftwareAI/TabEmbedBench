from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import polars as pl

from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_benchmark(
    embedding_model: Optional[BaseEmbeddingGenerator] = None,
    embedding_models: Optional[list[BaseEmbeddingGenerator]] = None,
    adbench_dataset_path: Optional[Union[str, Path]] = None,
    exclude_adbench_datasets: Optional[list[str]] = None,
    exclude_adbench_image_datasets: bool = True,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    save_embeddings: bool = False,
    run_outlier: bool = True,
    run_task_specific: bool = True,
) -> pl.DataFrame:
    """
    Run a benchmark pipeline for embedding models on specified datasets and tasks.

    This function allows evaluating a single embedding model or a list of embedding models over a series
    of predefined datasets and benchmarking tasks. The benchmarking process includes outlier detection
    and task-specific evaluations depending on the parameters provided. It manages embedding creation,
    task-specific dataset preparation, and result aggregation for further analysis.

    Args:
        embedding_model (Optional[BaseEmbeddingGenerator]): A single embedding model to process.
        embedding_models (Optional[list[BaseEmbeddingGenerator]]): A list of embedding models to process.
        adbench_dataset_path (Optional[Union[str, Path]]): Path to the ADBench dataset directory.
        exclude_adbench_datasets (Optional[list[str]]): List of dataset names to exclude from ADBench evaluation.
        exclude_adbench_image_datasets (bool): Flag indicating whether to exclude image datasets in ADBench.
        tabarena_version (str): Version identifier for the TabArena framework.
        tabarena_lite (bool): Flag indicating whether to use TabArena lite mode
                              for processing smaller datasets.
        upper_bound_dataset_size (int): Maximum size of dataset entries considered for evaluation.
        save_embeddings (bool): Flag determining whether to save computed embeddings during evaluations.
        run_outlier (bool): Flag to toggle the outlier detection benchmark execution.
        run_task_specific (bool): Flag to toggle task-specific benchmark execution.

    Returns:
        pl.DataFrame: A combined dataframe containing the results of the outlier and task-specific benchmarks.

    Raises:
        ValueError: If neither `embedding_model` nor `embedding_models` is provided, or if both are provided.
        ValueError: If a provided model lacks the `compute_embeddings`, `name`, or `preprocess_data` attributes.
        ValueError: If no eligible outlier models are available when `run_outlier` is enabled.
    """
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

    if run_outlier:
        outlier_embedding_models = []
        for model in models_to_process:
            if not model.task_only:
                outlier_embedding_models.append(model)
        if len(outlier_embedding_models) > 0:
            result_outlier_df = run_outlier_benchmark(
                embedding_models=outlier_embedding_models,
                dataset_paths=adbench_dataset_path,
                save_embeddings=save_embeddings,
                exclude_datasets=exclude_adbench_datasets,
                exclude_image_datasets=exclude_adbench_image_datasets,
                upper_bound_dataset_size=upper_bound_dataset_size,
            )
        else:
            raise ValueError("No outlier models provided.")
    else:
        result_outlier_df = pl.DataFrame()
    if run_task_specific:
        result_tabarena_df = run_tabarena_benchmark(
            embedding_models=models_to_process,
            tabarena_version=tabarena_version,
            tabarena_lite=tabarena_lite,
            upper_bound_dataset_size=upper_bound_dataset_size,
            save_embeddings=save_embeddings,
        )
    else:
        result_tabarena_df = pl.DataFrame()

    combined_results = pl.concat(
        [result_outlier_df, result_tabarena_df], how="diagonal"
    )

    return combined_results

if __name__ == "__main__":
    from tabembedbench.embedding_models.tabpfn_embedding import UniversalTabPFNEmbedding

    tabpfn = UniversalTabPFNEmbedding()

    run_benchmark(
        embedding_model=tabpfn,
        upper_bound_dataset_size=500,
        adbench_dataset_path="/Users/lkl/PycharmProjects/TabEmbedBench/data/adbench_tabular_datasets",
    )
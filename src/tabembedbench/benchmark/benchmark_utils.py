from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List
import shutil

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.dataset_utils import (
    download_adbench_tabular_datasets,
    get_data_description,
)
from tabembedbench.utils.logging_utils import setup_logger

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(__name__, f"log/benchmark_{timestamp}.log")

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


def run_outlier_benchmark(
    model: Optional[BaseEmbeddingGenerator] = None,
    models: Optional[List[BaseEmbeddingGenerator]] = None,
    dataset_paths: Optional[Union[str, Path]] = None,
    save_embeddings: bool = False,
    save_embeddings_path: Optional[Union[str, Path]] = None,
    upper_bound_dataset_size: int = 100000,
    exclude_datasets: Optional[list[str]] = None,
) -> pl.DataFrame:
    """
    Executes an outlier detection benchmark on provided datasets using a specified model.
    This function iterates through datasets in the given path, processes each dataset, runs
    the detection model, and aggregates the benchmark results.

    Args:
        model: A model implementing the BaseEmbeddingGenerator interface. This model is used
            for outlier detection across the datasets.
        models: A list of models implementing the BaseEmbeddingGenerator interface.
        dataset_paths: Path to the directory containing dataset files in `.npz` format. If no
            path is provided, a default path to ADBench tabular datasets is used. The datasets
            will be downloaded if they do not exist at the default location.
        random_state: Seed for random number generation to ensure reproducibility.
        save_embeddings: A boolean flag indicating whether to save the generated embeddings
            during the benchmark process.
        save_embeddings_path: Path where embeddings and other intermediate results will be saved, if
            `save_embeddings` is set to True.
        upper_bound_dataset_size:
        exclude_datasets:

    Returns:
        pl.DataFrame: A DataFrame containing the combined benchmark results for all datasets.
    """
    if model is None and models is None:
        raise ValueError("Either model or models must be provided.")
    if model is not None and models is not None:
        raise ValueError("Only one of the parameters 'model' or 'models' should be provided, not both.")

    if model is not None:
        models_to_process = [model]
    else:
        models_to_process = models

    for model_element in models_to_process:
        if not hasattr(model_element, "compute_embeddings"):
            raise ValueError(
                "There is an element within the list of models that does not have a function 'compute_embeddings'."
            )

    logger.info("Running outlier benchmark...")
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            logger.warning("Downloading ADBench tabular datasets...")
            download_adbench_tabular_datasets(dataset_paths)
    else:
        dataset_paths = Path(dataset_paths)
    logger.info(f"Dataset paths: {dataset_paths}")

    if save_embeddings_path is None:
        save_embeddings_path = dataset_paths / "embeddings"
        if save_embeddings:
            save_embeddings_path.mkdir(parents=True, exist_ok=True)

    npz_files = list(dataset_paths.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} .npz files: {[f.name for f in npz_files]}")

    benchmark_result_df = None

    for dataset_file in dataset_paths.glob("*.npz"):
        logger.info(f"Running benchmark for {dataset_file.name}...")
        dataset = np.load(dataset_file)

        X = dataset["X"]
        y = dataset["y"]

        with np.load(dataset_file) as dataset:
            if dataset["X"].shape[0] > upper_bound_dataset_size:
                logger.warning(
                    f"Skipping {dataset_file.name} - dataset size {dataset['X'].shape[0]} exceeds limit {upper_bound_dataset_size}"
                )
                continue

            # Now load the full data
            X = dataset["X"]
            y = dataset["y"]

        dataset_description = get_data_description(X, y, dataset_file.stem)
        logger.info(
            f"Samples: {dataset_description['samples']}, Features: {dataset_description['features']}"
        )
        dataset_df = pl.DataFrame(dataset_description)

        for model in models_to_process:
            logger.info(f"Starting experiment for dataset {dataset_file.stem} with model {model.name}...")
            result_df = run_experiment(
                model,
                X=X,
                y=y,
                save_embeddings=save_embeddings,
                save_embeddings_path=save_embeddings_path,
                dataset_name=dataset_file.stem,
            )

            dataset_df = dataset_df.join(result_df, how="cross")

            dataset_df = (
                dataset_df.with_columns(
                    pl.lit(model.name).alias("embedding_model")
                )
            )

            if benchmark_result_df is None:
                benchmark_result_df = dataset_df
            else:
                benchmark_result_df = _align_schemas_and_concat(
                    benchmark_result_df, dataset_df
                )

            if benchmark_result_df is None:
                raise ValueError("Benchmark result DataFrame is empty.")
            logger.info("Number of rows added: ", benchmark_result_df.height)

    return benchmark_result_df


def run_experiment(
    model: BaseEmbeddingGenerator,
    X: Union[torch.Tensor, np.ndarray],
    y: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_embeddings: bool = False,
    save_embeddings_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
) -> pl.DataFrame:
    result_dict = dict()
    result_dict["algorithm"] = []
    result_dict["neighbors"] = []

    X_embed = model.compute_embeddings(X)

    if save_embeddings:
        save_file_path = Path(save_embeddings_path) / f"{model.name}" / f"{dataset_name}_embeddings.npz"
        if save_file_path.exists() and save_file_path.is_dir():
            shutil.rmtree(save_file_path)

        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(file=save_file_path, x=X_embed, y=y)

    num_neighbors_list = [i for i in range(1, 15)]

    logger.info(f"Running LocalOutlierFactor for dataset {dataset_name}")
    try:
        for num_neighbors in num_neighbors_list:
            lof = LocalOutlierFactor(
                n_neighbors=num_neighbors,
                n_jobs=-1,
            )

            lof.fit(X_embed)

            neg_outlier_factor = (-1)*lof.negative_outlier_factor_

            score_outlier_factor = compute_metrics(y, neg_outlier_factor)

            result_dict["algorithm"].append("lof")
            result_dict["neighbors"].append(num_neighbors)

            for key, item in score_outlier_factor.items():
                if key in result_dict.keys():
                    result_dict[key].append(item)
                else:
                    result_dict[key] = [item]

        result_df = pl.DataFrame(result_dict)
    except Exception as e:
        logger.error(
            f"Error in run_experiement for dataset {dataset_name}: {str(e)}",
            exc_info=True,
        )

        # TODO: better handling of error for different metrics.
        empty_result = {"algorithm": [], "neighbors": [], "auc_score": []}

        result_df = pl.DataFrame(empty_result)

    return result_df


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics based on provided true labels and predicted labels.

    Args:
        y_true: List or array of true binary labels (0 or 1).
        y_pred: List or array of predicted probabilities or scores.

    Returns:
        dict: A dictionary containing the computed AUC score under the key
        'auc_score'.
    """
    return {"auc_score": roc_auc_score(y_true, y_pred)}


def _align_schemas_and_concat(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    """
    Aligns schemas of two DataFrames and concatenates them safely.

    Args:
        df1: First DataFrame
        df2: Second DataFrame to concatenate

    Returns:
        pl.DataFrame: Concatenated DataFrame with aligned schemas
    """
    # Get all column names from both DataFrames
    all_columns = set(df1.columns) | set(df2.columns)

    # Create a common schema by determining the "widest" type for each column
    common_schema = {}

    for col in all_columns:
        df1_dtype = df1.schema.get(col) if col in df1.columns else None
        df2_dtype = df2.schema.get(col) if col in df2.columns else None

        if df1_dtype is None:
            common_schema[col] = df2_dtype
        elif df2_dtype is None:
            common_schema[col] = df1_dtype
        else:
            # Choose the wider type between the two
            common_schema[col] = _get_wider_dtype(df1_dtype, df2_dtype)

    # Cast both DataFrames to the common schema
    df1_aligned = _cast_to_schema(df1, common_schema)
    df2_aligned = _cast_to_schema(df2, common_schema)

    return pl.concat([df1_aligned, df2_aligned])


def _get_wider_dtype(dtype1, dtype2):
    """
    Returns the wider of two dtypes to prevent data loss.
    """
    # Handle integer types
    if str(dtype1).startswith("Int") and str(dtype2).startswith("Int"):
        # Extract bit width and return the larger one
        width1 = int(str(dtype1)[3:]) if str(dtype1)[3:].isdigit() else 32
        width2 = int(str(dtype2)[3:]) if str(dtype2)[3:].isdigit() else 32
        return pl.Int64 if max(width1, width2) > 32 else pl.Int32

    # Handle float types
    if str(dtype1).startswith("Float") and str(dtype2).startswith("Float"):
        return pl.Float64  # Always use Float64 for safety

    # Handle string types
    if str(dtype1) == "String" or str(dtype2) == "String":
        return pl.String

    # If types are different categories, default to string
    if dtype1 != dtype2:
        return pl.String

    return dtype1


def _cast_to_schema(df: pl.DataFrame, target_schema: dict) -> pl.DataFrame:
    """
    Casts DataFrame columns to match the target schema.
    """
    cast_expressions = []

    for col_name, target_dtype in target_schema.items():
        if col_name in df.columns:
            current_dtype = df.schema[col_name]
            if current_dtype != target_dtype:
                cast_expressions.append(pl.col(col_name).cast(target_dtype))
            else:
                cast_expressions.append(pl.col(col_name))
        else:
            # Add missing column with null values
            cast_expressions.append(pl.lit(None).cast(target_dtype).alias(col_name))

    return df.select(cast_expressions)

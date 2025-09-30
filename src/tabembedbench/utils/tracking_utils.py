import logging
from pathlib import Path

import numpy as np
import polars as pl


RESULT_DF_SCHEMA = {
    "dataset_name": pl.Categorical,
    "dataset_size": pl.UInt64,
    "embedding_model": pl.Categorical,
    "num_neighbors": pl.UInt64,
    "auc_score": pl.Float64,
    "mse_score": pl.Float64,
    "time_to_compute_train_embeddings": pl.Float64,
    "distance_metric": pl.Categorical,
    "task": pl.Categorical,
    "algorithm": pl.Categorical,
    "algorithm_parameters": pl.String,
    "emb_dim": pl.UInt64,
    "prediction_time": pl.Float64,
    "time_to_compute_test_embeddings": pl.Float64,
}


def get_batch_dict_result_df():
    """
    Initializes and returns a dictionary for batch processing and an empty DataFrame
    with a predefined schema for storing results.

    The function creates a batch dictionary to store details such as dataset name,
    size, embedding model, evaluation metrics, and other related information. This
    dictionary facilitates batch processing by serving as a container for intermediate
    results. Additionally, it initializes an empty DataFrame with the specified schema
    to store final computed results.

    Returns:
        tuple: A tuple containing the batch dictionary and the empty result DataFrame.

    """
    batch_dict = {key: [] for key in RESULT_DF_SCHEMA.keys()}

    result_df = pl.DataFrame(
        schema=RESULT_DF_SCHEMA,
    )

    return batch_dict, result_df


def update_batch_dict(
    batch_dict: dict[str, list],
    dataset_name: str,
    dataset_size: int,
    embedding_model_name: str,
    num_neighbors: int,
    time_to_compute_train_embeddings: float,
    task: str,
    algorithm: str,
    auc_score: float = None,
    mse_score: float = None,
    distance_metric: str = "euclidean",
    embedding_dimension: int = None,
    prediction_time: float = None,
    time_to_compute_test_embeddings: float = None,
):
    """
    Updates the provided batch dictionary with results and parameters of an embedding
    benchmark task. The function appends relevant data corresponding to the input
    parameters to the lists in `batch_dict`. This function also handles conditional
    appending based on the presence of optional parameters like `auc_score`.

    Args:
        batch_dict (dict[str, list]): A dictionary containing lists to store
            benchmark results and parameters of the embedding task.
        dataset_name (str): The name of the dataset being processed.
        dataset_size (int): The size, in number of records, of the dataset.
        embedding_model_name (str): The name of the embedding model being evaluated.
        num_neighbors (int): The number of neighbors used in the benchmark.
        time_to_compute_train_embeddings (float): The time taken to compute the embeddings, in seconds.
        task (str): The type of task being performed during the benchmark run.
        algorithm (str): The algorithm used for the benchmark task.
        auc_score (float, optional): The AUC (Area Under Curve) score achieved
            during evaluation. Defaults to None.
        mse_score (float, optional): The MSR (Mean Square Residual) score achieved
            during evaluation. Defaults to None.
        distance_metric (str, optional): The distance metric used, such as
            "euclidean" or "cosine". Defaults to "euclidean".
        embedding_dimension (int, optional): The dimensionality of the embeddings.
            Defaults to None.
        prediction_time: (float, optional): The time taken to make predictions using
            the embeddings. Defaults to None.
        time_to_compute_test_embeddings (float, optional): The time taken to compute
            embeddings for the test dataset. Defaults to None.
    """
    batch_dict["dataset_name"].append(dataset_name)
    batch_dict["dataset_size"].append(dataset_size)
    batch_dict["embedding_model"].append(embedding_model_name)
    batch_dict["num_neighbors"].append(num_neighbors)
    batch_dict["time_to_compute_train_embeddings"].append(time_to_compute_train_embeddings)
    batch_dict["distance_metric"].append(distance_metric)
    batch_dict["task"].append(task)
    batch_dict["algorithm"].append(algorithm)
    batch_dict["emb_dim"].append(embedding_dimension)
    batch_dict["prediction_time"].append(prediction_time)
    batch_dict["time_to_compute_test_embeddings"].append(time_to_compute_test_embeddings)

    if auc_score is not None:
        batch_dict["auc_score"].append(auc_score)
        batch_dict["mse_score"].append(np.inf)
    else:
        batch_dict["auc_score"].append((-1) * np.inf)
        batch_dict["mse_score"].append(mse_score)

    return batch_dict


def update_result_df(
    batch_dict: dict[str, list],
    result_df: pl.DataFrame,
    logger: logging.Logger = None,
):
    """
    Updates the result dataframe with new batch data and resets the batch dictionary.

    This function appends the data from `batch_dict` to the `result_df`, provided that
    the "dataset_name" key in the `batch_dict` contains valid data. It then resets the
    `batch_dict` to an empty state for further data accumulation. If a logger is
    provided, debug messages are recorded to aid in tracking the update process.

    Args:
        batch_dict (dict[str, list]): A dictionary containing batch data to be added
            to the result dataframe. Each key must match the schema defined in
            `RESULT_DF_SCHEMA`.
        result_df (pl.DataFrame): The result dataframe to which the batch data will be
            appended.
        logger (logging.Logger, optional): A logger object for recording debug messages
            during execution. Defaults to None.

    Returns:
        tuple[dict[str, list], pl.DataFrame]: A tuple containing the reset batch
        dictionary and the updated result dataframe.
    """
    logger.debug("Updating result dataframe...")
    if not batch_dict["dataset_name"]:
        return batch_dict, result_df

    batch_df = pl.from_dict(
        batch_dict,
        schema=RESULT_DF_SCHEMA,
    )

    result_df = pl.concat([result_df, batch_df], how="vertical")

    result_df = result_df.unique(maintain_order=True)

    batch_dict = {key: [] for key in RESULT_DF_SCHEMA.keys()}

    return batch_dict, result_df


def save_result_df(
        result_df: pl.DataFrame,
        output_path: str | Path,
        benchmark_name:str,
        timestamp: str,
):
    """
    Saves the provided Polars DataFrame to both Parquet and CSV formats in the
    specified output directory. The output file names are dynamically generated
    using the provided benchmark name and timestamp.

    Args:
        result_df (pl.DataFrame): The Polars DataFrame to be saved.
        output_path (str | Path): The directory path where the results
            will be saved.
        benchmark_name (str): A name identifier for the benchmark, used
            to generate the output file name.
        timestamp (str): A timestamp identifier to include in the output
            file name.
    """
    output_path = Path(output_path)

    output_file = Path(output_path / f"results_{benchmark_name}_{timestamp}")

    parquet_file = output_file.with_suffix(".parquet")
    csv_file = output_file.with_suffix(".csv")

    result_df.write_parquet(parquet_file)
    result_df.write_csv(csv_file)

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
    "msr_score": pl.Float64,
    "time_to_compute_embeddings": pl.Float64,
    "benchmark": pl.Categorical,
    "distance_metric": pl.Categorical,
    "task": pl.Categorical,
    "algorithm": pl.Categorical,
}

EMPTY_BATCH_DICT = {
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
        "algorithm": [],
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
    batch_dict = EMPTY_BATCH_DICT

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
    compute_time: float,
    task: str,
    algorithm: str,
    auc_score: float = None,
    msr_score: float = None,
    distance_metric: str = "euclidean",
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
        compute_time (float): The time taken to compute the embeddings, in seconds.
        task (str): The type of task being performed during the benchmark run.
        algorithm (str): The algorithm used for the benchmark task.
        auc_score (float, optional): The AUC (Area Under Curve) score achieved
            during evaluation. Defaults to None.
        msr_score (float, optional): The MSR (Mean Square Residual) score achieved
            during evaluation. Defaults to None.
        distance_metric (str, optional): The distance metric used, such as
            "euclidean" or "cosine". Defaults to "euclidean".
    """
    batch_dict["dataset_name"].append(dataset_name)
    batch_dict["dataset_size"].append(dataset_size)
    batch_dict["embedding_model"].append(embedding_model_name)
    batch_dict["num_neighbors"].append(num_neighbors)
    batch_dict["time_to_compute_embeddings"].append(compute_time)
    batch_dict["benchmark"].append("tabarena")
    batch_dict["distance_metric"].append(distance_metric)
    batch_dict["task"].append(task)
    batch_dict["algorithm"].append(algorithm)

    if auc_score is not None:
        batch_dict["auc_score"].append(auc_score)
        batch_dict["msr_score"].append(np.inf)
    else:
        batch_dict["auc_score"].append((-1) * np.inf)
        batch_dict["msr_score"].append(msr_score)


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

    batch_dict = EMPTY_BATCH_DICT

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

    if parquet_file.exists():
        parquet_file.unlink()
    if csv_file.exists():
        csv_file.unlink()

    result_df.write_parquet(parquet_file)
    result_df.write_csv(csv_file)


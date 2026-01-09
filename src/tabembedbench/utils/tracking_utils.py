import logging
import os
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl
import torch


def save_dataframe(
    dataframe: pl.DataFrame,
    output_path: str | Path,
    dataframe_name: str,
    timestamp: str,
    save_to_gcs: bool = False,
    bucket_name: str = None,
):
    """
    Saves the provided Polars DataFrame to both Parquet and CSV formats in the
    specified output directory. The output file names are dynamically generated
    using the provided benchmark name and timestamp.

    Args:
        dataframe (pl.DataFrame): The Polars DataFrame to be saved.
        output_path (str | Path): The directory path where the results
            will be saved.
        dataframe_name (str): A name identifier for the benchmark, used
            to generate the output file name.
        timestamp (str): A timestamp identifier to include in the output
            file name.
    """
    if save_to_gcs:
        save_to_google_cloud_storage(
            dataframe=dataframe,
            output_path=output_path,
            bucket_name=bucket_name,
            dataframe_name=dataframe_name,
            timestamp=timestamp,
        )
        return None
    output_path = Path(output_path)

    output_file = Path(output_path / f"{dataframe_name}_{timestamp}")

    parquet_file = output_file.with_suffix(".parquet")
    csv_file = output_file.with_suffix(".csv")

    dataframe.write_parquet(parquet_file)
    dataframe.write_csv(csv_file)


def save_to_google_cloud_storage(
    dataframe: pl.DataFrame,
    output_path: str | Path,
    bucket_name: str,
    dataframe_name: str,
    timestamp: str,
) -> None:
    """
    Saves the given DataFrame to Google Cloud Storage in both Parquet and CSV formats.

    This function writes the provided `result_df` as a Parquet file and a CSV file
    to the specified Google Cloud Storage bucket and path. If the bucket name is
    not explicitly provided, it attempts to retrieve the bucket name from the
    environment variable `GCS_BUCKET_NAME`. Any errors during the process will be
    logged.

    Args:
        dataframe: A Polars DataFrame to be written to Google Cloud Storage.
        output_path: The relative directory path in the GCS bucket where the files
            will be stored.
        bucket_name: The name of the Google Cloud Storage bucket. If not provided,
            it will default to the value of the environment variable
            `GCS_BUCKET_NAME`.
        dataframe_name: A unique identifier to include in the file names.
        timestamp: A timestamp string to append to the file names.

    """
    try:
        if isinstance(output_path, Path):
            output_path = str(output_path)

        bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")

        if not bucket_name:
            logging.error("No GCS bucket name provided.")
            return

        gcs_path = (
            f"gs://{bucket_name}/{output_path}/result_{dataframe_name}_{timestamp}"
        )

        dataframe.write_parquet(gcs_path + ".parquet")
        dataframe.write_csv(gcs_path + ".csv")
    except Exception as e:
        logging.error(e)


class MemoryTracker:
    """Tracks CPU and GPU memory usage of embedding models on datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._start_memory = 0
        self._peak_memory = 0
        self._start_gpu_memory = {}

    def start_tracking(self) -> None:
        """
        Starts tracking memory usage for both CPU and GPU.

        This method initializes memory tracking for the current process. On the CPU side, it uses
        the `tracemalloc` Python module to capture the initial memory snapshot. For GPU memory,
        it resets the peak memory statistics of all available CUDA devices and records the
        initial amount of memory allocated per GPU.

        Raises:
            RuntimeError: If GPU memory tracking is attempted but CUDA is not available.

        """
        tracemalloc.start()
        self._start_memory = tracemalloc.get_traced_memory()[0]

        # GPU memory tracking (if available)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            for i in range(torch.cuda.device_count()):
                self._start_gpu_memory[i] = torch.cuda.memory_allocated(i)

    def stop_tracking(self) -> Dict[str, float]:
        """
        Stops resource tracking and collects memory statistics for both CPU and GPU.

        This method calculates the memory usage and peak memory details during the tracking
        period and returns a dictionary of the collected statistics. GPU statistics are only
        collected if CUDA is available, otherwise the method notes that GPU tracking is not available.

        Returns:
            Dict[str, float]: A dictionary containing the following keys:
                - "cpu_memory_used_mb": The memory used by the CPU in MB during the tracking period.
                - "cpu_peak_memory_mb": The peak memory used by the CPU in MB during the tracking period.
                - "gpu_X_memory_used_mb": The memory used by each GPU X in MB during the tracking period
                  (one for each available device).
                - "gpu_X_max_allocated_mb": The maximum memory allocated for each GPU X in MB at any point
                  during the tracking period.
                - "gpu_X_reserved_mb": The reserved memory for each GPU X in MB during the tracking period.
                - "gpu_available": A boolean value indicating whether GPU is available.
        """
        stats = {}

        # CPU memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used = (current - self._start_memory) / 1024 / 1024  # MB
        peak_memory = peak / 1024 / 1024  # MB

        stats["cpu_memory_used_mb"] = memory_used
        stats["cpu_peak_memory_mb"] = peak_memory

        # GPU memory stats (if available)
        if torch.cuda.is_available():
            total_gpu_used = 0
            total_gpu_peak = 0

            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                start = self._start_gpu_memory.get(i, 0) / 1024 / 1024  # MB
                gpu_used = allocated - start

                max_allocated = torch.cuda.max_memory_allocated(i) / 1024 / 1024  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB

                stats[f"gpu_{i}_memory_used_mb"] = gpu_used
                stats[f"gpu_{i}_max_allocated_mb"] = max_allocated
                stats[f"gpu_{i}_reserved_mb"] = reserved

                total_gpu_used += gpu_used
                # For peak, we take the MAX peak across all cards (usually the limiting factor)
                total_gpu_peak = max(total_gpu_peak, max_allocated)

            stats["gpu_total_used_mb"] = total_gpu_used
            stats["gpu_max_peak_single_card_mb"] = total_gpu_peak
        else:
            stats["gpu_available"] = False

        return stats

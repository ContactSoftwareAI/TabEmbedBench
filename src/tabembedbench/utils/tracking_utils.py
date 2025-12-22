import logging
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl
import torch


def save_result_df(
    result_df: pl.DataFrame,
    output_path: str | Path,
    benchmark_name: str,
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


class MemoryTracker:
    """Tracks CPU and GPU memory usage of embedding models on datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._start_memory = 0
        self._peak_memory = 0
        self._start_gpu_memory = {}

    def start_tracking(self) -> None:
        """Start CPU and GPU memory tracking."""
        # CPU memory tracking
        tracemalloc.start()
        self._start_memory = tracemalloc.get_traced_memory()[0]

        # GPU memory tracking (if available)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            for i in range(torch.cuda.device_count()):
                self._start_gpu_memory[i] = torch.cuda.memory_allocated(i)

    def stop_tracking(self) -> Dict[str, float]:
        """Stop tracking and return CPU and GPU memory stats."""
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
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                start = self._start_gpu_memory.get(i, 0) / 1024 / 1024  # MB
                gpu_used = allocated - start

                max_allocated = torch.cuda.max_memory_allocated(i) / 1024 / 1024  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB

                stats[f"gpu_{i}_memory_used_mb"] = gpu_used
                stats[f"gpu_{i}_max_allocated_mb"] = max_allocated
                stats[f"gpu_{i}_reserved_mb"] = reserved
        else:
            stats["gpu_available"] = False

        return stats

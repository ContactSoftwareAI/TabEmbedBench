from pathlib import Path

import polars as pl


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

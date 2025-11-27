import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from itertools import cycle

logger = logging.getLogger("Results processing ")


def setup_publication_style():
    """Set up matplotlib/seaborn for publication-quality figures.

    Configures matplotlib and seaborn settings to produce publication-ready
    visualizations with appropriate font sizes, styles, and context.
    """
    sns.set_context("paper")
    sns.set_style("whitegrid")

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def separate_by_task_type(df: pl.DataFrame):
    """Separate a DataFrame into binary, multiclass, and regression task subsets.

    Args:
        df: A Polars DataFrame containing results with 'classification_type'
            and 'task' columns.

    Returns:
        tuple: A tuple containing three Polars DataFrames:
            - binary_results_df: Results for binary classification tasks
            - multiclass_results_df: Results for multiclass classification tasks
            - regression_results_df: Results for regression tasks
    """
    binary_results_df = df.filter(pl.col("classification_type") == "binary")

    multiclass_results_df = df.filter(pl.col("classification_type") == "multiclass")

    regression_results_df = df.filter(pl.col("task") == "regression")

    return binary_results_df, multiclass_results_df, regression_results_df


def create_color_mapping(models_to_keep: list[str]):
    colors = [
        "#0080C5",
        "#FCB900",
        "#92C108",
        "#E3075A",
        "#36AEE7",
        "#F29100",
        "#DBDC2E",
        "#EE5CA1",
        "#005E9E",
        "#EA5A02",
        "#639C2E",
        "#A90E4E",
        "#003254",
    ]
    color_mapping = dict(zip(models_to_keep, cycle(colors)))
    return color_mapping


def keeping_models(df: pl.DataFrame, keep_models: list[str]):
    """Filter DataFrame to keep only specified embedding models.

    Args:
        df: A Polars DataFrame containing an 'embedding_model' column.
        keep_models: List of embedding model names to retain.

    Returns:
        pl.DataFrame: Filtered DataFrame containing only rows with embedding
            models in the keep_models list.
    """
    return df.filter(pl.col("embedding_model").is_in(keep_models))


def rename_models(df: pl.DataFrame, renaming_mapping: dict):
    """Rename embedding models according to a mapping dictionary.

    Args:
        df: A Polars DataFrame containing an 'embedding_model' column.
        renaming_mapping: Dictionary mapping old model names to new names.

    Returns:
        pl.DataFrame: DataFrame with renamed embedding models.
    """
    return df.with_columns(pl.col("embedding_model").replace(renaming_mapping))


def save_fig(
    ax,
    data_path: str,
    file_name: str,
):
    """Save a matplotlib figure to a PDF file.

    Creates the output directory if it doesn't exist and saves the figure
    with publication-quality settings.

    Args:
        ax: Matplotlib axes object containing the plot to save.
        data_path: Directory path where the figure will be saved.
        file_name: Name of the output file (without extension).
    """
    setup_publication_style()
    fig = ax.get_figure()

    data_path = Path(data_path)

    if not data_path.exists():
        data_path.mkdir(parents=True)

    file_name = Path(data_path / f"{file_name}.pdf")

    fig.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )


def clean_results(df: pl.DataFrame):
    """Clean results by filtering datasets with inconsistent algorithm coverage.

    Ensures that all embedding models have been evaluated with the same set
    of algorithms for each dataset. Datasets with inconsistent algorithm
    coverage across embedding models are filtered out and a warning is logged.

    Args:
        df: A Polars DataFrame containing benchmark results with columns:
            'dataset_name', 'embedding_model', and 'algorithm'.

    Returns:
        pl.DataFrame: Filtered DataFrame containing only datasets where all
            embedding models were evaluated with the same algorithms.
    """
    set_included_dataset = set(df.get_column("dataset_name").to_list())
    unique_counts = df.group_by(["dataset_name", "embedding_model"]).agg(
        pl.col("algorithm").n_unique().alias("num_unique_algorithms")
    )

    agg_counts = (
        unique_counts.group_by("dataset_name")
        .agg(
            [
                pl.col("num_unique_algorithms").min().alias("min_unique_algorithms"),
                pl.col("num_unique_algorithms").max().alias("max_unique_algorithms"),
            ]
        )
        .with_columns(
            (pl.col("max_unique_algorithms") == pl.col("min_unique_algorithms")).alias(
                "equal"
            )
        )
    )

    valid_datasets = (
        agg_counts.filter(pl.col("equal"))
        .unique("dataset_name")
        .get_column("dataset_name")
        .to_list()
    )

    set_valid_datasets = set(valid_datasets)

    if len(set_valid_datasets) != len(set_included_dataset):
        logger.warning(
            f"The following dataset were not evaluated on all "
            f"evaluators for each embedding model: "
            f"{set_included_dataset - set_valid_datasets}."
            f" - Consider the logs to check for any errors."
        )

    return df.filter(pl.col("dataset_name").is_in(valid_datasets))


def create_descriptive_dataframe(
    df: pl.DataFrame,
    metric_col: str,
):
    """
    Computes a descriptive statistics dataframe grouped by specific features.

    This function takes a dataframe and calculates descriptive statistics for the provided
    metric column, grouped by 'embedding_model' and 'algorithm'. Additionally, it computes
    the average time to compute training embeddings and provides the number of unique datasets.

    Args:
        df (pl.DataFrame): Input dataframe containing the required columns for computation.
        metric_col (str): The column in the dataframe whose descriptive statistics are
            to be calculated.

    Returns:
        pl.DataFrame: A dataframe containing grouped descriptive statistics for the specified
            metric column and additional computed statistics.
    """
    descriptive_statistic_df = df.group_by(["embedding_model", "algorithm"]).agg(
        pl.col(metric_col).mean().alias(f"average_{metric_col}"),
        pl.col(metric_col).std().alias(f"std_{metric_col}"),
        pl.col(metric_col).min().alias(f"min_{metric_col}"),
        pl.col(metric_col).max().alias(f"max_{metric_col}"),
        pl.col(metric_col).median().alias(f"median_{metric_col}"),
        pl.col("time_to_compute_embedding")
        .mean()
        .alias(f"averaged_embedding_compute_time"),
        pl.col("dataset_name").n_unique().alias("num_datasets"),
    )

    return descriptive_statistic_df


def create_outlier_ratio_dataframe(
    df: pl.DataFrame,
    bin_edges: list,
    metric_col: str,
):
    binning_expr = pl.lit(None, dtype=pl.Int64)
    for i in range(len(bin_edges), -1, -1):
        if i == len(bin_edges):
            lower_bound = bin_edges[i - 1]
            condition = pl.col("outlier_ratio") >= lower_bound
            bin_label = f">={lower_bound}"
        elif i == 0:
            upper_bound = bin_edges[i]
            condition = pl.col("outlier_ratio") < upper_bound
            bin_label = f"<{upper_bound}"
        else:
            lower_bound = bin_edges[i - 1]
            upper_bound = bin_edges[i]
            condition = (pl.col("outlier_ratio") >= lower_bound) & (
                pl.col("outlier_ratio") < upper_bound
            )
            bin_label = f"[{lower_bound},{upper_bound})"
        binning_expr = (
            pl.when(condition).then(pl.lit(bin_label)).otherwise(binning_expr)
        )
    temp_df = df.with_columns(binning_expr.alias("outlier_ratio_bin_based"))
    descriptive_statistic_df = temp_df.group_by(
        ["embedding_model", "outlier_ratio_bin_based"]
    ).agg(
        pl.col(metric_col).mean().alias(f"average_{metric_col}"),
        pl.col(metric_col).std().alias(f"std_{metric_col}"),
        pl.col(metric_col).min().alias(f"min_{metric_col}"),
        pl.col(metric_col).max().alias(f"max_{metric_col}"),
        pl.col(metric_col).median().alias(f"median_{metric_col}"),
        pl.col("time_to_compute_embedding")
        .mean()
        .alias(f"averaged_embedding_compute_time"),
        pl.col("dataset_name").n_unique().alias("num_datasets"),
    )

    return descriptive_statistic_df


def create_outlier_plots(
    df: pl.DataFrame,
    data_path: str | Path = "data",
    name_mapping: dict | None = None,
    color_mapping: dict | None = None,
    models_to_keep: list | None = None,
    algorithm_order: list | None = None,
    bin_edges: list | None = None,
):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    outlier_path = data_path / "outlier_plots"
    outlier_path.mkdir(parents=True, exist_ok=True)

    if name_mapping is not None:
        df = rename_models(df, name_mapping)
        df = keeping_models(df, models_to_keep)

    if models_to_keep is None:
        models_to_keep = df.get_column("embedding_model").unique().to_list()

    if color_mapping is None:
        color_mapping = create_color_mapping(models_to_keep)

    setup_publication_style()

    if "algorithm_metric" in df.columns:
        df = df.filter(
            (pl.col("algorithm_metric") == "euclidean")
            | pl.col("algorithm_metric").is_null()
        )
        grouped_columns = [
            "algorithm",
            "embedding_model",
            "algorithm_metric",
            "dataset_name",
            "outlier_ratio",
            "time_to_compute_embedding",
        ]
    else:
        grouped_columns = [
            "algorithm",
            "embedding_model",
            "dataset_name",
            "outlier_ratio",
            "time_to_compute_embedding",
        ]

    agg_result = df.group_by(grouped_columns).agg(
        pl.col("auc_score").max().alias("auc_score"),
    )

    agg_result = clean_results(agg_result)

    descriptive_df = create_descriptive_dataframe(agg_result, "auc_score")
    descriptive_df.write_csv(Path(outlier_path / "outlier_descriptive.csv"))

    if bin_edges:
        outlier_ratio_df = create_outlier_ratio_dataframe(
            agg_result, bin_edges, "auc_score"
        )
        outlier_ratio_df.write_csv(Path(outlier_path / "outlier_ratio.csv"))

    boxplot = sns.boxplot(
        data=agg_result,
        x="algorithm",
        y="auc_score",
        hue="embedding_model",
        palette=color_mapping,
        hue_order=models_to_keep,
        order=algorithm_order,
    )

    boxplot.set_xlabel("")
    boxplot.set_ylabel("AUC Score")
    boxplot.legend(title="Embedding Models")
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha="right")

    save_fig(boxplot, outlier_path, "outlier_algorithm_comparison")
    plt.close()


def create_tabarena_plots(
    df: pl.DataFrame,
    data_path: str | Path = "data",
    name_mapping: dict | None = None,
    color_mapping: dict | None = None,
    models_to_keep: list | None = None,
    algorithm_order_classification: list | None = None,
    algorithm_order_regression: list | None = None,
) -> None:
    """
    Creates and saves a series of visualizations and descriptive statistics for different
    machine learning tasks and algorithms, including binary classification, multiclass
    classification, and regression. The function processes the given data, applies filtering
    and transformations, and generates boxplots and descriptive statistics for evaluation.

    Args:
        df (pl.DataFrame): Input DataFrame containing the results to process and visualize.
            The DataFrame is expected to include relevant columns for models, algorithms,
            metrics, and scores.
        data_path (str | Path, optional): Path to the directory where the plots and statistics
            files will be saved. Defaults to "data".
        name_mapping (dict | None, optional): Dictionary mapping original model names to
            new names for visualization purposes. Supports renaming models for better clarity in
            plots. Defaults to None.
        color_mapping (dict | None, optional): Dictionary mapping model names to specific colors
            for visualization. If None, colors will be assigned automatically using a colorblind
            palette. Defaults to None.
        models_to_keep (list | None, optional): List of model names to keep in the dataset for
            plotting. Filters datasets to include only specified models. Defaults to None.
        algorithm_order_classification (list | None, optional): Order of algorithms to display
            on the x-axis for binary and multiclass classification plots. Adjusts plot appearance
            for consistent ordering. Defaults to None.
        algorithm_order_regression (list | None, optional): Order of algorithms to display on
            the x-axis for regression plots. Adjusts plot appearance for consistent ordering.
            Defaults to None.
    """
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    tabarena_path = data_path / "tabarena_plots"
    tabarena_path.mkdir(parents=True, exist_ok=True)

    if name_mapping is not None:
        df = rename_models(df, name_mapping)
        df = keeping_models(df, models_to_keep)

    if models_to_keep is None:
        models_to_keep = df.get_column("embedding_model").unique().to_list()

    if color_mapping is None:
        color_mapping = create_color_mapping(models_to_keep)

    # Filter "euclidean" metric and "distance" weight
    if "algorithm_metric" in df.columns and "algorithm_weights" in df.columns:
        df = df.filter(
            (
                (pl.col("algorithm_metric") == "euclidean")
                | pl.col("algorithm_metric").is_null()
            )
            & (
                (pl.col("algorithm_weights") == "distance")
                | pl.col("algorithm_weights").is_null()
            )
        )
        grouped_columns = [
            "algorithm",
            "embedding_model",
            "algorithm_metric",
            "dataset_name",
            "time_to_compute_embedding",
        ]
    else:
        grouped_columns = [
            "algorithm",
            "embedding_model",
            "dataset_name",
            "time_to_compute_embedding",
        ]

    binary, multiclass, regression = separate_by_task_type(df)

    # Clean results for each task type
    binary = clean_results(binary)
    multiclass = clean_results(multiclass)
    regression = clean_results(regression)

    setup_publication_style()

    # Boxplot for binary classification
    binary_agg_result = binary.group_by(grouped_columns).agg(
        pl.col("auc_score").max().alias("auc_score"),
    )

    boxplot = sns.boxplot(
        data=binary_agg_result,
        x="algorithm",
        y="auc_score",
        hue="embedding_model",
        palette=color_mapping,
        hue_order=models_to_keep,
        order=algorithm_order_classification,
    )

    boxplot.set_xlabel("")
    boxplot.set_ylabel("AUC Score")
    boxplot.legend(title="Embedding Models")
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha="right")

    save_fig(boxplot, tabarena_path, "binary_clf_algorithm_comparison")
    plt.close()

    # Boxplot for multiclass classification
    multiclass_agg_result = multiclass.group_by(grouped_columns).agg(
        pl.col("auc_score").max().alias("auc_score"),
    )

    boxplot = sns.boxplot(
        data=multiclass_agg_result,
        x="algorithm",
        y="auc_score",
        hue="embedding_model",
        palette=color_mapping,
        hue_order=models_to_keep,
        order=algorithm_order_classification,
    )

    boxplot.set_xlabel("")
    boxplot.set_ylabel("AUC Score")
    boxplot.legend(title="Embedding Models")
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha="right")

    save_fig(boxplot, tabarena_path, "multiclass_clf_algorithm_comparison")
    plt.close()

    # Boxplot for regression
    regression_agg_result = regression.group_by(grouped_columns).agg(
        pl.col("mape_score").max().alias("mape_score"),
    )

    boxplot = sns.boxplot(
        data=regression_agg_result,
        x="algorithm",
        y="mape_score",
        hue="embedding_model",
        palette=color_mapping,
        hue_order=models_to_keep,
        order=algorithm_order_regression,
    )

    boxplot.set_xlabel("")
    boxplot.set_ylabel("MAPE Score")
    boxplot.legend(title="Embedding Models")
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha="right")

    # Speichere den kombinierten Plot
    save_fig(boxplot, tabarena_path, "regression_algorithm_comparison")
    plt.close()

    logger.info("All visualizations completed successfully")

    # Save descriptive statistics
    descriptive_binary_df = create_descriptive_dataframe(binary_agg_result, "auc_score")
    descriptive_binary_df.write_csv(
        Path(tabarena_path / "binary_auc_score_descriptive.csv")
    )
    descriptive_multiclass_df = create_descriptive_dataframe(
        multiclass_agg_result, "auc_score"
    )
    descriptive_multiclass_df.write_csv(
        Path(tabarena_path / "multiclass_auc_score_descriptive.csv")
    )
    descriptive_regression_df = create_descriptive_dataframe(
        regression_agg_result, "mape_score"
    )
    descriptive_regression_df.write_csv(
        Path(tabarena_path / "regression_auc_score_descriptive.csv")
    )

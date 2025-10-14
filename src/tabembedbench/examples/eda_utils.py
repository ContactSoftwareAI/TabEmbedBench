import logging
from pathlib import Path

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

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
    binary_results_df = (
        df.filter(
            pl.col("classification_type") == "binary"
        )
    )

    multiclass_results_df = (
        df.filter(
            pl.col("classification_type") == "multiclass"
        )
    )

    regression_results_df = (
        df.filter(
            pl.col("task") == "regression"
        )
    )

    return binary_results_df, multiclass_results_df, regression_results_df

def keeping_models(df: pl.DataFrame, keep_models: list[str]):
    """Filter DataFrame to keep only specified embedding models.
    
    Args:
        df: A Polars DataFrame containing an 'embedding_model' column.
        keep_models: List of embedding model names to retain.
    
    Returns:
        pl.DataFrame: Filtered DataFrame containing only rows with embedding
            models in the keep_models list.
    """
    return df.filter(
        pl.col("embedding_model").is_in(keep_models)
    )

def rename_models(df: pl.DataFrame, renaming_mapping: dict):
    """Rename embedding models according to a mapping dictionary.
    
    Args:
        df: A Polars DataFrame containing an 'embedding_model' column.
        renaming_mapping: Dictionary mapping old model names to new names.
    
    Returns:
        pl.DataFrame: DataFrame with renamed embedding models.
    """
    return df.with_columns(
        pl.col("embedding_model").replace(renaming_mapping)
    )

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
    unique_counts = (
        df.group_by(
            ["dataset_name", "embedding_model"]
        ).agg(
            pl.col("algorithm").n_unique().alias("num_unique_algorithms")
        )
    )

    agg_counts = (
        unique_counts.group_by("dataset_name")
        .agg(
            [
                pl.col("num_unique_algorithms").min().alias("min_unique_algorithms"),
                pl.col("num_unique_algorithms").max().alias(
                    "max_unique_algorithms"),
            ]
        )
        .with_columns(
            (pl.col("max_unique_algorithms") == pl.col(
                "min_unique_algorithms")).alias("equal")
        )
    )

    valid_datasets = (
        agg_counts.filter(pl.col("equal"))
        .unique("dataset_name")
        .get_column("dataset_name").to_list()
    )

    set_valid_datasets = set(valid_datasets)

    if len(set_valid_datasets) != len(set_included_dataset):
        logger.warning(
            f"The following dataset were not evaluated on all "
            f"evaluators for each embedding model: "
            f"{set_included_dataset - set_valid_datasets}."
            f" - Consider the logs to check for any errors."
        )

    return df.filter(
        pl.col("dataset_name").is_in(valid_datasets)
    )

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
    descriptive_statistic_df = (
        df.group_by(["embedding_model", "algorithm"])
        .agg(
            pl.col(metric_col).mean().alias(f"average_{metric_col}"),
            pl.col(metric_col).std().alias(f"std_{metric_col}"),
            pl.col(metric_col).min().alias(f"min_{metric_col}"),
            pl.col(metric_col).max().alias(f"max_{metric_col}"),
            pl.col(metric_col).median().alias(f"median_{metric_col}"),
            pl.col("time_to_compute_train_embedding").mean().alias(
                f"average_embedding_compute_time_training"),
            pl.col("dataset_name").n_unique().alias("num_datasets"),
        )
    )

    return descriptive_statistic_df

def create_detailed_boxplots(
        df: pl.DataFrame,
        metric_col: str,
        metric_name: str,
        algorithm: str | None = None,
        title: str | None = None,
        embedding_model_palette: dict[str, str] | None = None,
        agg_algorithm: str = "maximize",
        embedding_model_order: list | None = None
):
    """Create boxplots comparing embedding model performance.
    
    Generates boxplots showing the distribution of a performance metric across
    different embedding models. Optionally filters by algorithm and aggregates
    results per dataset.
    
    Args:
        df: A Polars DataFrame containing benchmark results.
        metric_col: Name of the column containing the metric to plot.
        metric_name: Display name for the metric (used in y-axis label).
        algorithm: Optional algorithm name to filter results. If provided,
            results are aggregated by dataset. Defaults to None.
        title: Optional title prefix for the plot. Defaults to None.
        embedding_model_palette: Optional dictionary mapping model names to
            colors. Defaults to None.
        agg_algorithm: Aggregation method when filtering by algorithm.
            Either 'maximize' or 'minimize'. Defaults to 'maximize'.
        embedding_model_order: Optional list specifying the order of models
            on the x-axis. Defaults to None.
    
    Returns:
        matplotlib.axes.Axes: The boxplot axes object.
    
    Raises:
        ValueError: If agg_algorithm is not 'maximize' or 'minimize'.
    """
    if algorithm:
        df = df.filter(pl.col("algorithm") == algorithm)
        plot_title = f"{title} ({metric_name}, {algorithm})"
        if agg_algorithm == "maximize":
            df = df.group_by(["embedding_model", "dataset_name", "algorithm"]).agg(
                pl.col(metric_col).max()
            )
        elif agg_algorithm == "minimize":
            df = df.group_by(["embedding_model", "dataset_name", "algorithm"]).agg(
                pl.col(metric_col).min()
            )
        else:
            raise ValueError("agg_algorithm must be 'maximize' or 'minimize'.")
    else:
        plot_title = f"{title} ({metric_name})"

    if title is None:
        plot_title = None

    boxplot = sns.boxplot(
        data=df,
        x="embedding_model",
        y=metric_col,
        hue="embedding_model",
        palette=embedding_model_palette,
        order=embedding_model_order
    )

    boxplot.set_xlabel("Embedding Model")

    if metric_name:
        boxplot.set_ylabel(metric_name)

    if title:
        boxplot.set_title(plot_title)

    plt.setp(boxplot.get_xticklabels(), rotation=45, ha='right')

    return boxplot

def create_dataset_performance_heatmaps(
        df: pl.DataFrame,
        metric_col: str,
        metric_name: str | None = None,
        algorithm: str | None = None,
        title: str | None = None,
        agg_algorithm: str = "maximize",
        embedding_model_order: list | None = None
):
    """Create heatmaps showing performance across datasets and models.
    
    Generates a heatmap with datasets on the y-axis and embedding models on
    the x-axis, with cell colors representing performance metric values.
    
    Args:
        df: A Polars DataFrame containing benchmark results.
        metric_col: Name of the column containing the metric to plot.
        metric_name: Optional display name for the metric. Defaults to None.
        algorithm: Optional algorithm name to filter results. If provided,
            results are aggregated by dataset. Defaults to None.
        title: Optional title prefix for the plot. Defaults to None.
        agg_algorithm: Aggregation method when filtering by algorithm.
            Either 'maximize' or 'minimize'. Defaults to 'maximize'.
        embedding_model_order: Optional list specifying the order of models
            on the x-axis. Defaults to None.
    
    Returns:
        matplotlib.axes.Axes: The heatmap axes object.
    
    Raises:
        ValueError: If agg_algorithm is not 'maximize' or 'minimize'.
    """
    filtered_df = df

    if algorithm:
        filtered_df = df.filter((pl.col("algorithm") == algorithm))
        plot_title = f"{title} ({metric_name}, {algorithm})"
        if agg_algorithm == "maximize":
            filtered_df = filtered_df.group_by(
                ["embedding_model", "dataset_name", "algorithm"]
            ).agg(pl.col(metric_col).max())
        elif agg_algorithm == "minimize":
            filtered_df = filtered_df.group_by(
                ["embedding_model", "dataset_name", "algorithm"]
            ).agg(pl.col(metric_col).min())
        else:
            raise ValueError("agg_algorithm must be 'maximize' or 'minimize'.")
    else:
        plot_title = f"{title} ({metric_name})"

    pivoted_df = filtered_df.pivot(
        on="embedding_model",
        index="dataset_name",
        values=metric_col
    )

    if title is None:
        plot_title = None

    pivoted_df = pivoted_df.select(["dataset_name"] + embedding_model_order)
    heatmap = sns.heatmap(
        data=pivoted_df.to_pandas().set_index("dataset_name"),
        cmap='viridis',
    )

    heatmap.set_xlabel("Embedding Model")
    heatmap.set_ylabel("Dataset")

    if title:
        heatmap.set_title(plot_title)

    plt.setp(heatmap.get_yticklabels(), rotation=0, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return heatmap

def create_neighbors_effect_analysis(
    df: pl.DataFrame,
    metric_col: str,
    metric_name: str,
    num_neighbors_col: str = "algorithm_num_neighbors",
    algorithm: str | None = None,
    title: str | None = None,
    embedding_model_palette: dict[str, str] | None = None,
    embedding_model_order: list | None = None
):
    """Create line plots analyzing the effect of number of neighbors on performance.
    
    Generates line plots showing how performance metrics vary with the number
    of neighbors (K) for different embedding models. Useful for analyzing
    K-nearest neighbor algorithms.
    
    Args:
        df: A Polars DataFrame containing benchmark results.
        metric_col: Name of the column containing the metric to plot.
        metric_name: Display name for the metric (used in y-axis label).
        num_neighbors_col: Name of the column containing the number of
            neighbors. Defaults to 'algorithm_num_neighbors'.
        algorithm: Optional algorithm name to filter results. Defaults to None.
        title: Optional title prefix for the plot. Defaults to None.
        embedding_model_palette: Optional dictionary mapping model names to
            colors. Defaults to None.
        embedding_model_order: Optional list specifying the order of models
            in the legend. Defaults to None.
    
    Returns:
        matplotlib.axes.Axes: The line plot axes object.
    
    Raises:
        ValueError: If the num_neighbors_col contains no non-null values.
    """
    filtered_df = df
    if algorithm:
        filtered_df = df.filter(pl.col("algorithm") == algorithm)
        plot_title = f"{title} ({metric_name}, {algorithm})"
    else:
        plot_title = f"{title} ({metric_name})"

    non_null_count = filtered_df.select(
        pl.col(num_neighbors_col).is_not_null().sum()
    ).item()
    if non_null_count == 0:
        raise ValueError(
            f"Column '{num_neighbors_col}' exists but contains no non-null values"
        )

    lineplot = sns.lineplot(
        data=filtered_df,
        x=num_neighbors_col,
        y=metric_col,
        hue="embedding_model",
        palette=embedding_model_palette,
        hue_order=embedding_model_order,
        errorbar=None
    )

    lineplot.set_xlabel("Number of Neighbors (K)")

    if metric_name:
        lineplot.set_ylabel(metric_name)

    if title:
        lineplot.set_title(plot_title)

    lineplot.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    return lineplot

def create_critical_difference_plot(
        df: pl.DataFrame,
        metric_col: str,
        metric_name: str,
        algorithm: str | None = None,
        title: str | None = None,
):
    if algorithm:
        filtered_df = df.filter(pl.col("algorithm") == algorithm)
        plot_title = f"{title} ({metric_name}, {algorithm})"
    else:
        filtered_df = (
            df.group_by(
                ["dataset_name", "embedding_model"]
            ).agg(
                pl.col(metric_col).mean().alias("mean_score")
            )
            .select(
                ["dataset_name", "embedding_model", "mean_score"]
            )
        )

        metric_col = "mean_score"

    pivoted_df = (
        filtered_df.pivot(
            on="embedding_model",
            index="dataset_name",
            values=metric_col
        )
    )
    labels = pivoted_df.select(pl.exclude("dataset_name")).columns
    scores = pivoted_df.select(pl.exclude("dataset_name")).to_numpy()

def create_plots(
        df: pl.DataFrame,
        data_path: str,
        mapping: dict
):
    data_path = Path(data_path)
    models_to_keep = [item for item in mapping.values()]

    df = rename_models(df, mapping)

    df = keeping_models(df, models_to_keep)

    colors = sns.color_palette("colorblind", n_colors=len(models_to_keep))

    color_mapping = {
        model: f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for model, (r, g, b) in zip(models_to_keep, colors)
    }

    binary, multiclass, regression = separate_by_task_type(df)

    # Clean results for each task type
    binary = clean_results(binary)
    multiclass = clean_results(multiclass)
    regression = clean_results(regression)

    setup_publication_style()

    # Get unique algorithms for each task type
    binary_algorithms = binary.get_column("algorithm").unique().to_list()
    multiclass_algorithms = multiclass.get_column("algorithm").unique().to_list()
    regression_algorithms = regression.get_column("algorithm").unique().to_list()

    # Process binary classification tasks
    for algorithm in binary_algorithms:
        logger.info(f"Processing binary classification with {algorithm}")

        descriptive_df = create_descriptive_dataframe(binary, "auc_score")
        print(descriptive_df)

        descriptive_df.write_csv(
            Path(data_path / "binary_auc_score_descriptive.csv")
        )

        # Create boxplot
        ax_box = create_detailed_boxplots(
            binary,
            metric_col="auc_score",
            metric_name="AUC Score",
            algorithm=algorithm,
            title="Binary Classification Performance",
            embedding_model_palette=color_mapping,
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_box, data_path, f"binary_boxplot_{algorithm}")
        plt.close()

        # Create heatmap
        ax_heat = create_dataset_performance_heatmaps(
            binary,
            metric_col="auc_score",
            metric_name="AUC Score",
            algorithm=algorithm,
            title="Binary Classification Performance",
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_heat, data_path, f"binary_heatmap_{algorithm}")
        plt.close()

        try:
            ax_neighbors = create_neighbors_effect_analysis(
                binary,
                metric_col="auc_score",
                metric_name="AUC Score",
                algorithm=algorithm,
                title="Binary Classification Performance",
                embedding_model_palette=color_mapping,
                embedding_model_order=models_to_keep,
            )
            save_fig(ax_neighbors, data_path, f"binary_neighbors_{algorithm}")
            plt.close()
        except ValueError as e:
            logger.warning(f"Skipping neighbors analysis for binary {algorithm}: {e}")

    # Process multiclass classification tasks
    for algorithm in multiclass_algorithms:
        logger.info(f"Processing multiclass classification with "
                    f"{algorithm}_auc_score")
        descriptive_df = create_descriptive_dataframe(multiclass,
                                                      "auc_score")
        print(descriptive_df)

        descriptive_df.write_csv(
            Path(data_path / "multiclass_auc_score_descriptive.csv")
        )

        # Create boxplot
        ax_box = create_detailed_boxplots(
            multiclass,
            metric_col="auc_score",
            metric_name="AUC Score",
            algorithm=algorithm,
            title="Multiclass Classification Performance",
            embedding_model_palette=color_mapping,
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_box, data_path, f"multiclass_boxplot_{algorithm}_auc_score")
        plt.close()

        # Create heatmap
        ax_heat = create_dataset_performance_heatmaps(
            multiclass,
            metric_col="auc_score",
            metric_name="AUC Score",
            algorithm=algorithm,
            title="Multiclass Classification Performance",
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_heat, data_path, f"multiclass_heatmap_"
                                     f"{algorithm}_auc_score")
        plt.close()

        try:
            ax_neighbors = create_neighbors_effect_analysis(
                multiclass,
                metric_col="auc_score",
                metric_name="AUC Score",
                algorithm=algorithm,
                title="Multiclass Classification Performance",
                embedding_model_palette=color_mapping,
                embedding_model_order=models_to_keep,
            )
            save_fig(ax_neighbors, data_path, f"multiclass_neighbors"
                                              f"_{algorithm}_auc_score")
            plt.close()
        except ValueError as e:
            logger.warning(f"Skipping neighbors analysis for multiclass"
                           f" {algorithm}: {e}")

        descriptive_df = create_descriptive_dataframe(multiclass,
                                                      "log_loss_score")
        print(descriptive_df)

        descriptive_df.write_csv(Path(data_path /
                                      "multiclass_log_loss_descriptive.csv"))

        ax_box = create_detailed_boxplots(
            multiclass,
            metric_col="log_loss_score",
            metric_name="log-Loss",
            algorithm=algorithm,
            title="Multiclass Classification Performance",
            embedding_model_palette=color_mapping,
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_box, data_path, f"multiclass_boxplot_{algorithm}_log_loss")
        plt.close()

        # Create heatmap
        ax_heat = create_dataset_performance_heatmaps(
            multiclass,
            metric_col="log_loss_score",
            metric_name="log-Loss",
            algorithm=algorithm,
            title="Multiclass Classification Performance",
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_heat, data_path, f"multiclass_heatmap_{algorithm}_log_loss")
        plt.close()

        try:
            ax_neighbors = create_neighbors_effect_analysis(
                multiclass,
                metric_col="log_loss_score",
                metric_name="log-Loss",
                algorithm=algorithm,
                title="Multiclass Classification Performance",
                embedding_model_palette=color_mapping,
                embedding_model_order=models_to_keep,
            )
            save_fig(ax_neighbors, data_path, f"multiclass_neighbors"
                                              f"_{algorithm}_log_loss")
            plt.close()
        except ValueError as e:
            logger.warning(f"Skipping neighbors analysis for multiclass"
                           f" {algorithm}: {e}")

    # Process regression tasks
    for algorithm in regression_algorithms:
        logger.info(f"Processing regression with {algorithm}")

        descriptive_df = create_descriptive_dataframe(regression, "mape_score")
        print(descriptive_df)

        descriptive_df.write_csv(Path(data_path /
                                 "regression_descriptive.csv"))

        # Create boxplot (minimizing RMSE)
        ax_box = create_detailed_boxplots(
            regression,
            metric_col="mape_score",
            metric_name="MAPE",
            algorithm=algorithm,
            title="Regression Performance",
            embedding_model_palette=color_mapping,
            agg_algorithm="minimize",
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_box, data_path, f"regression_boxplot_{algorithm}")
        plt.close()

        # Create heatmap (minimizing RMSE)
        ax_heat = create_dataset_performance_heatmaps(
            regression,
            metric_col="mape_score",
            metric_name="MAPE",
            algorithm=algorithm,
            title="Regression Performance",
            agg_algorithm="minimize",
            embedding_model_order=models_to_keep,
        )
        save_fig(ax_heat, data_path, f"regression_heatmap_{algorithm}")
        plt.close()

        # Create neighbors effect analysis
        try:
            ax_neighbors = create_neighbors_effect_analysis(
                regression,
                metric_col="mape_score",
                metric_name="MAPE",
                algorithm=algorithm,
                title="Regression Performance",
                embedding_model_palette=color_mapping,
                embedding_model_order=models_to_keep,
            )
            save_fig(ax_neighbors, data_path, f"regression_neighbors_{algorithm}")
            plt.close()
        except ValueError as e:
            logger.warning(f"Skipping neighbors analysis for regression {algorithm}: {e}")

    logger.info("All visualizations completed successfully")


def main():
    """Main function to generate comprehensive benchmark visualizations.
    
    Loads benchmark results, processes them by task type (binary classification,
    multiclass classification, regression), and generates various visualizations
    including boxplots, heatmaps, and neighbor effect analyses for each
    algorithm and metric combination.
    
    The function:
        1. Loads results from a parquet file
        2. Renames and filters embedding models
        3. Separates results by task type
        4. Cleans results to ensure consistent algorithm coverage
        5. Generates visualizations for each task type and algorithm
        6. Saves all figures as PDF files
    """
    import os

    load_dotenv()

    tabarena_result_parquet_path = os.getenv("TABARENA_RESULT_PATH")

    include_sphere = os.getenv("INCLUDE_SPHEREBASED_EMBEDDINGS", "False").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    df = pl.read_parquet(
        tabarena_result_parquet_path
    )

    mapping = {
        "tabicl-classifier-v1.1-0506_preprocessed": "TabICL",
        "TabVectorizerEmbedding": "TableVectorizer",
        "TabPFN": "TabPFN",
    }

    data_path = Path("data/tabembedbench_20251007_190514/figures")

    if include_sphere:
        for n in range(3, 10):
             mapping[f"sphere-model-d{2**n}"] = f"Sphere-Based (Dim {2**n})"
        data_path = Path("data/tabembedbench_20251007_190514/figures"
                      "/sphere_included")

    if not data_path.exists():
        data_path.mkdir(parents=True)

    create_plots(df, data_path, mapping)

if __name__ == "__main__":
    new_result = pl.read_csv("/Users/lkl/PycharmProjects/TabEmbedBench/data/result_tabpfn_embedding_363702/tabpfn_embedding_task_363702_20251010_040852.csv")
    old_result = pl.read_csv("/Users/lkl/PycharmProjects/TabEmbedBench/data/tabembedbench_20251007_190514/results_TabArena_20251007_190514.csv")
    print(new_result)

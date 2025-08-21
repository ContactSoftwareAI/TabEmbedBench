import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl


def create_boxplot(
    result_df: pl.DataFrame, algorithm: str, score: str, embedding_model: str
):
    result_df = result_df.cast({score: pl.Float64, "neighbors": pl.Int64})

    filtered_results = result_df.filter(
        (pl.col("algorithm") == algorithm)
        & (pl.col("embedding_model") == embedding_model)
    ).select(["neighbors", score, "dataset", "samples", "features"])

    stats_by_neighbors = (
        filtered_results.group_by("neighbors")
        .agg(
            [
                pl.col(score).mean().alias(f"avg_{score}"),
            ]
        )
        .sort("neighbors")
    )

    fig_px = px.box(
        data_frame=filtered_results,
        x="neighbors",
        y=f"{score}",
        color="neighbors",
        points="suspectedoutliers",
        hover_data=["dataset", "samples", "features"],
        title=f"Average {score} over datasets by Number of Neighbors, embedding model: {embedding_model}",
    )

    fig_px.update_traces(
        hovertemplate="<b>Dataset:</b> %{customdata[0]}<br>"
        + "<b>Samples:</b> %{customdata[1]}<br>"
        + "<b>Features:</b> %{customdata[2]}<br>"
        + f"<b>{score}:</b> %{{y}}<br>"
        + "<extra></extra>"  # Removes the trace box
    )

    fig_px.add_scatter(
        x=stats_by_neighbors["neighbors"],
        y=stats_by_neighbors[f"avg_{score}"],
        mode="lines+markers",
        name="Average",
        line=dict(color="black", width=2),
        marker=dict(color="black", size=6),
        hovertemplate="Neighbors: %{x}<br>Average: %{y:.3f}<extra></extra>",
    )

    return fig_px


def create_quantile_lines_chart(
    result_df: pl.DataFrame, algorithm: str, score: str, embedding_model: str
):
    result_df = result_df.cast({score: pl.Float64, "neighbors": pl.Int64})

    filtered_results = result_df.filter(
        (pl.col("algorithm") == algorithm)
        & (pl.col("embedding_model") == embedding_model)
    ).select(["neighbors", score, "dataset", "samples", "features"])

    # Calculate statistics by neighbors
    stats_by_neighbors = (
        filtered_results.group_by("neighbors")
        .agg(
            [
                pl.col(score).mean().alias(f"avg_{score}"),
                pl.col(score).median().alias(f"median_{score}"),
                pl.col(score).quantile(0.25).alias(f"q1_{score}"),
                pl.col(score).quantile(0.75).alias(f"q3_{score}"),
            ]
        )
        .sort("neighbors")
    )

    # Create the base figure
    fig_px = px.line(
        title=f"Statistical Summary of {score} by Number of Neighbors, embedding model: {embedding_model}",
        labels={"x": "Number of Neighbors", "y": score},
    )

    # Add the four lines
    fig_px.add_scatter(
        x=stats_by_neighbors["neighbors"],
        y=stats_by_neighbors[f"avg_{score}"],
        mode="lines+markers",
        name="Average",
        line=dict(color="red", width=2),
        marker=dict(color="red", size=6),
        hovertemplate="Neighbors: %{x}<br>Average: %{y:.3f}<extra></extra>",
    )

    fig_px.add_scatter(
        x=stats_by_neighbors["neighbors"],
        y=stats_by_neighbors[f"median_{score}"],
        mode="lines+markers",
        name="Median",
        line=dict(color="blue", width=2),
        marker=dict(color="blue", size=6),
        hovertemplate="Neighbors: %{x}<br>Median: %{y:.3f}<extra></extra>",
    )

    fig_px.add_scatter(
        x=stats_by_neighbors["neighbors"],
        y=stats_by_neighbors[f"q1_{score}"],
        mode="lines+markers",
        name="Q1 (25th percentile)",
        line=dict(color="green", width=2),
        marker=dict(color="green", size=6),
        hovertemplate="Neighbors: %{x}<br>Q1: %{y:.3f}<extra></extra>",
    )

    fig_px.add_scatter(
        x=stats_by_neighbors["neighbors"],
        y=stats_by_neighbors[f"q3_{score}"],
        mode="lines+markers",
        name="Q3 (75th percentile)",
        line=dict(color="orange", width=2),
        marker=dict(color="orange", size=6),
        hovertemplate="Neighbors: %{x}<br>Q3: %{y:.3f}<extra></extra>",
    )

    # Update layout
    fig_px.update_layout(
        xaxis_title="Number of Neighbors",
        yaxis_title=score,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig_px


def create_multi_model_quantile_lines_chart(
    result_df: pl.DataFrame, algorithm: str, score: str
):
    """
    Create a chart with four lines (avg, median, Q1, Q3) for each embedding model.
    """
    result_df = result_df.cast({score: pl.Float64, "neighbors": pl.Int64})

    filtered_results = result_df.filter(pl.col("algorithm") == algorithm).select(
        ["neighbors", score, "dataset", "samples", "features", "embedding_model"]
    )

    # Calculate statistics by neighbors and embedding model
    stats_by_neighbors_and_model = (
        filtered_results.group_by(["neighbors", "embedding_model"])
        .agg(
            [
                pl.col(score).mean().alias(f"avg_{score}"),
                pl.col(score).median().alias(f"median_{score}"),
                pl.col(score).quantile(0.25).alias(f"q1_{score}"),
                pl.col(score).quantile(0.75).alias(f"q3_{score}"),
            ]
        )
        .sort(["embedding_model", "neighbors"])
    )

    # Create figure
    fig = go.Figure()

    # Define colors for different models
    model_colors = px.colors.qualitative.Set1
    models = stats_by_neighbors_and_model["embedding_model"].unique().to_list()

    # Define line styles for different statistics
    line_styles = {
        "avg": dict(dash=None),  # solid line
        "median": dict(dash="dash"),  # dashed line
        "q1": dict(dash="dot"),  # dotted line
        "q3": dict(dash="dashdot"),  # dash-dot line
    }

    stat_names = {
        "avg": "Average",
        "median": "Median",
        "q1": "Q1 (25th percentile)",
        "q3": "Q3 (75th percentile)",
    }

    # Add traces grouped by statistic type
    for stat_key, stat_name in stat_names.items():
        for i, model in enumerate(models):
            model_data = stats_by_neighbors_and_model.filter(
                pl.col("embedding_model") == model
            )
            base_color = model_colors[i % len(model_colors)]

            fig.add_trace(
                go.Scatter(
                    x=model_data["neighbors"].to_list(),
                    y=model_data[f"{stat_key}_{score}"].to_list(),
                    mode="lines+markers",
                    name=f"{model}",
                    legendgroup=stat_name,
                    legendgrouptitle_text=stat_name,
                    line=dict(color=base_color, width=2, **line_styles[stat_key]),
                    marker=dict(color=base_color, size=4),
                    hovertemplate=f"<b>Model:</b> {model}<br><b>Neighbors:</b> %{{x}}<br><b>{stat_name}:</b> %{{y:.3f}}<extra></extra>",
                )
            )

    # Update layout
    fig.update_layout(
        title=f"Statistical Summary of {score} by Number of Neighbors (All Models)",
        xaxis_title="Number of Neighbors",
        yaxis_title=score,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            groupclick="togglegroup",  # This enables group toggling
        ),
        margin=dict(r=200),  # Add right margin for legend
        hovermode="x unified",
    )

    return fig

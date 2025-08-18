import numpy as np
import plotly.express as px
import polars as pl


def create_boxplot(result_df: pl.DataFrame, algorithm: str, score: str):
    filtered_results = result_df.filter(pl.col("algorithm") == algorithm).select(
        ["neighbors", score, "dataset", "samples", "features"]
    )

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
        title=f"Average {score} over datasets by Number of Neighbors",
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

    fig_px.show()


if __name__ == "__main__":
    sample_dict = {
        "algorithm": ["lof", "lof", "lof", "lof_with_train_containment"] * 50,
        "neighbors": [i for i in range(1, 21)] * 10,
        "auc_score": np.random.rand(200),
        "dataset": ["bla", "blurgh", "ba", "gd"] * 50,
        "samples": [100, 120, 30, 10000] * 50,
        "features": [4, 2, 6, 8] * 50,
    }

    result_df = pl.DataFrame(sample_dict)

    fig = create_boxplot(result_df, "lof", "auc_score")

    fig.show()

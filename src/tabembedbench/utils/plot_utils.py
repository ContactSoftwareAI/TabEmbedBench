import polars as pl
import scikit_posthocs as sp
import matplotlib.pyplot as plt


def create_critical_diagram(
        result_df: pl.DataFrame,
        algorithm: str,
        metric: str = "auc_score",
        output_file: str = None,
        dpi: int = 300,
):
    filtered_algorithm_df = (
        result_df.filter(
            pl.col("algorithm") == algorithm,
            pl.col("embedding_model") != "tabicl-classifier-v1.1-0506_with_tabvectorizer_preprocessed"
        )
        .select(
            "dataset_name", "embedding_model", metric
        )
    )

    pivoted_dataframe = filtered_algorithm_df.pivot(
        index="dataset_name",
        on="embedding_model",
        values=metric,
    )

    return pivoted_dataframe


if __name__ == "__main__":
    outlier_result = pl.read_parquet("/Users/lkl/PycharmProjects/TabEmbedBench/data/tabembedbench_20251007_115000/results_ADBench_Tabular_20251007_115000.parquet")

    data_frame = create_critical_diagram(
        result_df=outlier_result,
        algorithm="DeepSVDD-dynamic",
        metric="auc_score",
        output_file="deepsvdd_critical_diagram.pdf",  # PDF for publication
        dpi=300
    )
    print(data_frame.head())
    print(data_frame.columns)
    from aeon.visualisation import plot_critical_difference

    columns = [estimator for estimator in data_frame.columns if estimator !=
               "dataset_name"]

    print(columns)

    fig, ax = plot_critical_difference(
        scores=data_frame.select(pl.exclude("dataset_name")).to_numpy(),
        labels=columns,
        width=20,
        textspace=5,
        reverse=False,
    )

    fig.show()

    tabarena_result = pl.read_parquet("/Users/lkl/PycharmProjects/TabEmbedBench/data/tabembedbench_20251007_190514/results_TabArena_20251007_190514.parquet")

    mapping = {
        "sphere-model-d8": "SphereBased Embedding (Dim 8)",
        "sphere-model-d16": "SphereBased Embedding (Dim 16)",
        "sphere-model-d32": "SphereBased Embedding (Dim 32)",
        "sphere-model-d64": "SphereBased Embedding (Dim 64)",
        "sphere-model-d128": "SphereBased Embedding (Dim 128)",
        "sphere-model-d256": "SphereBased Embedding (Dim 256)",
        "sphere-model-d512": "SphereBased Embedding (Dim 512)",
        "TabVectorizerEmbedding": "TableVectorizer",
        "tabicl-classifier-v1.1-0506_preprocessed": "TabICL"
    }

    print(tabarena_result.columns)

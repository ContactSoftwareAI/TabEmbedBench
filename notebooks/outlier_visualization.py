import polars as pl
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_ADBench_Tabular_20251007_115000.parquet"
save_path_to_balanced_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705"


def setup_publication_style():
    """Set up matplotlib/seaborn for publication-quality figures."""
    sns.set_context("paper")
    sns.set_style("whitegrid")

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def save_fig(ax, data_path: str, file_name: str):
    """Save a matplotlib figure to a PDF file."""
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


def get_models_to_keep():
    """Gibt die Liste der umbenannten Modelle in der richtigen Reihenfolge zurück"""
    return ["TabICL", "TabVectorizer", "TabPFN"]


def create_color_mapping(models_to_keep: list[str]) -> dict[str, str]:
    """Erstellt die gleiche Farbzuordnung wie in eda_utils.py"""
    colors = sns.color_palette("colorblind", n_colors=len(models_to_keep))

    color_mapping = {
        model: f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for model, (r, g, b) in zip(models_to_keep, colors)
    }

    return color_mapping


def load_benchmark_data(file_path: str) -> pl.DataFrame:
    """Lädt die Benchmark-Daten aus der Parquet-Datei."""
    try:
        data = pl.read_parquet(file_path)
        print(f"Daten erfolgreich geladen: {data.shape} (Zeilen, Spalten)")
        print(f"Spalten: {data.columns}")
        return data
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return None


def rename_embedding_models(df: pl.DataFrame) -> pl.DataFrame:
    """Benennt Embedding-Modell-Namen für bessere Lesbarkeit um."""
    return df.with_columns([
        pl.col("embedding_model")
        .str.replace("tabicl-classifier-v1.1-0506_preprocessed", "TabICL")
        .str.replace("TabVectorizerEmbedding", "TabVectorizer")
        .alias("embedding_model")
    ])


def add_outlier_ratio_category(df: pl.DataFrame) -> pl.DataFrame:
    """Fügt eine Kategorie für Outlier Ratio hinzu."""
    return df.with_columns([
        pl.when(pl.col("outlier_ratio") < 0.05)
        .then(pl.lit("< 0.05"))
        .when(pl.col("outlier_ratio") < 0.1)
        .then(pl.lit("0.05 - 0.1"))
        .when(pl.col("outlier_ratio") < 0.25)
        .then(pl.lit("0.1 - 0.25"))
        .otherwise(pl.lit(">= 0.25"))
        .alias("outlier_ratio_category")
    ])


def create_balanced_algorithm_comparison_data(df: pl.DataFrame, score_col: str = "auc_score",
                                              selected_metrics: list[str] = None,
                                              selected_embedding_models: list[str] = None,
                                              save_files: bool = True,
                                              output_dir: str = "filtered_results"):
    """
    Erstellt ausgewogene Daten für den Algorithmus-Vergleich, indem LOF-Scores
    über alle Nachbar-Werte gemittelt werden und IsolationForest über algorithm_n_estimators aggregiert wird.
    Andere Algorithmen werden direkt übernommen.
    Filtert optional nach bestimmten Embedding-Modellen und speichert die Ergebnisse.

    Args:
        df: Input DataFrame
        score_col: Spaltenname für Score-Werte
        selected_embedding_models: Liste der zu behaltenden Embedding-Modelle (None = alle)
        save_files: Ob Dateien gespeichert werden sollen
        output_dir: Verzeichnis für Ausgabedateien
    """
    # Konvertiere output_dir zu Path-Objekt
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )
    
    if selected_metrics is not None:
        print(f"Filtere nach Distanzmetrik: {selected_metrics}")
        valid_data = valid_data.filter(
            pl.col("algorithm_metric").is_in(selected_metrics) |
            pl.col("algorithm_metric").is_null()
        )

    # Filtere nach ausgewählten Embedding-Modellen
    if selected_embedding_models is not None:
        print(f"Filtere nach Embedding-Modellen: {selected_embedding_models}")
        valid_data = valid_data.filter(pl.col("embedding_model").is_in(selected_embedding_models))

    # Ermittle alle verfügbaren Algorithmen
    all_algorithms = valid_data["algorithm"].unique().to_list()
    print(f"Gefundene Algorithmen: {all_algorithms}")

    balanced_data_parts = []

    lof_data = (
        valid_data
        .filter(pl.col("algorithm") == "LocalOutlierFactor")
        .with_columns(
            # Erstelle Ranking nach Score (höchste zuerst)
            pl.col(score_col).rank(method="max", descending=True).over(
                ["dataset_name", "embedding_model", "algorithm_metric", "algorithm",
                 "dataset_size", "embed_dim", "outlier_ratio"]
            ).alias("score_rank")
        )
        .filter(pl.col("score_rank") == 1)  # Nur die besten
        .group_by(
            ["dataset_name", "embedding_model", "algorithm_metric", "algorithm",
             "dataset_size", "embed_dim", "outlier_ratio"]
        )
        .first()  # Falls mehrere den gleichen Maximalwert haben
        .select([
            "dataset_name", "embedding_model", "algorithm_metric", "algorithm",
            "dataset_size", "embed_dim", "outlier_ratio", score_col,
            "time_to_compute_train_embedding", "algorithm_n_neighbors", "task"
        ])
        .rename({"algorithm_n_neighbors": "algorithm_param"})
    )

    if len(lof_data) > 0:
        balanced_data_parts.append(lof_data)
        print(f"✓ LocalOutlierFactor verarbeitet: {len(lof_data)} Experimente")

    # Für IsolationForest: Wähle die Zeile mit dem maximalen Score
    isolation_data = (
        valid_data
        .filter(pl.col("algorithm") == "IsolationForest")
        .with_columns(
            # Erstelle Ranking nach Score (höchste zuerst)
            pl.col(score_col).rank(method="max", descending=True).over(
                ["dataset_name", "embedding_model", "algorithm_metric", "algorithm",
                 "dataset_size", "embed_dim", "outlier_ratio"]
            ).alias("score_rank")
        )
        .filter(pl.col("score_rank") == 1)  # Nur die besten
        .group_by(
            ["dataset_name", "embedding_model", "algorithm_metric", "algorithm",
             "dataset_size", "embed_dim", "outlier_ratio"]
        )
        .first()  # Falls mehrere den gleichen Maximalwert haben
        .select([
            "dataset_name", "embedding_model", "algorithm_metric", "algorithm",
            "dataset_size", "embed_dim", "outlier_ratio", score_col,
            "time_to_compute_train_embedding", "algorithm_n_estimators", "task"
        ])
        .rename({"algorithm_n_estimators": "algorithm_param"})
    )

    if len(isolation_data) > 0:
        balanced_data_parts.append(isolation_data)
        print(f"✓ IsolationForest verarbeitet: {len(isolation_data)} Experimente")

    # Für alle anderen Algorithmen: Übernehme alle Zeilen direkt mit allen Parametern
    other_algorithms = [alg for alg in all_algorithms if alg not in ["LocalOutlierFactor", "IsolationForest"]]

    for algorithm in other_algorithms:
        other_data = valid_data.filter(pl.col("algorithm") == algorithm)

        # Füge leere algorithm_param Spalte hinzu für Konsistenz
        other_data = other_data.with_columns(pl.lit(None).alias("algorithm_param"))

        # Behalte nur die Spalten, die auch die aggregierten DataFrames haben
        other_data = other_data.select([
            "dataset_name", "embedding_model", "algorithm_metric", "algorithm",
            "dataset_size", "embed_dim", "outlier_ratio", score_col,
            "time_to_compute_train_embedding", "algorithm_param", "task"
        ])

        balanced_data_parts.append(other_data)
        print(f"✓ {algorithm} direkt übernommen: {len(other_data)} Experimente")

    # Kombiniere alle Datensätze
    if balanced_data_parts:
        balanced_data = pl.concat(balanced_data_parts)
        balanced_data = add_outlier_ratio_category(balanced_data)
        print(f"✅ Balanced Data erstellt: {len(balanced_data)} Experimente")
    else:
        print("⚠️ Keine Algorithmen gefunden!")
        return pl.DataFrame()

    if save_files:
        # Erstelle aktuellen Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Speichere als Parquet
        parquet_path = output_dir_path / f"results_ADBench_Tabular_{timestamp}_balanced.parquet"
        balanced_data.write_parquet(parquet_path)
        print(f"💾 Parquet gespeichert: {parquet_path}")

        # Speichere als CSV
        csv_path = output_dir_path / f"results_ADBench_Tabular_{timestamp}_balanced.csv"
        balanced_data.write_csv(csv_path)
        print(f"💾 CSV gespeichert: {csv_path}")

    return balanced_data


def create_score_distribution_boxplot(
    df: pl.DataFrame,
    score_col: str = "auc_score",
    embedding_model_palette: dict[str, str] | None = None,
    embedding_model_order: list | None = None
):
    """Erstellt ein Boxplot mit Matplotlib/Seaborn Stil."""
    setup_publication_style()

    title = f"Distribution of AUC Score with respect to embedding model"

    boxplot = sns.boxplot(
        data=df,
        x="embedding_model",
        y=score_col,
        hue="embedding_model",
        palette=embedding_model_palette,
        order=embedding_model_order
    )

    boxplot.set_xlabel("Embedding Model")
    boxplot.set_ylabel("AUC Score")
    boxplot.set_title(title)
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha='right')

    return boxplot



def create_algorithm_comparison(
    df: pl.DataFrame,
    score_col: str = "auc_score",
    embedding_model_palette: dict[str, str] | None = None,
    embedding_model_order: list | None = None
):
    """Erstellt einen Algorithmus-Vergleich mit Seaborn."""
    setup_publication_style()

    boxplot = sns.boxplot(
        data=df,
        x="algorithm",
        y=score_col,
        hue="embedding_model",
        palette=embedding_model_palette,
        hue_order=embedding_model_order
    )

    boxplot.set_xlabel("Algorithm")
    boxplot.set_ylabel("AUC Score")
    boxplot.set_title(f"Algorithm Comparison: AUC Score")
    plt.setp(boxplot.get_xticklabels(), rotation=45, ha='right')

    return boxplot



def create_neighbors_effect_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
    """Analysiert den Nachbarn-Effekt mit Seaborn."""
    setup_publication_style()

    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null() &
        (pl.col("algorithm_n_neighbors") > 0)
    )

    lineplot = sns.lineplot(
        data=valid_data,
        x="algorithm_n_neighbors",
        y=score_col,
        hue="embedding_model",
        markers=True,
        errorbar=None
    )

    lineplot.set_xlabel("Number of Neighbors (K)")
    lineplot.set_ylabel(score_col)
    lineplot.set_title(f"Averaged AUC Score with respect to Number of Neighbors (K)")
    lineplot.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    return lineplot


# def create_dataset_difficulty_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
#     """Analysiert die Schwierigkeit verschiedener Datensätze mit ausgewogenen Daten."""
#
#     # Berechne durchschnittliche Performance pro Datensatz
#     dataset_difficulty = (df
#                           .group_by("dataset_name")
#                           .agg([
#         pl.col(score_col).mean().alias(f"avg_{score_col}"),
#         pl.col("dataset_size").first().alias("size"),
#         pl.col("embed_dim").first().alias("features")
#     ])
#                           .sort(f"avg_{score_col}"))
#
#     fig = px.bar(
#         dataset_difficulty.to_pandas(),
#         x="dataset_name",
#         y=f"avg_{score_col}",
#         hover_data=["size", "features"],
#         title=f"Datensatz-Schwierigkeit (durchschnittliche {score_col}, ausgewogen)",
#         labels={"dataset_name": "Datensatz"}
#     )
#
#     fig.update_xaxes(tickangle=45)
#     fig.update_layout(height=600, xaxis_tickangle=45)
#     return fig


def generate_summary_statistics(df: pl.DataFrame):
    """Erstellt eine Zusammenfassung der wichtigsten Statistiken basierend auf ausgewogenen Daten."""

    print("\n=== BENCHMARK ERGEBNISSE ZUSAMMENFASSUNG ===")
    print(f"Rohdaten: {len(df)} Experimente")
    print(f"Anzahl der Datensätze: {df['dataset_name'].n_unique()}")
    print(f"Anzahl der Embedding-Modelle: {df['embedding_model'].n_unique()}")
    print(f"Anzahl der Algorithmen: {df['algorithm'].n_unique()}")

    print(f"\nEmbedding-Modelle: {df['embedding_model'].unique().to_list()}")
    print(f"Algorithmen: {df['algorithm'].unique().to_list()}")
    print(f"Distanzmetriken: {df['algorithm_metric'].unique().to_list()}")


    # auc_score Score Statistiken (nur gültige Werte)
    valid_auc_score = df.filter(
        (pl.col("auc_score") > -np.inf) &
        (pl.col("auc_score") < np.inf) &
        pl.col("auc_score").is_not_null()
    )

    if len(valid_auc_score) > 0:
        print(f"\n--- auc_score Score Statistiken ---")
        print(f"Mittelwert: {valid_auc_score['auc_score'].mean():.4f}")
        print(f"Median: {valid_auc_score['auc_score'].median():.4f}")
        print(f"Min: {valid_auc_score['auc_score'].min():.4f}")
        print(f"Max: {valid_auc_score['auc_score'].max():.4f}")
        print(f"Standardabweichung: {valid_auc_score['auc_score'].std():.4f}")

        # Performance nach Embedding-Modell und Outlier Ratio
        print(f"\n--- Performance nach Embedding-Modell und Outlier Ratio ---")
        model_outlier_performance = (valid_auc_score
                                     .group_by(["embedding_model", "outlier_ratio_category"])
                                     .agg([
            pl.col("auc_score").mean().alias("avg_auc_score"),
            pl.col("auc_score").std().alias("std_auc_score"),
            pl.col("auc_score").count().alias("num_experiments")
        ])
                                     .sort(["embedding_model", "outlier_ratio_category"]))

        for row in model_outlier_performance.iter_rows(named=True):
            print(f"{row['embedding_model']} (Ratio {row['outlier_ratio_category']}): "
                  f"auc_score={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f}, "
                  f"Experimente={row['num_experiments']}")

        # Performance nach Embedding-Modell und Algorithmus
        print(f"\n--- Performance nach Embedding-Modell und Algorithmus ---")
        model_algorithm_performance = (valid_auc_score
                                       .group_by(["embedding_model", "algorithm"])
                                       .agg([
            pl.col("auc_score").mean().alias("avg_auc_score"),
            pl.col("auc_score").std().alias("std_auc_score"),
            pl.col("auc_score").count().alias("num_experiments")
        ])
                                       .sort(["embedding_model", "algorithm"]))

        for row in model_algorithm_performance.iter_rows(named=True):
            print(f"{row['embedding_model']} ({row['algorithm']}): "
                  f"auc_score={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f}, "
                  f"Experimente={row['num_experiments']}")


    print(f"\n--- Mehrdimensionale Performance-Analyse ---")

    # 1. Hauptanalyse: Alle Dimensionen
    full_performance = (valid_auc_score
                        .group_by(["embedding_model", "algorithm", "outlier_ratio_category"])
                        .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                        .sort(["embedding_model", "algorithm", "outlier_ratio_category"]))

    print("=== Detaillierte Matrix (Embedding × Algorithmus × Outlier Ratio) ===")
    for row in full_performance.iter_rows(named=True):
        print(f"{row['embedding_model']} | {row['algorithm']} | Ratio {row['outlier_ratio_category']}: "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")

    # 1. Randverteilung: Nach Embedding-Modell und Outlier Ratio
    print(f"\n=== Randverteilung: Embedding-Modell × Outlier Ratio ===")
    model_outlier_marginal = (valid_auc_score
                              .group_by(["embedding_model", "outlier_ratio_category"])
                              .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                              .sort(["embedding_model", "outlier_ratio_category"]))

    for row in model_outlier_marginal.iter_rows(named=True):
        print(f"{row['embedding_model']} (Ratio {row['outlier_ratio_category']}): "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")

    # 2. Randverteilung: Nach Embedding-Modell und Algorithmus
    print(f"\n=== Randverteilung: Embedding-Modell × Algorithmus ===")
    model_algorithm_marginal = (valid_auc_score
                                .group_by(["embedding_model", "algorithm"])
                                .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                                .sort(["embedding_model", "algorithm"]))

    for row in model_algorithm_marginal.iter_rows(named=True):
        print(f"{row['embedding_model']} ({row['algorithm']}): "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")

    # 3. Randverteilung: Nach Embedding-Modell (Gesamt)
    print(f"\n=== Randverteilung: Embedding-Modell (Gesamt) ===")
    model_total_marginal = (valid_auc_score
                            .group_by(["embedding_model"])
                            .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                            .sort("avg_auc_score", descending=True))

    for row in model_total_marginal.iter_rows(named=True):
        print(f"{row['embedding_model']}: "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")

    # 4. Randverteilung: Nach Algorithmus (Gesamt)
    print(f"\n=== Randverteilung: Algorithmus (Gesamt) ===")
    algorithm_total_marginal = (valid_auc_score
                                .group_by(["algorithm"])
                                .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                                .sort("avg_auc_score", descending=True))

    for row in algorithm_total_marginal.iter_rows(named=True):
        print(f"{row['algorithm']}: "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")

    # 5. Randverteilung: Nach Outlier Ratio (Gesamt)
    print(f"\n=== Randverteilung: Outlier Ratio (Gesamt) ===")
    outlier_total_marginal = (valid_auc_score
                              .group_by(["outlier_ratio_category"])
                              .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                              .sort("avg_auc_score", descending=True))

    for row in outlier_total_marginal.iter_rows(named=True):
        print(f"Outlier Ratio {row['outlier_ratio_category']}: "
              f"AUC={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f} (n={row['num_experiments']})")


    # Berechnungszeit Statistiken (ausgewogene Daten)
    valid_time = df.filter(
        pl.col("time_to_compute_train_embedding").is_not_null() &
        (pl.col("time_to_compute_train_embedding") > 0)
    )

    if len(valid_time) > 0:
        print(f"\n--- Berechnungszeit Statistiken (ausgewogen) ---")
        print(f"Mittelwert: {valid_time['time_to_compute_train_embedding'].mean():.4f} Sekunden")
        print(f"Median: {valid_time['time_to_compute_train_embedding'].median():.4f} Sekunden")
        print(f"Min: {valid_time['time_to_compute_train_embedding'].min():.4f} Sekunden")
        print(f"Max: {valid_time['time_to_compute_train_embedding'].max():.4f} Sekunden")

    # Performance pro Embedding-Modell (ausgewogene Daten)
    print(f"\n--- Performance nach Embedding-Modell (ausgewogen) ---")
    model_performance = (valid_auc_score
                         .group_by("embedding_model")
                         .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("time_to_compute_train_embedding").mean().alias("avg_time"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                         .sort("avg_auc_score", descending=True))

    for row in model_performance.iter_rows(named=True):
        print(f"{row['embedding_model']}: "
              f"auc_score={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f}, "
              f"Zeit={row['avg_time']:.4f}s, "
              f"Experimente={row['num_experiments']}")


def main():
    """Hauptfunktion zur Ausführung der Visualisierungen."""
    # Lade die Daten
    data = load_benchmark_data(path_to_data_file)

    if data is None:
        return

    selected_embedding_models = ["TabVectorizerEmbedding", "tabicl-classifier-v1.1-0506_preprocessed", "TabPFN"]
    selected_metrics = ["euclidean", None]
    # Erstelle ausgewogene Daten für konsistente Statistiken
    balanced_data = create_balanced_algorithm_comparison_data(data, "auc_score", selected_metrics, selected_embedding_models, True, save_path_to_balanced_file)
    balanced_data = rename_embedding_models(balanced_data)

    # Zeige grundlegende Statistiken
    generate_summary_statistics(balanced_data)

    # Definiere Modellreihenfolge und Farben
    models_to_keep = ["TabICL", "TabVectorizer", "TabPFN"]  # Gleiche Reihenfolge wie in eda_utils.py
    color_mapping = create_color_mapping(models_to_keep)

    # Setze Publication Style
    setup_publication_style()

    # Erstelle Visualisierungen
    print("\nErstelle Visualisierungen...")

    # 1. Score-Verteilungs-Boxplot
    ax1 = create_score_distribution_boxplot(
        balanced_data, "auc_score", embedding_model_palette=color_mapping,
        embedding_model_order=models_to_keep
    )
    save_fig(ax1, "benchmark_visualizations", "score_distribution_boxplot")
    plt.close()

    # 2. Algorithmus-Vergleich
    ax2 = create_algorithm_comparison(
        balanced_data, "auc_score",
        embedding_model_palette=color_mapping,
        embedding_model_order=models_to_keep
    )
    save_fig(ax2, "benchmark_visualizations", "algorithm_comparison")
    plt.close()

    ax3 = create_neighbors_effect_analysis(data, "auc_score")
    save_fig(ax3, "benchmark_visualizations", "neighbors_effect_analysis")
    plt.close()
    #
    # # 9. Datensatz-Schwierigkeit-Analyse
    # fig10 = create_dataset_difficulty_analysis(balanced_data, "auc_score")
    # fig10.show()


print("Visualisierungen erfolgreich erstellt!")


if __name__ == "__main__":
    main()
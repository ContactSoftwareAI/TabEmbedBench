import polars as pl
import plotly.express as px
from pathlib import Path
import numpy as np
import plotly.io as pio  # Für PNG Export hinzufügen


# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_ADBench_Tabular_20251007_115000.parquet"
save_path_to_balanced_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705"

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
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    # Filtere nach ausgewählten Embedding-Modellen
    if selected_embedding_models is not None:
        print(f"Filtere nach Embedding-Modellen: {selected_embedding_models}")
        valid_data = valid_data.filter(pl.col("embedding_model").is_in(selected_embedding_models))

        if len(valid_data) == 0:
            print(f"⚠️ Keine Daten nach Filterung gefunden!")
            return pl.DataFrame()

        # Ermittle alle verfügbaren Algorithmen
    all_algorithms = valid_data["algorithm"].unique().to_list()
    print(f"Gefundene Algorithmen: {all_algorithms}")

    balanced_data_parts = []

    # Für LocalOutlierFactor: Aggregiere über algorithm_n_neighbors und nehme maximalen AUC-Wert
    lof_data = (
        valid_data
        .filter(pl.col("algorithm") == "LocalOutlierFactor")
        .group_by(
            ["dataset_name", "embedding_model", "algorithm_metric", "algorithm", "dataset_size", "embed_dim", "outlier_ratio"])
        .agg([
            pl.col(score_col).max().alias(score_col),
            pl.col("time_to_compute_train_embedding").mean().alias("time_to_compute_train_embedding"),
            pl.col("algorithm_n_neighbors").first().alias("algorithm_param"),
            pl.col("task").first().alias("task")
        ])
    )
    if len(lof_data) > 0:
        balanced_data_parts.append(lof_data)
        print(f"✓ LocalOutlierFactor verarbeitet: {len(lof_data)} Experimente")

    # Für IsolationForest: Aggregiere über algorithm_n_estimators
    isolation_data = (
        valid_data
        .filter(pl.col("algorithm") == "IsolationForest")
        .group_by(
            ["dataset_name", "embedding_model", "algorithm_metric", "algorithm", "dataset_size", "embed_dim", "outlier_ratio"])
        .agg([
            pl.col(score_col).max().alias(score_col),
            pl.col("time_to_compute_train_embedding").mean().alias("time_to_compute_train_embedding"),
            pl.col("algorithm_n_estimators").filter(pl.col(score_col) == pl.col(score_col).max()).first().alias(
                "algorithm_param"),
            pl.col("task").first().alias("task")
        ])
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

        # Speichere als Parquet
        parquet_path = output_path / parquet_filename
        balanced_data.write_parquet(parquet_path)
        print(f"💾 Parquet gespeichert: {parquet_path}")

        # Speichere als CSV
        csv_path = output_path / csv_filename
        balanced_data.write_csv(csv_path)
        print(f"💾 CSV gespeichert: {csv_path}")

    return balanced_data


def create_score_distribution_boxplot(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt ein ausgewogenes Boxplot für die Verteilung der Scores nach Embedding-Modell."""

    title = f"Verteilung der {score_col} nach Embedding-Modell"

    fig = px.box(
        df.to_pandas(),
        x="embedding_model",
        y=score_col,
        color="embedding_model",
        title=title,
        points="outliers",
        hover_data=["dataset_name", "algorithm", "algorithm_metric"]
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False, height=500)
    return fig


def create_algorithm_comparison(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt einen ausgewogenen Vergleich der verschiedenen Algorithmen."""

    # Gruppierte Boxplots mit ausgewogenen Daten
    fig = px.box(
        df.to_pandas(),
        x="algorithm",
        y=score_col,
        color="embedding_model",
        title=f"Algorithmus-Vergleich: {score_col}",
        hover_data=["dataset_name", "algorithm_metric"]
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600)
    return fig


def create_neighbors_effect_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
    """Analysiert den Effekt der Anzahl von Nachbarn auf die Performance."""
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null() &
        (pl.col("algorithm_n_neighbors") > 0)  # Filtere IsolationForest (algorithm_n_neighbors=0) heraus
    )

    # Berechne Durchschnittswerte pro Nachbaranzahl und Embedding-Modell
    avg_scores = (valid_data
                  .group_by(["algorithm_n_neighbors", "embedding_model"])
                  .agg(pl.col(score_col).mean().alias(f"avg_{score_col}"))
                  .sort(["embedding_model", "algorithm_n_neighbors"]))

    fig = px.line(
        avg_scores.to_pandas(),
        x="algorithm_n_neighbors",
        y=f"avg_{score_col}",
        color="embedding_model",
        markers=True,
        title=f"Durchschnittliche {score_col} nach Anzahl Nachbarn (LocalOutlierFactor)",
        hover_data=["embedding_model"]
    )

    fig.update_layout(height=500)
    return fig


def create_dataset_difficulty_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
    """Analysiert die Schwierigkeit verschiedener Datensätze mit ausgewogenen Daten."""

    # Berechne durchschnittliche Performance pro Datensatz
    dataset_difficulty = (df
                          .group_by("dataset_name")
                          .agg([
        pl.col(score_col).mean().alias(f"avg_{score_col}"),
        pl.col("dataset_size").first().alias("size"),
        pl.col("embed_dim").first().alias("features")
    ])
                          .sort(f"avg_{score_col}"))

    fig = px.bar(
        dataset_difficulty.to_pandas(),
        x="dataset_name",
        y=f"avg_{score_col}",
        hover_data=["size", "features"],
        title=f"Datensatz-Schwierigkeit (durchschnittliche {score_col}, ausgewogen)",
        labels={"dataset_name": "Datensatz"}
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600, xaxis_tickangle=45)
    return fig


def generate_summary_statistics(df: pl.DataFrame):
    """Erstellt eine Zusammenfassung der wichtigsten Statistiken basierend auf ausgewogenen Daten."""

    print("\n=== BENCHMARK ERGEBNISSE ZUSAMMENFASSUNG (AUSGEWOGENE DATEN) ===")
    print(f"Rohdaten: {len(df)} Experimente")
    print(f"Ausgewogene Daten: {len(df)} Experimente")
    print(f"Anzahl der Datensätze: {df['dataset_name'].n_unique()}")
    print(f"Anzahl der Embedding-Modelle: {df['embedding_model'].n_unique()}")
    print(f"Anzahl der Algorithmen: {df['algorithm'].n_unique()}")

    print(f"\nEmbedding-Modelle: {df['embedding_model'].unique().to_list()}")
    print(f"Algorithmen: {df['algorithm'].unique().to_list()}")
    print(f"Distanzmetriken: {df['algorithm_metric'].unique().to_list()}")

    # Outlier Ratio Verteilung
    outlier_ratio_dist = (df
                          .group_by("outlier_ratio_category")
                          .agg(pl.col("dataset_name").count().alias("count"))
                          .sort("outlier_ratio_category"))

    print(f"\n--- Outlier Ratio Verteilung ---")
    for row in outlier_ratio_dist.iter_rows(named=True):
        print(f"Outlier Ratio {row['outlier_ratio_category']}: {row['count']} Experimente")

    # auc_score Score Statistiken (nur gültige Werte)
    valid_auc_score = df.filter(
        (pl.col("auc_score") > -np.inf) &
        (pl.col("auc_score") < np.inf) &
        pl.col("auc_score").is_not_null()
    )

    if len(valid_auc_score) > 0:
        print(f"\n--- auc_score Score Statistiken (ausgewogen) ---")
        print(f"Mittelwert: {valid_auc_score['auc_score'].mean():.4f}")
        print(f"Median: {valid_auc_score['auc_score'].median():.4f}")
        print(f"Min: {valid_auc_score['auc_score'].min():.4f}")
        print(f"Max: {valid_auc_score['auc_score'].max():.4f}")
        print(f"Standardabweichung: {valid_auc_score['auc_score'].std():.4f}")

        # Performance nach Outlier Ratio
        print(f"\n--- Performance nach Outlier Ratio (ausgewogen) ---")
        outlier_ratio_performance = (valid_auc_score
                                     .group_by("outlier_ratio_category")
                                     .agg([
            pl.col("auc_score").mean().alias("avg_auc_score"),
            pl.col("auc_score").std().alias("std_auc_score"),
            pl.col("auc_score").count().alias("num_experiments"),
            pl.col("outlier_ratio").mean().alias("avg_outlier_ratio")
        ])
                                     .sort("outlier_ratio_category"))

        for row in outlier_ratio_performance.iter_rows(named=True):
            print(f"Outlier Ratio {row['outlier_ratio_category']}: "
                  f"auc_score={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f}, "
                  f"Durchschn. Ratio={row['avg_outlier_ratio']:.4f}, "
                  f"Experimente={row['num_experiments']}")

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

    # Algorithmus-Performance (ausgewogene Daten)
    print(f"\n--- Performance nach Algorithmus (ausgewogen) ---")
    algorithm_performance = (valid_auc_score
                             .group_by("algorithm")
                             .agg([
        pl.col("auc_score").mean().alias("avg_auc_score"),
        pl.col("auc_score").std().alias("std_auc_score"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                             .sort("avg_auc_score", descending=True))

    for row in algorithm_performance.iter_rows(named=True):
        print(f"{row['algorithm']}: "
              f"auc_score={row['avg_auc_score']:.4f}±{row['std_auc_score']:.4f}, "
              f"Experimente={row['num_experiments']}")


def main():
    """Hauptfunktion zur Ausführung der Visualisierungen."""
    # Lade die Daten
    data = load_benchmark_data(path_to_data_file)

    if data is None:
        return

    selected_embedding_models = ["TabVectorizerEmbedding", "tabicl-classifier-v1.1-0506_preprocessed", "TabPFN"]
    # Erstelle ausgewogene Daten für konsistente Statistiken
    balanced_data = create_balanced_algorithm_comparison_data(data, "auc_score", selected_embedding_models)
    balanced_data = rename_embedding_models(balanced_data)

    # Zeige grundlegende Statistiken
    generate_summary_statistics(balanced_data)

    # Erstelle Visualisierungen
    print("\nErstelle Visualisierungen...")


    # 1. Score-Verteilungs-Boxplot
    fig1 = create_score_distribution_boxplot(balanced_data, "auc_score")
    fig1.show()

    # 4. Algorithmus-Vergleich
    fig5 = create_algorithm_comparison(balanced_data, "auc_score")
    fig5.show()

    # 6. Nachbarn-Effekt-Analyse
    fig7 = create_neighbors_effect_analysis(data, "auc_score")
    fig7.show()

    # 9. Datensatz-Schwierigkeit-Analyse
    fig10 = create_dataset_difficulty_analysis(balanced_data, "auc_score")
    fig10.show()

    # Speichere die Plots als png (optional)
    output_dir = Path("benchmark_visualizations")
    output_dir.mkdir(exist_ok=True)

    plots = [
        (fig1, "score_distribution_boxplot.png"),
        (fig5, "algorithm_comparison.png"),
        (fig7, "neighbors_effect_analysis.png"),
        (fig10, "dataset_difficulty_analysis.png")
    ]

    print(f"\nSpeichere Plots als PNG in {output_dir}...")
    for fig, filename in plots:
        try:
            # Für Plotly-Figures verwenden wir pio.write_image()
            pio.write_image(fig, output_dir / filename,
                          width=1200, height=800, scale=2,  # scale=2 für hohe Qualität
                          format='png')
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Fehler beim Speichern von {filename}: {e}")

print("Visualisierungen erfolgreich erstellt!")


if __name__ == "__main__":
    main()
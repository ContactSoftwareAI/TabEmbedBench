import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_ADBench_Tabular_20250918_144839.parquet"


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


def create_balanced_algorithm_comparison_data(df: pl.DataFrame, score_col: str = "auc_score"):
    """
    Erstellt ausgewogene Daten für den Algorithmus-Vergleich, indem LOF-Scores
    über alle Nachbar-Werte gemittelt werden.
    """
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    # Für LocalOutlierFactor: Mittle über alle Nachbar-Werte pro (dataset, embedding_model, distance_metric)
    lof_data = (
        valid_data
        .filter(pl.col("algorithm") == "LocalOutlierFactor")
        .group_by(["dataset_name", "embedding_model", "distance_metric", "algorithm", "dataset_size", "emb_dim"])
        .agg([
            pl.col(score_col).max().alias(score_col),
            pl.col("time_to_compute_train_embeddings").first().alias("time_to_compute_train_embeddings"),
            pl.col("prediction_time").max().alias("prediction_time"),  # Mittle auch die Prediction-Zeit
            pl.col("num_neighbors").first().alias("num_neighbors"),  # Füge num_neighbors hinzu
            pl.col("mse_score").first().alias("mse_score"),  # Füge mse_score hinzu
            pl.col("task").first().alias("task"),  # Füge task hinzu
            pl.col("time_to_compute_test_embeddings").first().alias("time_to_compute_test_embeddings")
            # Füge time_to_compute_test_embeddings hinzu
        ])
    )

    # Für IsolationForest: Nehme die Daten direkt und wähle nur die benötigten Spalten aus
    isolation_data = (
        valid_data
        .filter(pl.col("algorithm") == "IsolationForest")
        .select([
            "dataset_name", "embedding_model", "distance_metric", "algorithm",
            "dataset_size", "emb_dim", score_col, "time_to_compute_train_embeddings",
            "prediction_time", "num_neighbors", "mse_score", "task",
            "time_to_compute_test_embeddings"
        ])
    )

    # Kombiniere beide Datensätze
    balanced_data = pl.concat([lof_data, isolation_data])

    return balanced_data


def create_score_distribution_boxplot(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt ein ausgewogenes Boxplot für die Verteilung der Scores nach Embedding-Modell."""

    # Erstelle ausgewogene Daten
    balanced_data = create_balanced_algorithm_comparison_data(df, score_col)

    fig = px.box(
        balanced_data.to_pandas(),
        x="embedding_model",
        y=score_col,
        color="embedding_model",
        title=f"Verteilung der {score_col} nach Embedding-Modell (ausgewogen)",
        points="outliers",
        hover_data=["dataset_name", "algorithm", "distance_metric"]
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False, height=500)
    return fig


def create_computation_time_analysis(df: pl.DataFrame):
    """Erstellt ausgewogene Visualisierungen für die Berechnungszeit-Analyse."""

    # Erstelle ausgewogene Daten (für Embedding-Berechnungszeit ist das nicht nötig,
    # da diese pro (dataset, embedding_model) eindeutig ist)
    balanced_data = create_balanced_algorithm_comparison_data(df, "auc_score")

    # Boxplot für Berechnungszeit nach Embedding-Modell
    fig1 = px.box(
        balanced_data.to_pandas(),
        x="embedding_model",
        y="time_to_compute_train_embeddings",
        color="embedding_model",
        title="Berechnungszeit nach Embedding-Modell (ausgewogen)",
        log_y=True,
        hover_data=["dataset_name", "dataset_size"]
    )
    fig1.update_xaxes(tickangle=45)
    fig1.update_layout(showlegend=False, height=500)

    # Scatter-Plot: Berechnungszeit vs. Datensatzgröße
    fig2 = px.scatter(
        balanced_data.to_pandas(),
        x="dataset_size",
        y="time_to_compute_train_embeddings",
        color="embedding_model",
        symbol="algorithm",
        size="emb_dim",
        hover_data=["dataset_name", "algorithm"],
        title="Berechnungszeit vs. Datensatzgröße (ausgewogen)",
        log_x=True,
        log_y=True
    )
    fig2.update_layout(height=500)

    return fig1, fig2


def create_algorithm_comparison(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt einen ausgewogenen Vergleich der verschiedenen Algorithmen."""

    # Erstelle ausgewogene Daten
    balanced_data = create_balanced_algorithm_comparison_data(df, score_col)

    # Gruppierte Boxplots mit ausgewogenen Daten
    fig = px.box(
        balanced_data.to_pandas(),
        x="algorithm",
        y=score_col,
        color="embedding_model",
        title=f"Algorithmus-Vergleich: {score_col} (LOF gemittelt über Nachbar-Werte)",
        hover_data=["dataset_name", "distance_metric"]
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
        (pl.col("num_neighbors") > 0)  # Filtere IsolationForest (num_neighbors=0) heraus
    )

    # Berechne Durchschnittswerte pro Nachbaranzahl und Embedding-Modell
    avg_scores = (valid_data
                  .group_by(["num_neighbors", "embedding_model"])
                  .agg(pl.col(score_col).mean().alias(f"avg_{score_col}"))
                  .sort(["embedding_model", "num_neighbors"]))

    fig = px.line(
        avg_scores.to_pandas(),
        x="num_neighbors",
        y=f"avg_{score_col}",
        color="embedding_model",
        markers=True,
        title=f"Durchschnittliche {score_col} nach Anzahl Nachbarn (LocalOutlierFactor)",
        hover_data=["embedding_model"]
    )

    fig.update_layout(height=500)
    return fig



def create_distance_metric_comparison(df: pl.DataFrame, score_col: str = "auc_score"):
    """Vergleicht die Performance verschiedener Distanzmetriken."""
    # Filtere nur LocalOutlierFactor (da IsolationForest keine Distanzmetrik verwendet)
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null() &
        (pl.col("algorithm") == "LocalOutlierFactor") &
        (pl.col("distance_metric") != "")
    )

    if len(valid_data) == 0:
        print("Keine gültigen Daten für Distanzmetrik-Vergleich gefunden.")
        return None

    fig = px.box(
        valid_data.to_pandas(),
        x="distance_metric",
        y=score_col,
        color="embedding_model",
        title=f"Vergleich der Distanzmetriken: {score_col} (LocalOutlierFactor)",
        hover_data=["dataset_name", "num_neighbors"]
    )

    fig.update_layout(height=500)
    return fig


def create_dataset_difficulty_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
    """Analysiert die Schwierigkeit verschiedener Datensätze mit ausgewogenen Daten."""

    # Erstelle ausgewogene Daten
    balanced_data = create_balanced_algorithm_comparison_data(df, score_col)

    # Berechne durchschnittliche Performance pro Datensatz
    dataset_difficulty = (balanced_data
                          .group_by("dataset_name")
                          .agg([
        pl.col(score_col).mean().alias(f"avg_{score_col}"),
        pl.col("dataset_size").first().alias("size"),
        pl.col("emb_dim").first().alias("features")
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

    # Erstelle ausgewogene Daten für konsistente Statistiken
    balanced_data = create_balanced_algorithm_comparison_data(df, "auc_score")

    print("\n=== BENCHMARK ERGEBNISSE ZUSAMMENFASSUNG (AUSGEWOGENE DATEN) ===")
    print(f"Rohdaten: {len(df)} Experimente")
    print(f"Ausgewogene Daten: {len(balanced_data)} Experimente")
    print(f"Anzahl der Datensätze: {balanced_data['dataset_name'].n_unique()}")
    print(f"Anzahl der Embedding-Modelle: {balanced_data['embedding_model'].n_unique()}")
    print(f"Anzahl der Algorithmen: {balanced_data['algorithm'].n_unique()}")

    print(f"\nEmbedding-Modelle: {balanced_data['embedding_model'].unique().to_list()}")
    print(f"Algorithmen: {balanced_data['algorithm'].unique().to_list()}")
    print(f"Distanzmetriken: {balanced_data['distance_metric'].unique().to_list()}")

    # AUC Score Statistiken (nur gültige Werte)
    valid_auc = balanced_data.filter(
        (pl.col("auc_score") > -np.inf) &
        (pl.col("auc_score") < np.inf) &
        pl.col("auc_score").is_not_null()
    )

    if len(valid_auc) > 0:
        print(f"\n--- AUC Score Statistiken (ausgewogen) ---")
        print(f"Mittelwert: {valid_auc['auc_score'].mean():.4f}")
        print(f"Median: {valid_auc['auc_score'].median():.4f}")
        print(f"Min: {valid_auc['auc_score'].min():.4f}")
        print(f"Max: {valid_auc['auc_score'].max():.4f}")
        print(f"Standardabweichung: {valid_auc['auc_score'].std():.4f}")

    # Berechnungszeit Statistiken (ausgewogene Daten)
    valid_time = balanced_data.filter(
        pl.col("time_to_compute_train_embeddings").is_not_null() &
        (pl.col("time_to_compute_train_embeddings") > 0)
    )

    if len(valid_time) > 0:
        print(f"\n--- Berechnungszeit Statistiken (ausgewogen) ---")
        print(f"Mittelwert: {valid_time['time_to_compute_train_embeddings'].mean():.4f} Sekunden")
        print(f"Median: {valid_time['time_to_compute_train_embeddings'].median():.4f} Sekunden")
        print(f"Min: {valid_time['time_to_compute_train_embeddings'].min():.4f} Sekunden")
        print(f"Max: {valid_time['time_to_compute_train_embeddings'].max():.4f} Sekunden")

    # Performance pro Embedding-Modell (ausgewogene Daten)
    print(f"\n--- Performance nach Embedding-Modell (ausgewogen) ---")
    model_performance = (valid_auc
                         .group_by("embedding_model")
                         .agg([
        pl.col("auc_score").mean().alias("avg_auc"),
        pl.col("auc_score").std().alias("std_auc"),
        pl.col("time_to_compute_train_embeddings").mean().alias("avg_time"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                         .sort("avg_auc", descending=True))

    for row in model_performance.iter_rows(named=True):
        print(f"{row['embedding_model']}: "
              f"AUC={row['avg_auc']:.4f}±{row['std_auc']:.4f}, "
              f"Zeit={row['avg_time']:.4f}s, "
              f"Experimente={row['num_experiments']}")

    # Algorithmus-Performance (ausgewogene Daten)
    print(f"\n--- Performance nach Algorithmus (ausgewogen) ---")
    algorithm_performance = (valid_auc
                             .group_by("algorithm")
                             .agg([
        pl.col("auc_score").mean().alias("avg_auc"),
        pl.col("auc_score").std().alias("std_auc"),
        pl.col("auc_score").count().alias("num_experiments")
    ])
                             .sort("avg_auc", descending=True))

    for row in algorithm_performance.iter_rows(named=True):
        print(f"{row['algorithm']}: "
              f"AUC={row['avg_auc']:.4f}±{row['std_auc']:.4f}, "
              f"Experimente={row['num_experiments']}")



def main():
    """Hauptfunktion zur Ausführung der Visualisierungen."""
    # Lade die Daten
    data = load_benchmark_data(path_to_data_file)

    if data is None:
        return

    # Zeige grundlegende Statistiken
    generate_summary_statistics(data)

    # Erstelle Visualisierungen
    print("\nErstelle Visualisierungen...")

    # 1. Score-Verteilungs-Boxplot
    fig1 = create_score_distribution_boxplot(data, "auc_score")
    fig1.show()

    # 3. Berechnungszeit-Analyse
    fig3, fig4 = create_computation_time_analysis(data)
    fig3.show()
    fig4.show()

    # 4. Algorithmus-Vergleich
    fig5 = create_algorithm_comparison(data, "auc_score")
    fig5.show()

    # 6. Nachbarn-Effekt-Analyse
    fig7 = create_neighbors_effect_analysis(data, "auc_score")
    fig7.show()

    # 8. Distanzmetrik-Vergleich
    fig9 = create_distance_metric_comparison(data, "auc_score")
    if fig9:
        fig9.show()

    # 9. Datensatz-Schwierigkeit-Analyse
    fig10 = create_dataset_difficulty_analysis(data, "auc_score")
    fig10.show()

    # Speichere die Plots als HTML (optional)
    output_dir = Path("benchmark_visualizations")
    output_dir.mkdir(exist_ok=True)

    plots = [
        (fig1, "score_distribution_boxplot.html"),
        (fig3, "computation_time_by_model.html"),
        (fig4, "computation_time_vs_size.html"),
        (fig5, "algorithm_comparison.html"),
        (fig7, "neighbors_effect_analysis.html"),
        (fig10, "dataset_difficulty_analysis.html")
    ]

    if fig9:
        plots.append((fig9, "distance_metric_comparison.html"))

    print(f"\nSpeichere Plots in {output_dir}...")
    for fig, filename in plots:
        fig.write_html(output_dir / filename)

    print("Visualisierungen erfolgreich erstellt!")


if __name__ == "__main__":
    main()
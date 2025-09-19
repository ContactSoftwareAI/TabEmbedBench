import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_ADBench_Tabular_20250918_151705.parquet"


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


def create_score_distribution_boxplot(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt ein Boxplot für die Verteilung der Scores nach Embedding-Modell."""
    # Filtere ungültige Werte
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    fig = px.box(
        valid_data.to_pandas(),
        x="embedding_model",
        y=score_col,
        color="embedding_model",
        title=f"Verteilung der {score_col} nach Embedding-Modell",
        points="outliers",
        hover_data=["dataset_name", "algorithm", "num_neighbors"]
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False, height=500)
    return fig


def create_performance_by_dataset_size(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt ein Scatter-Plot für Performance vs. Datensatzgröße."""
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    fig = px.scatter(
        valid_data.to_pandas(),
        x="dataset_size",
        y=score_col,
        color="embedding_model",
        size="num_neighbors",
        hover_data=["dataset_name", "algorithm"],
        title=f"{score_col} vs. Datensatzgröße",
        log_x=True
    )

    fig.update_layout(height=500)
    return fig


def create_computation_time_analysis(df: pl.DataFrame):
    """Erstellt Visualisierungen für die Berechnungszeit-Analyse."""
    # Boxplot für Berechnungszeit nach Embedding-Modell
    fig1 = px.box(
        df.to_pandas(),
        x="embedding_model",
        y="time_to_compute_embeddings",
        color="embedding_model",
        title="Berechnungszeit nach Embedding-Modell",
        log_y=True,
        hover_data=["dataset_name", "dataset_size"]
    )
    fig1.update_xaxes(tickangle=45)
    fig1.update_layout(showlegend=False, height=500)

    # Scatter-Plot: Berechnungszeit vs. Datensatzgröße
    fig2 = px.scatter(
        df.to_pandas(),
        x="dataset_size",
        y="time_to_compute_embeddings",
        color="embedding_model",
        size="emb_dim",
        hover_data=["dataset_name", "algorithm"],
        title="Berechnungszeit vs. Datensatzgröße",
        log_x=True,
        log_y=True
    )
    fig2.update_layout(height=500)

    return fig1, fig2


def create_algorithm_comparison(df: pl.DataFrame, score_col: str = "auc_score"):
    """Erstellt einen Vergleich der verschiedenen Algorithmen."""
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    # Gruppierte Boxplots
    fig = px.box(
        valid_data.to_pandas(),
        x="algorithm",
        y=score_col,
        color="embedding_model",
        title=f"Algorithmus-Vergleich: {score_col}",
        hover_data=["dataset_name", "num_neighbors"]
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600)
    return fig


def create_correlation_heatmap(df: pl.DataFrame):
    """Erstellt eine Korrelations-Heatmap für numerische Spalten."""
    # Wähle numerische Spalten aus
    numeric_cols = ["dataset_size", "num_neighbors", "auc_score", "mse_score",
                    "time_to_compute_embeddings", "emb_dim"]

    # Filtere ungültige Werte
    correlation_data = df.select(numeric_cols).filter(
        (pl.col("auc_score") > -np.inf) &
        (pl.col("auc_score") < np.inf) &
        (pl.col("mse_score") > -np.inf) &
        (pl.col("mse_score") < np.inf)
    )

    # Berechne Korrelationsmatrix
    corr_matrix = correlation_data.to_pandas().corr()

    fig = px.imshow(
        corr_matrix,
        title="Korrelations-Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )

    fig.update_layout(height=500)
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


def create_embedding_dimension_analysis(df: pl.DataFrame, score_col: str = "auc_score"):
    """Analysiert den Effekt der Embedding-Dimension auf die Performance."""
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    # Berechne Durchschnittswerte pro Embedding-Dimension
    avg_scores = (valid_data
                  .group_by(["emb_dim", "algorithm"])
                  .agg([
        pl.col(score_col).mean().alias(f"avg_{score_col}"),
        pl.col("time_to_compute_embeddings").mean().alias("avg_time")
    ])
                  .sort(["algorithm", "emb_dim"]))

    fig = px.scatter(
        avg_scores.to_pandas(),
        x="emb_dim",
        y=f"avg_{score_col}",
        color="algorithm",
        size="avg_time",
        title=f"Durchschnittliche {score_col} vs. Embedding-Dimension",
        hover_data=["avg_time"],
        log_x=True
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
    """Analysiert die Schwierigkeit verschiedener Datensätze."""
    valid_data = df.filter(
        (pl.col(score_col) > -np.inf) &
        (pl.col(score_col) < np.inf) &
        pl.col(score_col).is_not_null()
    )

    # Berechne durchschnittliche Performance pro Datensatz
    dataset_difficulty = (valid_data
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
        title=f"Datensatz-Schwierigkeit (durchschnittliche {score_col})",
        labels={"dataset_name": "Datensatz"}
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600, xaxis_tickangle=45)
    return fig


def generate_summary_statistics(df: pl.DataFrame):
    """Erstellt eine Zusammenfassung der wichtigsten Statistiken."""
    print("\n=== BENCHMARK ERGEBNISSE ZUSAMMENFASSUNG ===")
    print(f"Gesamtzahl der Experimente: {len(df)}")
    print(f"Anzahl der Datensätze: {df['dataset_name'].n_unique()}")
    print(f"Anzahl der Embedding-Modelle: {df['embedding_model'].n_unique()}")
    print(f"Anzahl der Algorithmen: {df['algorithm'].n_unique()}")

    print(f"\nEmbedding-Modelle: {df['embedding_model'].unique().to_list()}")
    print(f"Algorithmen: {df['algorithm'].unique().to_list()}")
    print(f"Distanzmetriken: {df['distance_metric'].unique().to_list()}")

    # AUC Score Statistiken (nur gültige Werte)
    valid_auc = df.filter(
        (pl.col("auc_score") > -np.inf) &
        (pl.col("auc_score") < np.inf)
    )

    if len(valid_auc) > 0:
        print(f"\n--- AUC Score Statistiken ---")
        print(f"Mittelwert: {valid_auc['auc_score'].mean():.4f}")
        print(f"Median: {valid_auc['auc_score'].median():.4f}")
        print(f"Min: {valid_auc['auc_score'].min():.4f}")
        print(f"Max: {valid_auc['auc_score'].max():.4f}")

    # Berechnungszeit Statistiken
    print(f"\n--- Berechnungszeit Statistiken ---")
    print(f"Mittelwert: {df['time_to_compute_embeddings'].mean():.4f} Sekunden")
    print(f"Median: {df['time_to_compute_embeddings'].median():.4f} Sekunden")
    print(f"Min: {df['time_to_compute_embeddings'].min():.4f} Sekunden")
    print(f"Max: {df['time_to_compute_embeddings'].max():.4f} Sekunden")

    # Performance pro Embedding-Modell
    print(f"\n--- Performance nach Embedding-Modell ---")
    model_performance = (valid_auc
                         .group_by("embedding_model")
                         .agg([
        pl.col("auc_score").mean().alias("avg_auc"),
        pl.col("auc_score").std().alias("std_auc"),
        pl.col("time_to_compute_embeddings").mean().alias("avg_time")
    ])
                         .sort("avg_auc", descending=True))

    for row in model_performance.iter_rows(named=True):
        print(f"{row['embedding_model']}: "
              f"AUC={row['avg_auc']:.4f}±{row['std_auc']:.4f}, "
              f"Zeit={row['avg_time']:.4f}s")


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

    # 2. Performance vs. Datensatzgröße
    fig2 = create_performance_by_dataset_size(data, "auc_score")
    fig2.show()

    # 3. Berechnungszeit-Analyse
    fig3, fig4 = create_computation_time_analysis(data)
    fig3.show()
    fig4.show()

    # 4. Algorithmus-Vergleich
    fig5 = create_algorithm_comparison(data, "auc_score")
    fig5.show()

    # 5. Korrelations-Heatmap
    fig6 = create_correlation_heatmap(data)
    fig6.show()

    # 6. Nachbarn-Effekt-Analyse
    fig7 = create_neighbors_effect_analysis(data, "auc_score")
    fig7.show()

    # 7. Embedding-Dimension-Analyse
    fig8 = create_embedding_dimension_analysis(data, "auc_score")
    fig8.show()

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
        (fig2, "performance_vs_dataset_size.html"),
        (fig3, "computation_time_by_model.html"),
        (fig4, "computation_time_vs_size.html"),
        (fig5, "algorithm_comparison.html"),
        (fig6, "correlation_heatmap.html"),
        (fig7, "neighbors_effect_analysis.html"),
        (fig8, "embedding_dimension_analysis.html"),
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
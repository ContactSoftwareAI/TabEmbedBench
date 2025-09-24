import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Pfad zur Parquet-Datei (anpassen Sie diesen Pfad entsprechend)
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_TabArena_20250919_061130.parquet"


def load_tabarena_data(file_path: str) -> pl.DataFrame:
    """Lädt die TabArena-Benchmark-Daten aus der Parquet-Datei."""
    try:
        # Lade Parquet-Datei
        data = pl.read_parquet(file_path)
        print(f"Daten erfolgreich geladen: {data.shape} (Zeilen, Spalten)")
        print(f"Spalten: {data.columns}")
        return data
    except Exception as e:
        print(f"Fehler beim Laden der Parquet-Datei: {e}")
        # Fallback zu CSV falls Parquet fehlschlägt
        try:
            csv_path = file_path.replace(".parquet", ".csv")
            data = pl.read_csv(csv_path)
            print(f"Daten als CSV geladen: {data.shape} (Zeilen, Spalten)")
            print(f"Spalten: {data.columns}")
            return data
        except Exception as e2:
            print(f"Fehler beim Laden als CSV: {e2}")
            return None


def separate_classification_regression(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Trennt die Daten in Classification und Regression Datensätze."""
    if "task" not in df.columns:
        print("Warnung: Keine Task-Spalte gefunden. Versuche automatische Erkennung...")
        # Automatische Erkennung basierend auf verfügbaren Score-Spalten
        classification_data = df.filter(
            pl.col("auc_score").is_not_null() &
            (pl.col("auc_score") != -np.inf)
        )
        regression_data = df.filter(
            pl.col("mse_score").is_not_null() &
            (pl.col("mse_score") != np.inf)
        )
    else:
        classification_data = df.filter(pl.col("task") == "Supervised Classification")
        regression_data = df.filter(pl.col("task") == "Supervised Regression")

    print(f"Classification Experimente: {len(classification_data)}")
    print(f"Regression Experimente: {len(regression_data)}")

    return classification_data, regression_data


def create_classification_performance_overview(df: pl.DataFrame) -> go.Figure:
    """Erstellt eine Performance-Übersicht für Classification Tasks."""
    # Filtere gültige AUC-Daten
    valid_data = df.filter(
        pl.col("auc_score").is_not_null() &
        (pl.col("auc_score") != -np.inf) &
        (pl.col("auc_score") > 0)
    )

    if len(valid_data) == 0:
        print("Keine gültigen Classification-Daten gefunden")
        return None

    # Berechne Durchschnittswerte pro Embedding-Modell
    summary = (valid_data
               .group_by(["embedding_model"])
               .agg([
        pl.col("auc_score").mean().alias("avg_auc"),
        pl.col("auc_score").std().alias("std_auc"),
        pl.col("auc_score").count().alias("count")
    ])
               .sort("avg_auc", descending=True))

    fig = px.bar(
        summary.to_pandas(),
        x="embedding_model",
        y="avg_auc",
        error_y="std_auc",
        title="Classification Performance Overview (AUC Score)",
        labels={"avg_auc": "Average AUC Score", "embedding_model": "Embedding Model"},
        color="avg_auc",
        color_continuous_scale="viridis"
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500, showlegend=False)
    return fig


def create_regression_performance_overview(df: pl.DataFrame) -> go.Figure:
    """Erstellt eine Performance-Übersicht für Regression Tasks."""
    # Filtere gültige MSE-Daten
    valid_data = df.filter(
        pl.col("mse_score").is_not_null() &
        (pl.col("mse_score") != np.inf) &
        (pl.col("mse_score") >= 0)
    )

    if len(valid_data) == 0:
        print("Keine gültigen Regression-Daten gefunden")
        return None

    # Berechne Durchschnittswerte pro Embedding-Modell
    summary = (valid_data
               .group_by(["embedding_model"])
               .agg([
        pl.col("mse_score").mean().alias("avg_mse"),
        pl.col("mse_score").std().alias("std_mse"),
        pl.col("mse_score").count().alias("count")
    ])
               .sort("avg_mse", descending=False))  # Für MSE: niedrigere Werte sind besser

    fig = px.bar(
        summary.to_pandas(),
        x="embedding_model",
        y="avg_mse",
        error_y="std_mse",
        title="Regression Performance Overview (MSE Score)",
        labels={"avg_mse": "Average MSE Score", "embedding_model": "Embedding Model"},
        color="avg_mse",
        color_continuous_scale="viridis_r"  # Umgekehrte Skala da niedrigere MSE besser ist
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500, showlegend=False)
    return fig


def create_classification_boxplot(df: pl.DataFrame) -> go.Figure:
    """Erstellt Boxplots für Classification Performance."""
    valid_data = df.filter(
        pl.col("auc_score").is_not_null() &
        (pl.col("auc_score") != -np.inf) &
        (pl.col("auc_score") > 0)
    )

    if len(valid_data) == 0:
        return None

    fig = px.box(
        valid_data.to_pandas(),
        x="embedding_model",
        y="auc_score",
        color="embedding_model",
        title="Classification Performance Distribution (AUC Score)",
        points="outliers",
        hover_data=["dataset_name"] if "dataset_name" in df.columns else None
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False, height=500)
    return fig


def create_regression_boxplot(df: pl.DataFrame) -> go.Figure:
    """Erstellt Boxplots für Regression Performance."""
    valid_data = df.filter(
        pl.col("mse_score").is_not_null() &
        (pl.col("mse_score") != np.inf) &
        (pl.col("mse_score") >= 0)
    )

    if len(valid_data) == 0:
        return None

    fig = px.box(
        valid_data.to_pandas(),
        x="embedding_model",
        y="mse_score",
        color="embedding_model",
        title="Regression Performance Distribution (MSE Score)",
        points="outliers",
        hover_data=["dataset_name"] if "dataset_name" in df.columns else None,
        log_y=True  # Log-Skala für MSE oft sinnvoll
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False, height=500)
    return fig


def create_classification_dataset_heatmap(df: pl.DataFrame) -> go.Figure:
    """Erstellt eine Heatmap für Classification Performance pro Datensatz."""
    if "dataset_name" not in df.columns:
        return None

    valid_data = df.filter(
        pl.col("auc_score").is_not_null() &
        (pl.col("auc_score") != -np.inf)
    )

    if len(valid_data) == 0:
        return None

    # Berechne durchschnittliche AUC pro Datensatz und Modell
    heatmap_data = (valid_data
                    .group_by(["dataset_name", "embedding_model"])
                    .agg(pl.col("auc_score").mean().alias("avg_auc"))
                    .pivot(values="avg_auc", index="dataset_name", columns="embedding_model"))

    heatmap_df = heatmap_data.to_pandas().set_index("dataset_name")

    fig = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        color_continuous_scale="RdYlBu",
        title="Classification Performance Heatmap (AUC Score)",
        labels=dict(x="Embedding Model", y="Dataset", color="AUC Score")
    )

    fig.update_layout(
        height=max(400, len(heatmap_df.index) * 25),
        xaxis_tickangle=45
    )
    return fig


def create_regression_dataset_heatmap(df: pl.DataFrame) -> go.Figure:
    """Erstellt eine Heatmap für Regression Performance pro Datensatz."""
    if "dataset_name" not in df.columns:
        return None

    valid_data = df.filter(
        pl.col("mse_score").is_not_null() &
        (pl.col("mse_score") != np.inf)
    )

    if len(valid_data) == 0:
        return None

    # Berechne durchschnittliche MSE pro Datensatz und Modell
    heatmap_data = (valid_data
                    .group_by(["dataset_name", "embedding_model"])
                    .agg(pl.col("mse_score").mean().alias("avg_mse"))
                    .pivot(values="avg_mse", index="dataset_name", columns="embedding_model"))

    heatmap_df = heatmap_data.to_pandas().set_index("dataset_name")

    fig = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        color_continuous_scale="RdYlBu_r",  # Umgekehrte Skala für MSE
        title="Regression Performance Heatmap (MSE Score)",
        labels=dict(x="Embedding Model", y="Dataset", color="MSE Score")
    )

    fig.update_layout(
        height=max(400, len(heatmap_df.index) * 25),
        xaxis_tickangle=45
    )
    return fig


def create_neighbors_analysis(df: pl.DataFrame, task_type: str) -> go.Figure:
    """Analysiert den Effekt der Nachbarnanzahl getrennt nach Task-Typ."""
    if "num_neighbors" not in df.columns:
        return None

    if task_type == "classification":
        score_col = "auc_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != -np.inf) &
            pl.col("num_neighbors").is_not_null() &
            (pl.col("num_neighbors") > 0)
        )
        title = "K-Nachbarn Effekt auf Classification Performance (AUC)"
    else:  # regression
        score_col = "mse_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != np.inf) &
            pl.col("num_neighbors").is_not_null() &
            (pl.col("num_neighbors") > 0)
        )
        title = "K-Nachbarn Effekt auf Regression Performance (MSE)"

    if len(valid_data) == 0:
        return None

    # Gruppiere nach Nachbarn und Embedding-Modell
    analysis = (valid_data
                .group_by(["num_neighbors", "embedding_model"])
                .agg(pl.col(score_col).mean().alias(f"avg_{score_col}"))
                .sort(["embedding_model", "num_neighbors"]))

    fig = px.line(
        analysis.to_pandas(),
        x="num_neighbors",
        y=f"avg_{score_col}",
        color="embedding_model",
        markers=True,
        title=title,
        labels={"num_neighbors": "Anzahl Nachbarn (K)", f"avg_{score_col}": f"Durchschnittliche {score_col.upper()}"}
    )

    fig.update_layout(height=500)
    return fig


def create_distance_metric_comparison(df: pl.DataFrame, task_type: str) -> go.Figure:
    """Vergleicht Distanzmetriken getrennt nach Task-Typ."""
    if "distance_metric" not in df.columns:
        return None

    if task_type == "classification":
        score_col = "auc_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != -np.inf)
        )
        title = "Distanzmetrik-Vergleich für Classification (AUC)"
    else:  # regression
        score_col = "mse_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != np.inf)
        )
        title = "Distanzmetrik-Vergleich für Regression (MSE)"

    valid_data = valid_data.filter(
        pl.col("distance_metric").is_not_null() &
        (pl.col("distance_metric") != "")
    )

    if len(valid_data) == 0:
        return None

    fig = px.box(
        valid_data.to_pandas(),
        x="distance_metric",
        y=score_col,
        color="embedding_model",
        title=title,
        labels={"distance_metric": "Distanzmetrik"},
        hover_data=["dataset_name"] if "dataset_name" in df.columns else None
    )

    fig.update_layout(height=500)
    return fig


def create_computation_time_analysis(df: pl.DataFrame, task_type: str) -> tuple[go.Figure, go.Figure]:
    """Analysiert Berechnungszeiten getrennt nach Task-Typ."""
    time_cols = [col for col in df.columns if "time" in col.lower()]

    if not time_cols:
        return None, None

    time_col = "time_to_compute_train_embeddings" if "time_to_compute_train_embeddings" in time_cols else time_cols[0]

    valid_data = df.filter(
        pl.col(time_col).is_not_null() &
        (pl.col(time_col) > 0)
    )

    if len(valid_data) == 0:
        return None, None

    # Boxplot für Berechnungszeiten
    fig1 = px.box(
        valid_data.to_pandas(),
        x="embedding_model",
        y=time_col,
        color="embedding_model",
        title=f"{task_type.title()} - Berechnungszeiten nach Embedding-Modell",
        log_y=True,
        hover_data=["dataset_name"] if "dataset_name" in df.columns else None
    )
    fig1.update_xaxes(tickangle=45)
    fig1.update_layout(showlegend=False, height=500)

    # Scatter-Plot: Zeit vs. Datensatzgröße
    fig2 = None
    if "dataset_size" in df.columns:
        fig2 = px.scatter(
            valid_data.to_pandas(),
            x="dataset_size",
            y=time_col,
            color="embedding_model",
            hover_data=["dataset_name"] if "dataset_name" in df.columns else None,
            title=f"{task_type.title()} - Berechnungszeit vs. Datensatzgröße",
            log_x=True,
            log_y=True
        )
        fig2.update_layout(height=500)

    return fig1, fig2


def generate_task_statistics(df: pl.DataFrame, task_type: str):
    """Generiert Statistiken für einen spezifischen Task-Typ."""
    if task_type == "classification":
        score_col = "auc_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != -np.inf)
        )
    else:  # regression
        score_col = "mse_score"
        valid_data = df.filter(
            pl.col(score_col).is_not_null() &
            (pl.col(score_col) != np.inf)
        )

    if len(valid_data) == 0:
        print(f"Keine gültigen {task_type}-Daten gefunden")
        return

    print(f"\n=== {task_type.upper()} STATISTIKEN ===")
    print(f"Anzahl Experimente: {len(valid_data)}")

    if "dataset_name" in df.columns:
        print(f"Anzahl Datensätze: {valid_data['dataset_name'].n_unique()}")

    # Score-Statistiken
    print(f"\n--- {score_col.upper()} Statistiken ---")
    print(f"Mittelwert: {valid_data[score_col].mean():.4f}")
    print(f"Median: {valid_data[score_col].median():.4f}")
    print(f"Min: {valid_data[score_col].min():.4f}")
    print(f"Max: {valid_data[score_col].max():.4f}")
    print(f"Standardabweichung: {valid_data[score_col].std():.4f}")

    # Performance pro Modell
    print(f"\n--- Performance nach Embedding-Modell ---")
    model_performance = (valid_data
                         .group_by("embedding_model")
                         .agg([
        pl.col(score_col).mean().alias("avg_score"),
        pl.col(score_col).std().alias("std_score"),
        pl.col(score_col).count().alias("num_experiments")
    ])
                         .sort("avg_score", descending=score_col == "auc_score"))

    for row in model_performance.iter_rows(named=True):
        print(f"{row['embedding_model']}: "
              f"{score_col.upper()}={row['avg_score']:.4f}±{row['std_score']:.4f}, "
              f"Experimente={row['num_experiments']}")


def main():
    """Hauptfunktion zur Ausführung der Visualisierungen."""
    # Lade die Daten
    data = load_tabarena_data(path_to_data_file)

    if data is None:
        print(f"Datei {path_to_data_file} konnte nicht geladen werden.")
        return

    # Trenne Classification und Regression
    class_data, reg_data = separate_classification_regression(data)

    # Generiere Statistiken für beide Task-Typen
    if len(class_data) > 0:
        generate_task_statistics(class_data, "classification")

    if len(reg_data) > 0:
        generate_task_statistics(reg_data, "regression")

    # Erstelle Visualisierungen
    print("\nErstelle Visualisierungen...")
    plots_to_save = []

    # CLASSIFICATION VISUALISIERUNGEN
    if len(class_data) > 0:
        print("\n--- Classification Visualisierungen ---")

        # 1. Performance Overview
        print("1. Classification Performance Overview...")
        fig1 = create_classification_performance_overview(class_data)
        if fig1:
            fig1.show()
            plots_to_save.append((fig1, "classification_performance_overview.html"))

        # 2. Boxplot
        print("2. Classification Boxplot...")
        fig2 = create_classification_boxplot(class_data)
        if fig2:
            fig2.show()
            plots_to_save.append((fig2, "classification_boxplot.html"))

        # 3. Dataset Heatmap
        print("3. Classification Dataset Heatmap...")
        fig3 = create_classification_dataset_heatmap(class_data)
        if fig3:
            fig3.show()
            plots_to_save.append((fig3, "classification_dataset_heatmap.html"))

        # 4. Neighbors Analysis
        print("4. Classification Neighbors Analysis...")
        fig4 = create_neighbors_analysis(class_data, "classification")
        if fig4:
            fig4.show()
            plots_to_save.append((fig4, "classification_neighbors_analysis.html"))

        # 5. Distance Metric Comparison
        print("5. Classification Distance Metrics...")
        fig5 = create_distance_metric_comparison(class_data, "classification")
        if fig5:
            fig5.show()
            plots_to_save.append((fig5, "classification_distance_metrics.html"))

        # 6. Computation Time Analysis
        print("6. Classification Computation Times...")
        fig6, fig7 = create_computation_time_analysis(class_data, "classification")
        if fig6:
            fig6.show()
            plots_to_save.append((fig6, "classification_time_by_model.html"))
        if fig7:
            fig7.show()
            plots_to_save.append((fig7, "classification_time_vs_size.html"))

    # REGRESSION VISUALISIERUNGEN
    if len(reg_data) > 0:
        print("\n--- Regression Visualisierungen ---")

        # 1. Performance Overview
        print("1. Regression Performance Overview...")
        fig8 = create_regression_performance_overview(reg_data)
        if fig8:
            fig8.show()
            plots_to_save.append((fig8, "regression_performance_overview.html"))

        # 2. Boxplot
        print("2. Regression Boxplot...")
        fig9 = create_regression_boxplot(reg_data)
        if fig9:
            fig9.show()
            plots_to_save.append((fig9, "regression_boxplot.html"))

        # 3. Dataset Heatmap
        print("3. Regression Dataset Heatmap...")
        fig10 = create_regression_dataset_heatmap(reg_data)
        if fig10:
            fig10.show()
            plots_to_save.append((fig10, "regression_dataset_heatmap.html"))

        # 4. Neighbors Analysis
        print("4. Regression Neighbors Analysis...")
        fig11 = create_neighbors_analysis(reg_data, "regression")
        if fig11:
            fig11.show()
            plots_to_save.append((fig11, "regression_neighbors_analysis.html"))

        # 5. Distance Metric Comparison
        print("5. Regression Distance Metrics...")
        fig12 = create_distance_metric_comparison(reg_data, "regression")
        if fig12:
            fig12.show()
            plots_to_save.append((fig12, "regression_distance_metrics.html"))

        # 6. Computation Time Analysis
        print("6. Regression Computation Times...")
        fig13, fig14 = create_computation_time_analysis(reg_data, "regression")
        if fig13:
            fig13.show()
            plots_to_save.append((fig13, "regression_time_by_model.html"))
        if fig14:
            fig14.show()
            plots_to_save.append((fig14, "regression_time_vs_size.html"))

    # Speichere die Plots als HTML
    output_dir = Path("tabarena_visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"\nSpeichere {len(plots_to_save)} Plots in {output_dir}...")
    for fig, filename in plots_to_save:
        try:
            fig.write_html(output_dir / filename)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Fehler beim Speichern von {filename}: {e}")

    print("\nTabArena-Visualisierungen erfolgreich erstellt!")
    print(f"HTML-Dateien gespeichert in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
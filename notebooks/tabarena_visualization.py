import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_TabArena_20250925_134113.parquet"


def load_tabarena_results(file_path: str) -> pl.DataFrame:
    """Lädt die TabArena-Experiment-Ergebnisse aus der Parquet-Datei."""
    try:
        data = pl.read_parquet(file_path)

        # Entferne physiochemical_protein Datensatz
        data = data.filter(pl.col("dataset_name") != "physiochemical_protein")
        data = data.filter(pl.col("dataset_name") != "superconductivity")

        print(f"Daten erfolgreich geladen: {data.shape} (Zeilen, Spalten)")
        print(f"Spalten: {data.columns}")

        # Zeige einen Überblick über die Tasks
        if "task" in data.columns:
            tasks = data["task"].unique().to_list()
            print(f"Verfügbare Tasks: {tasks}")

        return data
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return None


def separate_by_task_type(df: pl.DataFrame) -> dict:
    """Trennt die Daten nach Task-Typ und identifiziert die entsprechenden Metriken."""
    results = {}

    # Identifiziere verfügbare Spalten für verschiedene Metriken
    columns = df.columns

    # Binary Classification (AUC)
    if "auc_score" in columns:
        binary_class_data = df.filter(
            pl.col("auc_score").is_not_null() &
            (pl.col("auc_score") != -np.inf) &
            (pl.col("auc_score") > 0)
        )
        if len(binary_class_data) > 0:
            results["binary_classification"] = binary_class_data
            print(f"Binary Classification Experimente (AUC): {len(binary_class_data)}")

    # Multi-Class Classification (Log-Loss)
    if "log_loss_score" in columns:
        multi_class_data = df.filter(
            pl.col("log_loss_score").is_not_null() &
            (pl.col("log_loss_score") != np.inf) &
            (pl.col("log_loss_score") > 0)
        )
        if len(multi_class_data) > 0:
            results["multi_class_classification"] = multi_class_data
            print(f"Multi-Class Classification Experimente (Log-Loss): {len(multi_class_data)}")

    # Regression (MAPE)
    if "mape_score" in columns:
        regression_data = df.filter(
            pl.col("mape_score").is_not_null() &
            (pl.col("mape_score") != np.inf) &
            (pl.col("mape_score") >= 0)
        )
        if len(regression_data) > 0:
            results["regression"] = regression_data
            print(f"Regression Experimente (MAPE): {len(regression_data)}")

    return results


def create_detailed_boxplots(task_data: dict) -> dict:
    """Erstellt detaillierte Boxplots für jeden Task-Typ."""
    plots = {}

    for task_type, data in task_data.items():
        # Bestimme Metrik
        if task_type == "binary_classification":
            metric_col = "auc_score"
            title = "Binary Classification Performance (AUC Score)"
        elif task_type == "multi_class_classification":
            metric_col = "log_loss_score"
            title = "Multi-Class Classification Performance (Log-Loss)"
        elif "regression" in task_type:
            if "mape_score" in data.columns:
                metric_col = "mape_score"
                title = "Regression Performance (MAPE Score)"
            else:
                metric_col = "mse_score"
                title = "Regression Performance (MSE Score)"

        fig = px.box(
            data.to_pandas(),
            x="embedding_model",
            y=metric_col,
            color="embedding_model",
            title=title,
            points="outliers",
            hover_data=["dataset_name"] if "dataset_name" in data.columns else None
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            showlegend=False,
            height=500,
            yaxis_title=metric_col.upper().replace("_", " ")
        )

        plots[task_type] = fig

    return plots


def create_dataset_performance_heatmaps(task_data: dict) -> dict:
    """Erstellt Heatmaps für Performance pro Datensatz und Modell."""
    heatmaps = {}

    for task_type, data in task_data.items():
        if "dataset_name" not in data.columns:
            continue

        # Bestimme Metrik
        if task_type == "binary_classification":
            metric_col = "auc_score"
            title = "Binary Classification Performance Heatmap (AUC)"
        elif task_type == "multi_class_classification":
            metric_col = "log_loss_score"
            title = "Multi-Class Classification Performance Heatmap (Log-Loss)"
        elif "regression" in task_type:
            if "mape_score" in data.columns:
                metric_col = "mape_score"
                title = "Regression Performance Heatmap (MAPE)"
            else:
                metric_col = "mse_score"
                title = "Regression Performance Heatmap (MSE)"

        # Pivot-Tabelle für Heatmap erstellen
        heatmap_data = (data
                        .group_by(["dataset_name", "embedding_model"])
                        .agg(pl.col(metric_col).mean().alias(f"avg_{metric_col}"))
                        .pivot(values=f"avg_{metric_col}",
                               index="dataset_name",
                               columns="embedding_model"))

        heatmap_df = heatmap_data.to_pandas().set_index("dataset_name")

        # Bestimme Farbskala basierend auf Metrik
        if task_type == "binary_classification":
            colorscale = "RdYlBu"  # Höhere Werte besser
        else:
            colorscale = "RdYlBu_r"  # Niedrigere Werte besser

        fig = px.imshow(
            heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            color_continuous_scale=colorscale,
            title=title,
            labels=dict(x="Embedding Model", y="Dataset", color=metric_col.upper())
        )

        fig.update_layout(
            height=max(400, len(heatmap_df.index) * 25),
            xaxis_tickangle=45
        )

        heatmaps[task_type] = fig

    return heatmaps


def create_neighbors_effect_analysis(task_data: dict) -> dict:
    """Analysiert den Effekt der K-Nachbarn für alle Task-Typen."""
    neighbor_plots = {}

    for task_type, data in task_data.items():
        if "num_neighbors" not in data.columns:
            continue

        # Filtere nur gültige Nachbar-Werte
        valid_data = data.filter(
            pl.col("num_neighbors").is_not_null() &
            (pl.col("num_neighbors") > 0)
        )

        if len(valid_data) == 0:
            continue

        # Bestimme Metrik
        if task_type == "binary_classification":
            metric_col = "auc_score"
            title = "K-Neighbors Effect on Binary Classification (AUC)"
        elif task_type == "multi_class_classification":
            metric_col = "log_loss_score"
            title = "K-Neighbors Effect on Multi-Class Classification (Log-Loss)"
        elif "regression" in task_type:
            if "mape_score" in valid_data.columns:
                metric_col = "mape_score"
                title = "K-Neighbors Effect on Regression (MAPE)"
            else:
                metric_col = "mse_score"
                title = "K-Neighbors Effect on Regression (MSE)"

        # Gruppiere nach Nachbarn und Modell
        neighbor_analysis = (valid_data
                             .group_by(["num_neighbors", "embedding_model"])
                             .agg(pl.col(metric_col).mean().alias(f"avg_{metric_col}"))
                             .sort(["embedding_model", "num_neighbors"]))

        fig = px.line(
            neighbor_analysis.to_pandas(),
            x="num_neighbors",
            y=f"avg_{metric_col}",
            color="embedding_model",
            markers=True,
            title=title,
            labels={
                "num_neighbors": "Number of Neighbors (K)",
                f"avg_{metric_col}": f"Average {metric_col.upper().replace('_', ' ')}"
            }
        )

        fig.update_layout(height=500)
        neighbor_plots[task_type] = fig

    return neighbor_plots


def generate_comprehensive_statistics(task_data: dict):
    """Generiert umfassende Statistiken für alle Task-Typen."""
    print("\n" + "=" * 80)
    print("EXPERIMENT ERGEBNISSE - COMPREHENSIVE OVERVIEW")
    print("=" * 80)

    for task_type, data in task_data.items():
        print(f"\n--- {task_type.replace('_', ' ').upper()} ---")
        print(f"Anzahl Experimente: {len(data)}")

        if "dataset_name" in data.columns:
            print(f"Anzahl Datensätze: {data['dataset_name'].n_unique()}")

        print(f"Anzahl Embedding-Modelle: {data['embedding_model'].n_unique()}")
        print(f"Embedding-Modelle: {', '.join(data['embedding_model'].unique().to_list())}")

        # Bestimme Metrik für Statistiken
        if task_type == "binary_classification":
            metric_col = "auc_score"
        elif task_type == "multi_class_classification":
            metric_col = "log_loss_score"
        elif "regression" in task_type:
            if "mape_score" in data.columns:
                metric_col = "mape_score"
            else:
                metric_col = "mse_score"

        # Grundlegende Statistiken
        print(f"\n{metric_col.upper().replace('_', ' ')} Statistiken:")
        print(f"  Mittelwert: {data[metric_col].mean():.4f}")
        print(f"  Median: {data[metric_col].median():.4f}")
        print(f"  Min: {data[metric_col].min():.4f}")
        print(f"  Max: {data[metric_col].max():.4f}")
        print(f"  Std: {data[metric_col].std():.4f}")

        # Performance pro Modell
        print(f"\nPerformance nach Embedding-Modell:")
        model_performance = (data
        .group_by("embedding_model")
        .agg([
            pl.col(metric_col).mean().alias("avg_score"),
            pl.col(metric_col).std().alias("std_score"),
            pl.col(metric_col).count().alias("num_experiments")
        ]))

        # Sortiere basierend auf Metrik (AUC: absteigend, andere: aufsteigend)
        ascending = task_type != "binary_classification"
        model_performance = model_performance.sort("avg_score", descending=not ascending)

        for row in model_performance.iter_rows(named=True):
            print(f"  {row['embedding_model']:<25}: "
                  f"{metric_col.upper()}={row['avg_score']:.4f}±{row['std_score']:.4f} "
                  f"({row['num_experiments']} experiments)")


def main():
    """Hauptfunktion zur Erstellung aller Visualisierungen."""
    print("TabArena Experiment Visualizer")
    print("=" * 50)

    # Lade Daten
    data = load_tabarena_results(path_to_data_file)
    if data is None:
        return

    # Separiere nach Task-Typ
    task_data = separate_by_task_type(data)
    if not task_data:
        print("Keine gültigen Task-Daten gefunden!")
        return

    # Generiere Statistiken
    generate_comprehensive_statistics(task_data)

    print("\n" + "=" * 50)
    print("ERSTELLE VISUALISIERUNGEN...")
    print("=" * 50)

    plots_to_save = []


    # 2. Detaillierte Boxplots
    print("2. Erstelle detaillierte Boxplots...")
    boxplot_figs = create_detailed_boxplots(task_data)
    for task_type, fig in boxplot_figs.items():
        fig.show()
        plots_to_save.append((fig, f"{task_type}_detailed_boxplot.html"))

    # 3. Dataset-Performance Heatmaps
    print("3. Erstelle Dataset-Performance Heatmaps...")
    heatmap_figs = create_dataset_performance_heatmaps(task_data)
    for task_type, fig in heatmap_figs.items():
        fig.show()
        plots_to_save.append((fig, f"{task_type}_dataset_heatmap.html"))

    # 4. Nachbarn-Effekt Analyse
    print("4. Erstelle K-Neighbors Effekt Analyse...")
    neighbor_figs = create_neighbors_effect_analysis(task_data)
    for task_type, fig in neighbor_figs.items():
        fig.show()
        plots_to_save.append((fig, f"{task_type}_neighbors_effect.html"))

    # Speichere alle Plots
    output_dir = Path("experiment_visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"\n5. Speichere {len(plots_to_save)} Visualisierungen...")
    for fig, filename in plots_to_save:
        try:
            fig.write_html(output_dir / filename)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Fehler beim Speichern von {filename}: {e}")

    print(f"\n🎉 Alle Visualisierungen erfolgreich erstellt!")
    print(f"📁 HTML-Dateien gespeichert in: {output_dir.absolute()}")

    # Zusammenfassung
    print(f"\n📊 ZUSAMMENFASSUNG:")
    print(f"   • {len(task_data)} Task-Typen analysiert")
    total_experiments = sum(len(data) for data in task_data.values())
    print(f"   • {total_experiments} Experimente insgesamt")
    print(f"   • {len(plots_to_save)} Visualisierungen erstellt")


if __name__ == "__main__":
    main()
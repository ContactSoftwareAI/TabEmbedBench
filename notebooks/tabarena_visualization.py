import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

# Pfad zur Parquet-Datei
path_to_data_file = r"C:\Users\fho\Documents\code\TabData\TabEmbedBench\data\tabembedbench_20250918_151705\results_TabArena_20251001_030811_fixed.parquet"


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


def create_balanced_tabarena_data(df: pl.DataFrame, metric_col: str):
    """
    Erstellt ausgewogene Daten für TabArena-Experimente, indem der beste Wert
    über alle K-Nachbar-Werte pro Experiment genommen wird.
    """
    valid_data = df.filter(
        pl.col(metric_col).is_not_null() &
        (pl.col(metric_col) != np.inf) &
        (pl.col(metric_col) != -np.inf)
    )

    if len(valid_data) == 0:
        return valid_data

    # Für jeden Experiment-Typ nehme den besten Wert über alle K-Nachbar-Werte
    # Bestimme ob höhere oder niedrigere Werte besser sind basierend auf der Metrik
    if metric_col == "auc_score":
        # Für AUC: höhere Werte sind besser -> max()
        agg_func = pl.col(metric_col).max()
    else:
        # Für log_loss und mape: niedrigere Werte sind besser -> min()
        agg_func = pl.col(metric_col).min()

    # Gruppiere nach experiment-definierenden Eigenschaften und nimm besten Wert
    balanced_data = (
        valid_data
        .group_by([
            "dataset_name",
            "embedding_model",
            "metric",
            "task"
        ])
        .agg([
            agg_func.alias(metric_col),
            pl.col("time_to_compute_train_embedding").mean().alias("time_to_compute_train_embeddings"),
            pl.col("embed_dim").first().alias("embedding_dimension"),
            pl.col("dataset_size").first().alias("dataset_size"),
            pl.col("num_neighbors").first().alias("num_neighbors"),  # Info über die beste K
        ])
    )

    return balanced_data


def separate_by_task_type(df: pl.DataFrame) -> dict:
    """Trennt die Daten nach Task-Typ und identifiziert die entsprechenden Metriken."""
    results = {}
    columns = df.columns

    # Binary Classification (AUC)
    if "auc_score" in columns:
        binary_class_data = df.filter(
            pl.col("auc_score").is_not_null() &
            (pl.col("auc_score") != -np.inf) &
            (pl.col("auc_score") > 0)
        )
        if len(binary_class_data) > 0:
            balanced_binary = create_balanced_tabarena_data(binary_class_data, "auc_score")
            results["binary_classification"] = balanced_binary
            print(f"Binary Classification Experimente (AUC, balanced): {len(balanced_binary)}")

    # Multi-Class Classification (Log-Loss)
    if "log_loss_score" in columns:
        multi_class_data = df.filter(
            pl.col("log_loss_score").is_not_null() &
            (pl.col("log_loss_score") != np.inf) &
            (pl.col("log_loss_score") > 0)
        )
        if len(multi_class_data) > 0:
            balanced_multi = create_balanced_tabarena_data(multi_class_data, "log_loss_score")
            results["multi_class_classification"] = balanced_multi
            print(f"Multi-Class Classification Experimente (Log-Loss, balanced): {len(balanced_multi)}")

    # Regression (MAPE)
    if "mape_score" in columns:
        regression_data = df.filter(
            pl.col("mape_score").is_not_null() &
            (pl.col("mape_score") != np.inf) &
            (pl.col("mape_score") >= 0)
        )
        if len(regression_data) > 0:
            balanced_regression = create_balanced_tabarena_data(regression_data, "mape_score")
            results["regression"] = balanced_regression
            print(f"Regression Experimente (MAPE, balanced): {len(balanced_regression)}")

    return results


def create_balanced_boxplots(task_data: dict, selected_models: list[str] = None
) -> dict:
    """Erstellt Boxplots basierend auf den besten K-Werten pro Experiment."""
    plots = {}

    for task_type, data in task_data.items():
        # Filtere nach ausgewählten Modellen, falls angegeben
        if selected_models is not None:
            data = data.filter(pl.col("embedding_model").is_in(selected_models))

            # Überprüfe, ob nach Filterung noch Daten vorhanden sind
            if len(data) == 0:
                print(f"Keine Daten für {task_type} mit den ausgewählten Modellen: {selected_models}")
                continue

        # Bestimme Metrik
        if task_type == "binary_classification":
            metric_col = "auc_score"
            title = "Binary Classification Performance (beste AUC über K-Werte)"
        elif task_type == "multi_class_classification":
            metric_col = "log_loss_score"
            title = "Multi-Class Classification Performance (beste Log-Loss über K-Werte)"
        elif "regression" in task_type:
            if "mape_score" in data.columns:
                metric_col = "mape_score"
                title = "Regression Performance (beste MAPE über K-Werte)"
            else:
                metric_col = "mse_score"
                title = "Regression Performance (beste MSE über K-Werte)"

        fig = px.box(
            data.to_pandas(),
            x="embedding_model",
            y=metric_col,
            color="embedding_model",
            title=title,
            points="outliers",
            hover_data=["dataset_name", "num_neighbors"] if "dataset_name" in data.columns else ["num_neighbors"]
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


# def generate_comprehensive_statistics(task_data: dict):
#     """Generiert umfassende Statistiken für alle Task-Typen."""
#     print("\n" + "=" * 80)
#     print("EXPERIMENT ERGEBNISSE - COMPREHENSIVE OVERVIEW")
#     print("=" * 80)
#
#     for task_type, data in task_data.items():
#         print(f"\n--- {task_type.replace('_', ' ').upper()} ---")
#         print(f"Anzahl Experimente: {len(data)}")
#
#         if "dataset_name" in data.columns:
#             print(f"Anzahl Datensätze: {data['dataset_name'].n_unique()}")
#
#         print(f"Anzahl Embedding-Modelle: {data['embedding_model'].n_unique()}")
#         print(f"Embedding-Modelle: {', '.join(data['embedding_model'].unique().to_list())}")
#
#         # Bestimme Metrik für Statistiken
#         if task_type == "binary_classification":
#             metric_col = "auc_score"
#         elif task_type == "multi_class_classification":
#             metric_col = "log_loss_score"
#         elif "regression" in task_type:
#             if "mape_score" in data.columns:
#                 metric_col = "mape_score"
#             else:
#                 metric_col = "mse_score"
#
#         # Grundlegende Statistiken
#         print(f"\n{metric_col.upper().replace('_', ' ')} Statistiken:")
#         print(f"  Mittelwert: {data[metric_col].mean():.4f}")
#         print(f"  Median: {data[metric_col].median():.4f}")
#         print(f"  Min: {data[metric_col].min():.4f}")
#         print(f"  Max: {data[metric_col].max():.4f}")
#         print(f"  Std: {data[metric_col].std():.4f}")
#
#         # Performance pro Modell
#         print(f"\nPerformance nach Embedding-Modell:")
#         model_performance = (data
#         .group_by("embedding_model")
#         .agg([
#             pl.col(metric_col).mean().alias("avg_score"),
#             pl.col(metric_col).std().alias("std_score"),
#             pl.col(metric_col).count().alias("num_experiments")
#         ]))
#
#         # Sortiere basierend auf Metrik (AUC: absteigend, andere: aufsteigend)
#         ascending = task_type != "binary_classification"
#         model_performance = model_performance.sort("avg_score", descending=not ascending)
#
#         for row in model_performance.iter_rows(named=True):
#             print(f"  {row['embedding_model']:<25}: "
#                   f"{metric_col.upper()}={row['avg_score']:.4f}±{row['std_score']:.4f} "
#                   f"({row['num_experiments']} experiments)")


def generate_balanced_statistics(task_data: dict):
    """Generiert Statistiken basierend auf den besten K-Werten pro Experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT ERGEBNISSE - BALANCED OVERVIEW (BESTE K-WERTE)")
    print("=" * 80)

    for task_type, data in task_data.items():
        print(f"\n--- {task_type.replace('_', ' ').upper()} (BALANCED) ---")
        print(f"Anzahl Experimente (nach Balancing): {len(data)}")

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
        print(f"\n{metric_col.upper().replace('_', ' ')} Statistiken (balanced):")
        print(f"  Mittelwert: {data[metric_col].mean():.4f}")
        print(f"  Median: {data[metric_col].median():.4f}")
        print(f"  Min: {data[metric_col].min():.4f}")
        print(f"  Max: {data[metric_col].max():.4f}")
        print(f"  Std: {data[metric_col].std():.4f}")

        # Performance pro Modell
        print(f"\nPerformance nach Embedding-Modell (balanced):")
        model_performance = (data
        .group_by("embedding_model")
        .agg([
            pl.col(metric_col).mean().alias("avg_score"),
            pl.col(metric_col).std().alias("std_score"),
            pl.col(metric_col).count().alias("num_experiments"),
            pl.col("num_neighbors").mean().alias("avg_best_k")
        ]))

        # Sortiere basierend auf Metrik (AUC: absteigend, andere: aufsteigend)
        ascending = task_type != "binary_classification"
        model_performance = model_performance.sort("avg_score", descending=not ascending)

        for row in model_performance.iter_rows(named=True):
            print(f"  {row['embedding_model']:<25}: "
                  f"{metric_col.upper()}={row['avg_score']:.4f}±{row['std_score']:.4f} "
                  f"{row['num_experiments']} experiments)")


def main():
    """Hauptfunktion zur Erstellung aller Visualisierungen."""
    print("TabArena Experiment Visualizer")
    print("=" * 50)

    # Lade Daten
    data = load_tabarena_results(path_to_data_file)
    if data is None:
        return

    # Separiere nach Task-Typ (ursprünglich)
    task_data_raw = separate_by_task_type(data)
    # Separiere nach Task-Typ (balanced - beste K-Werte)
    task_data_balanced = separate_by_task_type(data)

    if not task_data_balanced:
        print("Keine gültigen Task-Daten gefunden!")
        return

    # Generiere Statistiken für balanced Daten
    generate_balanced_statistics(task_data_balanced)

    # # Optional: Auch Raw-Statistiken anzeigen
    # print("\n" + "=" * 50)
    # print("VERGLEICH: RAW vs BALANCED")
    # print("=" * 50)
    # generate_comprehensive_statistics(task_data_raw)

    print("\n" + "=" * 50)
    print("ERSTELLE VISUALISIERUNGEN...")
    print("=" * 50)

    plots_to_save = []

    selected_models = ["tabicl-classifier-v1.1-0506_preprocessed", "TabVectorizerEmbedding"]
    # 1. Balanced Boxplots (beste K-Werte)
    print("1. Erstelle balanced Boxplots (beste K-Werte)...")
    balanced_boxplot_figs = create_balanced_boxplots(task_data_balanced, selected_models)
    for task_type, fig in balanced_boxplot_figs.items():
        fig.show()
        plots_to_save.append((fig, f"{task_type}_balanced_boxplot.html"))

    # 2. Balanced Dataset-Performance Heatmaps
    print("2. Erstelle balanced Dataset-Performance Heatmaps...")
    balanced_heatmap_figs = create_dataset_performance_heatmaps(task_data_balanced)
    for task_type, fig in balanced_heatmap_figs.items():
        fig.show()
        plots_to_save.append((fig, f"{task_type}_balanced_heatmap.html"))

    # # 4. Nachbarn-Effekt Analyse (bleibt wie es ist)
    # print("4. Erstelle K-Neighbors Effekt Analyse...")
    # neighbor_figs = create_neighbors_effect_analysis(task_data_raw)
    # for task_type, fig in neighbor_figs.items():
    #     fig.show()
    #     plots_to_save.append((fig, f"{task_type}_neighbors_effect.html"))

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
    print(f"   • {len(task_data_balanced)} Task-Typen analysiert (balanced)")
    total_balanced = sum(len(data) for data in task_data_balanced.values())
    total_raw = sum(len(data) for data in task_data_raw.values())
    print(f"   • {total_raw} Experimente roh → {total_balanced} Experimente balanced")
    print(f"   • {len(plots_to_save)} Visualisierungen erstellt")


if __name__ == "__main__":
    main()
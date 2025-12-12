import scikit_posthocs as sp
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import os
import re
from datetime import datetime
from pathlib import Path
import seaborn as sns

embedding_models = ['TabICL',
                    'TabPFN',
                    'TableVectorizer',
                    'Random Line (Dim 64)',
                    'Random Line (Dim 192)',
                    'Random Line (Dim 512)',
                    'Random Circle (Dim 64)',
                    'Random Circle (Dim 192)',
                    'Random Circle (Dim 512)',
                    'Sphere-Based Line (Dim 64)',
                    'Sphere-Based Line (Dim 192)',
                    'Sphere-Based Line (Dim 512)',
                    'Sphere-Based Circle (Dim 64)',
                    'Sphere-Based Circle (Dim 192)',
                    'Sphere-Based Circle (Dim 512)',
                    'Text-Based'
                    ]
colors = [
    "#0080C5", # CIM Database 1
    "#F29100", # Project Office 1
    "#92C108", # Workspaces 1
    "#A90E4E", # IoT 3
    "#A90E4E", # IoT 3
    "#A90E4E", # IoT 3
    "#E3075A", # IoT 1
    "#E3075A", # IoT 1
    "#E3075A", # IoT 1
    "#FCB900", # Project Office 2
    "#FCB900", # Project Office 2
    "#FCB900", # Project Office 2
    "#EA5A02", # Project Office 3
    "#EA5A02", # Project Office 3
    "#EA5A02", # Project Office 3
    "#003254", # CONTACT 1
    "#005E9E", # CIM Database 3
    "#36AEE7", # CIM Database 2
    "#DBDC2E", # Workspaces 2
    "#639C2E", # Workspaces 3
    "#CC171D", # CONTACT rot
    "#EE5CA1", # IoT 2
]
color_dict = {embedding_models[i]: colors[i] for i in range(len(embedding_models))}
print(color_dict)

maximize = ["auc_score"]
names = {"auc_score": "AUC Score",
         "mape_score": "MAPE Score"}

timestamps = ["20251112_161222","20251204_094445","20251205_071358","20251208_084628","20251208_151933","20251210_073948"]
directory = f"C:/Users/arf/TabEmbedBench/src/tabembedbench/examples/data"
result_outlier_files = [f"{directory}/tabembedbench_{timestamp}/results_ADBench_Tabular_{timestamp}.csv" for timestamp in timestamps]
result_tabarena_files = [f"{directory}/tabembedbench_{timestamp}/results_TabArena_{timestamp}.csv" for timestamp in timestamps]
output_dir = Path(f"{directory}/figures_{datetime.now().strftime("%Y%m%d_%H%M%S")}")
output_dir.mkdir(parents=True, exist_ok=True)

benchmark = {"Outlier": {"result_files": result_outlier_files,
                          "task": "Outlier Detection",
                          "measure": "auc_score",
                          "algorithms": ["LocalOutlierFactor","IsolationForest","DeepSVDD","DeepSVDD-dynamic"]},
             "Binary Classification": {"result_files": result_tabarena_files,
                                        "task": "classification",
                                        "classification_type": "binary",
                                        "measure": "auc_score",
                                        "algorithms": ["KNNClassifier","MLPClassifier"]},
             "Multiclass Classification": {"result_files": result_tabarena_files,
                                            "task": "classification",
                                            "classification_type": "multiclass",
                                            "measure": "auc_score",
                                            "algorithms": ["KNNClassifier","MLPClassifier"]},
             "Regression": {"result_files": result_tabarena_files,
                             "task": "regression",
                             "measure": "mape_score",
                             "algorithms": ["KNNRegressor","MLPRegressor"]}
             }

data_complete_cum = []

for b in benchmark:
    if all(os.path.exists(filename) for filename in benchmark[b]["result_files"]):
        print(f"{b}:")
        pl_data = [pl.read_csv(filename) for filename in benchmark[b]["result_files"]]
        for algorithm in benchmark[b]["algorithms"]:
            print(algorithm)
            pl_data_temp = []
            for i in range(len(pl_data)):
                pl_data_temp.append(pl_data[i].filter(pl.col("algorithm") == algorithm).filter(pl.col("task") == benchmark[b]["task"]))
                if "classification_type" in benchmark[b]:
                    pl_data_temp[i] = pl_data_temp[i].filter(pl.col("classification_type") == benchmark[b]["classification_type"])
            data = pd.concat([(
                pl_data_temp[i].select(["dataset_name",benchmark[b]["measure"],"embedding_model"])
                .to_pandas()) for i in range(len(pl_data))])
#            data = pd.concat([(
#                pl_data[i].filter(pl.col("algorithm") == algorithm).filter(pl.col("task") == benchmark[b]["task"]).filter(pl.col("classification_type") == benchmark[b]["classification_type"])
#                .select(["dataset_name",benchmark[b]["measure"],"embedding_model"])
#                .to_pandas()) for i in range(len(pl_data))])
            if benchmark[b]["measure"] in maximize:
                data = data.loc[data.groupby(['dataset_name','embedding_model'])[benchmark[b]["measure"]].idxmax()].drop_duplicates().reset_index(drop=True)
            else:
                data = data.loc[data.groupby(['dataset_name','embedding_model'])[benchmark[b]["measure"]].idxmin()].drop_duplicates().reset_index(drop=True)

            pivot = data.pivot_table(index=['dataset_name'],
                                     columns='embedding_model',
                                     values=benchmark[b]["measure"])
            incomplete_datasets = pivot[pivot.isnull().any(axis=1)].index
            data_complete = data[~data['dataset_name'].isin(incomplete_datasets)]
            print("data_complete:")
            print(data_complete)

            data_cum = pd.DataFrame(data_complete)
            if benchmark[b]["measure"] not in maximize:
                data_cum[benchmark[b]["measure"]] = (-1) * data_cum[benchmark[b]["measure"]]
            data_cum["dataset_name"] = f"{b}_{algorithm}_" + data_cum["dataset_name"].astype(str)
            data_complete_cum.append(data_cum.rename(columns={benchmark[b]["measure"]: "measure"}))

            pivot = data_complete.pivot_table(index=['dataset_name'],
                                              columns='embedding_model',
                                              values=benchmark[b]["measure"])

            print("pivot:")
            print(pivot)

            nemenyi_friedman = sp.posthoc_nemenyi_friedman(data_complete, y_col=benchmark[b]["measure"], block_col='dataset_name', group_col='embedding_model', block_id_col='dataset_name', melted=True)

            print("nemenyi_friedman:")
            print(nemenyi_friedman)

            rankmat = pivot.rank(axis='columns', ascending=(benchmark[b]["measure"] not in maximize))
            print("rankmat:")
            print(rankmat)
            meanranks = rankmat.mean()
            print("meanranks:")
            print(meanranks)

            fig, ax = plt.subplots(figsize=(10, 3))

            plt.title(b+": "+algorithm)
            sp.critical_difference_diagram(meanranks, nemenyi_friedman, label_props={}, ax=ax)
            # Get all text objects (algorithm labels)
            texts = [t for t in ax.texts if t.get_text().strip()]

            # Get lines except grouping bars
            elbows = [e for e in ax.lines if e.get_color() != 'k']

            # Get the axis markers (scatter points)
            markers = ax.collections

            # Process each text label
            for i, text in enumerate(texts):
                original_text = text.get_text()
                cleaned_text = re.sub(r'\s*\([0-9.]+\)\s*', '', original_text).strip()
                text.set_text(cleaned_text)

                if cleaned_text in color_dict and i < len(elbows):
                    color = color_dict[cleaned_text]
                    text.set_color(color)
                    elbows[i].set_color(color)

                    # Color the corresponding marker (scatter point on axis)
                    if i < len(markers):
                        markers[i].set_color(color)
                        markers[i].set_edgecolor(color)
                        markers[i].set_facecolor(color)

            plt.tight_layout()

            plt.savefig(os.path.join(output_dir,f"{b}_{algorithm}_cd.pdf"),dpi=300,pad_inches=0.02)
            plt.close()

            plt.figure(figsize=(8, 8))  # Adjust figure size for better readability

            boxplot = sns.boxplot(
                data=data_complete,
                x='embedding_model',  # The categorical variable to create individual boxes for
                y=benchmark[b]["measure"],  # The numerical variable whose distribution each box will represent
                hue='embedding_model',
                order=embedding_models,  # Apply the custom order to the x-axis categories
                hue_order=embedding_models,  # Crucial: ensures color mapping and legend order match x-axis
                palette=color_dict  # Use your custom color dictionary
            # The 'hue' parameter is typically used when you want to split each x-category
                # further by another categorical variable, leading to grouped boxes.
                # If you just want one box per embedding_model, you don't need 'hue' based on the same column.
            )

            # 4. Add labels, title, and other customizations
            boxplot.set_title(b+": "+algorithm)
            boxplot.set_ylabel(names[benchmark[b]["measure"]])
            boxplot.set_xlabel("")

            # Rotate x-axis labels if they are long to prevent overlap
            plt.xticks(rotation=45, ha='right')  # 'ha' (horizontal alignment) helps position rotated labels

            # Add a grid for better readability of the y-axis values
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()  # Adjust layout to prevent labels/title from being cut off

            plt.savefig(os.path.join(output_dir,f"{b}_{algorithm}_boxplot.pdf"),dpi=300,pad_inches=0.02)
            plt.close()

data_complete_cum = pd.concat(data_complete_cum, ignore_index=True)
print(data_complete_cum)
pivot = data_complete_cum.pivot_table(index=['dataset_name'],
                                      columns='embedding_model',
                                      values="measure")

print("pivot:")
print(pivot)

nemenyi_friedman = sp.posthoc_nemenyi_friedman(data_complete_cum, y_col="measure", block_col='dataset_name',
                                               group_col='embedding_model', block_id_col='dataset_name', melted=True)

print("nemenyi_friedman:")
print(nemenyi_friedman)

rankmat = pivot.rank(axis='columns', ascending=False)
print("rankmat:")
print(rankmat)
meanranks = rankmat.mean()
print("meanranks:")
print(meanranks)

fig, ax = plt.subplots(figsize=(10, 3))

plt.title("Cumulative")
sp.critical_difference_diagram(meanranks, nemenyi_friedman, label_props={}, ax=ax)
# Get all text objects (algorithm labels)
texts = [t for t in ax.texts if t.get_text().strip()]

# Get lines except grouping bars
elbows = [e for e in ax.lines if e.get_color() != 'k']

# Get the axis markers (scatter points)
markers = ax.collections

# Process each text label
for i, text in enumerate(texts):
    original_text = text.get_text()
    cleaned_text = re.sub(r'\s*\([0-9.]+\)\s*', '', original_text).strip()
    text.set_text(cleaned_text)

    if cleaned_text in color_dict and i < len(elbows):
        color = color_dict[cleaned_text]
        text.set_color(color)
        elbows[i].set_color(color)

        # Color the corresponding marker (scatter point on axis)
        if i < len(markers):
            markers[i].set_color(color)
            markers[i].set_edgecolor(color)
            markers[i].set_facecolor(color)

plt.tight_layout()

plt.savefig(os.path.join(output_dir, f"Cumulative_cd.pdf"), dpi=300, pad_inches=0.02)
plt.close()

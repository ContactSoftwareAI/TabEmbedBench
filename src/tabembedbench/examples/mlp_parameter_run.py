import polars as pl
import openml
import numpy as np
from sklearn.metrics import roc_auc_score
from logging import getLogger

from tabembedbench.evaluators.mlp_evaluator import PyTorchMLPWrapper
from tabembedbench.embedding_models import (
    TabICLEmbedding,
    TabVectorizerEmbedding,
    AbstractEmbeddingGenerator
)
from tabembedbench.utils.torch_utils import get_device
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical
from pathlib import Path


logger = getLogger("Afterwords_run")

def run_after_benchmark_experiments(
        result_df: pl.DataFrame,
        embedding_models: list[AbstractEmbeddingGenerator],
        output_path: str,
        epochs: int = 1000,
        device= None
):
    logger.info("Start")
    print("Start")
    device = device if device is not None else get_device()

    filtered_result = (
        result_df.filter(
            pl.col("algorithm") == "MLPClassifier",
            pl.col("classification_type") == "binary"
        )
    )

    included_datasets = filtered_result.get_column("dataset_name").unique().to_list()

    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = benchmark_suite.tasks

    final_result_df = pl.DataFrame()

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if dataset.name not in included_datasets:
            continue
        dataset_results = (
            filtered_result.filter(
                pl.col("dataset_name") == dataset.name
            )
        )

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        categorical_indices = np.nonzero(categorical_indicator)[0]
        categorical_indices = categorical_indices.tolist()

        train_indices, test_indices = task.get_train_test_split_indices(
            fold=1,
            repeat=1,
        )

        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        # Preprocess data
        numerical_transformer = TransformToNumerical()
        X_train = numerical_transformer.fit_transform(X_train)
        X_test = numerical_transformer.transform(X_test)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        for embedding_model in embedding_models:
            logger.info(f"Run for {dataset.name}, {embedding_model.name}")
            print(f"Run for {dataset.name}, {embedding_model.name}")
            model_result = dataset_results.filter(
                pl.col("embedding_model") == embedding_model.name
            )

            if len(model_result) == 1:
                X_train_emb, _, X_test_emb, _ = embedding_model.generate_embeddings(
                    X_train=X_train, X_test=X_test,
                    categorical_indices=categorical_indices, outlier=False
                )

                hidden_dim = (
                    model_result.select(
                        pl.selectors.starts_with("algorithm_best_hidden_dim")
                    )
                )
                hidden_layer_dims = []
                for hidden_layer in sorted(hidden_dim.columns):
                    value = hidden_dim.item(0, hidden_layer)
                    if value is not None:
                        hidden_layer_dims.append(value)

                row = model_result.row(0, named=True)

                if len(hidden_layer_dims) != row["algorithm_best_n_layers"]:
                    raise ValueError()

                evaluator = PyTorchMLPWrapper(
                    input_dim=X_train_emb.shape[1],
                    hidden_dims=hidden_layer_dims,
                    output_dim=len(np.unique(y_train)),
                    dropout=row["algorithm_best_dropout"],
                    learning_rate=row["algorithm_best_learning_rate"],
                    batch_size=row["algorithm_best_batch_size"],
                    epochs=epochs,
                    task_type="classification",
                    device=device,
                    early_stopping_patience=200
                )

                evaluator.fit(X_train_emb, y_train)

                y_pred = evaluator.predict(X_test_emb)

                auc_score = roc_auc_score(y_test, y_pred)
                print(auc_score)
                result_dict = {}
                logger.info(len(evaluator.loss_history_))
                print(len(evaluator.loss_history_))
                for epoch_dict in evaluator.loss_history_:
                    result_dict["embedding_model"] = embedding_model.name
                    result_dict["dataset_name"] = dataset.name
                    result_dict["epoch"] = epoch_dict["epoch"]
                    result_dict["loss"] = epoch_dict["loss"]
                    result_dict["auc_score"] = auc_score

                    new_row = pl.DataFrame(result_dict)
                    final_result_df = pl.concat([final_result_df, new_row], how="diagonal")

                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)

                output_file = Path(
                    output_path / f"after_result_experiments_tabarena"
                )

                parquet_file = output_file.with_suffix(".parquet")
                csv_file = output_file.with_suffix(".csv")

                final_result_df.write_parquet(parquet_file)
                final_result_df.write_csv(csv_file)
            else:
                continue


if __name__ == "__main__":
    import torch
    file_path = "/Users/lkl/PycharmProjects/TabEmbedBench/data/tabembedbench_20251007_190514/results_TabArena_20251007_190514.parquet"
    result_df = pl.read_parquet(file_path)

    output_path = "/Users/lkl/PycharmProjects/TabEmbedBench/data/tabembedbench_20251007_190514/"

    tabicl_with_preproccessing = TabICLEmbedding(preprocess_tabicl_data=True)

    tabicl_with_preproccessing.name = "tabicl-classifier-v1.1-0506_preprocessed"

    tablevector = TabVectorizerEmbedding()
    tablevector.name = "TabVectorizerEmbedding"

    embedding_models = [tabicl_with_preproccessing, tablevector]

    run_after_benchmark_experiments(
        result_df,
        embedding_models,
        epochs=500,
        output_path=output_path,
        device=torch.device("mps")
    )






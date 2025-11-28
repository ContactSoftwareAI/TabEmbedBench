from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_absolute_percentage_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabembedbench.benchmark import AbstractBenchmark
from tabembedbench.evaluators import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class DatasetBenchmark(AbstractBenchmark):
    def __init__(
        self,
        dataset_path: str,
        target_column,
        feature_columns = None,
        categorical_columns = None,
        numerical_columns = None,
        test_size: float = 0.2,
        random_state: int = 42,
        dataset_name: str = None,
        result_dir: str = None,
        task_type: str = "Supervised Classification",
        timestamp: str = TIMESTAMP,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 100000,
        upper_bound_num_features: int = 500,
    ):
        dataset_name = dataset_name if dataset_name else Path(dataset_path).stem
        result_dir = result_dir if result_dir else f"result_{dataset_name}"

        super().__init__(
            logger_name=f"Dataset_{dataset_name}",
            task_type=task_type,
            result_dir=result_dir,
            timestamp=timestamp,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def _load_datasets(self, **kwargs) -> list[pd.DataFrame]:
        return [pd.read_csv(self.dataset_path)]

    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str | None]:
        return False, None

    def _prepare_dataset(self, dataset: pd.DataFrame, **kwargs) -> Iterator[dict]:
        y = dataset[self.target_column]
        X = dataset.drop(columns=[self.target_column])
        if not self.feature_columns:
            self.feature_columns = X.columns.tolist()

        X = X[self.feature_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.task_type == "Supervised Classification":
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

        yield {
            "X": None,
            "X_train": X_train,
            "X_test": X_test,
            "y": None,
            "y_train": y_train,
            "y_test": y_test,
            "dataset_name": self.dataset_name,
            "dataset_size": X.shape[0],
            "num_features": X.shape[-1],
            "metadata": {
                "task_type": self.task_type,
                "categorical_column_names": self.categorical_columns,
                "numerical_columns": self.numerical_columns,
            }
        }

    def _get_benchmark_name(self) -> str:
        return f"Run_Dataset_{self.dataset_name}"

    def _evaluate(
        self,
        embeddings: tuple,
        evaluator: AbstractEvaluator,
        data_split: dict,
    ) -> dict:
        """Evaluate embeddings for classification or regression.

        Args:
            embeddings: Tuple of (train_embeddings, test_embeddings, compute_time).
            evaluator: The evaluator to use.
            data_split: Dictionary with data and metadata.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings, test_embeddings, compute_time = embeddings
        y_train = data_split["y_train"]
        y_test = data_split["y_test"]
        task_type = data_split["metadata"]["task_type"]

        # Train evaluator
        prediction_train, _ = evaluator.get_prediction(
            train_embeddings,
            y_train,
            train=True,
        )

        # Get test predictions
        test_prediction, _ = evaluator.get_prediction(
            test_embeddings,
            train=False,
        )

        # Build result dictionary
        result_dict = {
            "dataset_name": [data_split["dataset_name"]],
            "dataset_size": [data_split["dataset_size"]],
            "num_features": [data_split["num_features"]],
            "embed_dim": [train_embeddings.shape[-1]],
            "time_to_compute_embedding": [compute_time],
            "algorithm": [evaluator._name],
        }

        # Compute task-specific metrics
        if task_type == "Supervised Regression":
            mape_score = mean_absolute_percentage_error(y_test, test_prediction)
            result_dict["task"] = ["regression"]
            result_dict["mape_score"] = [mape_score]

        elif task_type == "Supervised Classification":
            n_classes = test_prediction.shape[1]
            if n_classes == 2:
                auc_score = roc_auc_score(y_test, test_prediction[:, 1])
                result_dict["task"] = ["classification"]
                result_dict["classification_type"] = ["binary"]
            else:
                auc_score = roc_auc_score(y_test, test_prediction, multi_class="ovr")
                log_loss_score = log_loss(y_test, test_prediction)
                result_dict["task"] = ["classification"]
                result_dict["classification_type"] = ["multiclass"]
                result_dict["log_loss_score"] = [log_loss_score]
            result_dict["auc_score"] = [auc_score]

        # Add evaluator parameters
        evaluator_params = evaluator.get_parameters()
        for key, value in evaluator_params.items():
            result_dict[f"algorithm_{key}"] = [value]

        return result_dict

if __name__ == "__main__":
    from tabembedbench.examples.eurips_run import get_evaluators, get_embedding_models

    embedding_models = get_embedding_models(debug=True)
    evaluators = get_evaluators(debug=True)

    csv_path = ""
    target_column = ""
    task_type = "Supervised Classification"

    dataset_benchmark = DatasetBenchmark(
        dataset_path=csv_path,
        target_column=target_column,
        task_type=task_type,
    )

    result = dataset_benchmark.run_benchmark(
        embedding_models=embedding_models,
        evaluators=evaluators,
    )

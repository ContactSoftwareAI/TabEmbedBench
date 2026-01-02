from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_absolute_percentage_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabembedbench.benchmark import AbstractBenchmark
from tabembedbench.evaluators import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class DatasetBenchmark(AbstractBenchmark):
    """
    Handles dataset benchmarking operations.

    This class is responsible for managing datasets, preparing them, and evaluating
    embeddings using an evaluator. It supports operations like loading datasets,
    splitting into train and test datasets, and applying task-specific transformations
    (such as encoding for classification). The primary aim of this class is to provide
    a benchmarking framework for datasets in supervised learning tasks, including
    classification and regression.

    Attributes:
        dataset_name (str): Name of the dataset.
        dataset_path (str): Path to the dataset file.
        target_column (str): Name of the target variable column.
        feature_columns (list[str] or None): List of feature column names. If None, all
            columns except the target column will be used.
        categorical_columns (list[str] or None): List of categorical feature column names.
        numerical_columns (list[str] or None): List of numerical feature column names.
        target_column (str): Name of the target variable column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility of train-test splits.
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        numerical_columns: Optional[list[str]] = None,
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
        if "set" in dataset:
            data_train = dataset[dataset["set"]!="test"]
            data_test = dataset[dataset["set"]=="test"]
            y_train = data_train[self.target_column]
            y_test = data_test[self.target_column]
            X_train = data_train.drop(columns=[self.target_column])
            if not self.feature_columns:
                self.feature_columns = X_train.columns.tolist()
            X_train = X_train[self.feature_columns]
            X_test = data_test.drop(columns=[self.target_column])
            X_test = X_test[self.feature_columns]

        else:

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
            "dataset_size": dataset.shape[0],
            "num_features": X_train.shape[-1],
            "metadata": {
                "task_type": self.task_type,
                "categorical_column_names": self.categorical_columns,
                "numerical_columns": self.numerical_columns,
            },
        }

    def _get_benchmark_name(self) -> str:
        return f"Run_Dataset_{self.dataset_name}"

    def _get_evaluator_prediction(
        self,
        embeddings: tuple,
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> dict:
        """Evaluate embeddings for classification or regression.

        Args:
            embeddings: Tuple of (train_embeddings, test_embeddings, compute_time).
            evaluator: The evaluator to use.
            dataset_configurations: Dictionary with data and metadata.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings, test_embeddings, compute_time = embeddings
        y_train = dataset_configurations["y_train"]
        y_test = dataset_configurations["y_test"]
        task_type = dataset_configurations["metadata"]["task_type"]

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
            "dataset_name": [dataset_configurations["dataset_name"]],
            "dataset_size": [dataset_configurations["dataset_size"]],
            "num_features": [dataset_configurations["num_features"]],
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
    categorical_columns = []
    numerical_columns = []

    dataset_benchmark = DatasetBenchmark(
        dataset_path=csv_path,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        feature_columns=categorical_columns+numerical_columns,
        feature_columns=feature_columns if len(feature_columns) > 0 else None,
        task_type=task_type,
    )

    result = dataset_benchmark.run_benchmark(
        embedding_models=embedding_models,
        evaluators=evaluators,
    )

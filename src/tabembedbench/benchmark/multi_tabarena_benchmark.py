from datetime import datetime
from pathlib import Path

import numpy as np
import openml
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench import evaluators
from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator
from tabembedbench.evaluators.abstractevaluator import AbstractEvaluator


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class MultiTabArenaBenchmark(AbstractBenchmark):
    def __init__(
            self,
            tabarena_version: str = "tabarena-v0.1",
            tabarena_lite: bool = True,
            result_dir: str | Path = "result_tabarena",
            timestamp: str = TIMESTAMP,
            save_result_dataframe: bool = True,
            upper_bound_num_samples: int = 100000,
            upper_bound_num_features: int = 500,
            max_number_of_dataset: int = 5,
            use_all_datasets: bool = True,
            random_seed: int = 42
    ):
        super().__init__(
            logger_name="TabEmbedBench_MultiTabArena",
            result_dir=result_dir,
            timestamp=timestamp,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )

        self.tabarena_version = tabarena_version
        self.tabarena_lite = tabarena_lite
        self.benchmark_suite = None
        self.task_ids = None
        self.max_number_of_dataset = max_number_of_dataset
        self.use_all_datasets = use_all_datasets
        self.random_seed = random_seed

        self.rng = np.random.default_rng(self.random_seed)

    def run_benchmark(
            self,
            embedding_models: list[AbstractEmbeddingGenerator],
            evaluators: list[AbstractEvaluator],
            **kwargs,
    ) -> pl.DataFrame:
        dataset_batches = self._load_datasets(**kwargs)

        for dataset_batch in dataset_batches:
            self._process_batch(
                dataset_batch,
                embedding_models,
                evaluators,
                **kwargs
            )

        self.logger.info(f"{self._get_benchmark_name()} benchmark completed.")
        return self.result_df

    def _process_batch(
            self,
            dataset_batch,
            embedding_models: list[AbstractEmbeddingGenerator],
            evaluators: list[AbstractEvaluator],
            **kwargs
    ):
        for batch in dataset_batch:
            prepared_batch = self._prepare_batch(batch)

            for embedding_model in embedding_models:
                for prepared_dataset in prepared_batch:
                    X_train_emb = []
                    X_test_emb = []
                    y_train_emb = []
                    y_test_emb = []
                    try:
                        embeddings_train, time_train_embeddings, embeddings_test, time_test_embeddings, \
                            = embedding_model.generate_embeddings(
                            X_train=prepared_dataset.X_train,
                            X_test=prepared_dataset.X_test,
                        )
                    except Exception as e:
                        print(e)

                    X_train_emb.append(embeddings_train)
                    X_test_emb.append(embeddings_test)
                    y_train_emb.append(prepared_dataset.y_train)
                    y_test_emb.append(prepared_dataset.y_test)

            #TODO: concat X_train_emb, and X_test_emb to get final input for
            # evaluators, same for y_train.

                result_dict = {}

                for evaluator in evaluators:
                    prediction_train, _ = evaluator.get_prediction(
                        X_train_emb,
                        y_train_emb,
                        train=True,
                    )

                    prediction_test, _ = evaluator.get_prediction(
                        X_test_emb,
                        train=False,
                    )

                    n_classes = prediction_test.shape[1]
                    if n_classes == 2:
                        auc_score = roc_auc_score(y_test_emb, prediction_test[:,
                        1])
                        result_dict["task"] = ["classification"]
                        result_dict["classification_type"] = ["binary"]
                    else:
                        auc_score = roc_auc_score(
                            y_test_emb, prediction_test, multi_class="ovr"
                        )
                        log_loss_score = log_loss(y_test_emb, prediction_test)
                        result_dict["task"] = ["classification"]
                        result_dict["classification_type"] = ["multiclass"]
                        result_dict["log_loss_score"] = [log_loss_score]
                    result_dict["auc_score"] = [auc_score]

    def _prepare_batch(self, batch):
        label_offset = 0

        prepared_batch = []

        for task_id in batch:
            prepared_dataset = {}

            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )

            train_indices, test_indices = task.get_train_test_split_indices(
                fold=1,
                repeat=1,
            )

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

            numerical_transformer = TransformToNumerical()
            X_train = numerical_transformer.fit_transform(X_train)
            X_test = numerical_transformer.transform(X_test)

            # Encode labels
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            # Offset labels to avoid overlap across datasets
            max_label = np.max(y_train)
            y_train = y_train + label_offset
            y_test = y_test + label_offset
            label_offset += max_label + 1

            prepared_dataset["X_train"] = X_train
            prepared_dataset["y_train"] = y_train
            prepared_dataset["X_test"] = X_test
            prepared_dataset["y_test"] = y_test

            prepared_batch.append(prepared_dataset)

        return prepared_batch

    def _load_datasets(self, **kwargs):
        """Load TabArena tasks and create batches of datasets.

        Returns:
            List of dictionaries containing batch information.
        """
        self.benchmark_suite = openml.study.get_suite(self.tabarena_version)
        self.task_ids = self.benchmark_suite.tasks

        binary_classification_datasets = []
        multiclass_datasets = []

        for task_id in self.task_ids:
            task = openml.tasks.get_task(task_id)
            if task.task_type == "Supervised Classification":
                dataset = task.get_dataset()
                if dataset.qualities["NumberOfInstances"] > self.upper_bound_num_samples:
                    continue
                if dataset.qualities["NumberOfFeatures"] > self.upper_bound_num_features:
                    continue

                num_class = len(dataset.retrieve_class_labels())

                if num_class > 2:
                    multiclass_datasets.append(task_id)
                else:
                    binary_classification_datasets.append(task_id)
            else:
                continue

        binary_batches = self._get_dataset_choices(binary_classification_datasets)
        multiclass_batches = self._get_dataset_choices(multiclass_datasets)

        # Create dataset info for each batch
        all_batches = []
        
        for batch in binary_batches:
            batch_info = self._get_batches_metadata(batch, "binary")
            all_batches.append(batch_info)
        
        for batch in multiclass_batches:
            batch_info = self._get_batches_metadata(batch, "multiclass")
            all_batches.append(batch_info)

        return all_batches

    def _get_dataset_choices(self, classification_ids):
        ids_array = np.array(classification_ids)
        self.rng.shuffle(ids_array)

        batches = []
        start = 0
        while start < len(ids_array):
            remaining = len(ids_array) - start
            if remaining > 1:
                size = self.rng.integers(
                    2, min(self.max_number_of_dataset, remaining)+1
                )
                batches.append(ids_array[start:start+size])
            else:
                size = 1
                batches.append(ids_array[start:start+1])
            start += size

        return batches

    def _get_batches_metadata(self, task_ids: np.ndarray, classification_type: str) -> dict:
        """Get metadata for a batch of tasks.

        Args:
            task_ids: Array of task IDs in the batch.
            classification_type: Type of classification ('binary' or 'multiclass').

        Returns:
            Dictionary containing batch metadata.
        """
        tasks = []
        datasets = []
        total_samples = 0
        total_features = 0

        batch_metadata = []
        
        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            batch_metadata.append({
                "task_id": task_id,
                "task": task,
                "dataset_id": dataset.qualities["ID"],
            })
        
        return {
            "task_ids": task_ids.tolist(),
            "tasks": tasks,
            "datasets": datasets,
            "classification_type": classification_type,
            "num_datasets": len(task_ids),
            "total_samples": total_samples,
            "total_features": total_features,
            "folds": 1,
            "repeats": 1,
        }

    def _should_skip_dataset(self, dataset_info, **kwargs) -> tuple[bool, str | None]:
        """Check if a batch should be skipped.

        Args:
            dataset_info: Dictionary containing batch information.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (should_skip, reason).
        """
        # For multi-dataset batches, we already filtered during loading
        # Just log the batch info
        batch_name = f"Batch_{dataset_info['num_datasets']}_datasets_{dataset_info['classification_type']}"
        self.logger.info(
            f"Starting experiments for {batch_name} with {dataset_info['total_samples']} total samples"
        )
        return False, None

    def _prepare_data(self, dataset_info, **kwargs):
        """Prepare data from multiple datasets in a batch.

        This method stacks multiple datasets together, offsetting labels
        so they don't overlap across datasets.

        Args:
            dataset_info: Dictionary containing batch information.
            **kwargs: Additional parameters (unused).

        Returns:
            Generator yielding prepared data for each fold/repeat combination.
        """
        tasks = dataset_info["tasks"]
        datasets = dataset_info["datasets"]
        folds = dataset_info["folds"]
        repeats = dataset_info["repeats"]

        # Iterate through all folds and repeats
        for repeat in range(repeats):
            for fold in range(folds):
                # Lists to accumulate data from all datasets in the batch
                X_train_list = []
                X_test_list = []
                y_train_list = []
                y_test_list = []
                all_categorical_indices = []
                
                label_offset = 0
                feature_offset = 0
                
                for task, dataset in zip(tasks, datasets):
                    # Get data
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        target=task.target_name, dataset_format="dataframe"
                    )

                    categorical_indices = np.nonzero(categorical_indicator)[0]
                    # Offset categorical indices by current feature offset
                    offset_categorical_indices = (categorical_indices + feature_offset).tolist()
                    all_categorical_indices.extend(offset_categorical_indices)

                    # Get train/test split
                    train_indices, test_indices = task.get_train_test_split_indices(
                        fold=fold,
                        repeat=repeat,
                    )

                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]

                    # Preprocess data
                    numerical_transformer = TransformToNumerical()
                    X_train = numerical_transformer.fit_transform(X_train)
                    X_test = numerical_transformer.transform(X_test)

                    # Encode labels
                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)

                    # Offset labels to avoid overlap across datasets
                    max_label = np.max(y_train)
                    y_train = y_train + label_offset
                    y_test = y_test + label_offset
                    label_offset += max_label + 1

                    # Accumulate data
                    X_train_list.append(X_train)
                    X_test_list.append(X_test)
                    y_train_list.append(y_train)
                    y_test_list.append(y_test)
                    
                    # Update feature offset for next dataset
                    feature_offset += X_train.shape[1]

                # Stack all datasets together
                X_train_stacked = np.hstack(X_train_list)
                X_test_stacked = np.hstack(X_test_list)
                y_train_stacked = np.concatenate(y_train_list)
                y_test_stacked = np.concatenate(y_test_list)

                # Create batch name for identification
                batch_name = f"Batch_{dataset_info['num_datasets']}_datasets_{dataset_info['classification_type']}"

                yield {
                    "data": X_train_stacked,
                    "dataset_name": batch_name,
                    "dataset_size": X_train_stacked.shape[0],
                    "num_features": X_train_stacked.shape[1],
                    "task_type": "Supervised Classification",
                    "embedding_kwargs": {
                        "X_test": X_test_stacked,
                        "categorical_indices": all_categorical_indices,
                    },
                    "eval_kwargs": {
                        "y_train": y_train_stacked,
                        "y_test": y_test_stacked,
                        "task_type": "Supervised Classification",
                        "num_classes": label_offset,
                        "classification_type": dataset_info["classification_type"],
                    },
                }

    def _evaluate_embeddings(
        self,
        embedding_results,
        evaluator: AbstractEvaluator,
        dataset_info: dict,
        **kwargs,
    ) -> dict:
        """Evaluate embeddings for the batched classification task.

        Args:
            embedding_results: Tuple of (train_embeddings, compute_time, test_embeddings, test_compute_time).
            evaluator: The evaluator to use.
            dataset_info: Dictionary with dataset metadata.
            **kwargs: Additional parameters including 'y_train', 'y_test', 'num_classes', and 'classification_type'.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings = embedding_results[0]
        test_embeddings = embedding_results[2]
        y_train = kwargs.get("y_train")
        y_test = kwargs.get("y_test")
        num_classes = kwargs.get("num_classes")
        classification_type = kwargs.get("classification_type")

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
            **dataset_info,
        }

        # Compute metrics for multiclass classification
        # (we ignore binary since we're combining datasets)
        if test_prediction.shape[1] == num_classes:
            auc_score = roc_auc_score(y_test, test_prediction, multi_class="ovr")
            log_loss_score = log_loss(y_test, test_prediction)
            result_dict["task"] = ["classification"]
            result_dict["classification_type"] = [classification_type]
            result_dict["num_classes"] = [num_classes]
            result_dict["auc_score"] = [auc_score]
            result_dict["log_loss_score"] = [log_loss_score]
        else:
            # Handle edge case where prediction shape doesn't match
            self.logger.warning(
                f"Prediction shape {test_prediction.shape[1]} doesn't match "
                f"expected num_classes {num_classes}"
            )
            result_dict["task"] = ["classification"]
            result_dict["classification_type"] = [classification_type]
            result_dict["num_classes"] = [num_classes]
            result_dict["auc_score"] = [np.nan]
            result_dict["log_loss_score"] = [np.nan]

        return result_dict

    def _get_benchmark_name(self) -> str:
        """Get the benchmark name for result saving.

        Returns:
            String identifier for the benchmark.
        """
        return "MultiTabArena"

    def _is_evaluator_compatible(self, evaluator: AbstractEvaluator, **kwargs) -> bool:
        """Check if evaluator is compatible with the current task.

        Args:
            evaluator: The evaluator to check.
            **kwargs: Additional parameters.

        Returns:
            True if evaluator supports classification tasks.
        """
        task_type = kwargs.get("task_type")
        if task_type is None:
            return False
        return task_type == evaluator.task_type

    def _get_task_configuration(self, dataset, task) -> tuple[int, int]:
        """Get the number of folds and repeats for a task.

        Args:
            dataset: OpenML dataset object.
            task: OpenML task object.

        Returns:
            Tuple of (n_folds, n_repeats).
        """
        if self.tabarena_lite:
            return 1, 1

        _, folds, _ = task.get_split_dimensions()
        n_samples = dataset.qualities["NumberOfInstances"]

        if n_samples < 2_500:
            tabarena_repeats = 10
        elif n_samples > 250_000:
            tabarena_repeats = 1
        else:
            tabarena_repeats = 3

        return folds, tabarena_repeats


def run_multi_tabarena_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    max_number_of_dataset: int = 5,
    use_all_datasets: bool = True,
    random_seed: int = 42,
    result_dir: str | Path = "result_multi_tabarena",
    save_result_dataframe: bool = True,
    timestamp: str = TIMESTAMP,
) -> pl.DataFrame:
    """Run the Multi-TabArena benchmark for a set of embedding models.

    This function evaluates the performance of specified embedding models on batches
    of tasks from the TabArena benchmark. Multiple datasets are combined into single
    batches with offset labels to create more challenging multi-dataset scenarios.

    Args:
        embedding_models: List of embedding model instances that inherit from
            AbstractEmbeddingGenerator. Each model will be evaluated for its performance.
        evaluators: List of evaluators to use for evaluation.
        tabarena_version: The version identifier for the TabArena benchmark study.
            Defaults to "tabarena-v0.1".
        tabarena_lite: Boolean indicating whether to run in lite mode. If True, uses fewer
            splits and repetitions for quicker evaluations. Defaults to True.
        upper_bound_num_samples: Integer representing the maximum dataset size to
            consider for benchmarking. Datasets larger than this value will be skipped.
            Defaults to 100000.
        upper_bound_num_features: Integer representing the maximum number of features
            considered for benchmarking. Datasets with more features than this value
            will be skipped. Defaults to 500.
        max_number_of_dataset: Maximum number of datasets to combine in a single batch.
            Defaults to 5.
        use_all_datasets: Whether to use all available datasets. Defaults to True.
        random_seed: Random seed for reproducibility. Defaults to 42.
        result_dir: Directory path for saving results.
        save_result_dataframe: Whether to save results to disk.
        timestamp: Timestamp string for result file naming.

    Returns:
        polars.DataFrame: A dataframe summarizing the benchmark results. The columns
            include batch information, embedding model names, metrics such as AUC and
            log loss scores, embedding computation time, and benchmark type.
    """
    benchmark = MultiTabArenaBenchmark(
        tabarena_version=tabarena_version,
        tabarena_lite=tabarena_lite,
        result_dir=result_dir,
        timestamp=timestamp,
        save_result_dataframe=save_result_dataframe,
        upper_bound_num_samples=upper_bound_num_samples,
        upper_bound_num_features=upper_bound_num_features,
        max_number_of_dataset=max_number_of_dataset,
        use_all_datasets=use_all_datasets,
        random_seed=random_seed,
    )

    return benchmark.run_benchmark(embedding_models, evaluators)

if __name__ == "__main__":
    from tabembedbench.embedding_models import (SphereBasedEmbedding,)
    embedding_models = []

    sphere_model = SphereBasedEmbedding(embed_dim=2**3)
    sphere_model.name = f"sphere-model-d{2**3}"
    embedding_models.append(sphere_model)



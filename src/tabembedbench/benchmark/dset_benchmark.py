from typing import Any, Callable, Iterator

import numpy as np
import openml

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.benchmark.constants import SUPERVISED_MULTICLASSIFICATION
from tabembedbench.evaluators import AbstractEvaluator


class DatasetSeparationBenchmark(AbstractBenchmark):
    def __init__(
        self,
        highest_collection_size: int = 10,
        random_seed: int = 42,
    ):
        super().__init__(
            name="Dataset Separation Benchmark",
            task_type=[SUPERVISED_MULTICLASSIFICATION],
        )
        self.highest_collection_size = highest_collection_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def _load_datasets(self, **kwargs) -> list[dict[str, Any]]:
        self.benchmark_suite = openml.study.get_suite("tabarena-v0.1")
        self.task_ids = self.benchmark_suite.tasks

        datasets = []
        for task_id in self.task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            datasets.append(
                {
                    "task_id": task_id,
                    "task": task,
                    "dataset": dataset,
                    "num_samples": int(dataset.qualities["NumberOfInstances"]),
                    "num_features": int(dataset.qualities["NumberOfFeatures"]),
                }
            )

        collections = []
        start_idx = 0
        pool_size = len(datasets)

        while start_idx < pool_size:
            upper_collection_limit = min(
                self.highest_collection_size, pool_size - start_idx
            )
            if upper_collection_limit > 3:
                pass
            current_chunk_size = self.rng.integers(3, high=upper_collection_limit)

            end_idx = min(start_idx + current_chunk_size, pool_size)

            collections.append(datasets[start_idx:end_idx])
            start_idx = end_idx

        return collections

    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str]:
        pass

    def _prepare_dataset(self, dataset) -> Iterator[dict]:
        pass

    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        pass

    def _get_evaluator_prediction(
        self,
        embeddings: tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        pass

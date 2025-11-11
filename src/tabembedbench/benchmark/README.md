# Benchmark Module

This module provides a simple and efficient way to benchmark tabular embedding models on multiple tasks and datasets.

## Overview

It evaluates embedding models on:

- **Outlier Detection**: Using ADBench tabular datasets
- **Classification & Regression**: Using TabArena supervised learning tasks

## Quick Start

### Basic Usage

```python
from tabembedbench.benchmark import run_benchmark
from tabembedbench.embedding_models import TableVectorizerEmbedding
from tabembedbench.evaluators.outlier import LocalOutlierFactorEvaluator
from tabembedbench.evaluators.knn_classifier import KNNClassifierEvaluator

# Create models and evaluators
models = [TableVectorizerEmbedding()]
evaluators = [
    LocalOutlierFactorEvaluator(
        model_params={
            "n_neighbors": 5, "contamination": 0.1, "metric": "euclidean"
        }
    ), KNNClassifierEvaluator(
        num_neighbors=5, weights="distance", metric="euclidean"
    )
]

# Run benchmark
outlier_results, tabarena_results, result_dir = run_benchmark(
    embedding_models=models, evaluator_algorithms=evaluators
)
```

### Run Only Outlier Detection

```python
from tabembedbench.benchmark import run_outlier_benchmark

results = run_outlier_benchmark(
    embedding_models=models,
    evaluators=evaluators,
    upper_bound_num_samples=10000
)
```

### Run Only TabArena Tasks

```python
from tabembedbench.benchmark import run_tabarena_benchmark

results = run_tabarena_benchmark(
    embedding_models=models,
    evaluators=evaluators,
    tabarena_lite=True
)
```

## Architecture

The module uses a simple, clear architecture:

### 1. AbstractBenchmark (Base Class)

The base class provides a **linear workflow**:

```python
def run_benchmark(self, embedding_models, evaluators):
    datasets = self._load_datasets()
    
    for dataset in datasets:
        if should_skip_dataset(dataset):
            continue
            
        for data_split in self._prepare_dataset(dataset):
            for model in embedding_models:
                embeddings = generate_embeddings(model, data_split)
                
                for evaluator in evaluators:
                    if is_compatible(evaluator, data_split):
                        results = evaluate(embeddings, evaluator, data_split)
                        add_result(results)
    
    return results
```

**Key Features**:
- Clear, step-by-step workflow
- Easy to understand and modify
- Handles errors gracefully
- Saves results automatically

### 2. OutlierBenchmark

Evaluates models on outlier detection using ADBench datasets.

**What it does**:
1. Loads `.npz` files from ADBench
2. Generates embeddings for each dataset
3. Uses evaluators to detect outliers
4. Computes AUC scores
5. Saves results

### 3. TabArenaBenchmark

Evaluates models on classification and regression using TabArena datasets.

**What it does**:
1. Downloads tasks from OpenML
2. Creates train/test splits
3. Generates embeddings
4. Uses evaluators for prediction
5. Computes metrics (AUC for classification, MAPE for regression)
6. Saves results

### 4. run_benchmark (Main Orchestrator)

Coordinates both benchmarks and manages resources.

**What it does**:
1. Sets up logging and directories
2. Validates models and evaluators
3. Runs outlier benchmark (if enabled)
4. Runs TabArena benchmark (if enabled)
5. Cleans up resources (GPU cache, model state)
6. Returns combined results

## Standardized Data Format

All benchmarks use a **consistent data format** from `_prepare_dataset()`:

```python
{
    'X': full_data or None,              # For outlier detection
    'X_train': train_data or None,       # For supervised learning
    'X_test': test_data or None,
    'y': full_labels or None,
    'y_train': train_labels or None,
    'y_test': test_labels or None,
    'dataset_name': str,
    'dataset_size': int,
    'num_features': int,
    'metadata': dict                     # Task-specific info
}
```

This makes it easy to:
- Understand what data is available
- Add new benchmark types
- Debug issues

## Key Parameters

### Common Parameters

- `embedding_models`: List of models to evaluate
- `evaluators`: List of evaluators to use
- `upper_bound_num_samples`: Skip datasets larger than this (default: 10000)
- `upper_bound_num_features`: Skip datasets with more features (default: 500)
- `save_result_dataframe`: Whether to save results (default: True)

### Outlier-Specific Parameters

- `dataset_paths`: Path to ADBench datasets
- `exclude_datasets`: List of dataset filenames to skip
- `exclude_image_datasets`: Whether to skip image datasets (default: True)

### TabArena-Specific Parameters

- `tabarena_version`: OpenML suite version (default: "tabarena-v0.1")
- `tabarena_lite`: Use fewer folds for faster execution (default: True)
- `run_tabpfn_subset`: Run only TabPFN subset (default: False)

## Results

### Result Format

Results are returned as Polars DataFrames with columns:

**Common Columns:**
- `dataset_name`: Name of the dataset
- `dataset_size`: Number of samples
- `num_features`: Number of features
- `embedding_model`: Name of the model
- `embed_dim`: Embedding dimension
- `time_to_compute_embedding`: Time to generate embeddings
- `algorithm`: Name of the evaluator

**Task-Specific Columns:**
- **Outlier Detection**: `auc_score`, `outlier_ratio`
- **Classification**: `auc_score`, `classification_type` (binary/multiclass)
- **Regression**: `mape_score`

### Result Storage

Results are automatically saved to:

```
data/tabembedbench_YYYYMMDD_HHMMSS/
├── results_ADBench_Tabular_YYYYMMDD_HHMMSS.parquet
├── results_TabArena_YYYYMMDD_HHMMSS.parquet
└── logs/
    └── benchmark_complete_YYYYMMDD_HHMMSS.log
```

## Memory Efficiency

The module is **memory-efficient** by design:

1. **Lazy Loading**: Datasets are loaded one at a time
2. **Generator Pattern**: `_prepare_dataset()` yields data splits instead of returning all at once
3. **Sequential Processing**: Only one dataset in memory at a time
4. **Automatic Cleanup**: GPU cache and model state cleared between datasets

## Extending the Benchmark

To create a custom benchmark, implement 5 methods:

```python
from tabembedbench.benchmark import AbstractBenchmark

class CustomBenchmark(AbstractBenchmark):
    def __init__(self, ...):
        super().__init__(
            logger_name="MyBenchmark",
            task_type="My Task Type"
        )
    
    def _load_datasets(self):
        # Return list of datasets
        return [dataset1, dataset2, ...]
    
    def _should_skip_dataset(self, dataset):
        # Return (should_skip, reason)
        return False, None
    
    def _prepare_dataset(self, dataset):
        # Yield data splits in standardized format
        yield {
            'X': X,
            'y': y,
            'dataset_name': "...",
            'dataset_size': ...,
            'num_features': ...,
            'metadata': {}
        }
    
    def _evaluate(self, embeddings, evaluator, data_split):
        # Return evaluation results
        return {'auc_score': [...], ...}
    
    def _get_benchmark_name(self):
        # Return benchmark identifier
        return "MyBenchmark"
```

## Configuration Best Practices

### Custom Evaluators

```python
from tabembedbench.evaluators.knn_classifier import KNNClassifierEvaluator

# Create evaluator with specific parameters
evaluator = KNNClassifierEvaluator(
    n_trials=50,  # Optuna trials
    cv_folds=5,   # Cross-validation folds
    random_state=42
)
```

## API Reference

### run_benchmark

```python
run_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluator_algorithms: list[AbstractEvaluator],
    tabarena_specific_embedding_models: list[AbstractEmbeddingGenerator] | None = None,
    adbench_dataset_path: str | Path | None = None,
    exclude_adbench_datasets: list[str] | None = None,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    upper_bound_num_features: int = 500,
    run_outlier: bool = True,
    run_supervised: bool = True,
    data_dir: str | Path = "data",
    save_logs: bool = True,
    run_tabpfn_subset: bool = False,
    logging_level: int = logging.INFO,
) -> Tuple[pl.DataFrame, pl.DataFrame, Path]
```

### run_outlier_benchmark

```python
run_outlier_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    dataset_paths: str | Path | None = None,
    exclude_datasets: list[str] | None = None,
    exclude_image_datasets: bool = True,
    upper_bound_num_samples: int = 10000,
    upper_bound_num_features: int = 500,
    save_result_dataframe: bool = True,
    result_dir: str | Path = "result_outlier",
    timestamp: str = TIMESTAMP,
) -> pl.DataFrame
```

### run_tabarena_benchmark

```python
run_tabarena_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    result_dir: str | Path = "result_tabarena",
    save_result_dataframe: bool = True,
    timestamp: str = TIMESTAMP,
    run_tabpfn_subset: bool = True,
) -> pl.DataFrame
```

## Additional Resources

- **AbstractEmbeddingGenerator**: See `embedding_models/abstractembedding.py`
- **AbstractEvaluator**: See `evaluators/abstractevaluator.py`

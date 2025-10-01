# Benchmark Module

This module contains the core benchmarking infrastructure for TabEmbedBench. It provides comprehensive evaluation capabilities for tabular embedding models across multiple tasks and datasets, with unified result collection and analysis.

## Overview

The benchmark module orchestrates systematic evaluation of embedding models through:
- **Outlier Detection Benchmarks**: Using ADBench tabular datasets
- **Task-Specific Benchmarks**: Using TabArena classification and regression tasks
- **Unified Result Management**: Consolidated data collection and storage
- **Performance Monitoring**: Timing, memory usage, and resource tracking
- **Flexible Configuration**: Customizable evaluation parameters and constraints

The module supports both individual benchmark components and comprehensive multi-task evaluations.

## Core Components

### 1. Main Benchmark Orchestrator (`run_benchmark.py`)

The central orchestration system that coordinates all benchmark activities.

#### Key Functions

##### `run_benchmark(embedding_models, evaluator_algorithms, **kwargs)`
Main entry point for comprehensive benchmarking.

**Parameters:**
- `embedding_models`: List of embedding model instances to evaluate
- `evaluator_algorithms`: List of evaluator instances
- `run_outlier`: Whether to run outlier detection benchmarks (default: True)
- `run_task_specific`: Whether to run TabArena task-specific benchmarks (default: True)
- `save_result_dataframe`: Whether to save results to file (default: True)
- `exclude_adbench_datasets`: List of ADBench datasets to exclude
- `upper_bound_dataset_size`: Maximum dataset size to process
- `upper_bound_num_features`: Maximum number of features to process
- `logging_level`: Logging verbosity level
- `save_logs`: Whether to save logs to file

**Returns:** Polars DataFrame with comprehensive benchmark results

**Features:**
- Automatic evaluator generation if not provided
- Resource management and GPU cache clearing
- Comprehensive logging and error handling
- Result validation and consolidation
- Timing and performance monitoring

##### `benchmark_context(models_to_process, main_logger, context_name="benchmark")`
Context manager for benchmark execution with proper resource management.

**Features:**
- Automatic resource cleanup
- Error handling and recovery
- Progress tracking
- Memory management

##### `validate_embedding_models(embedding_models)`
Validates embedding model instances before benchmark execution.

**Validation Checks:**
- Interface compliance (AbstractEmbeddingGenerator)
- Required method implementation
- Model configuration validation
- Name uniqueness verification

##### `validate_evaluator_models(evaluator_algorithms)`
Validates evaluator instances before benchmark execution.

**Validation Checks:**
- Interface compliance (AbstractEvaluator)
- Parameter configuration validation
- Task compatibility verification
- Evaluator uniqueness checks

### 2. Outlier Detection Benchmark (`outlier_benchmark.py`)

Specialized benchmark for outlier detection using ADBench tabular datasets.

#### Key Functions

##### `run_outlier_benchmark(embedding_models, evaluator_algorithms=None, **kwargs)`
Executes outlier detection benchmark across multiple datasets and models.

**Parameters:**
- `embedding_models`: List of embedding models to evaluate
- `evaluator_algorithms`: List of outlier detection evaluators
- `adbench_datasets_dir`: Directory containing ADBench datasets
- `exclude_adbench_datasets`: Datasets to exclude from evaluation
- `upper_bound_dataset_size`: Maximum dataset size constraint
- `upper_bound_num_features`: Maximum feature count constraint
- `neighbors`: Range of neighbor counts to test
- `neighbors_step`: Step size for neighbor count sweep
- `distance_metrics`: List of distance metrics to evaluate

**Returns:** Polars DataFrame with outlier detection results

**Features:**
- Automatic dataset loading and validation
- Multiple distance metric evaluation
- Neighbor count parameter sweeps
- AUROC performance measurement
- Computation time tracking
- Memory usage monitoring

**Result Columns:**
- `dataset_name`: Name of the dataset
- `dataset_size`: Number of samples in dataset
- `num_features`: Number of features in dataset
- `model_name`: Name of embedding model
- `evaluator_name`: Name of evaluator algorithm
- `neighbors`: Number of neighbors used
- `distance_metric`: Distance metric used
- `auc_score`: AUROC performance score
- `compute_time`: Embedding computation time
- `evaluation_time`: Evaluator execution time
- `benchmark_label`: Benchmark identifier
- `task_type`: Task type ("outlier_detection")
- `algorithm_{parameter}`: Column for each algorithm parameter

### 3. TabArena Task-Specific Benchmark (`tabarena_benchmark.py`)

Comprehensive benchmark using OpenML's TabArena suite for classification and regression tasks.

#### Key Functions

##### `run_tabarena_benchmark(embedding_models, evaluator_algorithms=None, **kwargs)`
Executes TabArena benchmark across classification and regression tasks.

**Parameters:**
- `embedding_models`: List of embedding models to evaluate
- `evaluator_algorithms`: List of task-specific evaluators
- `tabarena_suite`: OpenML suite identifier (default: "tabarena-v0.1")
- `tabarena_lite`: Whether to use lite mode for faster execution
- `upper_bound_dataset_size`: Maximum dataset size constraint
- `upper_bound_num_features`: Maximum feature count constraint
- `neighbors`: Range of neighbor counts to test
- `neighbors_step`: Step size for neighbor count sweep
- `distance_metrics`: List of distance metrics to evaluate

**Returns:** Polars DataFrame with task-specific results

**Features:**
- Automatic OpenML dataset downloading
- Train/test split management
- Feature preprocessing and encoding
- Multi-task evaluation (classification/regression)
- Cross-validation support
- Performance metric computation

##### `_evaluate_classification(embeddings_train, embeddings_test, y_train, y_test, **kwargs)`
Evaluates classification performance using KNN classifier.

**Parameters:**
- `embeddings_train`: Training embeddings
- `embeddings_test`: Test embeddings
- `y_train`: Training labels
- `y_test`: Test labels
- `num_neighbors`: Number of neighbors for KNN
- `weights`: Weighting scheme ("uniform" or "distance")
- `distance_metric`: Distance metric for neighbor search
- `algorithm_{parameter}`: Column for each algorithm parameter

**Returns:** Dictionary with classification metrics (ROC-AUC, accuracy, etc.)

##### `_evaluate_regression(embeddings_train, embeddings_test, y_train, y_test, **kwargs)`
Evaluates regression performance using KNN regressor.

**Parameters:**
- `embeddings_train`: Training embeddings
- `embeddings_test`: Test embeddings
- `y_train`: Training targets
- `y_test`: Test targets
- `num_neighbors`: Number of neighbors for KNN
- `weights`: Weighting scheme ("uniform" or "distance")
- `distance_metric`: Distance metric for neighbor search

**Returns:** Dictionary with regression metrics (MSE, MAE, R²)

##### `_get_task_configuration(dataset, tabarena_lite: bool, task) -> tuple[int, int]`
Determines cross-validation configuration for datasets.

**Parameters:**
- `dataset`: OpenML dataset object
- `tabarena_lite`: Whether to use reduced configuration
- `task`: Task type ("classification" or "regression")

**Returns:** Tuple of (n_folds, n_repeats)

**Configuration Logic:**
- Lite mode: Reduced folds/repeats for faster execution
- Full mode: Comprehensive cross-validation
- Dataset-specific adjustments based on size

## Integration with Framework Components

### Embedding Model Integration
1. **Model Validation**: Ensures models implement required interfaces
2. **Preprocessing**: Handles model-specific data preprocessing
3. **Embedding Generation**: Manages embedding computation and timing
4. **State Management**: Resets models between datasets
5. **Error Handling**: Graceful handling of model failures

### Evaluator Integration
1. **Automatic Generation**: Creates evaluators if not provided
2. **Parameter Sweeps**: Systematic exploration of evaluator parameters
3. **Task Matching**: Matches evaluators to appropriate tasks
4. **Performance Measurement**: Comprehensive metric computation
5. **Result Aggregation**: Consolidates results across evaluators

### Data Management
1. **Dataset Loading**: Automatic downloading and caching
2. **Format Validation**: Ensures data compatibility
3. **Preprocessing**: Standardized data preparation
4. **Size Constraints**: Enforces computational limits
5. **Quality Checks**: Validates data integrity

## Result Management

### Data Structure
Results are stored in Polars DataFrames with standardized schemas:

**Common Columns:**
- `model_name`: Embedding model identifier
- `evaluator_name`: Evaluator algorithm identifier
- `dataset_name`: Dataset identifier
- `dataset_size`: Number of samples
- `num_features`: Number of features
- `compute_time`: Embedding computation time
- `evaluation_time`: Evaluator execution time
- `benchmark_label`: Benchmark type identifier
- `task_type`: Task category

**Task-Specific Columns:**
- **Classification**: `auc_score`, `accuracy`, `f1_score`
- **Regression**: `mse`, `mae`, `r2_score`
- **Outlier Detection**: `auc_score`, `precision`, `recall`

### Storage Options
- **Parquet Format**: Efficient columnar storage with compression
- **CSV Format**: Human-readable format for analysis
- **JSON Format**: Structured format for web applications
- **Database Integration**: Direct storage to SQL databases

### File Organization
```
data/
├── results/
│   ├── results_YYYYMMDD_HHMMSS.parquet
│   ├── outlier_results_YYYYMMDD_HHMMSS.parquet
│   └── tabarena_results_YYYYMMDD_HHMMSS.parquet
├── logs/
│   └── benchmark_YYYYMMDD_HHMMSS.log
└── datasets/
    ├── adbench/
    └── tabarena/
```

## Performance Optimization

### Resource Management
- **GPU Memory**: Automatic cache clearing between models
- **CPU Memory**: Efficient data structure usage
- **Disk Space**: Compressed result storage
- **Network**: Cached dataset downloads

### Computational Efficiency
- **Parallel Processing**: Multi-threaded evaluation where possible
- **Batch Processing**: Efficient batch embedding computation
- **Lazy Evaluation**: Deferred computation for large datasets
- **Caching**: Intelligent caching of intermediate results

### Scalability Features
- **Dataset Filtering**: Size and feature constraints
- **Lite Modes**: Reduced evaluation for development
- **Incremental Execution**: Resume interrupted benchmarks
- **Distributed Computing**: Support for cluster execution

## Usage Examples

### Basic Benchmark Execution
```python
from tabembedbench.benchmark import run_benchmark
from tabembedbench.embedding_models import SphereBasedEmbedding, TabICLEmbedding

# Create embedding models
models = [
    SphereBasedEmbedding(embed_dim=64),
    TabICLEmbedding(preprocess_tabicl_data=True)
]

# Run comprehensive benchmark
results = run_benchmark(
    embedding_models=models,
    run_outlier=True,
    run_task_specific=True,
    save_result_dataframe=True
)
```

### Outlier Detection Only
```python
from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark

results = run_outlier_benchmark(
    embedding_models=models,
    exclude_adbench_datasets=["3_backdoor.npz"],
    upper_bound_dataset_size=10000
)
```

### TabArena Tasks Only
```python
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark

results = run_tabarena_benchmark(
    embedding_models=models,
    tabarena_lite=True,
    upper_bound_dataset_size=5000
)
```

### Custom Evaluator Configuration
```python
from tabembedbench.evaluators import KNNClassifierEvaluator, LocalOutlierFactorEvaluator

# Create custom evaluators
evaluators = [
    KNNClassifierEvaluator(num_neighbors=5, metric="euclidean"),
    LocalOutlierFactorEvaluator(model_params={"n_neighbors": 10})
]

results = run_benchmark(
    embedding_models=models,
    evaluator_algorithms=evaluators
)
```

## Configuration Best Practices

### Model Selection
- **Diverse Architectures**: Include different embedding approaches
- **Baseline Models**: Include simple baselines for comparison
- **Resource Constraints**: Consider computational requirements
- **Task Alignment**: Match models to intended use cases

### Parameter Tuning
- **Neighbor Ranges**: Test multiple neighbor counts (5-50)
- **Distance Metrics**: Include euclidean and cosine metrics
- **Evaluation Depth**: Balance thoroughness with execution time
- **Cross-Validation**: Use appropriate fold counts for dataset size

### Resource Management
- **Dataset Limits**: Set appropriate size constraints
- **Memory Monitoring**: Monitor GPU and CPU memory usage
- **Execution Time**: Plan for long-running benchmarks

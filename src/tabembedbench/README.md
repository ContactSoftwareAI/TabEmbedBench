# TabEmbedBench Module

Core module implementing the TabEmbedBench benchmarking framework for evaluating tabular data embedding methods. This module provides the complete infrastructure for systematic evaluation of embedding quality across outlier detection, classification, and regression tasks.

## Module Overview

The `tabembedbench` module is organized into five key components:

```
tabembedbench/
├── benchmark/           # Benchmarking orchestration and execution
├── embedding_models/    # Embedding model implementations
├── evaluators/          # Evaluation algorithms
├── examples/            # Usage examples and demonstrations
└── utils/               # Shared utilities and helper functions
```

## Architecture

### Three-Layer Design

1. **Embedding Layer** (`embedding_models/`): Generates vector representations from tabular data
2. **Evaluation Layer** (`evaluators/`): Assesses embedding quality through downstream tasks
3. **Orchestration Layer** (`benchmark/`): Coordinates systematic evaluation across datasets

This separation enables:
- Independent development of embedding methods
- Standardized evaluation protocols
- Extensible benchmark configurations

## Quick Start

### Basic Benchmark Execution

```python
from tabembedbench.benchmark import run_benchmark
from tabembedbench.embedding_models import TabICLEmbedding, TableVectorizerEmbedding
from tabembedbench.evaluators import KNNClassifierEvaluator, LocalOutlierFactorEvaluator

# Configure models
models = [
    TabICLEmbedding(preprocess_tabicl_data=True),
    TableVectorizerEmbedding(optimize=False)
]

# Configure evaluators
evaluators = [
    KNNClassifierEvaluator(num_neighbors=5, metric="euclidean"),
    LocalOutlierFactorEvaluator(model_params={"n_neighbors": 10})
]

# Run comprehensive benchmark
outlier_results, tabarena_results, result_dir = run_benchmark(
    embedding_models=models,
    evaluator_algorithms=evaluators,
    run_outlier=True,
    run_supervised=True
)
```

### Using Example Scripts

```bash
# Full benchmark with all models
uv run src/tabembedbench/examples/eurips_run.py

# Quick test in debug mode
uv run src/tabembedbench/examples/eurips_run.py --debug

# Constrained execution
uv run src/tabembedbench/examples/eurips_run.py --max-samples 5000 --max-features 100
```

## Component Modules

### 1. Benchmark Module

**Purpose**: Orchestrates systematic evaluation across datasets and tasks.

**Key Components**:
- `AbstractBenchmark`: Base class defining benchmark workflow
- `OutlierBenchmark`: ADBench outlier detection evaluation
- `TabArenaBenchmark`: OpenML TabArena task evaluation
- `run_benchmark()`: Main orchestrator for multi-task benchmarking

**Workflow**:
```
Load Datasets → Preprocess → Generate Embeddings → Evaluate → Collect Results
```

**Documentation**: See [`benchmark/README.md`](benchmark/README.md)

### 2. Embedding Models Module

**Purpose**: Implements tabular data embedding generators.

**Available Models**:
- `TabICLEmbedding`: Neural in-context learning embeddings
- `UniversalTabPFNEmbedding`: Prior-fitted network embeddings
- `TableVectorizerEmbedding`: Traditional vectorization approach

**Common Interface**:
All models inherit from `AbstractEmbeddingGenerator`:
```python
class CustomEmbedding(AbstractEmbeddingGenerator):
    def _preprocess_data(self, X, train=True): ...
    def _compute_embeddings(self, X): ...
    def reset_embedding_model(self): ...
```

**Documentation**: See [`embedding_models/README.md`](embedding_models/README.md)

### 3. Evaluators Module

**Purpose**: Assesses embedding quality through downstream tasks.

**Available Evaluators**:
- `KNNClassifierEvaluator`: Classification via k-nearest neighbors
- `KNNRegressorEvaluator`: Regression via k-nearest neighbors
- `LocalOutlierFactorEvaluator`: Density-based outlier detection
- `IsolationForestEvaluator`: Tree-based outlier detection

**Common Interface**:
All evaluators inherit from `AbstractEvaluator`:
```python
class CustomEvaluator(AbstractEvaluator):
    def get_prediction(self, embeddings, y=None, train=True): ...
    def reset_evaluator(self): ...
    def get_parameters(self): ...
```

**Documentation**: See [`evaluators/README.md`](evaluators/README.md)

### 4. Examples Module

**Purpose**: Demonstrates framework usage and configuration patterns.

**Key Examples**:
- `eurips_run.py`: Comprehensive benchmark configuration
  - Model setup with parameter sweeps
  - Evaluator configuration
  - CLI interface with debug modes

**Usage Patterns**:
```python
from tabembedbench.examples.eurips_run import get_embedding_models, get_evaluators

models = get_embedding_models(debug=False)
evaluators = get_evaluators(debug=False)
```

**Documentation**: See [`examples/README.md`](examples/README.md)

### 5. Utils Module

**Purpose**: Shared utilities for data handling, logging, and visualization.

**Key Utilities**:

| Module | Purpose |
|--------|---------|
| `dataset_utils.py` | Dataset downloading and management |
| `preprocess_utils.py` | Data preprocessing and type inference |
| `torch_utils.py` | PyTorch device management and GPU utilities |
| `logging_utils.py` | Unified logging configuration |
| `tracking_utils.py` | Result tracking and storage |
| `eda_utils.py` | Visualization and exploratory analysis |

**Common Functions**:
```python
from tabembedbench.utils.torch_utils import get_device, empty_gpu_cache
from tabembedbench.utils.logging_utils import setup_unified_logging
from tabembedbench.utils.dataset_utils import download_adbench_tabular_datasets

# Device management
device = get_device()
empty_gpu_cache(device)

# Logging setup
logger = setup_unified_logging(log_level=logging.INFO)
```

## Data Flow

```
Input Data (Tabular)
      ↓
Embedding Model (AbstractEmbeddingGenerator)
      ↓
Embeddings (Vector Representations)
      ↓
Evaluator (AbstractEvaluator)
      ↓
Predictions & Metrics
      ↓
Results DataFrame (Polars)
```

## Extension Points

### Adding New Embedding Models

1. Inherit from `AbstractEmbeddingGenerator`
2. Implement required methods: `_preprocess_data()`, `_compute_embeddings()`, `reset_embedding_model()`
3. Test with existing benchmarks
4. Add to model registry

### Adding New Evaluators

1. Inherit from `AbstractEvaluator`
2. Implement required methods: `get_prediction()`, `reset_evaluator()`, `get_parameters()`
3. Define task type (classification, regression, outlier_detection)
4. Validate with multiple embedding models

### Adding New Benchmarks

1. Inherit from `AbstractBenchmark`
2. Implement dataset loading and preparation logic
3. Define evaluation metrics and result format
4. Integrate with `run_benchmark()` orchestrator

## Result Format

All benchmarks return Polars DataFrames with standardized columns:

**Common Columns**:
- `dataset_name`: Dataset identifier
- `dataset_size`: Number of samples
- `num_features`: Number of features
- `embedding_model`: Model identifier
- `embed_dim`: Embedding dimension
- `time_to_compute_embedding`: Computation time (seconds)
- `algorithm`: Evaluator identifier

**Task-Specific Metrics**:
- **Outlier Detection**: `auc_score`, `outlier_ratio`
- **Classification**: `auc_score`, `classification_type`
- **Regression**: `mape_score`

Results are automatically saved to:
```
data/tabembedbench_YYYYMMDD_HHMMSS/
├── results_ADBench_Tabular_*.parquet
├── results_TabArena_*.parquet
└── logs/
```

## Design Principles

### 1. Abstraction Through Interfaces
- `AbstractEmbeddingGenerator` for all embedding models
- `AbstractEvaluator` for all evaluation algorithms
- `AbstractBenchmark` for all benchmark types

### 2. Separation of Concerns
- Embedding generation isolated from evaluation
- Benchmark orchestration separate from algorithm implementation
- Utilities shared across components

### 3. Extensibility
- Easy addition of new models, evaluators, and benchmarks
- Plugin-style architecture
- Minimal coupling between components

### 4. Memory Efficiency
- Lazy dataset loading
- Generator-based data iteration
- Automatic GPU cache management
- Sequential processing with state cleanup

### 5. Reproducibility
- Comprehensive logging
- Result versioning with timestamps
- Automatic metadata tracking
- Standardized random seeds

## Dependencies

**Core Requirements**:
- Python ≥ 3.12
- NumPy: Array operations
- Polars: Efficient data handling
- Scikit-learn: ML algorithms and metrics

**Neural Model Requirements**:
- PyTorch: Deep learning backend
- CUDA (optional): GPU acceleration

**Utilities**:
- Click: CLI interfaces
- Logging: Execution tracking

## Performance Considerations

### Computational Efficiency
- **GPU Acceleration**: Neural models support CUDA
- **Batch Processing**: Efficient data handling
- **Parallel Evaluation**: Multi-threaded where possible

### Memory Management
- **Streaming Data**: Generator-based iteration
- **Resource Cleanup**: Automatic state reset
- **Cache Management**: GPU memory clearing

### Scalability
- **Dataset Constraints**: Configurable size limits
- **Feature Limits**: Configurable dimensionality bounds
- **Distributed Support**: Compatible with cluster execution

## Development Workflow

### Running Tests
```bash
# Quick validation with debug mode
uv run src/tabembedbench/examples/eurips_run.py --debug

# Test specific benchmark
python -c "
from tabembedbench.benchmark import run_outlier_benchmark
from tabembedbench.embedding_models import TableVectorizerEmbedding
results = run_outlier_benchmark([TableVectorizerEmbedding()])
"
```

### Code Quality
```bash
# Lint code
uv run ruff check src/tabembedbench

# Format code
uv run ruff format src/tabembedbench
```

## Module Initialization

The module uses minimal `__init__.py` files to allow flexible imports:

```python
# Import core components
from tabembedbench.benchmark import run_benchmark
from tabembedbench.embedding_models import TabICLEmbedding
from tabembedbench.evaluators import KNNClassifierEvaluator
```

## Further Documentation

Each submodule contains detailed documentation:

- **Benchmark Framework**: [`benchmark/README.md`](benchmark/README.md)
- **Embedding Models**: [`embedding_models/README.md`](embedding_models/README.md)
- **Evaluators**: [`evaluators/README.md`](evaluators/README.md)
- **Examples**: [`examples/README.md`](examples/README.md)
- **Utils**: [`utils/README.md`](utils/README.md)

For general project information, installation, and setup, see the main [`README.md`](../../README.md) at the project root.

## Authors

- Lars Kleinemeier
- Frederik Hoppe

# TabEmbedBench

TabEmbedBench is a comprehensive benchmarking framework designed for evaluating tabular data embedding methods. The framework provides systematic evaluation capabilities across multiple tasks including outlier detection, classification, and regression, with support for both neural and traditional embedding approaches.

## Key Features

- **Comprehensive Benchmarking**: Evaluate embedding models on outlier detection (ADBench) and supervised learning tasks (TabArena)
- **Multiple Embedding Models**: Support for TabICL, TabPFN, and TableVectorizer with both neural and traditional approaches
- **Flexible Evaluation**: Multiple evaluators including KNN, MLP with hyperparameter optimization, LOF, Isolation Forest, and DeepSVDD
- **Unified Results**: Consolidated result collection with automatic timing and performance tracking
- **Extensible Architecture**: Easy integration of new embedding models and evaluation methods
- **Advanced Features**: GPU acceleration, memory management with logging, and configurable dataset exclusion

## Prerequisites

- Python 3.12 or higher
- uv package manager

## Setting up UV

UV is a fast Python package manager and project manager. 
If you don't have UV installed, you can install it using one of the following methods:

### Option 1: Using pip
```bash
pip install uv
```
### Option 2: Using the installer script (recommended)
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Option 3: Using Homebrew (macOS/Linux)
``` bash
brew install uv
```
For more installation options, visit the [official UV documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Installation

### Development Installation
1. **Clone the repository:**
``` bash
   git clone <repository-url>
   cd tabembedbench
```

2. **Install the package in development mode:**
``` bash
   uv sync --dev
```
This command will:
- Create a virtual environment if one doesn't exist
- Install all dependencies specified in `pyproject.toml`
- Install the package in editable mode for development

3. **Activate the virtual environment:**
``` bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
```
Alternatively, you can run commands directly with UV without activating:
``` bash
   uv run python your_script.py
```

## Quick Start

### Basic Usage
```python
from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models import TabICLEmbedding, TableVectorizerEmbedding

# Create embedding model instances
models = [
    TabICLEmbedding(preprocess_tabicl_data=True),
    TableVectorizerEmbedding()
]

# Run comprehensive benchmark
results_df = run_benchmark(
    embedding_models=models,
    run_outlier=True,
    run_task_specific=True,
    save_result_dataframe=True
)

# Results are saved to data/results/ and returned as Polars DataFrame
print(results_df.head())
```

### Command Line Usage

#### Using UV (Recommended)
```bash
# Run the example benchmark script with uv
uv run src/tabembedbench/examples/eurips_run.py

# Run in debug mode for quick testing
uv run src/tabembedbench/examples/eurips_run.py --debug

# Run only outlier detection
uv run src/tabembedbench/examples/eurips_run.py --no-run-task-specific

# Run only task-specific benchmarks
uv run src/tabembedbench/examples/eurips_run.py --no-run-outlier

# Limit dataset size and features
uv run src/tabembedbench/examples/eurips_run.py --max-samples 1000 --max-features 50

# Combine multiple options
uv run src/tabembedbench/examples/eurips_run.py --debug --max-samples 500 --max-features 20
```

#### Using Python directly (after environment activation)
```bash
# First activate the environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Then run the script
python src/tabembedbench/examples/eurips_run.py

# Run in debug mode for quick testing
python src/tabembedbench/examples/eurips_run.py --debug

# Run only outlier detection
python src/tabembedbench/examples/eurips_run.py --no-run-task-specific

# Limit dataset size and features
python src/tabembedbench/examples/eurips_run.py --max-samples 1000 --max-features 50
```

## Project Structure

```
tabembedbench/
├── src/
│   └── tabembedbench/
│       ├── benchmark/           # Benchmarking framework
│       │   ├── README.md             # Comprehensive benchmark documentation
│       │   ├── run_benchmark.py      # Main benchmark orchestrator
│       │   ├── outlier_benchmark.py  # ADBench outlier detection
│       │   └── tabarena_benchmark.py # TabArena task evaluation
│       ├── embedding_models/    # Embedding model implementations
│       │   ├── README.md             # Embedding models documentation
│       │   ├── abstractembedding.py  # AbstractEmbeddingGenerator base class
│       │   ├── tabicl_embedding.py   # TabICL neural embedding
│       │   ├── tabpfn_embedding.py   # TabPFN prior-fitted networks
│       │   └── tablevectorizer_embedding.py # Traditional vectorization
│       ├── evaluators/          # Evaluation algorithms
│       │   ├── README.md             # Evaluators documentation
│       │   ├── abstractevaluator.py  # AbstractEvaluator base class
│       │   ├── knn_classifier.py     # KNN classification evaluator
│       │   ├── knn_regressor.py      # KNN regression evaluator
│       │   ├── mlp_classifier.py     # MLP classification evaluator
│       │   ├── mlp_regressor.py      # MLP regression evaluator
│       │   └── outlier.py            # LOF, Isolation Forest, and DeepSVDD evaluators
│       ├── examples/            # Usage examples and demonstrations
│       │   ├── README.md             # Examples documentation
│       │   └── eurips_run.py         # Comprehensive example script
│       └── utils/               # Utility functions
│           ├── README.md             # Utils documentation
│           ├── config.py             # Configuration enums
│           ├── dataset_utils.py      # Dataset handling
│           ├── embedding_utils.py    # Embedding utilities
│           ├── logging_utils.py      # Logging functionality
│           ├── plot_utils.py         # Visualization tools
│           ├── preprocess_utils.py   # Data preprocessing
│           ├── torch_utils.py        # PyTorch utilities
│           └── tracking_utils.py     # Result tracking
├── examples/                    # Additional examples
├── notebooks/                   # Analysis notebooks
└── pyproject.toml              # Project configuration
```

## Available Embedding Models

All models implement the `AbstractEmbeddingGenerator` interface:

### Neural Models
- **TabICLEmbedding**: Neural embedding based on Tabular In-Context Learning architecture
- **TabPFNEmbedding**: Prior-fitted networks with pre-trained knowledge transfer

### Traditional Models  
- **TableVectorizerEmbedding**: Classical feature engineering baseline using table vectorization techniques

For detailed documentation on each model, see [`src/tabembedbench/embedding_models/README.md`](src/tabembedbench/embedding_models/README.md).

## Evaluation Framework

### Available Evaluators

#### Supervised Learning Evaluators
- **KNNClassifierEvaluator**: K-nearest neighbors classification with configurable distance metrics
- **KNNRegressorEvaluator**: K-nearest neighbors regression with configurable distance metrics
- **MLPClassifierEvaluator**: Multi-layer perceptron neural network classifier with Optuna-based hyperparameter optimization
- **MLPRegressorEvaluator**: Multi-layer perceptron neural network regressor with Optuna-based hyperparameter optimization

#### Outlier Detection Evaluators
- **LocalOutlierFactorEvaluator**: Density-based outlier detection using LOF algorithm
- **IsolationForestEvaluator**: Tree-based outlier detection using ensemble methods
- **DeepSVDDEvaluator**: Deep learning-based outlier detection using Support Vector Data Description

For detailed documentation on evaluators, see [`src/tabembedbench/evaluators/README.md`](src/tabembedbench/evaluators/README.md).

### Benchmark Types

#### Outlier Detection Benchmark
- Uses ADBench tabular datasets
- Evaluates with Local Outlier Factor (LOF) and Isolation Forest
- Multiple distance metrics and neighbor counts
- Returns AUROC scores and computation times

#### Task-specific Benchmark (TabArena)
- Uses OpenML's TabArena suite
- Classification and regression tasks
- KNN and MLP-based evaluation with parameter sweeps
- Supports both full and lite modes for development
- Configurable dataset exclusion for focused benchmarking

For comprehensive benchmark documentation, see [`src/tabembedbench/benchmark/README.md`](src/tabembedbench/benchmark/README.md).

## Documentation

Each module includes comprehensive documentation:

- **[Benchmark Module](src/tabembedbench/benchmark/README.md)**: Complete benchmarking framework documentation
- **[Embedding Models](src/tabembedbench/embedding_models/README.md)**: All embedding model implementations and usage
- **[Evaluators](src/tabembedbench/evaluators/README.md)**: Evaluation algorithms and interfaces
- **[Examples](src/tabembedbench/examples/README.md)**: Usage examples and configuration patterns
- **[Utils](src/tabembedbench/utils/README.md)**: Utility functions and helper classes

## Development Tools

The project includes several development tools configured in `pyproject.toml`:

- **Ruff**: For code linting and formatting
```bash
uv run ruff check .      # Check for issues
uv run ruff format .     # Format code
```

## Contributing

### Adding New Embedding Models
1. Inherit from `AbstractEmbeddingGenerator`
2. Implement all abstract methods:
   - `task_only` (property)
   - `_preprocess_data(X, train=True)`
   - `_compute_embeddings(X)`
   - `reset_embedding_model()`
3. Follow the established naming conventions
4. Test integration with the benchmarking framework
5. Add documentation following the existing patterns

### Adding New Evaluators
1. Inherit from `AbstractEvaluator`
2. Implement required methods:
   - `get_prediction(embeddings, y=None, train=True)`
   - `reset_evaluator()`
   - `get_parameters()`
3. Follow task-specific naming conventions
4. Test with multiple embedding models

For detailed contribution guidelines, see the respective module documentation.

## Performance Considerations

- **GPU Support**: Neural models (TabICL, TabPFN) support GPU acceleration
- **Memory Management**: Automatic GPU cache clearing between datasets with memory logging
- **Scalability**: Configurable dataset size and feature limits via command-line arguments
- **Parallel Processing**: Multi-threaded evaluation where possible
- **Hyperparameter Optimization**: Automated tuning for MLP-based evaluators using Optuna

## Results and Analysis

Results are automatically saved in structured formats:
- **Parquet files**: Efficient columnar storage in `data/results/`
- **Comprehensive metrics**: Performance scores, timing, and metadata
- **Visualization support**: Built-in plotting utilities for analysis

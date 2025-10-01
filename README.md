# TabEmbedBench

TabEmbedBench is a comprehensive benchmarking framework designed for evaluating tabular data embedding methods. The framework provides systematic evaluation capabilities across multiple tasks including outlier detection, classification, and regression, with support for both neural and traditional embedding approaches.

## Key Features

- **Comprehensive Benchmarking**: Evaluate embedding models on outlier detection (ADBench) and task-specific benchmarks (TabArena)
- **Multiple Embedding Models**: Support for TabICL, TabPFN, SphereBasedEmbedding, and TabVectorizer
- **Flexible Evaluation**: Configurable evaluators with parameter sweeps for thorough assessment
- **Unified Results**: Consolidated result collection with automatic timing and performance tracking
- **Extensible Architecture**: Easy integration of new embedding models and evaluation methods

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
from tabembedbench.embedding_models import TabICLEmbedding, SphereBasedEmbedding

# Create embedding model instances
models = [
    TabICLEmbedding(preprocess_tabicl_data=True),
    SphereBasedEmbedding(embed_dim=64)
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
```bash
# Run the example benchmark script
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
│       │   ├── readme.md             # Embedding models documentation
│       │   ├── abstractembedding.py  # AbstractEmbeddingGenerator base class
│       │   ├── tabicl_embedding.py   # TabICL neural embedding
│       │   ├── tabpfn_embedding.py   # TabPFN prior-fitted networks
│       │   ├── spherebased_embedding.py # Geometric embeddings
│       │   └── tabvectorizer_embedding.py # Traditional vectorization
│       ├── evaluators/          # Evaluation algorithms
│       │   ├── README.md             # Evaluators documentation
│       │   ├── abstractevaluator.py  # AbstractEvaluator base class
│       │   ├── classifier.py         # KNN classification evaluator
│       │   ├── regression.py         # KNN regression evaluator
│       │   └── outlier.py            # LOF and Isolation Forest evaluators
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
- **UniversalTabPFNEmbedding**: Prior-fitted networks with pre-trained knowledge transfer

### Traditional Models  
- **SphereBasedEmbedding**: Geometric embedding using spherical projections
- **TabVectorizerEmbedding**: Traditional vectorization approach with optimization support

For detailed documentation on each model, see [`src/tabembedbench/embedding_models/readme.md`](src/tabembedbench/embedding_models/readme.md).

## Evaluation Framework

### Available Evaluators
- **KNNClassifierEvaluator**: K-nearest neighbors classification
- **KNNRegressorEvaluator**: K-nearest neighbors regression  
- **LocalOutlierFactorEvaluator**: Density-based outlier detection
- **IsolationForestEvaluator**: Tree-based outlier detection

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
- KNN-based evaluation with parameter sweeps
- Supports both full and lite modes for development

For comprehensive benchmark documentation, see [`src/tabembedbench/benchmark/README.md`](src/tabembedbench/benchmark/README.md).

## Documentation

Each module includes comprehensive documentation:

- **[Benchmark Module](src/tabembedbench/benchmark/README.md)**: Complete benchmarking framework documentation
- **[Embedding Models](src/tabembedbench/embedding_models/readme.md)**: All embedding model implementations and usage
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
- **Memory Management**: Automatic GPU cache clearing between datasets
- **Scalability**: Configurable dataset size and feature limits
- **Parallel Processing**: Multi-threaded evaluation where possible

## Results and Analysis

Results are automatically saved in structured formats:
- **Parquet files**: Efficient columnar storage in `data/results/`
- **Comprehensive metrics**: Performance scores, timing, and metadata
- **Visualization support**: Built-in plotting utilities for analysis

## Authors

- Lars Kleinemeier
- Frederik Hoppe

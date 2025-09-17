# TabEmbedBench

TabEmbedBench is a benchmarking framework designed for evaluating tabular data embedding methods. 
The project includes wrappers for some Tabular Foundation models to extract their embeddings.

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
1. **Install the package in development mode:**
``` bash
   uv sync --dev
```
This command will:
- Create a virtual environment if one doesn't exist
- Install all dependencies specified in `pyproject.toml`
- Install the package in editable mode for development

1. **Activate the virtual environment:**
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

```Python
from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models.tabicl_embedding import (
    get_tabicl_embedding_model,
)

# Create embedding model instances
models = [get_tabicl_embedding_model(preprocess_data=True)]

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


## Project Structure
``` 
tabembedbench/
├── src/
│   └── tabembedbench/
│       ├── benchmark/           # Benchmarking framework
│       │   ├── run_benchmark.py      # Main benchmark orchestrator
│       │   ├── outlier_benchmark.py  # ADBench outlier detection
│       │   └── tabarena_benchmark.py # TabArena task evaluation
│       ├── embedding_models/    # Embedding model implementations
│       │   ├── base.py               # BaseEmbeddingGenerator abstract class
│       │   ├── tabicl_embedding.py   # TabICL wrapper
│       │   ├── tabpfn_embedding.py   # TabPFN wrapper
│       │   ├── spherebased_embedding.py # Geometric embeddings
│       │   └── tabvectorizer_embedding.py # TabVectorizer wrapper
│       └── utils/               # Utility functions
│           ├── dataset_utils.py      # Dataset handling
│           ├── embedding_utils.py    # Embedding utilities
│           ├── plot_utils.py         # Visualization tools
│           └── ...
├── example/                         # Usage examples
└── pyproject.toml                  # Project configuration
```
## Available Embedding Models
All models implement the interface: `BaseEmbeddingGenerator`
- **TabICL**: Uses TabICL to extract embeddings from tabular data with the 
  first two parts of the TabICL architecture.
- **Sphere-based Embedding**: Geometric embedding using spherical projections
- **TabVectorizer**: Standard implementation of TabVectorizer from skrub.

### WIP:
- **TabPFN**

## Benchmarking Framework
### Outlier Detection Benchmark
- Uses ADBench tabular datasets
- Evaluates with Local Outlier Factor (LOF)
- Multiple distance metrics and neighbor counts
- Returns AUC scores and computation times

### Task-specific Benchmark
- Uses OpenML's TabArena suite
- Classification and regression tasks
- KNN-based evaluation
- Supports both full and lite modes

## Development Tools
The project includes several development tools configured in : `pyproject.toml`
- **Ruff**: For code linting and formatting
``` bash
  uv run ruff check .      # Check for issues
  uv run ruff format .     # Format code
```
## Contributing
When adding new embedding models:
1. Inherit from `BaseEmbeddingGenerator`
2. Implement all abstract methods (, , , , ) `task_only``_get_default_name``_preprocess_data``_compute_embeddings``reset_embedding_model`
3. Follow the established naming conventions
4. Test integration with the benchmarking framework

## Authors
- Lars Kleinemeier
- Frederik Hoppe

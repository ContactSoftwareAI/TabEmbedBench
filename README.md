# TabEmbedBench

TabEmbedBench is a benchmarking framework for evaluating tabular data embedding models on outlier detection,
classification, and regression tasks. It utilizes datasets from ADBench and TabArena to provide comprehensive
performance evaluations.

## Prerequisites

- **Python**: 3.12 or 3.13 is required (as specified in `pyproject.toml`).
- **uv**: It is recommended to use [uv](https://github.com/astral-sh/uv) for fast and reliable Python package
  management.

## Installation

To install the project and its dependencies, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TabEmbedBench
   ```

2. **Install the package in editable mode with uv**:
   Using `uv sync` is the recommended way to set up the project. It automatically creates a virtual environment,
   installs all dependencies, and installs the `tabembedbench` package itself in **editable mode**.

   ```bash
   uv sync
   ```

   **What this does:**
    - **Editable Install**: The package is installed in a way that changes to the source code in `src/tabembedbench` are
      immediately reflected without needing to reinstall. This is ideal for development.
    - **Virtual Environment**: A `.venv` directory is created containing all project-specific dependencies, ensuring
      isolation from your system Python.
    - **Lockfile Consistency**: `uv` uses `uv.lock` to ensure that everyone working on the project has the exact same
      versions of all dependencies.

3. **(Optional) Install specific dependency groups**:
   If you only need production dependencies, you can use:
   ```bash
   uv sync --no-dev
   ```

## What it Does

TabEmbedBench is designed to simplify the evaluation of tabular embedding models. It provides:

- **Unified Interface**: A common API for different embedding models (TabPFN, TabStar, etc.).
- **Automated Benchmarking**: Tools to run models against standard datasets from ADBench and TabArena.
- **Multi-task Evaluation**: Support for Outlier Detection, Classification, and Regression.
- **Extensibility**: Easily add new models or evaluators by following the provided abstract base classes.

## Running Experiments

The project includes an example experiment script for the IJCAI benchmark.

### IJCAI Run

To run the IJCAI experiment script `src/tabembedbench/examples/ijcai_run.py`, use the following command:

```bash
uv run python src/tabembedbench/examples/ijcai_run.py
```

By default, the script is configured with `DEBUG = False`. If you want to run a quicker test, you can modify the `DEBUG`
flag in `src/tabembedbench/examples/ijcai_run.py` to `True`.

### Configuration

The experiments are configured via `BenchmarkConfig` and `DatasetConfig` objects within the script. You can adjust
parameters such as:

- `data_dir`: Local directory for dataset storage.
- `gcs_bucket`: Google Cloud Storage bucket for results (if applicable).
- `run_outlier` / `run_tabarena`: Toggle specific benchmark components.

## Project Structure

- `src/tabembedbench/`: Core source code.
    - `benchmark/`: Benchmark execution logic.
    - `embedding_models/`: Implementations of various tabular embedding models (e.g., TabPFN, TabStar, SphereBased).
    - `evaluators/`: Evaluation metrics and algorithms for classification, regression, and outlier detection.
    - `examples/`: Example scripts and benchmark runners (including `ijcai_run.py`).
- `tests/`: Unit and integration tests.
- `pyproject.toml`: Project metadata and dependency definitions.

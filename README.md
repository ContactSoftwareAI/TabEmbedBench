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

1. **Download the repository**

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
uv run python src/tabembedbench/examples/ijcai_run.py [OPTIONS]
```

### Command Line Options

The IJCAI benchmark runner supports several options to customize the run:

- `--debug`: Run in debug mode with minimal models and evaluators for testing.
- `--data-dir PATH`: Directory where results will be saved (default: `ijcai_run`).
- `--adbench-path PATH`: Path to ADBench tabular datasets (default: `data/adbench_tabular_datasets`).
- `--dataset-size INTEGER`: Upper bound on dataset size (number of samples) (default: `15000`).
- `--num-features INTEGER`: Upper bound on number of features (default: `500`).
- `--run-outlier / --no-run-outlier`: Enable/disable outlier detection benchmarking (default: `True`).
- `--run-tabarena / --no-run-tabarena`: Enable/disable TabArena benchmarking (default: `True`).
- `--exclude-adbench TEXT`: ADBench dataset names to exclude (can be used multiple times).
- `--exclude-tabarena TEXT`: TabArena dataset names to exclude (can be used multiple times).

**Example: Running a quick debug test**

```bash
uv run python src/tabembedbench/examples/ijcai_run.py --debug
```

**Example: Excluding specific datasets**

```bash
uv run python src/tabembedbench/examples/ijcai_run.py --exclude-adbench "Cardiotocography" --exclude-tabarena "blood"
```

### Configuration (Advanced)

While most parameters are available via command line, more complex configurations can still be adjusted within
`src/tabembedbench/examples/ijcai_run.py` by modifying the `get_embedding_models` and `get_evaluators` functions.

## Project Structure

- `src/tabembedbench/`: Core source code.
    - `benchmark/`: Benchmark execution logic.
    - `embedding_models/`: Implementations of various tabular embedding models (e.g., TabPFN, TabStar, SphereBased).
    - `evaluators/`: Evaluation metrics and algorithms for classification, regression, and outlier detection.
    - `examples/`: Example scripts and benchmark runners (including `ijcai_run.py`).
- `tests/`: Unit and integration tests.
- `pyproject.toml`: Project metadata and dependency definitions.

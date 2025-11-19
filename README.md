# TabEmbedBench

TabEmbedBench is a benchmarking framework for evaluating tabular data embedding methods across multiple tasks: outlier detection, classification, and regression. It provides a common API to run experiments with several embedding models and evaluator algorithms, and collects results in a consistent format.

## Overview

- Benchmarks two families of tasks:
  - Outlier detection (ADBench-style datasets)
  - Supervised tasks from TabArena (classification and regression)
- Provides several embedding generators (neural and traditional)
- Includes KNN/MLP evaluators for supervised tasks and LOF/Isolation Forest/DeepSVDD for outlier detection
- Scriptable via a Click-based CLI and a Python API


## Citations

If you use this code, please cite:

```
@misc{hoppe2025comparingtaskagnosticembeddingmodels,
      title={Comparing Task-Agnostic Embedding Models for Tabular Data}, 
      author={Frederik Hoppe and Lars Kleinemeier and Astrid Franz and Udo Göbel},
      year={2025},
      eprint={2511.14276},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2511.14276}, 
}
```

## Tech stack

- Language: Python
- Packaging/manager: `uv` (Astral) with `setuptools`
- Python version: `>=3.11, <3.14` (see `pyproject.toml`)
- Key libs: PyTorch, scikit-learn, polars, pyod, openml, tabpfn, tabicl, skrub, ray, optuna, seaborn

## Requirements

- Python 3.11 or 3.12 (3.13 also supported by the spec; stay `<3.14`)
- `uv` package manager installed
- Optional: CUDA-enabled GPU for faster neural models (PyTorch index for CUDA 12.8 is pre-configured via uv on Windows/Linux)

### Install uv
- Pip: `pip install uv`
- Installer: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Homebrew (macOS/Linux): `brew install uv`
See: https://docs.astral.sh/uv/getting-started/installation/

## Installation (development)

1) Clone the repo
```bash
git clone <repository-url>
cd TabEmbedBench
```

2) Sync dependencies and create venv
```bash
uv sync --dev
```
This will create `.venv/`, install dependencies from `pyproject.toml`, and install the package in editable mode.

3) Activate the environment (optional)
```bash
# Unix/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```
You can also run commands via `uv run` without activating the venv.

## How to run

### CLI (recommended)
A Click-based CLI is provided in `src/tabembedbench/examples/eurips_run.py`.

Using uv:
```bash
uv run src/tabembedbench/examples/eurips_run.py
```
Common options:
```bash
uv run src/tabembedbench/examples/eurips_run.py --debug
uv run src/tabembedbench/examples/eurips_run.py --max-samples 1000 --max-features 50
uv run src/tabembedbench/examples/eurips_run.py --no-run-outlier
uv run src/tabembedbench/examples/eurips_run.py --no-run-supervised
uv run src/tabembedbench/examples/eurips_run.py --adbench-data data/adbench_tabular_datasets --data-dir data
```
The CLI options (from the Click decorators):
- `--debug`
- `--max-samples` (default: 10000)
- `--max-features` (default: 200)
- `--run-outlier/--no-run-outlier` (default: run)
- `--run-supervised/--no-run-supervised` (default: run)
- `--adbench-data` (default: `data/adbench_tabular_datasets`)
- `--data-dir` (default: `data`)

Using Python directly (after activation):
```bash
python src/tabembedbench/examples/eurips_run.py --debug
```

### Python API
Minimal example using the high-level API in `benchmark/run_benchmark.py`:
```python
from tabembedbench.benchmark.run_benchmark import run_benchmark, DatasetConfig, BenchmarkConfig
from tabembedbench.embedding_models import TabICLEmbedding, TableVectorizerEmbedding, TabPFNEmbedding

models = [
    TabICLEmbedding(),
    TabPFNEmbedding(num_estimators=5),
    TableVectorizerEmbedding(),
]

dataset_config = DatasetConfig(
    adbench_dataset_path="data/adbench_tabular_datasets",
    upper_bound_dataset_size=1000,
    upper_bound_num_features=50,
)

benchmark_config = BenchmarkConfig(
    run_outlier=True,
    run_supervised=True,
    run_tabpfn_subset=True,
)

result_outlier, result_tabarena, result_dir = run_benchmark(
    embedding_models=models,
    evaluator_algorithms=[],  # see Evaluators below
    dataset_config=dataset_config,
    benchmark_config=benchmark_config,
)
```
Note: See the example script for how evaluators are constructed and passed in.

## Embedding models

Available in `src/tabembedbench/embedding_models/` and re-exported in `tabembedbench.embedding_models`:
- `TabICLEmbedding`
- `TabPFNEmbedding`
- `TableVectorizerEmbedding`

All implement the `AbstractEmbeddingGenerator` interface.

## Evaluators

Located in `src/tabembedbench/evaluators/` and used by the example runner:
- Supervised: `KNNClassifierEvaluator`, `KNNRegressorEvaluator`, `MLPClassifierEvaluator`, `MLPRegressorEvaluator`
- Outlier detection: `LocalOutlierFactorEvaluator`, `IsolationForestEvaluator`, `DeepSVDDEvaluator`

Refer to `src/tabembedbench/examples/eurips_run.py` for the complete evaluator grid and parameters.

## Results

- Results and plots are written under the chosen `--data-dir` (default `data/`).
- Example outputs in this repository show CSV and plots under timestamped `data/tabembedbench_*/` directories.

## Development

- Lint/format with Ruff (configured in `pyproject.toml`):
```bash
uv run ruff check .
uv run ruff format .
```
- Code style: see `tool.ruff` settings in `pyproject.toml`.

## Project structure

```
TabEmbedBench/
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── data/
└── src/
    └── tabembedbench/
        ├── benchmark/
        │   ├── run_benchmark.py
        │   ├── outlier_benchmark.py
        │   └── tabarena_benchmark.py
        ├── embedding_models/
        │   ├── __init__.py
        │   ├── abstractembedding.py
        │   ├── tabicl_embedding.py
        │   ├── tabpfn_embedding.py
        │   └── tablevectorizer_embedding.py
        ├── evaluators/
        │   ├── abstractevaluator.py
        │   ├── knn_classifier.py
        │   ├── knn_regressor.py
        │   ├── mlp_classifier.py
        │   ├── mlp_regressor.py
        │   └── outlier.py
        └── examples/
            └── eurips_run.py
```


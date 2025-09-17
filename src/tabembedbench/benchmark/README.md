# Benchmarks Module Overview

This directory contains the benchmarking utilities for TabEmbedBench. It provides scripts and functions to evaluate tabular embedding models on multiple tasks and datasets, and to save consolidated results for analysis.

## Contents

- `run_benchmark.py`
  - Orchestrates full benchmarking runs.
  - Combines Outlier Detection (ADBench tabular datasets) and Task‑specific (TabArena) evaluations in a single entry point.
  - Main function: `run_benchmark(...)`
    - Inputs: one or more embedding models, dataset and logging options.
    - Produces: a Polars DataFrame with results (optionally saved as Parquet in `data/results/`).

- `outlier_benchmark.py`
  - Outlier detection benchmark built on ADBench tabular datasets.
  - Main function: `run_outlier_benchmark(...)`
    - Loads `.npz` datasets, computes embeddings, and evaluates with Local Outlier Factor (LOF) across neighbor counts and distance metrics.
    - Returns a Polars DataFrame with columns like: dataset name/size, model, neighbors, AUC, compute time, benchmark label, distance metric, and task.
  - Helper: `_evaluate_local_outlier_factor(...)` computes AUROC using LOF scores.

- `tabarena_benchmark.py`
  - Task‑specific benchmark using OpenML’s TabArena suite (classification and regression tasks).
  - Main function: `run_tabarena_benchmark(...)`
    - Downloads/splits datasets via OpenML, transforms to numerical features, computes embeddings, and evaluates KNN classifiers/regressors with various neighbors and distance metrics.
    - Returns a Polars DataFrame with metrics (AUC for classification, MSR for regression), compute time, and metadata.
  - Helpers:
    - `_evaluate_classification(...)` (KNN + ROC‑AUC)
    - `_evaluate_regression(...)` (KNN + mean squared error)
    - `_get_task_configuration(...)` (controls folds/repeats, with a lite mode).

## How results and logs are handled

- Results: concatenated from individual benchmarks and (optionally) saved to `data/results/results_<timestamp>.parquet`.
- Logging: unified logging can be enabled; logs are stored under `data/logs/` when `save_logs=True`.
- GPU memory: if CUDA/MPS is detected, GPU caches are cleared between runs.

## Quick start

1. Prepare one or more embedding models implementing `BaseEmbeddingGenerator` (with `name`, `_preprocess_data`, and `_compute_embeddings`).
2. Call the umbrella function:
   ```text
   from tabembedbench.benchmark.run_benchmark import run_benchmark
   # Create one or more embedding model instances that implement BaseEmbeddingGenerator
   df = run_benchmark(
       embedding_models=[<your_models_here>],
       run_outlier=True,
       run_task_specific=True,
       save_result_dataframe=True,
   )
   ```
3. Inspect the returned Polars DataFrame or open the saved Parquet file in `data/results/`.

## Notes

- ADBench tabular datasets are downloaded on demand if not present (when running outlier benchmark with default paths).
- TabArena tasks are retrieved from OpenML (default: `tabarena-v0.1`), with a "lite" option for faster runs.
- Distance metrics commonly supported: `euclidean`, `cosine`.
- Neighbor counts are scanned in ranges controlled by `neighbors` and `neighbors_step` parameters.

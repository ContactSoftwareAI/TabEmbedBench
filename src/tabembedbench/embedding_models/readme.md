# Embedding Models

This folder contains the implementation of various embedding models designed for tabular data. These models generate vector representations of tabular data that can be used for downstream tasks like classification, regression, and outlier detection.

## Overview

All embedding models inherit from the `BaseEmbeddingGenerator` abstract base class, which defines a common interface for:
- Data preprocessing
- Embedding computation
- Model state management
- Naming conventions

## BaseEmbeddingGenerator

The `BaseEmbeddingGenerator` is an abstract base class that provides the foundation for all embedding models in this framework. It defines the core interface and functionality that all embedding generators must implement.

### Key Features

- **Abstract Interface**: Ensures consistent API across all embedding models
- **Data Preprocessing**: Standardized data preprocessing pipeline
- **Embedding Computation**: Core method for generating embeddings from input data
- **Model State Management**: Methods for resetting and managing model state
- **Flexible Naming**: Configurable naming system for model instances
- **Task Specification**: Distinguishes between task-specific and general-purpose models

### Core Methods

#### Abstract Methods (must be implemented by subclasses):

- `task_only` (property): Indicates whether the model is restricted to task-specific functionality
- `_get_default_name()`: Returns the default name for the model
- `preprocess_data(X, train=True)`: Preprocesses input data for training or inference
- `compute_embeddings(X)`: Generates embeddings for the input data
- `reset_embedding_model()`: Resets the model state after processing a dataset

#### Properties:

- `name`: Get/set the name of the embedding generator instance

### Usage Pattern

1. **Initialization**: Create an instance of a concrete embedding model
2. **Preprocessing**: Call `preprocess_data()` to prepare the data
3. **Embedding Generation**: Call `compute_embeddings()` to generate embeddings
4. **State Reset**: Call `reset_embedding_model()` between different datasets

## Available Implementations
The folder contains several concrete implementations:
1. **TabICL Embedding** (): Based on the TabICL (Tabular In-Context Learning) architecture `tabicl_embedding.py`
2. **TabPFN Embedding** (): Uses TabPFN (Tabular Prior-Fitted Networks) for embedding generation `tabpfn_embedding.py`
3. **Sphere-based Embedding** (): Geometric embedding approach using spherical projections `spherebased_embedding.py`

## Integration with Benchmarking Framework
These embedding models are designed to work seamlessly with the tabembedbench benchmarking framework, supporting:
- Outlier detection benchmarks
- TabArena task-specific evaluations
- Performance comparison across different embedding approaches
- Automated model evaluation pipelines

## Adding New Models
To add a new embedding model:
1. Create a new file in this directory
2. Inherit from `BaseEmbeddingGenerator`
3. Implement all abstract methods
4. Follow the established naming and interface conventions
5. Test integration with the benchmarking framework

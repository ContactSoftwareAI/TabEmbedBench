# Embedding Models Module

This module contains implementations of various embedding models designed for tabular data. These models generate vector representations of tabular data that can be used for downstream tasks like classification, regression, and outlier detection.

## Overview

All embedding models inherit from the `AbstractEmbeddingGenerator` abstract base class, which defines a common interface for:
- **Data Preprocessing**: Standardized data preparation pipelines
- **Embedding Computation**: Core methods for generating vector representations
- **Model State Management**: Methods for resetting and managing model state
- **Naming Conventions**: Consistent model identification and tracking

The module supports both neural and traditional embedding approaches, providing flexibility for different use cases and computational requirements.

## Base Class: AbstractEmbeddingGenerator

The `AbstractEmbeddingGenerator` class (`abstractembedding.py`) provides the foundation for all embedding models in this framework.

### Key Features

- **Abstract Interface**: Ensures consistent API across all embedding models
- **Data Preprocessing**: Standardized data preprocessing pipeline with train/test modes
- **Embedding Computation**: Core method for generating embeddings from input data
- **Model State Management**: Methods for resetting and managing model state between datasets
- **Flexible Naming**: Configurable naming system for model instances and experiment tracking
- **Validation**: Built-in validation for embedding quality and consistency

### Core Methods

#### Abstract Methods (must be implemented by subclasses):

##### `_preprocess_data(X, train=True)`
Preprocesses input data for training or inference.
- **Parameters**: 
  - `X`: Input data array
  - `train`: Whether this is training data (affects preprocessing behavior)
- **Returns**: Preprocessed data ready for embedding computation

##### `_compute_embeddings(X)`
Generates embeddings for the input data.
- **Parameters**: `X`: Preprocessed input data
- **Returns**: Numpy array of embeddings with shape (n_samples, embedding_dim)

##### `reset_embedding_model()`
Resets the model state after processing a dataset.
- **Purpose**: Ensures clean state between different datasets in benchmarks

#### Concrete Methods:

##### `compute_embeddings(X_train, X_test=None)`
Main interface for embedding computation with timing and validation.
- **Parameters**:
  - `X_train`: Training data for embedding computation
  - `X_test`: Optional test data for embedding computation
- **Returns**: Tuple of (train_embeddings, test_embeddings, computation_time)
- **Features**:
  - Automatic preprocessing
  - Timing measurement
  - Embedding validation
  - Error handling

##### `name` (property)
Get/set the name of the embedding generator instance.
- **Usage**: For experiment tracking and result identification

#### Static Validation Methods:

##### `_check_emb_shape(embeddings: np.ndarray) -> bool`
Validates embedding array shape and dimensions.

##### `_check_nan(embeddings: np.ndarray) -> bool`
Checks for NaN values in embeddings.

##### `_validate_embeddings(embeddings, context="")`
Comprehensive embedding validation with detailed error reporting.

### Usage Pattern

```python
# 1. Initialization
model = SomeEmbeddingModel(param1=value1)
model.name = "custom-model-name"

# 2. Embedding Generation
train_emb, test_emb, time_taken = model.generate_embeddings(X_train, X_test)

# 3. State Reset (between datasets)
model._reset_embedding_model()
```

## Available Implementations

### 1. TabICLEmbedding (`tabicl_embedding.py`)

Neural embedding model based on TabICL (Tabular In-Context Learning) architecture.

#### Key Features:
- **Neural Architecture**: Deep learning-based embedding computation
- **In-Context Learning**: Leverages contextual information for embeddings
- **Preprocessing Pipeline**: Integrated data preprocessing for optimal performance
- **GPU Support**: Optimized for GPU acceleration

#### Core Classes:
- `TabICLRowEmbedding(nn.Module)`: PyTorch module for row-level embeddings
- `TabICLEmbedding`: Main embedding generator class
- `OutlierPreprocessingPipeline`: Specialized preprocessing for outlier detection

#### Key Methods:
- `get_tabicl_model()`: Retrieves the underlying TabICL model
- `_preprocess_data()`: TabICL-specific data preprocessing
- `_compute_embeddings()`: Neural embedding computation

#### Parameters:
- `preprocess_tabicl_data`: Whether to apply TabICL-specific preprocessing
- Model-specific parameters passed to underlying TabICL architecture

#### Usage:

```python
model = TabICLEmbedding(preprocess_tabicl_data=True)
model.name = "tabicl-preprocessed"
embeddings = model.generate_embeddings(X_train, X_test)
```

### 2. TabPFNEmbedding (`tabpfn_embedding.py`)

Embedding model based on TabPFN (Tabular Prior-Fitted Networks) architecture.

#### Key Features:
- **Prior-Fitted Networks**: Leverages pre-trained knowledge for embeddings
- **Universal Architecture**: Works across diverse tabular datasets
- **Codebook Reduction**: Advanced dimensionality reduction techniques
- **Task-Specific Optimization**: Optimized for specific downstream tasks

#### Core Methods:
- `_compute_internal_embeddings()`: Implements column-wise embedding strategy
- `_preprocess_data()`: Converts data to float64
- `_fit_model()`: Identifies categorical features
- `_compute_embeddings()`: Computes embeddings for train/test data

#### Parameters:
- `num_estimators`: Number of estimators for ensemble predictions (default: 1)
- `estimator_agg`: Method for aggregating estimator outputs ('mean', 'first_element')
- `emb_agg`: Method for aggregating embeddings across columns ('mean', 'concat')

#### Column-wise Embedding Strategy:
The model treats each column alternately as a target variable while using other columns as features, fitting appropriate TabPFN models (classifier for categorical, regressor for numerical) and extracting embeddings.

#### Usage:

```python
model = TabPFNEmbedding(num_estimators=1, emb_agg='mean')
model.name = "tabpfn"
embeddings = model.generate_embeddings(X_train, X_test)
```

### 3. TableVectorizerEmbedding (`tablevectorizer_embedding.py`)

Traditional embedding approach using table vectorization techniques.

#### Key Features:
- **Vectorization-Based**: Uses traditional ML vectorization approaches
- **Optimization Support**: Optional hyperparameter optimization
- **Lightweight**: Minimal computational requirements
- **Fast Training**: Quick embedding computation

#### Core Methods:
- `_preprocess_data()`: Vectorizer-specific preprocessing
- `_compute_embeddings()`: Vectorization-based embedding computation
- `_optimize_tablevectorizer()`: Hyperparameter optimization

#### Parameters:
- `optimize`: Whether to perform hyperparameter optimization
- Vectorizer-specific parameters

#### Usage:

```python
model = TableVectorizerEmbedding(optimize=True)
model.name = "tablevectorizer-optimized"
embeddings = model.generate_embeddings(X_train, X_test)
```

## Integration with Benchmarking Framework

### Workflow Integration
1. **Model Registration**: Models are registered with the benchmark system
2. **Data Preprocessing**: Each model applies its specific preprocessing
3. **Embedding Generation**: Models compute embeddings for train/test data
4. **Validation**: Embeddings are validated for quality and consistency
5. **Evaluation**: Embeddings are passed to evaluators for performance assessment
6. **State Management**: Models are reset between different datasets

### Performance Tracking
- **Computation Time**: Automatic timing of embedding generation
- **Memory Usage**: Monitoring of memory consumption
- **GPU Utilization**: Tracking of GPU usage for neural models
- **Quality Metrics**: Validation of embedding properties

### Error Handling
- **Data Compatibility**: Automatic checking of data format requirements
- **Resource Management**: Handling of memory and GPU constraints
- **Graceful Degradation**: Fallback strategies for failed computations
- **Detailed Logging**: Comprehensive error reporting and debugging

## Model Selection Guidelines

### For Large Datasets:
- **TableVectorizerEmbedding**: Fast and memory-efficient

### For High-Quality Embeddings:
- **TabICLEmbedding**: State-of-the-art neural approach
- **UniversalTabPFNEmbedding**: Pre-trained knowledge transfer

### For Mixed Data Types:
- **TabICLEmbedding**: Robust preprocessing pipeline

### For GPU-Accelerated Computation:
- **TabICLEmbedding**: Optimized for GPU acceleration
- **UniversalTabPFNEmbedding**: GPU-compatible architecture

## Adding New Models

To add a new embedding model:

### 1. Create Model Class
```python
class MyEmbeddingModel(AbstractEmbeddingGenerator):
    def __init__(self, param1=default1, **kwargs):
        super().__init__()
        self.param1 = param1
        # Initialize model-specific components
    
    @property
    def _preprocess_data(self, X, train=True):
        # Implement preprocessing logic
        return processed_X
    
    def _compute_embeddings(self, X):
        # Implement embedding computation
        return embeddings
    
    def reset_embedding_model(self):
        # Reset model state
        pass
```

### 2. Follow Naming Conventions
- Use descriptive class names ending in "Embedding"
- Implement consistent parameter naming
- Provide meaningful default names

### 3. Add Documentation
- Document all parameters and methods
- Provide usage examples
- Explain model-specific features

### 4. Test Integration
- Test with benchmark framework
- Validate embedding quality
- Check performance characteristics

### 5. Update Module Exports
Add new model to `__init__.py` for easy importing.

## Dependencies

The embedding models module requires:
- **NumPy**: For numerical operations and array handling
- **PyTorch**: For neural network implementations (TabICL, TabPFN)
- **Scikit-learn**: For traditional ML components and interfaces
- **Pandas**: For data manipulation and preprocessing
- **Abstract Base Classes**: For interface definitions

## Performance Considerations

### Memory Management:
- Models automatically handle memory cleanup
- GPU cache clearing between datasets
- Efficient data structure usage

### Computational Efficiency:
- Vectorized operations where possible
- GPU acceleration for neural models
- Lazy loading of large models

### Scalability:
- Batch processing support
- Streaming data handling
- Distributed computation compatibility

## Notes

- All models are designed to work with numpy arrays as input
- State management is crucial - always call `reset_embedding_model()` between datasets
- Neural models require appropriate hardware (GPU recommended)
- Preprocessing is model-specific and handled automatically
- Embedding validation ensures quality and consistency across models
- The module supports both supervised and unsupervised embedding approaches

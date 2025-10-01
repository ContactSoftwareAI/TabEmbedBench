# Utils Module

This module contains utility functions and helper classes that support the core functionality of TabEmbedBench. The utilities provide essential services for data handling, preprocessing, logging, visualization, and system management across the framework.

## Overview

The utils module is organized into specialized utility files:
- **Data Management**: Dataset loading, validation, and preprocessing
- **Embedding Operations**: Embedding computation and aggregation utilities
- **System Management**: Device handling, logging, and resource management
- **Visualization**: Plotting and chart generation utilities
- **Result Tracking**: Experiment tracking and result management

## Available Utilities

### 1. Configuration (`config.py`)

Defines configuration enums and constants used throughout the framework.

#### Classes and Enums

##### `EmbAggregation(Enum)`
Enumeration of embedding aggregation methods for combining multiple embeddings.

**Available Methods:**
- `MEAN`: Average aggregation across embeddings
- `MAX`: Maximum value aggregation
- `MIN`: Minimum value aggregation
- `SUM`: Sum aggregation
- Additional aggregation strategies as needed

**Usage:**
```python
from tabembedbench.utils.config import EmbAggregation
agg_method = EmbAggregation.MEAN
```

### 2. Dataset Utilities (`dataset_utils.py`)

Provides functions for dataset management, loading, and validation.

#### Key Functions

##### `download_adbench_tabular_datasets()`
Downloads ADBench tabular datasets for outlier detection benchmarks.

**Features:**
- Automatic dataset downloading and caching
- Validation of dataset integrity
- Support for multiple dataset formats

##### `get_data_description()`
Retrieves metadata and descriptions for datasets.

**Returns:** Dataset characteristics, feature information, and task details

##### `read_data(data_path: str | Path) -> tuple[np.ndarray, np.ndarray]`
Loads dataset from file path.

**Parameters:**
- `data_path`: Path to dataset file

**Returns:** Tuple of (features, labels) as numpy arrays

##### `select_random_combined_datasets(datasets_dir: str) -> list[str | Path]`
Randomly selects datasets from a directory for benchmarking.

**Parameters:**
- `datasets_dir`: Directory containing datasets

**Returns:** List of selected dataset paths

##### `prepare_data_for_torch(X: np.ndarray, device: str = "cpu") -> torch.Tensor`
Converts numpy arrays to PyTorch tensors with proper device placement.

**Parameters:**
- `X`: Input data array
- `device`: Target device ("cpu", "cuda", "mps")

**Returns:** PyTorch tensor on specified device

##### `check_tabpfn_dataset_restrictions(X: np.ndarray) -> bool`
Validates dataset compatibility with TabPFN model requirements.

**Parameters:**
- `X`: Dataset features

**Returns:** Boolean indicating compatibility

##### `check_tabicl_dataset_restrictions(X: np.ndarray) -> bool`
Validates dataset compatibility with TabICL model requirements.

**Parameters:**
- `X`: Dataset features

**Returns:** Boolean indicating compatibility

### 3. Embedding Utilities (`embedding_utils.py`)

Provides functions for embedding computation, aggregation, and validation.

#### Key Functions

##### `compute_embeddings_aggregation()`
Computes aggregated embeddings using specified aggregation methods.

**Features:**
- Multiple aggregation strategies
- Batch processing support
- Memory-efficient computation

##### `embeddings_aggregation()`
Applies aggregation functions to embedding collections.

**Supports:** Mean, max, min, sum, and custom aggregation functions

##### `check_emb_aggregation(agg_func: EmbAggregation | str)`
Validates aggregation function specifications.

**Parameters:**
- `agg_func`: Aggregation method (enum or string)

**Returns:** Validated aggregation function

##### `validate_input(input_list)`
Validates input data for embedding computation.

**Features:**
- Type checking
- Shape validation
- Data quality assessment

##### `check_nan()`
Detects and handles NaN values in embeddings.

**Features:**
- NaN detection
- Replacement strategies
- Data integrity validation

### 4. Logging Utilities (`logging_utils.py`)

Provides comprehensive logging functionality for benchmarking and debugging.

#### Key Functions

##### `neighbors_progress(self, message, *args, **kwargs)`
Custom logging method for tracking neighbor-based algorithm progress.

**Features:**
- Progress tracking for KNN algorithms
- Detailed parameter logging
- Performance monitoring

##### `setup_unified_logging()`
Configures unified logging across the entire framework.

**Features:**
- Consistent log formatting
- Multiple output destinations
- Configurable log levels
- Timestamp and context tracking

##### `get_benchmark_logger(name: str) -> logging.Logger`
Creates specialized loggers for benchmark components.

**Parameters:**
- `name`: Logger name/identifier

**Returns:** Configured logger instance

**Features:**
- Component-specific logging
- Hierarchical logger organization
- Benchmark-specific formatting

### 5. Plot Utilities (`plot_utils.py`)

Provides visualization functions for benchmark results and analysis.

#### Key Functions

##### `create_boxplot()`
Generates boxplot visualizations for performance comparisons.

**Features:**
- Multi-model comparisons
- Statistical summaries
- Customizable styling
- Export capabilities

##### `create_quantile_lines_chart()`
Creates quantile-based line charts for performance analysis.

**Features:**
- Quantile visualization
- Trend analysis
- Confidence intervals
- Interactive plots

##### `create_multi_model_quantile_lines_chart()`
Generates comparative quantile charts across multiple models.

**Features:**
- Multi-model comparison
- Quantile overlays
- Performance ranking
- Statistical significance testing

### 6. Preprocessing Utilities (`preprocess_utils.py`)

Provides data preprocessing functions for tabular data preparation.

#### Key Functions

##### `infer_categorical_features()`
Automatically detects categorical features in datasets.

**Features:**
- Heuristic-based detection
- Type inference
- Threshold-based classification
- Manual override support

##### `infer_categorical_columns()`
Identifies categorical columns in structured data.

**Features:**
- Column-wise analysis
- Data type detection
- Cardinality-based inference
- Custom rules support

##### `_create_empty_array()`
Creates properly shaped empty arrays for data processing.

**Features:**
- Shape preservation
- Type consistency
- Memory optimization
- Initialization strategies

##### `_restore_original_format()`
Restores data to original format after processing.

**Features:**
- Format preservation
- Type restoration
- Structure maintenance
- Metadata retention

### 7. PyTorch Utilities (`torch_utils.py`)

Provides PyTorch-specific utilities for device management and optimization.

#### Key Functions

##### `get_device() -> torch.device`
Automatically detects and returns the best available device.

**Features:**
- CUDA detection
- MPS (Apple Silicon) support
- CPU fallback
- Device capability checking

**Returns:** Optimal PyTorch device

##### `empty_gpu_cache(device: torch.device)`
Clears GPU memory cache to prevent memory issues.

**Parameters:**
- `device`: Target device for cache clearing

**Features:**
- Memory management
- Cache optimization
- Multi-GPU support
- Error handling

### 8. Tracking Utilities (`tracking_utils.py`)

Provides experiment tracking and result management functionality.

#### Key Functions

##### `get_batch_dict_result_df()`
Creates structured DataFrames for batch result tracking.

**Features:**
- Standardized result format
- Metadata inclusion
- Performance metrics
- Experiment organization

##### `update_batch_dict()`
Updates batch tracking dictionaries with new results.

**Features:**
- Incremental updates
- Data validation
- Conflict resolution
- History preservation

##### `update_result_df()`
Updates result DataFrames with new experimental data.

**Features:**
- DataFrame operations
- Schema validation
- Data consistency
- Performance optimization

##### `save_result_df()`
Saves result DataFrames to persistent storage.

**Features:**
- Multiple format support (Parquet, CSV, etc.)
- Compression options
- Metadata preservation
- Backup strategies

## Integration Patterns

### Data Flow Integration
The utilities support the complete data flow:
1. **Dataset Loading**: `dataset_utils` handles data ingestion
2. **Preprocessing**: `preprocess_utils` prepares data for models
3. **Embedding Computation**: `embedding_utils` manages embedding operations
4. **Device Management**: `torch_utils` handles hardware optimization
5. **Result Tracking**: `tracking_utils` manages experiment results
6. **Visualization**: `plot_utils` creates analysis charts

### Cross-Module Dependencies
- **Embedding Models**: Use dataset and preprocessing utilities
- **Evaluators**: Leverage embedding and validation utilities
- **Benchmarks**: Integrate logging, tracking, and visualization utilities
- **Examples**: Demonstrate utility usage patterns

### Error Handling and Validation
All utilities include:
- Input validation
- Error handling
- Logging integration
- Performance monitoring
- Resource management

## Usage Examples

### Basic Data Loading
```python
from tabembedbench.utils.dataset_utils import read_data, prepare_data_for_torch

# Load dataset
X, y = read_data("path/to/dataset.npz")

# Prepare for PyTorch
X_tensor = prepare_data_for_torch(X, device="cuda")
```

### Embedding Aggregation
```python
from tabembedbench.utils.embedding_utils import embeddings_aggregation
from tabembedbench.utils.config import EmbAggregation

# Aggregate embeddings
aggregated = embeddings_aggregation(
    embeddings_list, 
    agg_func=EmbAggregation.MEAN
)
```

### Logging Setup
```python
from tabembedbench.utils.logging_utils import setup_unified_logging, get_benchmark_logger

# Setup logging
setup_unified_logging(level="INFO")

# Get component logger
logger = get_benchmark_logger("my_component")
logger.info("Starting benchmark")
```

### Result Tracking
```python
from tabembedbench.utils.tracking_utils import update_result_df, save_result_df

# Update results
result_df = update_result_df(result_df, new_results)

# Save results
save_result_df(result_df, "results/experiment_1.parquet")
```

### Device Management
```python
from tabembedbench.utils.torch_utils import get_device, empty_gpu_cache

# Get optimal device
device = get_device()

# Clear cache after processing
empty_gpu_cache(device)
```

## Best Practices

### Performance Optimization
- Use appropriate aggregation methods for embedding size
- Leverage device utilities for hardware optimization
- Implement proper memory management
- Cache frequently accessed data

### Error Handling
- Validate inputs before processing
- Use utility validation functions
- Implement graceful degradation
- Log errors appropriately

### Resource Management
- Clear GPU cache between experiments
- Use appropriate data types
- Optimize memory usage
- Monitor resource consumption

### Reproducibility
- Use consistent random seeds
- Log all configuration parameters
- Save intermediate results
- Document utility versions

## Dependencies

The utils module requires:
- **NumPy**: For numerical operations
- **PyTorch**: For tensor operations and device management
- **Pandas**: For DataFrame operations
- **Matplotlib/Plotly**: For visualization utilities
- **Logging**: For logging functionality
- **Pathlib**: For path operations

## Notes

- All utilities are designed to be framework-agnostic where possible
- Device utilities automatically detect available hardware
- Logging utilities provide consistent formatting across components
- Visualization utilities support both static and interactive plots
- Result tracking utilities support multiple storage formats
- All utilities include comprehensive error handling and validation

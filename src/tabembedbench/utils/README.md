# Utils Module

Shared utilities providing core infrastructure for data handling, logging, device management, preprocessing, result tracking, and visualization across the TabEmbedBench framework.

## Overview

The utils module contains six key utility components:

| Module | Purpose |
|--------|---------|
| `dataset_utils.py` | Dataset downloading and management |
| `torch_utils.py` | PyTorch device management and GPU utilities |
| `logging_utils.py` | Unified logging configuration |
| `preprocess_utils.py` | Data preprocessing and type inference |
| `tracking_utils.py` | Result tracking and storage |
| `eda_utils.py` | Visualization and exploratory analysis |

## Utilities Reference

### 1. Dataset Management (`dataset_utils.py`)

Handles downloading and managing benchmark datasets from external sources.

#### `download_adbench_tabular_datasets(save_path=None)`

Downloads tabular datasets for ADBench from the GitHub repository.

**Parameters:**
- `save_path` (str | Path | None): Directory to save datasets. Defaults to `./data/adbench_tabular_datasets`

**Behavior:**
- Downloads ADBench repository as zip file
- Extracts only Classical tabular datasets
- Creates directory structure automatically
- Cleans up temporary files

**Usage:**
```python
from tabembedbench.utils.dataset_utils import download_adbench_tabular_datasets

# Download to default location
download_adbench_tabular_datasets()

# Download to custom location
download_adbench_tabular_datasets(save_path="data/my_datasets")
```

**Data Source:**
- GitHub: https://github.com/Minqi824/ADBench
- Extracts: `Classical/` directory containing `.npz` files
- Skips: Image datasets and non-tabular data

### 2. PyTorch Device Management (`torch_utils.py`)

Provides utilities for managing PyTorch devices and GPU memory.

#### `get_device()`

Determines the appropriate compute device based on hardware availability.

**Returns:**
- `torch.device`: CUDA device if available, otherwise CPU

**Usage:**
```python
from tabembedbench.utils.torch_utils import get_device

device = get_device()
model = model.to(device)
```

#### `empty_gpu_cache(device)`

Clears GPU memory cache for the specified device.

**Parameters:**
- `device` (torch.device): Target device for cache clearing

**Supported Devices:**
- CUDA: Uses `torch.cuda.empty_cache()`
- MPS (Apple Silicon): Uses `torch.mps.empty_cache()`

**Usage:**
```python
from tabembedbench.utils.torch_utils import get_device, empty_gpu_cache

device = get_device()
# After heavy computation
empty_gpu_cache(device)
```

#### `log_gpu_memory(logger)`

Logs detailed GPU memory usage information.

**Parameters:**
- `logger` (logging.Logger): Logger instance for output

**Logged Information:**
- Allocated memory (GB)
- Reserved memory (GB)
- Maximum allocated memory (GB)
- Per-GPU statistics for multi-GPU systems

**Usage:**
```python
from tabembedbench.utils.torch_utils import log_gpu_memory
import logging

logger = logging.getLogger(__name__)
log_gpu_memory(logger)
```

### 3. Logging Configuration (`logging_utils.py`)

Unified logging setup for consistent output across the framework.

#### `setup_unified_logging(save_logs=True, log_dir="log", timestamp=None, logging_level=logging.INFO, capture_warnings=True)`

Configures unified logging with console and optional file handlers.

**Parameters:**
- `save_logs` (bool): Whether to save logs to file (default: True)
- `log_dir` (str): Directory for log files (default: "log")
- `timestamp` (str): Timestamp for log filename (auto-generated if None)
- `logging_level` (int): Logging level (default: logging.INFO)
- `capture_warnings` (bool): Capture Python warnings (default: True)

**Returns:**
- `str | None`: Log file path if `save_logs=True`, otherwise None

**Configured Loggers:**
- `TabEmbedBench_Main`: Main framework logger
- `TabEmbedBench_Outlier`: Outlier benchmark logger
- `TabEmbedBench_TabArena`: TabArena benchmark logger
- `py.warnings`: Python warnings logger

**Log Format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
```

**Usage:**
```python
from tabembedbench.utils.logging_utils import setup_unified_logging
import logging

# Basic setup
log_file = setup_unified_logging()

# Custom configuration
log_file = setup_unified_logging(
    save_logs=True,
    log_dir="my_logs",
    logging_level=logging.DEBUG,
    capture_warnings=True
)
```

#### `get_benchmark_logger(name)`

Retrieves a logger instance with custom benchmark logging methods.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Configured logger instance

**Custom Log Levels:**
- `NEIGHBORS_PROGRESS` (level 5): Special level for tracking neighbor computation progress

**Usage:**
```python
from tabembedbench.utils.logging_utils import get_benchmark_logger

logger = get_benchmark_logger("MyBenchmark")
logger.info("Starting benchmark...")
logger.neighbors_progress("Computing neighbors for dataset X")
```

### 4. Data Preprocessing (`preprocess_utils.py`)

Utilities for data type inference and preprocessing.

#### `infer_categorical_features(data, categorical_features=None)`

Infers categorical features from tabular data using multiple heuristics.

**Parameters:**
- `data` (np.ndarray | pd.DataFrame): Input data
- `categorical_features` (list[int] | None): Initial categorical feature indices

**Detection Criteria:**
1. Explicit categorical/object/string data types (pandas)
2. Low unique value count relative to dataset size
3. String values in numpy arrays
4. Features with ≤10 unique values

**Returns:**
- `list[int]`: Indices of detected categorical features

**Thresholds:**
- Max unique values for categorical: 10
- Min unique values for numerical: 10
- Minimum dataset size for ratio check: 100

**Usage:**
```python
from tabembedbench.utils.preprocess_utils import infer_categorical_features
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'NYC'],
    'score': [85.5, 90.2, 88.1]
})

cat_indices = infer_categorical_features(df)
# Returns: [1] (city column)
```

#### `infer_categorical_columns(data, max_unique_ratio=0.1, max_unique_count=200, return_split=False)`

Advanced categorical inference with configurable thresholds and data splitting.

**Parameters:**
- `data` (np.ndarray | torch.Tensor | pd.DataFrame): Input data
- `max_unique_ratio` (float): Maximum ratio of unique values to total samples (default: 0.1)
- `max_unique_count` (int): Maximum unique values for categorical (default: 200)
- `return_split` (bool): Return split numerical/categorical arrays (default: False)

**Returns:**
- If `return_split=False`: `list[int]` of categorical indices
- If `return_split=True`: `tuple[array, array]` of (numerical_data, categorical_data)

**Features:**
- Handles 2D and 3D arrays (batch processing)
- NaN-aware inference
- Preserves tensor devices and dtypes
- Returns empty arrays for missing categories

**Usage:**
```python
from tabembedbench.utils.preprocess_utils import infer_categorical_columns
import numpy as np

X = np.random.rand(1000, 10)
X[:, 2] = np.random.randint(0, 5, 1000)  # Categorical column

# Get indices
cat_indices = infer_categorical_columns(X)

# Split data
num_data, cat_data = infer_categorical_columns(X, return_split=True)
```

### 5. Result Tracking (`tracking_utils.py`)

Utilities for saving and managing benchmark results.

#### `save_result_df(result_df, output_path, benchmark_name, timestamp)`

Saves benchmark results to both Parquet and CSV formats.

**Parameters:**
- `result_df` (pl.DataFrame): Results dataframe to save
- `output_path` (str | Path): Output directory
- `benchmark_name` (str): Benchmark identifier for filename
- `timestamp` (str): Timestamp for filename uniqueness

**Output Files:**
- `results_{benchmark_name}_{timestamp}.parquet`: Binary format for efficient storage
- `results_{benchmark_name}_{timestamp}.csv`: Text format for easy inspection

**File Naming Pattern:**
```
results_ADBench_Tabular_20251118_090000.parquet
results_ADBench_Tabular_20251118_090000.csv
```

**Usage:**
```python
from tabembedbench.utils.tracking_utils import save_result_df
import polars as pl
from datetime import datetime

results = pl.DataFrame({
    'dataset': ['data1', 'data2'],
    'score': [0.85, 0.92]
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_result_df(
    result_df=results,
    output_path="data/results",
    benchmark_name="MyBenchmark",
    timestamp=timestamp
)
```

### 6. Exploratory Data Analysis (`eda_utils.py`)

Utilities for result visualization and analysis.

#### Style Configuration

##### `setup_publication_style()`

Configures matplotlib/seaborn for publication-quality figures.

**Settings:**
- Context: "paper"
- Style: "whitegrid"
- Font sizes: 9-11pt for readability
- Seaborn color palette: "colorblind"

#### Data Processing

##### `separate_by_task_type(df)`

Separates results by machine learning task type.

**Parameters:**
- `df` (pl.DataFrame): Results dataframe with 'classification_type' and 'task' columns

**Returns:**
- `tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]`: Binary, multiclass, regression results

##### `clean_results(df)`

Filters datasets with inconsistent algorithm coverage across models.

**Parameters:**
- `df` (pl.DataFrame): Results dataframe

**Returns:**
- `pl.DataFrame`: Cleaned results with consistent algorithm coverage

**Behavior:**
- Ensures all models evaluated with same algorithms per dataset
- Logs warnings for filtered datasets
- Maintains data integrity for fair comparisons

##### `create_descriptive_dataframe(df, metric_col)`

Computes descriptive statistics grouped by model and algorithm.

**Parameters:**
- `df` (pl.DataFrame): Results dataframe
- `metric_col` (str): Metric column for statistics

**Returns:**
- `pl.DataFrame`: Descriptive statistics including mean, std, min, max, median, embedding time, dataset count

#### Visualization Functions

##### `create_outlier_plots(df, data_path="data", name_mapping=None, color_mapping=None, models_to_keep=None, algorithm_order=None, bin_edges=None)`

Creates comprehensive visualizations for outlier detection results.

**Parameters:**
- `df` (pl.DataFrame): Outlier detection results
- `data_path` (str | Path): Output directory
- `name_mapping` (dict | None): Model name renaming
- `color_mapping` (dict | None): Custom colors per model
- `models_to_keep` (list | None): Models to include
- `algorithm_order` (list | None): X-axis ordering
- `bin_edges` (list | None): Bins for outlier ratio analysis

**Outputs:**
- `outlier_algorithm_comparison.pdf`: Boxplot comparison
- `outlier_descriptive.csv`: Descriptive statistics
- `outlier_ratio.csv`: Outlier ratio analysis (if bin_edges provided)

##### `create_tabarena_plots(df, data_path="data", name_mapping=None, color_mapping=None, models_to_keep=None, algorithm_order_classification=None, algorithm_order_regression=None)`

Creates visualizations for TabArena supervised learning results.

**Parameters:**
- `df` (pl.DataFrame): TabArena results
- `data_path` (str | Path): Output directory
- `name_mapping` (dict | None): Model name renaming
- `color_mapping` (dict | None): Custom colors per model
- `models_to_keep` (list | None): Models to include
- `algorithm_order_classification` (list | None): Classification plot ordering
- `algorithm_order_regression` (list | None): Regression plot ordering

**Outputs:**
- `binary_clf_algorithm_comparison.pdf`: Binary classification boxplot
- `multiclass_clf_algorithm_comparison.pdf`: Multiclass classification boxplot
- `regression_algorithm_comparison.pdf`: Regression boxplot
- `binary_auc_score_descriptive.csv`: Binary classification statistics
- `multiclass_auc_score_descriptive.csv`: Multiclass statistics
- `regression_auc_score_descriptive.csv`: Regression statistics

#### Helper Functions

##### `keeping_models(df, keep_models)`
Filters dataframe to specified models.

##### `rename_models(df, renaming_mapping)`
Renames models according to mapping dictionary.

##### `create_color_mapping(models_to_keep)`
Creates colorblind-friendly color palette.

##### `save_fig(ax, data_path, file_name)`
Saves matplotlib figure as publication-quality PDF.

## Usage Examples

### Complete Workflow

```python
from tabembedbench.utils.dataset_utils import download_adbench_tabular_datasets
from tabembedbench.utils.logging_utils import setup_unified_logging
from tabembedbench.utils.torch_utils import get_device, empty_gpu_cache
from tabembedbench.utils.tracking_utils import save_result_df
from tabembedbench.utils.eda_utils import create_outlier_plots
import polars as pl
from datetime import datetime

# 1. Setup logging
log_file = setup_unified_logging(
    save_logs=True,
    log_dir="logs",
    logging_level=logging.INFO
)

# 2. Download datasets
download_adbench_tabular_datasets(save_path="data/datasets")

# 3. Configure device
device = get_device()
print(f"Using device: {device}")

# 4. Run benchmark (your code)
results = run_my_benchmark()

# 5. Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_result_df(
    result_df=results,
    output_path="data/results",
    benchmark_name="MyBenchmark",
    timestamp=timestamp
)

# 6. Create visualizations
create_outlier_plots(
    df=results,
    data_path="data/visualizations",
    models_to_keep=["TabICL", "TabPFN", "TableVectorizer"]
)

# 7. Cleanup
empty_gpu_cache(device)
```

### Custom Visualization

```python
from tabembedbench.utils.eda_utils import (
    setup_publication_style,
    create_color_mapping,
    create_descriptive_dataframe
)
import polars as pl

# Load results
results = pl.read_parquet("results.parquet")

# Setup publication style
setup_publication_style()

# Create custom color mapping
models = ["Model A", "Model B", "Model C"]
colors = create_color_mapping(models)

# Generate descriptive statistics
stats = create_descriptive_dataframe(results, metric_col="auc_score")
print(stats)
```

### Preprocessing Pipeline

```python
from tabembedbench.utils.preprocess_utils import (
    infer_categorical_features,
    infer_categorical_columns
)
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Infer categorical features
cat_features = infer_categorical_features(df)
print(f"Categorical features: {cat_features}")

# Split numerical and categorical
num_data, cat_data = infer_categorical_columns(
    df.values,
    max_unique_ratio=0.1,
    return_split=True
)

print(f"Numerical shape: {num_data.shape}")
print(f"Categorical shape: {cat_data.shape}")
```

## Integration with Framework

### Benchmark Integration

The utils module is used throughout the benchmark pipeline:

```
Setup Logging → Download Datasets → Configure Device
       ↓                ↓                  ↓
   Logging Utils   Dataset Utils      Torch Utils
       ↓                ↓                  ↓
Infer Data Types → Run Embeddings → Track Results
       ↓                ↓                  ↓
Preprocess Utils   Torch Utils      Tracking Utils
       ↓                ↓                  ↓
  Save Results  → Create Plots  →  Analyze Results
       ↓                ↓                  ↓
Tracking Utils     EDA Utils         EDA Utils
```

### Module Dependencies

```python
# Core imports used across utils
import numpy as np
import polars as pl
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
```

## Design Principles

### 1. Separation of Concerns
- Each utility module handles a specific aspect of the framework
- No cross-dependencies between utility modules
- Clean interfaces for easy integration

### 2. Robustness
- Automatic directory creation
- Graceful error handling
- NaN-aware operations
- Device-agnostic GPU operations

### 3. Flexibility
- Configurable parameters with sensible defaults
- Support for multiple data formats (numpy, pandas, torch)
- Optional features (logging, splitting, visualization)

### 4. Efficiency
- Lazy loading and processing
- Memory-efficient data handling
- GPU cache management
- Polars for fast dataframe operations

### 5. Reproducibility
- Timestamp-based file naming
- Comprehensive logging
- Consistent random seeds (where applicable)
- Standardized output formats

## Performance Considerations

### Memory Efficiency
- Streaming data processing
- Efficient dataframe operations with Polars
- Automatic GPU cache clearing
- Minimal data copies

### Computation Speed
- Vectorized operations (numpy/torch)
- Parallel data loading where possible
- Efficient categorical inference
- Optimized visualization rendering

### Storage Optimization
- Dual format saving (Parquet + CSV)
- Parquet for efficient binary storage
- CSV for human readability
- Automatic directory organization

## Dependencies

**Core Requirements:**
- `numpy`: Numerical operations
- `polars`: Efficient dataframe operations
- `torch`: GPU management and tensor operations
- `pandas`: Data preprocessing compatibility
- `pathlib`: Cross-platform path handling
- `logging`: Unified logging infrastructure

**Visualization Requirements:**
- `matplotlib`: Plotting backend
- `seaborn`: Statistical visualizations

**Network Operations:**
- `requests`: Dataset downloading
- `zipfile`: Archive extraction

## Best Practices

### Logging
```python
# Always setup logging at the start
from tabembedbench.utils.logging_utils import setup_unified_logging
import logging

log_file = setup_unified_logging(logging_level=logging.INFO)
logger = logging.getLogger("MyModule")
logger.info("Starting process...")
```

### Device Management
```python
# Check device availability and clean up after use
from tabembedbench.utils.torch_utils import get_device, empty_gpu_cache

device = get_device()
# ... run computations ...
empty_gpu_cache(device)
```

### Data Preprocessing
```python
# Infer data types before processing
from tabembedbench.utils.preprocess_utils import infer_categorical_features

cat_features = infer_categorical_features(data)
# Use cat_features for model configuration
```

### Result Management
```python
# Always save results in both formats
from tabembedbench.utils.tracking_utils import save_result_df

save_result_df(results, output_path, benchmark_name, timestamp)
# Creates both .parquet (efficient) and .csv (readable) files
```

### Visualization
```python
# Setup publication style before plotting
from tabembedbench.utils.eda_utils import setup_publication_style

setup_publication_style()
# Now create your plots
```

## Common Patterns

### Benchmark Result Processing
```python
import polars as pl
from tabembedbench.utils.eda_utils import (
    clean_results,
    create_descriptive_dataframe,
    create_outlier_plots
)

# Load results
results = pl.read_parquet("results.parquet")

# Clean inconsistent results
cleaned = clean_results(results)

# Generate statistics
stats = create_descriptive_dataframe(cleaned, "auc_score")

# Create visualizations
create_outlier_plots(cleaned, data_path="output")
```

### Custom Model Naming
```python
from tabembedbench.utils.eda_utils import rename_models, keeping_models

# Define renaming mapping
name_mapping = {
    "TabICLEmbedding": "TabICL",
    "UniversalTabPFNEmbedding": "TabPFN"
}

# Rename and filter
results = rename_models(results, name_mapping)
results = keeping_models(results, ["TabICL", "TabPFN"])
```

## Notes

- All utility functions are designed to work independently
- File paths use `pathlib.Path` for cross-platform compatibility
- GPU operations gracefully handle CPU-only environments
- Logging is thread-safe for parallel execution
- Visualization functions create directories automatically
- All utilities support both numpy arrays and pandas DataFrames where applicable

# Examples Module

This module contains example scripts and demonstrations showing how to use the TabEmbedBench framework for benchmarking tabular embedding models. The examples provide practical implementations and usage patterns for running comprehensive embedding evaluations.

## Overview

The examples module serves as:
- **Getting Started Guide**: Demonstrates basic usage patterns
- **Configuration Examples**: Shows how to set up embedding models and evaluators
- **Benchmark Orchestration**: Illustrates complete benchmark workflows
- **Parameter Tuning**: Examples of hyperparameter sweeps and configurations

## Available Examples

### 1. EuRIPS Run (`eurips_run.py`)

A comprehensive example script that demonstrates a full benchmarking pipeline used for research experiments. This script showcases the complete workflow from model configuration to benchmark execution.

#### Key Functions

##### `get_embedding_models(debug=False)`
Configures and returns a list of embedding models for benchmarking.

**Models Included:**
- **TabPFNEmbedding**: TabPFN model
- **TabICLEmbedding**: TabICL model with preprocessing enabled
- **TabVectorizerEmbedding**: Table vectorization approach

**Debug Mode**: When `debug=True`, returns a reduced set of models for faster testing.

**Returns**: List of configured embedding model instances

##### `get_evaluators(debug=False)`
Configures and returns a comprehensive set of evaluators for different tasks.

**Evaluator Types:**
- **KNNRegressorEvaluator**: For regression tasks
- **KNNClassifierEvaluator**: For classification tasks  
- **LocalOutlierFactorEvaluator**: For outlier detection
- **IsolationForestEvaluator**: Alternative outlier detection

**Parameter Sweeps:**
- **Neighbor counts**: 5-45 neighbors (step size 5)
- **Distance metrics**: Euclidean
- **Weighting schemes**: distance-based
- **Ensemble sizes**: 50-250 estimators for Isolation Forest

**Debug Mode**: When `debug=True`, returns minimal evaluators for quick testing.

**Returns**: List of configured evaluator instances

##### `run_main(debug, max_samples, max_features, run_outlier, run_supervised)`
Orchestrates the complete benchmarking process.

**Parameters:**
- `debug`: Enable debug mode for faster execution
- `max_samples`: Upper bound on dataset size
- `max_features`: Upper bound on number of features
- `run_outlier`: Whether to run outlier detection benchmarks
- `run_supervised`: Whether to run TabArena supervised benchmarks

**Workflow:**
1. Configure embedding models and evaluators
2. Set up logging and constraints
3. Execute benchmark with specified parameters
4. Handle results and logging

##### `main()` - CLI Interface
Command-line interface using Click framework for easy execution.

**CLI Options:**
- `--debug`: Run in debug mode with reduced datasets
- `--max-samples`: Set maximum dataset size (default: 100,001)
- `--max-features`: Set maximum feature count (default: 500)
- `--run-outlier/--no-run-outlier`: Toggle outlier detection (default: True)
- `--run-supervised/--no-run-supervised`: Toggle TabArena tasks (default: True)

## Usage Examples

### Basic Execution
```bash
# Run full benchmark
python eurips_run.py

# Run in debug mode for quick testing
python eurips_run.py --debug

# Run only outlier detection
python eurips_run.py --no-run-supervised

# Limit dataset size and features
python eurips_run.py --max-samples 1000 --max-features 50
```

### Programmatic Usage
```python
from tabembedbench.examples.eurips_run import get_embedding_models, get_evaluators, run_main

# Get configured models and evaluators
models = get_embedding_models(debug=False)
evaluators = get_evaluators(debug=False)

# Run benchmark programmatically
run_main(
    debug=False,
    max_samples=10000,
    max_features=100,
    run_outlier=True,
    run_supervised=True
)
```

## Integration with Benchmarking Framework

### Model Configuration Patterns
The examples demonstrate best practices for:
- **Model Naming**: Consistent naming conventions for tracking results
- **Parameter Sweeps**: Systematic exploration of hyperparameter spaces
- **Debug Modes**: Reduced configurations for development and testing
- **Resource Management**: Handling memory and computational constraints

### Evaluator Setup Patterns
Examples show how to:
- **Create Evaluator Grids**: Systematic parameter combinations
- **supervised Configuration**: Different evaluators for different tasks
- **Performance Optimization**: Balancing thoroughness with execution time
- **Metric Selection**: Appropriate metrics for each evaluation type

### Benchmark Orchestration
The examples illustrate:
- **Multi-Task Evaluation**: Running both outlier detection and TabArena tasks
- **Result Management**: Handling and storing benchmark results
- **Logging Configuration**: Setting up comprehensive logging
- **Error Handling**: Robust execution with proper error management

## Configuration Best Practices

### Model Selection
- **Diverse Architectures**: Include different embedding approaches
- **Parameter Ranges**: Test multiple hyperparameter settings
- **Baseline Models**: Include simple baselines for comparison
- **Computational Balance**: Consider training time vs. performance trade-offs

### Evaluator Configuration
- **Comprehensive Coverage**: Test multiple neighbor counts and metrics
- **Task Alignment**: Match evaluators to intended use cases
- **Statistical Robustness**: Use sufficient parameter combinations
- **Computational Efficiency**: Balance thoroughness with execution time

### Execution Settings
- **Dataset Constraints**: Set appropriate size limits for available resources
- **Logging Levels**: Configure appropriate verbosity for debugging
- **Output Management**: Organize results for analysis
- **Reproducibility**: Use consistent random seeds and configurations

## Extending the Examples

### Adding New Models
```python
def get_embedding_models(debug=False):
    models = []
    
    # Add your custom model
    from my_models import MyCustomEmbedding
    custom_model = MyCustomEmbedding(param1=value1)
    custom_model.name = "my-custom-model"
    models.append(custom_model)
    
    # Include existing models
    models.extend(get_existing_models())
    return models
```

### Custom Evaluator Configurations
```python
def get_custom_evaluators():
    evaluators = []
    
    # Custom parameter ranges
    for neighbors in [3, 7, 15, 31]:
        for metric in ["euclidean", "manhattan", "cosine"]:
            evaluators.append(
                KNNClassifierEvaluator(
                    num_neighbors=neighbors,
                    metric=metric,
                    weights="distance"
                )
            )
    
    return evaluators
```

### Specialized Benchmark Runs
```python
def run_specialized_benchmark():
    # Focus on specific model types
    models = [get_neural_models(), get_traditional_models()]
    
    # Supervised evaluators
    evaluators = get_classification_evaluators()
    
    # Custom constraints
    results = run_benchmark(
        embedding_models=models,
        evaluator_algorithms=evaluators,
        include_only_datasets=["specific_dataset.npz"],
        custom_preprocessing=True
    )
```

## Dependencies

The examples module requires:
- **Click**: For command-line interface functionality
- **Logging**: For execution tracking and debugging
- **TabEmbedBench Core**: All benchmark and embedding modules
- **NumPy/Pandas**: For data handling (inherited from core modules)

## Notes

- Examples are designed to be both educational and production-ready
- Debug modes provide quick feedback during development
- CLI interfaces enable easy integration with experiment management systems
- Configuration patterns can be adapted for different research needs
- All examples follow the same architectural patterns as the core framework

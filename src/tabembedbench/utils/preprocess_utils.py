from typing import Union

import numpy as np
import pandas as pd
import torch


def infer_categorical_features(
    data: np.ndarray,
    categorical_features: list[int] | None = None,
) -> list[int]:
    """Infer the categorical features from the input data.

    Features are identified as categorical if any of these conditions are met:
    1. The feature index is in the provided categorical_features list AND has few unique values
    2. The feature has few unique values compared to the dataset size
    3. The feature has string/object/category data type (pandas DataFrame)
    4. The feature contains string values (numpy array)

    Parameters:
        data (np.ndarray or pandas.DataFrame): The input data.
        categorical_features (list[int], optional): Initial list of categorical feature indices.
            If None, will start with an empty list.

    Returns:
        list[int]: The indices of the categorical features.
    """
    if categorical_features is None:
        categorical_features = []

    max_unique_values_as_categorical_feature = 10
    min_unique_values_as_numerical_feature = 10

    _categorical_features: list[int] = []

    # First detect based on data type (string/object features)
    is_pandas = hasattr(data, "dtypes")

    if is_pandas:
        # Handle pandas DataFrame - use pandas' own type detection
        import pandas as pd

        for i, col_name in enumerate(data.columns):
            col = data[col_name]
            # Use pandas' built-in type checks for categorical features
            if (
                pd.api.types.is_categorical_dtype(col)
                or pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)
            ):
                _categorical_features.append(i)
    else:
        # Handle numpy array - check if any columns contain strings
        for i in range(data.shape[1]):
            if data.dtype == object:  # Check entire array dtype
                # Try to access first non-nan value to check its type
                col = data[:, i]
                for val in col:
                    if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        if isinstance(val, str):
                            _categorical_features.append(i)
                            break

    # Then detect based on unique values
    for i in range(data.shape[-1]):
        # Skip if already identified as categorical
        if i in _categorical_features:
            continue

        # Get unique values - handle differently for pandas and numpy
        n_unique = data.iloc[:, i].nunique() if is_pandas else len(np.unique(data[:, i]))

        # Filter categorical features, with too many unique values
        if (
            i in categorical_features
            and n_unique <= max_unique_values_as_categorical_feature
        ) or (
                i not in categorical_features
                and n_unique < min_unique_values_as_numerical_feature
                and data.shape[0] > 100
        ):
            _categorical_features.append(i)

    return _categorical_features


def infer_categorical_columns(
    data: np.ndarray | torch.Tensor | pd.DataFrame,
    max_unique_ratio: float = 0.1,
    max_unique_count: int = 200,
    return_split: bool = False,
) -> list[int] | tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    if hasattr(data, "detach"):
        is_tensor = True
        device = data.device
        dtype = data.dtype
        data_np = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        is_tensor = False
        data_np = data
    elif isinstance(data, pd.DataFrame):
        is_tensor = False
        data_np = data.to_numpy()
    else:
        raise ValueError("Input must be numpy array or torch tensor")

    if data_np.ndim not in [2, 3]:
        raise ValueError(f"Input must be 2D or 3D array, got {data_np.ndim}D array.")

    original_shape = data_np.shape

    if data_np.ndim == 3:
        batch_size, num_samples, num_features = data_np.shape
        data_2d = data_np.reshape(-1, num_features)
    else:
        num_samples, num_features = data_np.shape
        data_2d = data_np

    categorical_indices = []

    # Analyze each feature column
    for col_idx in range(num_features):
        column = data_2d[:, col_idx]

        # Skip columns with NaN values for simplicity
        valid_mask = ~np.isnan(column)
        if not np.any(valid_mask):
            continue

        valid_column = column[valid_mask]
        n_valid = len(valid_column)

        # Count unique values
        unique_values = np.unique(valid_column)
        n_unique = len(unique_values)

        # Apply categorical detection criteria
        unique_ratio = n_unique / n_valid if n_valid > 0 else 1.0

        is_categorical = (
            n_unique <= max_unique_count and unique_ratio <= max_unique_ratio
        )

        if is_categorical:
            categorical_indices.append(col_idx)

    # Return based on requested format
    if not return_split:
        return categorical_indices

    # Split into numerical and categorical arrays
    numerical_indices = [i for i in range(num_features) if i not in categorical_indices]

    if len(categorical_indices) == 0:
        # No categorical columns found
        categorical_data = _create_empty_array(
            original_shape,
            is_tensor,
            device if is_tensor else None,
            dtype if is_tensor else data_np.dtype,
        )
        numerical_data = _restore_original_format(
            data_np, original_shape, is_tensor, device if is_tensor else None
        )
    elif len(numerical_indices) == 0:
        # All columns are categorical
        numerical_data = _create_empty_array(
            original_shape,
            is_tensor,
            device if is_tensor else None,
            dtype if is_tensor else data_np.dtype,
        )
        categorical_data = _restore_original_format(
            data_np, original_shape, is_tensor, device if is_tensor else None
        )
    else:
        # Split the data
        if data_np.ndim == 3:
            numerical_data = data_np[:, :, numerical_indices]
            categorical_data = data_np[:, :, categorical_indices]
        else:
            numerical_data = data_np[:, numerical_indices]
            categorical_data = data_np[:, categorical_indices]

        # Convert back to original format if needed
        if is_tensor:
            numerical_data = torch.from_numpy(numerical_data).to(
                device=device, dtype=dtype
            )
            categorical_data = torch.from_numpy(categorical_data).to(
                device=device, dtype=dtype
            )

    return numerical_data, categorical_data


def _create_empty_array(
    original_shape: tuple[int, ...], is_tensor: bool, device: str | None, dtype
) -> Union[np.ndarray, "torch.Tensor"]:
    """Create an empty array with 0 features but maintaining other dimensions."""
    if len(original_shape) == 3:
        empty_shape = (original_shape[0], original_shape[1], 0)
    else:
        empty_shape = (original_shape[0], 0)

    if is_tensor:
        return torch.empty(empty_shape, device=device, dtype=dtype)
    return np.empty(empty_shape, dtype=dtype)


def _restore_original_format(
    data_np: np.ndarray,
    original_shape: tuple[int, ...],
    is_tensor: bool,
    device: str | None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Restore data to original tensor format if needed."""
    if is_tensor:
        return torch.from_numpy(data_np.reshape(original_shape)).to(device)
    return data_np.reshape(original_shape)

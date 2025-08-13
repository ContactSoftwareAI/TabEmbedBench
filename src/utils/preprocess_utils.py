import numpy as np
import pandas as pd
import torch

from typing import Union, List, Tuple, Optional


def infer_categorical_columns(
    X: Union[np.ndarray, torch.Tensor],
    max_unique_ratio: float = 0.1,
    max_unique_count: int = 200,
    return_split: bool = False,
) -> Union[
    List[int], Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
]:
    if hasattr(X, "detach"):
        is_tensor = True
        device = X.device
        dtype = X.dtype
        data_np = X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        is_tensor = False
        data_np = X
    elif isinstance(X, pd.DataFrame):
        is_tensor = False
        data_np = X.to_numpy()
    else:
        raise ValueError("Input must be numpy array or torch tensor")

    if data_np.ndim not in [2, 3]:
        raise ValueError(f"Input must be 2D or 3D array, got {data_np.ndim}D array.")

    original_shape = data_np.shape

    if data_np.ndim == 3:
        batch_size, num_samples, num_features = data_np.shape
        data_2d = data_np.reshape(-1, num_features)
        total_samples = batch_size * num_samples
    else:
        num_samples, num_features = data_np.shape
        data_2d = data_np
        total_samples = num_samples
        batch_size = None

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
    original_shape: Tuple[int, ...], is_tensor: bool, device: Optional[str], dtype
) -> Union[np.ndarray, "torch.Tensor"]:
    """Create an empty array with 0 features but maintaining other dimensions."""
    if len(original_shape) == 3:
        empty_shape = (original_shape[0], original_shape[1], 0)
    else:
        empty_shape = (original_shape[0], 0)

    if is_tensor:
        return torch.empty(empty_shape, device=device, dtype=dtype)
    else:
        return np.empty(empty_shape, dtype=dtype)


def _restore_original_format(
    data_np: np.ndarray,
    original_shape: Tuple[int, ...],
    is_tensor: bool,
    device: Optional[str],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Restore data to original tensor format if needed."""
    if is_tensor:
        return torch.from_numpy(data_np.reshape(original_shape)).to(device)
    return data_np.reshape(original_shape)

import numpy as np
import torch

from config import EmbAggregation
from typing import Union


def embeddings_aggregation(
    embeddings: Union[list[np.ndarray], list[torch.Tensor]],
    agg_func: EmbAggregation,
    axis: int = 0,
    quantile: float = 0.75,
) -> Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]]:
    if not isinstance(agg_func, EmbAggregation):
        raise ValueError("agg_func must be an instance of EmbAggregation.")

    list_type, list_shape = validate_input(embeddings)

    if list_type == np.ndarray:
        if agg_func == EmbAggregation.MEAN:
            return np.mean(embeddings, axis=0)
        elif agg_func == EmbAggregation.CONCAT:
            return np.concatenate(embeddings, axis=-1)
        elif agg_func == EmbAggregation.PERCENTILE:
            return np.quantile(embeddings, q=quantile, axis=0)
        elif agg_func == EmbAggregation.COLUMN:
            return embeddings
    elif list_type == torch.Tensor:
        if agg_func == EmbAggregation.MEAN:
            return torch.mean(torch.stack(embeddings), dim=0)
        elif agg_func == EmbAggregation.CONCAT:
            return torch.flatten(embeddings, start_dim=-1)
        elif agg_func == EmbAggregation.PERCENTILE:
            return torch.quantile(torch.stack(embeddings), q=quantile, dim=0)
        elif agg_func == EmbAggregation.COLUMN:
            return embeddings
    else:
        raise ValueError("embeddings must be a list of numpy arrays or torch tensors.")


def check_emb_aggregation(agg_func: Union[EmbAggregation, str]):
    """
    Validates and converts the input aggregation function to an EmbAggregation instance if
    it is provided as a string. Also ensures the input is a valid aggregation method.

    Args:
        agg_func (Union[EmbAggregation, str]): The aggregation function to check and validate.
            It can be either an instance of EmbAggregation or a string representing
            a valid aggregation method.

    Returns:
        EmbAggregation: The validated and converted aggregation method.

    Raises:
        ValueError: If agg_func is a string and not a valid aggregation method.
    """
    if isinstance(agg_func, str):
        try:
            agg_func = EmbAggregation(agg_func)
        except ValueError:
            valid_values = [e.value for e in EmbAggregation]
            raise ValueError(
                f"Invalid aggregation method: {agg_func}. "
                f"Valid options are: {valid_values}"
            )
    else:
        agg_func = agg_func

    return agg_func


def validate_input(input_list):
    """
    Determines the data type of elements within a list if all elements are of the
    same type.

    This function inspects all elements in the input list, checks if they belong
    to the same data type, and returns that data type if the condition is met.
    If the elements belong to different types, it raises a ValueError to indicate
    the inconsistency.

    Args:
        input_list (list): The list of items whose data type needs to be
            evaluated.

    Returns:
        type: The common data type of all elements in the input list if they are
            of the same type.

    Raises:
        ValueError: If the elements in the input list are not of the same
            data type.
    """
    first_type = type(input_list[0])
    first_shape = input_list[0].shape
    if all(isinstance(item, first_type) for item in input_list) and all(item.shape == first_shape for item in input_list):
        return first_type, first_shape
    else:
        raise ValueError(
            "All elements in the input list must be of the same type and have the same shape."
        )
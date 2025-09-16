
import numpy as np
import torch

from tabembedbench.utils.config import EmbAggregation


def compute_embeddings_aggregation(
    embeddings: list[np.ndarray] | list[torch.Tensor], agg_func: str = "mean"
):
    """Aggregates a list of embeddings using the specified aggregation function.

    This function takes a list of embeddings, which can be NumPy arrays or PyTorch tensors,
    and applies the specified aggregation function to them. The default aggregation function
    is "mean". The function ensures the desired aggregation method is valid before
    performing the aggregation.

    Args:
        embeddings (Union[list[np.ndarray], list[torch.Tensor]]): A list of embeddings to be aggregated.
        agg_func (str, optional): The aggregation function to use. Defaults to "mean".

    Returns:
        np.ndarray or torch.Tensor: The aggregated embedding, depending on the type of
        input embeddings.
    """
    agg_func = check_emb_aggregation(agg_func)
    return embeddings_aggregation(embeddings, agg_func)


def embeddings_aggregation(
    embeddings: list[np.ndarray] | list[torch.Tensor],
    agg_func: EmbAggregation,
    axis: int = 0,
    quantile: float = 0.75,
) -> np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor]:
    """Aggregates embeddings using the specified aggregation function. The function supports various
    aggregation methods such as mean, concatenation, percentile, and column-wise operations.

    Args:
        embeddings: Union[list[np.ndarray], list[torch.Tensor]]. A list of numpy arrays or torch tensors
            representing embeddings to be aggregated.
        agg_func: EmbAggregation. The aggregation function to apply to the embeddings. Must be an
            instance of the EmbAggregation enumeration.
        axis: int, optional. The axis along which to perform aggregation. Defaults to 0.
        quantile: float, optional. The quantile used when the aggregation function is set to
            `EmbAggregation.PERCENTILE`. Defaults to 0.75.

    Raises:
        ValueError: If `agg_func` is not an instance of EmbAggregation.
        ValueError: If `embeddings` is not a list of numpy arrays or torch tensors.

    Returns:
        Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]]. The aggregated embeddings.
        The type and shape depend on the input and the specified aggregation function.
    """
    if not isinstance(agg_func, EmbAggregation):
        raise ValueError("agg_func must be an instance of EmbAggregation.")

    list_type, list_shape = validate_input(embeddings)

    if list_type == np.ndarray:
        stacked = np.stack(embeddings)

        if agg_func == EmbAggregation.MEAN:
            return np.mean(stacked, axis=0)
        if agg_func == EmbAggregation.CONCAT:
            return np.concatenate(embeddings, axis=0)
        if agg_func == EmbAggregation.PERCENTILE:
            return np.quantile(np.stack(embeddings), q=quantile, axis=0)
        if agg_func == EmbAggregation.COLUMN:
            return embeddings
    elif list_type == torch.Tensor:
        if agg_func == EmbAggregation.MEAN:
            return torch.mean(torch.stack(embeddings), dim=0)
        if agg_func == EmbAggregation.CONCAT:
            return torch.flatten(embeddings, start_dim=-1)
        if agg_func == EmbAggregation.PERCENTILE:
            return torch.quantile(torch.stack(embeddings), q=quantile, dim=0)
        if agg_func == EmbAggregation.COLUMN:
            return embeddings
    else:
        raise ValueError("embeddings must be a list of numpy arrays or torch tensors.")


def check_emb_aggregation(agg_func: EmbAggregation | str):
    """Validates and converts the input aggregation function to an EmbAggregation instance if
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
    """Determines the data type of elements within a list if all elements are of the
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
    if all(isinstance(item, first_type) for item in input_list) and all(
        item.shape == first_shape for item in input_list
    ):
        return first_type, first_shape
    raise ValueError(
        "All elements in the input list must be of the same type and have the same shape."
    )


def check_nan(
    embeddings: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
):
    """Checks if any of the embeddings contain NaN values."""
    if isinstance(embeddings, torch.Tensor):
        return torch.isnan(embeddings).any()
    if isinstance(embeddings, np.ndarray):
        return np.isnan(embeddings).any()
    if isinstance(embeddings[0], torch.Tensor):
        return all([torch.isnan(embedding).any() for embedding in embeddings])
    return all([np.isnan(embedding).any() for embedding in embeddings])

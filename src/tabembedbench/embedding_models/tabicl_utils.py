import inspect
from typing import List, Optional, Union

import numpy as np
from tabicl.model.embedding import ColEmbedding
from tabicl.model.inference_config import InferenceConfig
from tabicl.model.interaction import RowInteraction
import torch
from torch import nn

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device


class TabICLEmbedding(nn.Module, BaseEmbeddingGenerator):
    """
    TabICLEmbedding is a neural network module for tabular data embedding. It is
    based on the TabICL architecture and uses the first two stages of TabICL to
    generate embeddings for the rows.

    This class combines the column embedding and row interaction modules into a
    single model for generating row embeddings. It uses the inference forward method
    from TabICL to generate row representations from the input data. It is not intended
    for training. It is designed to easily load the state dictionary and config
    from a trained TabICL model.

    Attributes:
        col_embedder (ColEmbedding): Module responsible for creating embeddings
            for columns in the input data.
        row_interactor (RowInteraction): Module responsible for processing the
            column embeddings and generating the final row-wise representations.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: Union[str, callable] = "gelu",
        norm_first: bool = True,
        **kwargs,
    ):
        """
        Initializes the model configuration by setting up the column embedding and row
        interaction modules with the specified parameters. Configurations are used to
        define the behavior of each module and the overall model.

        Args:
            embed_dim (int): Embedding dimension for both column embedding and row
                interaction modules.
            col_num_blocks (int): Number of blocks in the column embedding module.
            col_nhead (int): Number of attention heads in the column embedding module.
            col_num_inds (int): Number of induced tokens for column embedding.
            row_num_blocks (int): Number of blocks in the row interaction module.
            row_nhead (int): Number of attention heads in the row interaction module.
            row_num_cls (int): Number of classification tokens in the row interaction
                module.
            row_rope_base (float): Base value used for rotary position embedding
                (RoPE) in the row interaction module.
            ff_factor (int): Feedforward dimension factor used to scale the size of the
                intermediate layer in both modules.
            dropout (float): Dropout probability applied within the modules to prevent
                overfitting.
            activation (Union[str, callable]): Activation function used in feedforward
                layers of the modules. Can be a string or a callable function.
            norm_first (bool): If True, applies layer normalization before other
                operations in each layer.
        """
        super().__init__()

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        for param in self.col_embedder.parameters():
            param.requires_grad = False

        for param in self.row_interactor.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.eval()

    def forward(
        self,
        X: torch.Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass through the model, generating row representations
        from the input data. This method applies column embedding followed by row
        interaction to derive the final tensor output.

        Args:
            X (torch.Tensor): The input tensor for the model.
            feature_shuffles (Optional[List[List[int]]]): Optional feature shuffling
                configurations to modify input feature representation.
            inference_config (Optional[InferenceConfig]): Configuration for inference
                detailing how the column and row representations should be generated.

        Returns:
            torch.Tensor: The computed tensor of row representations after applying the
                prescribed transformations.
        """
        if inference_config is None:
            inference_config = InferenceConfig()

        column_representations = self.col_embedder(
            X, feature_shuffles=feature_shuffles, mgr_config=inference_config.COL_CONFIG
        )

        row_representations = self.row_interactor(
            column_representations, mgr_config=inference_config.ROW_CONFIG
        )
        return row_representations

    def _get_default_name(self) -> str:
        return "TabICL"

    def compute_embeddings(
        self, X: np.ndarray, device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Computes the embeddings for the given input array using the model's forward method.

        The method takes an input array and uses PyTorch to convert it into a tensor, passes
        it through the forward method of the model, and finally detaches and converts the
        result back to a NumPy array.

        Args:
            X (np.ndarray): The input array for which embeddings are to be computed. The array
                has to have the shape (num_datasets, num_samples, num_features) or
                (num_samples, num_features). The first dimension is optional.
            device (Optional[torch.device]): The device to use for computation. If None, uses
                the `get_default` method to get most suited device.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.

        Raises:
            ValueError: If the input array is not 2D or 3D.
        """
        if device is None:
            device = get_device()
            self.to(device)

        if len(X.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")

        X = torch.from_numpy(X).float().to(device)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)

        embeddings = self.forward(X).cpu().squeeze().numpy()
        return embeddings


def filter_params_for_class(cls, params_dict):
    """
    Filter parameters dictionary to only include parameters that are accepted
    by the class constructor.

    Args:
        cls: The class whose constructor signature to check
        params_dict: Dictionary of all parameters

    Returns:
        dict: Filtered dictionary with only valid parameters for the class
    """
    # Get the constructor signature
    sig = inspect.signature(cls.__init__)

    # Get parameter names (excluding 'self')
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Filter the parameters dictionary
    filtered_params = {k: v for k, v in params_dict.items() if k in valid_params}

    return filtered_params


def get_row_embeddings_model(state_dict: dict, config: dict):
    """
    Loads a row embedding model using the provided state dictionary and configuration,
    ensuring compatibility by filtering the configuration for the appropriate class attributes.

    Args:
        state_dict (dict): A dictionary containing model parameters to load into the model.
        config (dict): A configuration dictionary including parameters for initializing the
            model. Only parameters that match the expected class attributes will be used.

    Returns:
        TabICLEmbedding: An instance of TabICLEmbedding initialized with the filtered
            configuration and loaded with the provided state dictionary.
    """
    filtered_config = filter_params_for_class(TabICLEmbedding, config)
    row_embedding_model = TabICLEmbedding(**filtered_config)

    row_embedding_model.load_state_dict(state_dict, strict=False)
    return row_embedding_model

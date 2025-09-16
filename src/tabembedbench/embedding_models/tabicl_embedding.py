import inspect

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabicl.model.embedding import ColEmbedding
from tabicl.model.inference_config import InferenceConfig
from tabicl.model.interaction import RowInteraction
from tabicl.sklearn.preprocessing import PreprocessingPipeline
from torch import nn
from typing import Union

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device


class TabICLEmbedding(nn.Module, BaseEmbeddingGenerator):
    """TabICLEmbedding is a neural network module for tabular data embedding. It is
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
        normalize_embeddings: bool = False,
        preprocess_data: bool = False,
        **kwargs,
    ):
        """Initializes the model configuration by setting up the column embedding and row
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
            normalize_embeddings (bool):

        References:
            [1] Qu, J. et al. (2025). Tabicl: A tabular foundation model for in-context learning on large data.
                arXiv preprint arXiv:2502.05564.
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

        self.normalize_embeddings = normalize_embeddings
        self._preprocess_data = preprocess_data
        self.preprocess_pipeline = PreprocessingPipeline()
        self.eval()

    def forward(
        self,
        X: torch.Tensor,
        feature_shuffles: list[list[int]] | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> torch.Tensor:
        """Performs the forward pass through the model, generating row representations
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

    @property
    def task_only(self) -> bool:
        return False

    def preprocess_data(self, X: np.ndarray, train: bool = True):
        """Preprocesses the input data by applying normalization and preprocessing pipelines
        based on the mode (training or inference). If normalization is enabled, the
        numerical transformation is applied to the data during training, creating a
        reusable transformer. During inference, the existing transformer is reused.
        Additionally, if preprocessing is enabled, a preprocessing pipeline is applied
        to the data.

        Args:
            X (np.ndarray): Input data array to be preprocessed.
            train (bool): A flag indicating whether the data is used in training mode
                (default: True).

        Returns:
            np.ndarray: The preprocessed data.
        """
        X_preprocess = X

        if train and self._preprocess_data:
            X_preprocess = self.preprocess_pipeline.fit_transform(X_preprocess)
        else:
            if self._preprocess_data:
                X_preprocess = self.preprocess_pipeline.transform(X_preprocess)

        return X_preprocess

    def compute_embeddings(
        self,
        X: np.ndarray,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Computes the embeddings for the given input array using the model's forward method.

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

    def reset_embedding_model(self):
        self.preprocess_pipeline = PreprocessingPipeline()


def filter_params_for_class(cls, params_dict):
    """Filter parameters dictionary to only include parameters that are accepted
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


def get_row_embeddings_model(
    model_path: str | None = "auto",
    state_dict: dict | None = None,
    config: dict | None = None,
    preprocess_data: bool = False,
) -> TabICLEmbedding:
    """Loads and prepares a row embeddings model based on the provided model path, state dict, and
    configuration. The function can also optionally preprocess data for embeddings.

    The function supports two methods of loading the model:
    1. By specifying a model_path to a pre-trained Torch checkpoint, which provides the state_dict
       and config. Can also infer the model from a default hub repository if "auto" is given.
    2. By directly passing the state_dict and config. This allows using a custom or pre-loaded
       state_dict and config.

    In addition, the function supports preprocessing row-based data embeddings using a specified
    normalization technique if needed.

    Args:
        model_path (Optional[str]): Path to the pre-trained model file. If "auto", the function will
            download a preconfigured model checkpoint from a defined repository.
        state_dict (Optional[dict]): Pre-trained PyTorch model weights (state dict) if not using
            a model path. Must be specified along with the config if model_path is not provided.
        config (Optional[dict]): Configuration dictionary required to initialize the model. Must
            be specified along with the state_dict if model_path is not provided.
        preprocess_data (bool): Boolean flag indicating whether to preprocess the input data
            (normalize embeddings). Defaults to False.

    Returns:
        TabICLEmbedding: Instance of the TabICLEmbedding class initialized with the provided parameters
        and model weights.

    Raises:
        ValueError: Raised if neither model_path nor both state_dict and config are provided.
    """
    if model_path == "auto":
        model_ckpt_path = hf_hub_download(
            repo_id="jingang/TabICL-clf",
            filename="tabicl-classifier-v1.1-0506.ckpt",
        )

        model_ckpt = torch.load(model_ckpt_path)

        state_dict = model_ckpt["state_dict"]
        config = model_ckpt["config"]
    elif model_path is not None:
        state_dict = torch.load(model_path)["state_dict"]
        config = torch.load(model_path)["config"]
    else:
        if state_dict is None or config is None:
            raise ValueError(
                "Either model_path or state_dict and config must be provided"
            )

    filtered_config = filter_params_for_class(TabICLEmbedding, config)

    row_embedding_model = TabICLEmbedding(
        normalize_embeddings=preprocess_data,
        preprocess_data=preprocess_data,
        **filtered_config,
    )

    row_embedding_model.load_state_dict(state_dict, strict=False)

    return row_embedding_model

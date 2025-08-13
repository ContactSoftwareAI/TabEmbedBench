import torch

from utils.config import EmbAggregation
from tabicl.model.embedding import ColEmbedding
from tabicl.model.interaction import RowInteraction


class DataSetEmbeddings(ColEmbedding):
    """
    Provides embeddings for datasets with various aggregation methods.

    This class inherits from ColEmbedding and is used to generate embeddings for dataset
    columns with adjustable settings for aggregation methods, transformer blocks, and other
    embedding parameters. It includes functionality for aggregation using mean, concatenation,
    or a specified percentile and ensures its parameters are not trainable by default. This
    class is evaluated in inference mode. It requires the weights from the tabicl model to be
    loaded before inference.

    Attributes:
        percentile (float): Percentile value to use for aggregation when the aggregation
            method is set to `ColEmbAggregation.PERCENTILE`.
        col_emb_aggregation (ColEmbAggregation): Specifies the method to aggregate column
            embeddings. Options include MEAN, CONCAT, and PERCENTILE.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        reserve_cls_tokens: int = 4,
        col_emb_aggregation: EmbAggregation = EmbAggregation.MEAN,
        percentile: float = 0.75,
    ):
        super().__init__(
            embed_dim,
            num_blocks,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=reserve_cls_tokens,
        )

        self.percentile = percentile

        if isinstance(col_emb_aggregation, str):
            try:
                self.col_emb_aggregation = EmbAggregation(col_emb_aggregation)
            except ValueError:
                valid_values = [e.value for e in EmbAggregation]
                raise ValueError(
                    f"Invalid aggregation method: {col_emb_aggregation}. "
                    f"Valid options are: {valid_values}"
                )
        else:
            self.col_emb_aggregation = col_emb_aggregation

        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a forward pass for input embeddings based on the specified column
        embedding aggregation method. The method processes the input tensor and outputs
        a tensor based on one of the aggregation strategies: mean, concatenation,
        percentile, or retaining the original column embeddings.

        If the aggregation method is set to COLUMN, the output shape will be (B, C, D), where
            - B is the batch size,
            - C is the number of columns,
            - D is the embedding dimension.
        Otherwise, the output shape will be (B, D).

        Args:
            X: torch.Tensor
                The input tensor for the forward pass. It has the shape (B, N, C), where
                    - B is the batch size,
                    - N is the number of samples per datasets,
                    - C is the number of columns/features.
            **kwargs: dict
                Additional arguments passed to the method.

        Returns:
            torch.Tensor:
                Processed tensor based on the specified column embedding aggregation
                method.
        """
        X = super().forward(X)

        if self.col_emb_aggregation == EmbAggregation.MEAN:
            return torch.mean(X, dim=1)
        elif self.col_emb_aggregation == EmbAggregation.CONCAT:
            return torch.flatten(X, start_dim=1)
        elif self.col_emb_aggregation == EmbAggregation.PERCENTILE:
            return torch.quantile(X, q=self.percentile, dim=1)
        elif self.col_emb_aggregation == EmbAggregation.COLUMN:
            return X


def get_col_embedding(state_dict, config) -> ColEmbedding:
    """
    Constructs and initializes a column embedding model using the provided state dictionary and configuration.

    This function extracts the relevant state dictionary for the column embedding model by filtering
    keys that contain "col_embedder". Subsequently, it initializes a `ColEmbedding` model using
    the provided configuration parameters and loads the filtered state dictionary into the model.

    Args:
        state_dict (dict): A dictionary containing model state, which includes weights and parameters for
            all model components.
        config (dict): A dictionary containing configuration parameters required to initialize
            the `ColEmbedding` model. Expected keys include:
            - embed_dim (int): Dimension size for embeddings.
            - col_num_blocks (int): Number of blocks in the column embedding model.
            - col_nhead (int): Number of heads for the multi-head attention.
            - col_num_inds (int): Number of inducing points for attention mechanism.
            - ff_factor (int): Multiplicative factor for feedforward network dimension.
            - dropout (float): Dropout rate.
            - norm_first (bool): Whether to apply normalization before operations in blocks.
            - activation (str): Type of activation function to use.
            - row_num_cls (int): Reserved number of CLS token embeddings in a row.

    Returns:
        ColEmbedding: The initialized column embedding model with the state dictionary loaded.
    """
    col_emb_state_dict = {
        key.replace("col_embedder.", ""): item
        for key, item in state_dict.items()
        if "col_embedder" in key
    }

    col_emb_model = ColEmbedding(
        embed_dim=config["embed_dim"],
        num_blocks=config["col_num_blocks"],
        nhead=config["col_nhead"],
        num_inds=config["col_num_inds"],
        dim_feedforward=config["embed_dim"] * config["ff_factor"],
        dropout=config["dropout"],
        norm_first=config["norm_first"],
        activation=config["activation"],
        reserve_cls_tokens=config["row_num_cls"],
    )

    col_emb_model.load_state_dict(col_emb_state_dict)

    return col_emb_model


def get_row_interaction(state_dict, config) -> RowInteraction:
    """
    Processes the state dictionary and configuration to initialize a RowInteraction object
    and loads the appropriate state into it.

    This function creates a subset of the `state_dict` specific to the `row_interactor`
    by filtering keys that match the expected naming convention. It then initializes
    the `RowInteraction` object using configuration values and loads the state from the
    filtered dictionary.

    Args:
        state_dict (dict): The model's state dictionary containing all parameters
                           and their respective weights.
        config (dict): Configuration dictionary with required key-value pairs for
                       initializing the `RowInteraction` object.

    Returns:
        RowInteraction: An initialized and state-loaded `RowInteraction` object.
    """
    row_int_state_dict = {
        key.replace("row_interactor.", ""): item
        for key, item in state_dict.items()
        if "row_interactor" in key
    }

    row_interactor = RowInteraction(
        embed_dim=config["embed_dim"],
        num_blocks=config["row_num_blocks"],
        nhead=config["row_nhead"],
        num_cls=config["row_num_cls"],
        rope_base=config["row_rope_base"],
        dim_feedforward=config["embed_dim"] * config["ff_factor"],
        dropout=config["dropout"],
        norm_first=config["norm_first"],
        activation=config["activation"],
    )

    row_interactor.load_state_dict(row_int_state_dict)

    return row_interactor


def combine_col_embedder_row_interactor(
    col_emb_model: ColEmbedding, row_interactor: RowInteraction
):
    """
    Constructs and returns a sequential model consisting of a column embedding model
    and a row interaction model. The function combines the two components into a
    torch.nn.Sequential container, enabling sequential processing of inputs through
    the column embedding model followed by the row interaction model.

    Args:
        col_emb_model: A column embedding model that transforms input data into
            column-wise embeddings.
        row_interactor: A row interaction model that operates on the output from
            the column embedding model to generate final row-wise embeddings.

    Returns:
        torch.nn.Sequential: A sequential container combining the column embedding
            model and the row interaction model.
    """
    return torch.nn.Sequential(col_emb_model, row_interactor)


def get_row_embeddings_model(state_dict: dict, config: dict):
    """
    Builds and returns the row embeddings model by combining the column embeddings
    model and row interaction model. This function uses given state dictionary and
    configuration to initialize the column embedder and row interaction models,
    then combines them to create the final model.

    Args:
        state_dict (dict): State dictionary containing model parameters.
        config (dict): Configuration dictionary containing model settings.

    Returns:
        object: The combined row embeddings model.
    """
    col_emb_model = get_col_embedding(state_dict, config)
    row_interactor = get_row_interaction(state_dict, config)
    return combine_col_embedder_row_interactor(col_emb_model, row_interactor)


def get_dataset_embeddings(
    state_dict: dict,
    config: dict,
    col_emb_aggregation: EmbAggregation = EmbAggregation.MEAN,
    percentile: float = 0.75,
) -> DataSetEmbeddings:
    """
    Constructs and initializes a DataSetEmbeddings model using the provided state dictionary and configuration.

    Args:
        state_dict (dict): A dictionary containing model state, which includes weights and parameters for
            all model components.
        config (dict): A dictionary containing configuration parameters required to initialize
            the DataSetEmbeddings model.
        col_emb_aggregation (ColEmbAggregation): The aggregation method to use.
        percentile (float): Percentile value for PERCENTILE aggregation method.

    Returns:
        DataSetEmbeddings: The initialized model with the state dictionary loaded and training disabled.
    """
    col_emb_state_dict = {
        key.replace("col_embedder.", ""): item
        for key, item in state_dict.items()
        if "col_embedder" in key
    }

    dataset_emb_model = DataSetEmbeddings(
        embed_dim=config["embed_dim"],
        num_blocks=config["col_num_blocks"],
        nhead=config["col_nhead"],
        num_inds=config["col_num_inds"],
        dim_feedforward=config["embed_dim"] * config["ff_factor"],
        dropout=config["dropout"],
        norm_first=config["norm_first"],
        activation=config["activation"],
        reserve_cls_tokens=config["row_num_cls"],
        col_emb_aggregation=col_emb_aggregation,
        percentile=percentile,
    )

    dataset_emb_model.load_state_dict(col_emb_state_dict)

    return dataset_emb_model


def prepare_dataset(data):
    pass

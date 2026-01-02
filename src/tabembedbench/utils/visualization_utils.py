from pathlib import Path
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import umap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE


def reduce_embeddings(
    embeddings: np.ndarray,
    method: Literal["umap", "pca", "kpca", "tsne", "pca_then_umap"] = "umap",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Reduce embeddings to lower dimensions for visualization.

    Args:
        embeddings: High-dimensional embedding array (n_samples, n_features)
        method: Reduction method - "umap", "pca", "kpca", "tsne", or "pca_then_umap"
        n_components: Target number of dimensions (2 or 3)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the reducer:
            - For UMAP: n_neighbors, min_dist, metric
            - For tSNE: perplexity, learning_rate
            - For KernelPCA: kernel, gamma, coef0
            - For PCA: n_iter

    Returns:
        Reduced embedding array (n_samples, n_components)

    Raises:
        ValueError: If method is unknown
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    elif method == "kpca":
        # KernelPCA doesn't use random_state but uses n_init for randomness
        kernel = kwargs.pop("kernel", "rbf")
        gamma = kwargs.pop("gamma", None)
        coef0 = kwargs.pop("coef0", 1)

        reducer = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        reducer = umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        )
        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        # tSNE defaults: perplexity should be 5-50, typically (n_samples-1)/3
        perplexity = kwargs.pop("perplexity", min(30, (embeddings.shape[0] - 1) / 3))
        learning_rate = kwargs.pop("learning_rate", 200)

        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            learning_rate=learning_rate,
            **kwargs,
        )
        return reducer.fit_transform(embeddings)

    elif method == "pca_then_umap":
        # PCA first for computational efficiency, then UMAP for structure preservation
        pca_dims = min(50, embeddings.shape[1])
        pca = PCA(n_components=pca_dims, random_state=random_state)
        pca_embeddings = pca.fit_transform(embeddings)

        reducer = umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        )
        return reducer.fit_transform(pca_embeddings)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def create_embedding_plots(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    method: Literal["umap", "pca", "kpca", "tsne", "pca_then_umap"] = "umap",
    figsize: Tuple[int, int] = (10, 8),
    palette: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    **reduction_kwargs,
) -> plt.Figure:
    """
    Create a 2D scatter plot of embeddings colored by labels.

    Args:
        embeddings: High-dimensional embedding array (n_samples, n_features)
        labels: Label array (n_samples,) - can be int or str
        title: Title for the plot
        method: Dimensionality reduction method ("umap", "pca", "kpca", "tsne", "pca_then_umap")
        figsize: Figure size as (width, height)
        palette: Seaborn color palette name
        save_path: Path to save the figure (optional)
        **reduction_kwargs: Additional arguments for the reduction method

    Returns:
        Matplotlib figure object
    """
    # Reduce to 2D
    reduced = reduce_embeddings(
        embeddings, method=method, n_components=2, **reduction_kwargs
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels and color palette
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    palette = palette or sns.color_palette("husl", n_classes)

    # Plot each class
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=str(label),
            alpha=0.6,
            s=50,
            color=palette[idx] if isinstance(palette, list) else None,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_embedding_plots_3d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    method: Literal["umap", "pca", "kpca", "tsne", "pca_then_umap"] = "pca",
    figsize: Tuple[int, int] = (12, 9),
    palette: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    **reduction_kwargs,
) -> plt.Figure:
    """
    Create a 3D scatter plot of embeddings colored by labels.

    Args:
        embeddings: High-dimensional embedding array (n_samples, n_features)
        labels: Label array (n_samples,) - can be int or str
        title: Title for the plot
        method: Dimensionality reduction method ("umap", "pca", "kpca", "tsne", "pca_then_umap")
        figsize: Figure size as (width, height)
        palette: Seaborn color palette name
        save_path: Path to save the figure (optional)
        **reduction_kwargs: Additional arguments for the reduction method

    Returns:
        Matplotlib figure object
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Reduce to 3D
    reduced = reduce_embeddings(
        embeddings, method=method, n_components=3, **reduction_kwargs
    )

    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get unique labels and color palette
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    palette = palette or sns.color_palette("husl", n_classes)

    # Plot each class
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            reduced[mask, 2],
            label=str(label),
            alpha=0.6,
            s=30,
            color=palette[idx] if isinstance(palette, list) else None,
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_interactive_embedding_plot_3d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    method: Literal["umap", "pca", "kpca", "tsne", "pca_then_umap"] = "pca",
    save_path: Optional[str | Path] = None,
    **reduction_kwargs,
):
    """
    Create an interactive 3D scatter plot of embeddings using Plotly.

    Features:
    - Full 3D rotation and zoom
    - Hover to see label information
    - Smooth animations
    - Download as PNG

    Args:
        embeddings: High-dimensional embedding array (n_samples, n_features)
        labels: Label array (n_samples,) - can be int or str
        title: Title for the plot
        method: Dimensionality reduction method
        palette: Plotly color palette name
        save_path: Path to save the HTML file (optional)
        **reduction_kwargs: Additional arguments for the reduction method

    Returns:
        Plotly figure object

    Raises:
        ImportError: If Plotly is not installed
    """
    # Reduce to 3D
    reduced = reduce_embeddings(
        embeddings, method=method, n_components=3, **reduction_kwargs
    )

    # Get unique labels and create color mapping
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set2
    color_map = {
        str(label): colors[idx % len(colors)] for idx, label in enumerate(unique_labels)
    }

    # Create 3D scatter plot using go.Scatter3d (no pandas needed)
    fig = go.Figure()

    for label in unique_labels:
        mask = labels == label
        fig.add_trace(
            go.Scatter3d(
                x=reduced[mask, 0],
                y=reduced[mask, 1],
                z=reduced[mask, 2],
                mode="markers",
                name=str(label),
                marker=dict(
                    size=5,
                    opacity=0.7,
                    color=color_map[str(label)],
                ),
                text=[str(label)] * np.sum(mask),
                hovertemplate="<b>%{text}</b><br>Component 1: %{x:.3f}<br>Component 2: %{y:.3f}<br>Component 3: %{z:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            zaxis_title=f"{method.upper()} Component 3",
        ),
        hovermode="closest",
        width=1200,
        height=900,
        font=dict(size=12),
        showlegend=True,
    )

    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig

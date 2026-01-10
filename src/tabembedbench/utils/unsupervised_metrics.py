import numpy as np


def calculate_rank_me(embeddings: np.ndarray, epsilon: float = 1e-7) -> float:
    """Calculate the RankMe metric for assessing embedding quality.

    RankMe computes an entropy-based smooth rank measure of embeddings by analyzing
    the distribution of singular values. Higher values indicate better-quality
    representations with more informative dimensions.

    Based on: "RankMe: Assessing the Downstream Performance of Pretrained
    Self-Supervised Representations by Their Rank" (Garrido et al., 2023)

    Formula: RankMe(Z) = exp(-∑ p_k log(p_k))
    where p_k = σ_k(Z) / ||σ(Z)||_1 + ε

    Args:
        embeddings: Embedding matrix of shape (N, K) where N is the number of
                   samples and K is the embedding dimension.
        epsilon: Small constant added for numerical stability to avoid log(0).
                Default: 1e-7 (appropriate for float32).

    Returns:
        The RankMe score, a float value between 1 and min(N, K). Higher values
        indicate embeddings with higher effective rank and typically better
        downstream task performance.

    Raises:
        ValueError: If embeddings is not a 2D array.

    References:
    [1] Garrido, Q. et al. (2023). "RankMe: Assessing the Downstream Performance of Pretrained
        Self-Supervised Representations by Their Rank." arXiv preprint arXiv:2303.17309 (2023).
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array.")

    if embeddings.dtype == np.float16:
        embeddings = embeddings.astype(np.float32)

    singular_values = np.linalg.svd(embeddings, compute_uv=False)

    l1_norm = np.sum(singular_values)
    normalized_singular_values = singular_values / l1_norm + epsilon

    entropy = (-1) * np.sum(
        normalized_singular_values * np.log(normalized_singular_values)
    )

    return np.exp(entropy)


def calculate_alpha_req(embeddings: np.ndarray, epsilon: float = 1e-7) -> float:
    """Calculate the α-ReQ metric for assessing representation quality.

    α-ReQ measures the decay coefficient of the eigenspectrum of the embedding
    covariance matrix. Representations with α ≈ 1.0 typically exhibit the best
    downstream task performance, balancing between dense (α < 1) and sparse (α > 1)
    encodings.

    Based on: "α-ReQ: Assessing Representation Quality by Measuring Eigenspectrum
    Decay" (Agrawal et al., NeurIPS 2022)

    The eigenspectrum follows a power-law: λ_i ∝ i^(-α)
    Best performance is typically observed when α ∈ [0.8, 1.2]

    Args:
        embeddings: Embedding matrix of shape (N, D) where N is the number of
                   samples and D is the embedding dimension.
        epsilon: Small constant to filter near-zero eigenvalues for numerical
                stability. Default: 1e-10.

    Returns:
        The α decay coefficient, a float value typically between 0 and 3.
        Values near 1.0 indicate high-quality representations:
        - α < 0.8: Too dense/redundant representations
        - 0.8 ≤ α ≤ 1.2: Optimal "Goldilocks zone"
        - α > 1.2: Too sparse/collapsed representations

    Raises:
        ValueError: If embeddings is not a 2D array or has insufficient samples.

    References:
        [1] Agrawal, K.K. et al. (2022). "α-ReQ: Assessing Representation Quality by
            Measuring Eigenspectrum Decay." NeurIPS 2022.
        [2] Stringer, C. et al. (2019). "High-dimensional geometry of population
            responses in visual cortex." Nature, 571(7765), 361-365.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array.")

    num_samples, _ = embeddings.shape

    singular_values = np.linalg.svd(embeddings, compute_uv=False)
    eigenvalues = (singular_values**2) / num_samples

    eigenvalues = eigenvalues[eigenvalues > epsilon]

    if len(eigenvalues) < 2:
        raise ValueError("Insufficient non-zero eigenvalues for fitting power law.")

    indices = np.arange(1, len(eigenvalues) + 1)
    log_indices = np.log(indices)
    log_eigenvalues = np.log(eigenvalues)

    coefficients = np.polyfit(log_indices, log_eigenvalues, deg=1)

    slope = coefficients[0]

    alpha_req = -slope

    return float(alpha_req)

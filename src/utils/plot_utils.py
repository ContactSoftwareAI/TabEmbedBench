from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dimensionality_reduction(X, n_components=2, pca_components=100, random_state=42):
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=n_components, random_state=random_state)
    X_tsne = tsne.fit_transform(X_pca)

    return (X_pca, X_tsne)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Optional

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.preprocess_utils import infer_categorical_columns


class SphereBasedEmbedding(BaseEmbeddingGenerator):
    def __init__(self):
        super().__init__()

    def _get_default_name(self) -> str:
        return "Schalenmodell"

    def compute_embeddings(self, data: np.ndarray, embed_dim: Optional[int] = 512):
        cat_indices = infer_categorical_columns(data)

        return compute_embeddings(data, cat_indices, embed_dim)


def compute_embeddings(
    data: Union[pd.DataFrame, np.ndarray],
    categorical_indices: List[int],
    embed_dim: int,
) -> np.ndarray:
    """
    Erstellt Einbettungen für tabellarische Daten mit numerischen und kategorischen Spalten.

    Args:
        data: DataFrame oder NumPy Array mit den Daten
        categorical_indices: Liste der Indizes der kategorischen Spalten
        embed_dim: Dimension der Einbettungen
        random_seed: Seed für Reproduzierbarkeit

    Returns:
        np.ndarray: Zeileneinbettungen durch Mittelung der Spalteneinbettungen
    """
    # np.random.seed(random_seed)

    # Konvertiere zu NumPy Array falls DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data.copy()

    n_rows, n_cols = data_array.shape

    # Erstelle Einbettungen für jede Spalte
    column_embeddings = []

    for col_idx in range(n_cols):
        column_data = data_array[:, col_idx]

        if col_idx in categorical_indices:
            # Kategorische Spalte
            col_embedding = _embed_categorical_column(column_data, embed_dim)
            # plt.plot(col_embedding[:, 0], col_embedding[:, 1], 'o')
            # plt.show()
        else:
            # Numerische Spalte
            col_embedding = _embed_numerical_column(column_data, embed_dim)
            # plt.plot(col_embedding[:, 0], col_embedding[:, 1], 'o')
            # plt.show()

        column_embeddings.append(col_embedding)

    # Erstelle Zeileneinbettungen durch Mittelung
    row_embeddings = np.zeros((n_rows, embed_dim))
    for row_idx in range(n_rows):
        row_embedding = np.zeros(embed_dim)
        for col_idx in range(n_cols):
            row_embedding += column_embeddings[col_idx][row_idx]
        row_embeddings[row_idx] = row_embedding / n_cols

    return row_embeddings


def _generate_random_sphere_point(embed_dim: int) -> np.ndarray:
    """Generiert einen zufälligen Punkt auf der Einheitssphäre."""
    point = np.random.randn(embed_dim)
    return point / np.linalg.norm(point)


def _embed_numerical_column(column_data: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Erstellt Einbettungen für eine numerische Spalte.

    Jeder Wert wird auf einer Linie durch den Ursprung eingebettet,
    wobei die Linie durch einen zufälligen Punkt auf der Einheitssphäre definiert wird.
    """
    column_data = np.asarray(column_data, dtype=np.float64)

    # Finde Min, Max und Mitte
    col_min = np.min(column_data)
    col_max = np.max(column_data)
    col_mid = (col_max + col_min) / 2

    # Generiere zufälligen Punkt auf Einheitssphäre (repräsentiert die Mitte)
    sphere_point = _generate_random_sphere_point(embed_dim)

    # Erstelle Einbettungen für alle Werte
    n_values = len(column_data)
    embeddings = np.zeros((n_values, embed_dim))

    for i, value in enumerate(column_data):
        if col_max == col_min:
            # Alle Werte sind gleich - verwende Mittelpunkt
            radius = 1.0
        else:
            # Normiere Wert auf Radius zwischen 0.5 und 1.5
            normalized_value = (value - col_min) / (col_max - col_min)  # 0 bis 1
            radius = 0.5 + normalized_value * 1.0  # 0.5 bis 1.5

        # Platziere Punkt auf der Linie durch den Ursprung
        embeddings[i] = radius * sphere_point

    return embeddings


def _embed_categorical_column(column_data: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Erstellt Einbettungen für eine kategorische Spalte.

    Jede Kategorie erhält einen zufälligen Punkt in einer kleinen Kugel
    um einen zufälligen Punkt auf der Einheitssphäre.
    """
    # Finde einzigartige Kategorien
    unique_categories = np.unique(column_data)
    n_categories = len(unique_categories)

    # Generiere zufälligen Mittelpunkt auf Einheitssphäre
    center_point = _generate_random_sphere_point(embed_dim)

    # Erstelle Mapping von Kategorie zu Einbettung
    category_embeddings = {}

    for category in unique_categories:
        # Generiere zufälligen Punkt in kleiner Kugel (Radius 0.1) um Mittelpunkt
        random_offset = np.random.randn(embed_dim)
        random_offset = 0.1 * random_offset / np.linalg.norm(random_offset)

        category_embeddings[category] = center_point + random_offset

    # Erstelle Einbettungen für alle Werte
    n_values = len(column_data)
    embeddings = np.zeros((n_values, embed_dim))

    for i, value in enumerate(column_data):
        embeddings[i] = category_embeddings[value]

    return embeddings

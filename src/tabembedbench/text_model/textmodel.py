import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from tabembedbench.utils.google_embeddings import embed_text


class TextModel(TransformerMixin):
    """Text-based embedding generator for tabular data.

    This embedding model projects tabular data onto high-dimensional spheres by
    constructing sentences of each table column and embed them using an LLM.

    Attributes:
        column_names (list[str]): Names of all columns.
        n_cols (int | None): Number of columns in the fitted data.
    """

    def __init__(
        self
    ) -> None:
        """Initialize the text-based embedding generator.
        """
        super()
        self.column_names = None
        self.n_cols = None

    def fit(
        self,
        data: pd.DataFrame,
        y=None,
    ):
        """Fit the embedding model to the data.

        Args:
            data (pd.DataFrame | np.ndarray): Input data to fit.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.

        Returns:
            self: The fitted embedding model.
        """
        self.column_names = data.columns
        self.n_cols = len(data.columns)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms input data into row-wise embeddings by transforming each row
        into a sentence and apply an LLM.

        Args:
            data: Input data to be transformed into embeddings. Must be a Pandas
                DataFrame.

        Returns:
            np.ndarray: A NumPy array containing the row embeddings for the input data.

        Raises:
            ValueError: If the number of columns in the input data does not match the
                number of columns in the fitted data.
        """
        n_cols = len(data.columns)
        if n_cols != self.n_cols:
            raise ValueError("Number of columns in data does not match fitted data.")
        n_rows = len(data)

        row_texts = []
        for row_idx in data.index:
            row_text = ""
            for col_idx in range(n_cols-1):
                row_text += f"{self.column_names[col_idx]}: {data.loc[row_idx,data.columns[col_idx]]}, "
            row_text += f"{self.column_names[n_cols-1]}: {data.loc[row_idx,data.columns[n_cols-1]]}"
            row_texts.append(row_text)
        print(row_texts[0])
        success = False
        batchsize = 128
        while not success and batchsize >= 1:
            try:
                row_embeddings = embed_text(row_texts, batchsize=batchsize)
                success = True
            except Exception as e:
                print(f"An error occurred during embedding for batchsize {batchsize}: {e}")
                print("reducing batchsize")
                batchsize = int(batchsize/2)

        return np.array(row_embeddings)

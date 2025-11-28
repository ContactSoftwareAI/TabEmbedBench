from huggingface_hub import hf_hub_download
import numpy as np
from typing import Tuple, Union
import safetensors.torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from tabstar.arch.config import TabStarConfig, E5_SMALL
from tabstar.arch.fusion import NumericalFusion
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.arch.interaction import InteractionEncoder
from tabstar.training.dataloader import get_dataloader
import torch
import gc
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device
from tabstar.training.devices import clear_cuda_cache


class TabStarModel(PreTrainedModel):
    config_class = TabStarConfig

    def __init__(self, config: TabStarConfig):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained(E5_SMALL)
        self.tokenizer = AutoTokenizer.from_pretrained(E5_SMALL)
        self.numerical_fusion = NumericalFusion()
        self.tabular_encoder = InteractionEncoder()


    def forward(self, x_txt: np.ndarray, x_num: np.ndarray, d_output: int) -> Tensor:
        textual_embeddings = self.get_textual_embedding(x_txt)
        if not isinstance(x_num, Tensor):
            x_num = torch.tensor(x_num, dtype=textual_embeddings.dtype, device=textual_embeddings.device)
        embeddings = self.numerical_fusion(textual_embeddings=textual_embeddings, x_num=x_num)
        encoded = self.tabular_encoder(embeddings)
        target_tokens = encoded[:, :d_output]
        return target_tokens

    def get_textual_embedding(self, x_txt: np.array) -> Tensor:
        text_batch_size = 128
        while text_batch_size > 1:
            try:
                return self.get_textual_embedding_in_batches(x_txt, text_batch_size=text_batch_size)
            except torch.cuda.OutOfMemoryError:
                text_batch_size //= 2
                clear_cuda_cache()
                print(f"🤯 Reducing batch size to {text_batch_size} due to OOM")
        raise RuntimeError(f"🤯 OOM even with batch size 1!")

    def get_textual_embedding_in_batches(self, x_txt: np.array, text_batch_size: int) -> Tensor:
        # Get unique texts and mapping indices
        unique_texts, inverse_indices = np.unique(x_txt, return_inverse=True)
        num_unique_texts = len(unique_texts)
        embeddings = []
        for i in range(0, num_unique_texts, text_batch_size):
            batch_texts = unique_texts[i:i + text_batch_size].tolist()
            inputs = self.tokenizer(batch_texts, padding=True, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.text_encoder(**inputs)
            # Take the [CLS] token representation
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        embeddings = torch.cat(embeddings, dim=0)
        inverse_indices = torch.tensor(inverse_indices, dtype=torch.long, device=embeddings.device)
        # Map the unique embeddings back to the original positions and reshape to match the original dimension
        batch_size, seq_len = x_txt.shape
        embeddings = embeddings[inverse_indices].view(batch_size, seq_len, -1)
        if not tuple(embeddings.shape) == (batch_size, seq_len, self.config.d_model):
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
        return embeddings


class TabStarEmbedding(AbstractEmbeddingGenerator):
    def __init__(self, device: str | None = None):
        super().__init__(name="TabStar")
        self.preprocess_pipeline = None
        self.device = device if device is not None else get_device()
        self.use_amp = bool(self.device.type == "cuda")
        self.tabstar_row_embedder = self._get_model().to(self.device)

    def _get_model(self):
        model_ckpt_path = hf_hub_download(
            repo_id="alana89/TabSTAR",
            filename="model.safetensors"
        )
        config_class = TabStarConfig()
        tabstar_embedding_model = TabStarModel(config=config_class)
        weights = safetensors.torch.load_file(model_ckpt_path)
        # TODO: use model weights
        return tabstar_embedding_model

    def _preprocess_data(self, X: np.ndarray, train: bool = True, outlier: bool = False,
                         **kwargs) -> TabSTARData:
        X = DataFrame(X)
        X.columns = [str(col) for col in X.columns]
        y = Series(np.zeros(X.shape[0]))
        if train:
            self.preprocess_pipeline = TabSTARVerbalizer(is_cls=False)
            self.preprocess_pipeline.fit(X,y)
            X_preprocessed = self.preprocess_pipeline.transform(X,y=None)
        else:
            if self.preprocess_pipeline is None:
                raise ValueError("Preprocessing pipeline is not fitted")
            else:
                X_preprocessed = self.preprocess_pipeline.transform(X,y=None)

        return X_preprocessed

    def _fit_model(self, X_preprocessed: np.ndarray,
                   y_preprocessed: np.ndarray | None = None, train: bool = True,
                   **kwargs) -> None:
        self._is_fitted = True

    def _compute_embeddings(
            self,
            X_train_preprocessed: np.ndarray,
            X_test_preprocessed: np.ndarray | None = None,
            outlier: bool = False,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray | None]:

        self.tabstar_row_embedder.eval()
        embeddings_list = []

        if outlier:
            dataloader = get_dataloader(X_train_preprocessed, is_train=False, batch_size=128)
            for data in dataloader:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    embeddings_list.append(batch_predictions_numpy)

            embeddings = np.concatenate(embeddings_list, axis=0)

            return embeddings, None

        if self._is_fitted:
            dataloader_train = get_dataloader(X_train_preprocessed, is_train=False, batch_size=128)
            for data in dataloader_train:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num,
                                                                  d_output=data.d_output)
                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    embeddings_list.append(batch_predictions_numpy)

            embeddings_train = np.concatenate(embeddings_list, axis=0)

            embeddings_list = []
            dataloader_test = get_dataloader(X_test_preprocessed, is_train=False, batch_size=128)
            for data in dataloader_test:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num,
                                                                  d_output=data.d_output)
                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    embeddings_list.append(batch_predictions_numpy)

            embeddings_test = np.concatenate(embeddings_list, axis=0)

            return embeddings_train, embeddings_test
        else:
            raise ValueError("Model is not fitted")


    def _reset_embedding_model(self, *args, **kwargs):
        """Reset the embedding model to its initial state.

                Reinitializes all preprocessing pipelines to clear fitted state
                and moves model back to CPU to free GPU memory.
                """
        # Move model to CPU before deleting references
        if self.tabstar_row_embedder is not None:
            self.tabstar_row_embedder.cpu()

        # Clear preprocessing pipeline
        self.preprocess_pipeline = None
        #self._is_fitted = False

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        elif self.device == "mps":
            import torch

            torch.mps.empty_cache()

        # Move model back to the device for next use
        if self.tabstar_row_embedder is not None:
            self.tabstar_row_embedder.to(self.device)



# if __name__ == "__main__":
#     PATH = r'/home/frederik_hoppe_contact_software_/projects/tabembedbench/data/adbench_tabular_datasets/1_ALOI.npz'
#
#     data = np.load(PATH)
#
#     try:
#         X = data['X']  # Features
#         y = data['y']  # Target labels
#     except KeyError:
#         print("Available keys in the dataset:", list(data.files))
#         X = data[data.files[0]]
#         y = data[data.files[1]]
#
#     x_train = DataFrame(X)
#     x_train.columns = [f'feature_{i}' for i in range(x_train.shape[1])]
#     y_train = y #Series(y)
#
#     x_test = None
#     y_test = None
#
#     if x_test is None:
#         assert y_test is None, "If x_test is None, y_test must also be None"
#         x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
#
#     tabstar_model = TabStarEmbedding()
#     embeddings, _, _ = tabstar_model.generate_embeddings(x_train, x_test, outlier=True)
#
#     #tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
#     #tabstar = tabstar_cls()
#     #tabstar.fit(x_train, y_train)
#     #y_pred = tabstar.predict(x_test)
#     #metric = tabstar.score(X=x_test, y=y_test)
#     #print(f"Accuracy: {metric:.4f}")
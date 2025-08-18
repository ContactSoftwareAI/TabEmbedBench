from pathlib import Path
import numpy as np
import torch

from tabembedbench.benchmark.benchmark_utils import run_outlier_benchmark
from tabembedbench.embedding_models.tabicl_utils import get_row_embeddings_model
from tabembedbench.utils.torch_utils import get_device


model_ckpt_path = Path("../data/models/tabicl/tabicl-classifier-v1.1-0506.ckpt")

model_ckpt = torch.load(model_ckpt_path)

state_dict = model_ckpt["state_dict"]
config = model_ckpt["config"]

device = get_device()
print(device)

row_embedder = get_row_embeddings_model(state_dict=state_dict, config=config)

row_embedder.to(device)

run_outlier_benchmark(
    model=row_embedder,
    dataset_paths="../data/adbench_tabular_datasets"
)
from datetime import datetime

from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np
import torch

from tabembedbench.benchmark.benchmark_utils import run_outlier_benchmark
from tabembedbench.embedding_models.tabicl_utils import get_row_embeddings_model
from tabembedbench.embedding_models.spherebasedembedding_utils import SphereBasedEmbedding
from tabembedbench.utils.torch_utils import get_device

model_ckpt_path = Path("/data/models/tabicl/tabicl-classifier-v1.1-0506.ckpt")

model_name = model_ckpt_path.stem

print(model_name)
if not model_ckpt_path.exists():
    model_ckpt_path = hf_hub_download(
        repo_id="jingang/TabICL-clf",
        filename="tabicl-classifier-v1.1-0506.ckpt",
    )

model_ckpt = torch.load(model_ckpt_path)

state_dict = model_ckpt["state_dict"]
config = model_ckpt["config"]

device = get_device()
print(device)

row_embedder = get_row_embeddings_model(state_dict=state_dict, config=config)

row_embedder.name = model_name

sphere_embedder = SphereBasedEmbedding()

row_embedder.to(device)

models = [
    row_embedder,
    sphere_embedder
]

result_df = run_outlier_benchmark(
    models=models,
    save_embeddings=True
)

timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")

result_parquet = Path(f"data/results/results_{timestamp_compact}.parquet")
result_parquet.parent.mkdir(parents=True, exist_ok=True)

result_df.write_parquet(
    file=result_parquet
)
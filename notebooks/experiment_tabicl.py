import logging
from datetime import datetime
from pathlib import Path

from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models.spherebased_embedding import (
    SphereBasedEmbedding,
)
from tabembedbench.embedding_models.tabicl_embedding import get_row_embeddings_model
from tabembedbench.utils.torch_utils import get_device

device = get_device()

print(device)

row_embedder = get_row_embeddings_model(model_path="auto")

row_embedder.name = "tabicl-classifier-v1.1-0506"

row_embedder_preprocessed = get_row_embeddings_model(
    model_path="auto", preprocess_data=True
)

row_embedder_preprocessed.name = "tabicl-classifier-v1.1-0506" + "_preprocessed"

row_embedder.to(device)
row_embedder_preprocessed.to(device)

models = [row_embedder_preprocessed, row_embedder]

for n in range(3, 4):
    sphere_model = SphereBasedEmbedding(embed_dim=n)
    sphere_model.name = f"sphere-model-d{n}"
    models.append(sphere_model)

log_path = Path("data/logs")

result_df = run_benchmark(
    embedding_models=models,
    save_embeddings=False,
    exclude_adbench_datasets=["3_backdoor.npz"],
    exclude_adbench_image_datasets=True,
    upper_bound_dataset_size=100000,
    run_outlier=True,
    run_task_specific=True,
    save_logs=True,
    log_dir=log_path,
    logging_level=logging.DEBUG,
)

timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")

result_parquet = Path(f"data/results/results_{timestamp_compact}.parquet")
result_parquet.parent.mkdir(parents=True, exist_ok=True)

result_df.write_parquet(file=result_parquet)

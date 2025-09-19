import logging

from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models.spherebased_embedding import (
    SphereBasedEmbedding,
)
from tabembedbench.embedding_models.tabicl_embedding import get_tabicl_embedding_model
from tabembedbench.embedding_models.tabvectorizer_embedding import TabVectorizerEmbedding
from tabembedbench.utils.torch_utils import get_device


device = get_device()

tabicl_with_preproccessing = get_tabicl_embedding_model(
    model_path="auto", preprocess_data=True
)

tabicl_with_preproccessing.name = "tabicl-classifier-v1.1-0506_preprocessed"

tabicl_with_preproccessing.to(device)

tablevector = TabVectorizerEmbedding()

models = [tablevector, tabicl_with_preproccessing]

for n in range(5, 10):
    print(f"n={2**n}")
    sphere_model = SphereBasedEmbedding(embed_dim=2**n)
    sphere_model.name = f"sphere-model-d{2**n}"
    models.append(sphere_model)

result_df = run_benchmark(
    embedding_models=models,
    save_embeddings=False,
    exclude_adbench_datasets=["3_backdoor.npz"],
    exclude_adbench_image_datasets=True,
    upper_bound_dataset_size=100000,
    upper_bound_num_feautres=500,
    run_outlier=True,
    run_task_specific=True,
    save_logs=True,
    logging_level=logging.DEBUG,
)
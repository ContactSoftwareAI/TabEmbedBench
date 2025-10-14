import logging
from pathlib import Path
import seaborn as sns
import click

from tabembedbench.benchmark.run_benchmark import run_benchmark
from tabembedbench.embedding_models import (
    TabICLEmbedding,
    TabVectorizerEmbedding,
    TabPFNEmbedding,
    SphereBasedEmbedding
)
from tabembedbench.evaluators.classifier import KNNClassifierEvaluator
from tabembedbench.evaluators.outlier import (
    IsolationForestEvaluator,
    LocalOutlierFactorEvaluator,
    DeepSVDDEvaluator
)
from tabembedbench.evaluators.mlp_evaluator import (
    MLPClassifierEvaluator,
    MLPRegressorEvaluator
)
from tabembedbench.evaluators.regression import KNNRegressorEvaluator
from tabembedbench.examples.eda_utils import create_tabarena_plots, create_outlier_plots

logger = logging.getLogger("EuRIPS_Run_Benchmark")


def get_embedding_models(debug=False):
    embedding_models = []
    for n in range(3, 10):
        sphere_model = SphereBasedEmbedding(embed_dim=2**n)
        sphere_model.name = f"sphere-model-d{2**n}"
        embedding_models.append(sphere_model)

    tabicl_with_preproccessing = TabICLEmbedding(preprocess_tabicl_data=True)

    tabicl_with_preproccessing.name = "TabICL"

    tablevector = TabVectorizerEmbedding()
    tablevector.name = "TableVectorizer"

    tabpfn_embedder = TabPFNEmbedding()

    embedding_models.extend([
        tablevector,
        tabicl_with_preproccessing,
        tabpfn_embedder
    ])

    if debug:
        embedding_models = [tablevector, tabicl_with_preproccessing]
        sphere_model = SphereBasedEmbedding(embed_dim=8)
        sphere_model.name = "sphere-model-d8-debug"

        embedding_models.append(sphere_model)

        return embedding_models

    return embedding_models

def get_evaluators(debug=False):
    evaluator_algorithms = []

    deep_svdd_dynamic = DeepSVDDEvaluator(dynamic_hidden_neurons=True)
    deep_svdd_dynamic._name = "DeepSVDD-dynamic"

    if debug:
        evaluator_algorithms.extend(
            [
                KNNRegressorEvaluator(
                    num_neighbors=5, weights="uniform", metric="euclidean"
                ),
                KNNClassifierEvaluator(
                    num_neighbors=5, weights="uniform", metric="euclidean"
                ),
                MLPClassifierEvaluator(n_trials=5, cv_folds=2, verbose=False),
                MLPRegressorEvaluator(n_trials=5, cv_folds=2, verbose=False),
                DeepSVDDEvaluator(),
                deep_svdd_dynamic,
                LocalOutlierFactorEvaluator(
                    model_params={
                        "n_neighbors": 5,
                    }
                ),
            ]
        )
        return evaluator_algorithms

    for num_neighbors in range(0, 50, 5):
        if num_neighbors > 0:
            for metric in ["euclidean", "cosine"]:
                for weights in ["uniform", "distance"]:
                    evaluator_algorithms.extend([
                        KNNRegressorEvaluator(
                            num_neighbors=num_neighbors,
                            weights=weights,
                            metric=metric
                        ),
                        KNNClassifierEvaluator(
                            num_neighbors=num_neighbors,
                            weights=weights,
                            metric=metric
                        )
                    ])
                evaluator_algorithms.append(
                    LocalOutlierFactorEvaluator(
                        model_params={
                            "n_neighbors": num_neighbors,
                            "metric": metric,
                        }
                    )
                )
    for num_estimators in range(0, 300, 50):
        if num_estimators > 0:
            evaluator_algorithms.append(
                IsolationForestEvaluator(
                    model_params={
                        "n_estimators": num_estimators,
                    }
                )
            )

    evaluator_algorithms.extend(
        [
            MLPRegressorEvaluator(),
            MLPClassifierEvaluator(),
            deep_svdd_dynamic,
            DeepSVDDEvaluator(),
        ]
    )

    return evaluator_algorithms

def run_main(debug, max_samples, max_features, run_outlier,
             run_task_specific, adbench_dataset_path):
    if debug:
        logger.info("Running in DEBUG mode")
        max_samples = 800
        max_features = 50


    embedding_models = get_embedding_models(debug)

    evaluators = get_evaluators(debug)

    logger.info(f"Using {len(embedding_models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    outlier_result_df, tabarena_result_df, result_dir = run_benchmark(
        embedding_models=embedding_models,
        evaluator_algorithms=evaluators,
        adbench_dataset_path=adbench_dataset_path,
        exclude_adbench_datasets=["3_backdoor.npz"],
        upper_bound_dataset_size= max_samples,
        upper_bound_num_features= max_features,
        run_outlier=run_outlier,
        run_task_specific=run_task_specific,
        logging_level=logging.DEBUG,
    )

    models_to_keep = outlier_result_df.get_column("embedding_model").unique().to_list()
    colors = sns.color_palette("colorblind", n_colors=len(models_to_keep))
    color_mapping = {
        model: f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for model, (r, g, b) in zip(models_to_keep, colors)
    }

    result_sphere_dir = Path(result_dir / "sphere_included")
    result_sphere_dir.mkdir(exist_ok=True)

    create_tabarena_plots(tabarena_result_df, data_path=result_sphere_dir,
                          color_mapping=color_mapping)
    create_outlier_plots(outlier_result_df, data_path=result_sphere_dir, color_mapping=color_mapping)

    models_to_keep = ["TabICL", "TabPFN", "TableVectorizer"]
    color_mapping_small = {
        key: item for key, item in color_mapping.items() if key in models_to_keep
    }
    create_tabarena_plots(tabarena_result_df,
                          data_path=result_dir,
                          color_mapping=color_mapping_small,
                          models_to_keep=models_to_keep)
    create_outlier_plots(outlier_result_df,
                         data_path=result_dir,
                         color_mapping=color_mapping_small,
                         models_to_keep=models_to_keep)



@click.command()
@click.option('--debug', is_flag=True, help='Run in debug mode ')
@click.option('--max-samples', default=100001, help='Upper bound for dataset '
                                                  'size')
@click.option('--max-features', default=500, help='Upper bound for number of features')
@click.option('--run-outlier/--no-run-outlier', default=True, help='Run outlier detection')
@click.option('--run-task-specific/--no-run-task-specific', default=True, help='Run task-specific evaluations')
@click.option('--adbench-data', default='data/adbench_tabular_datasets',
              help='Run task-specific '
                                                 'evaluations')
def main(debug, max_samples, max_features, run_outlier, run_task_specific,
         adbench_data):
    run_main(
        debug=debug,
        max_samples=max_samples,
        max_features=max_features,
        run_outlier=run_outlier,
        run_task_specific=run_task_specific,
        adbench_dataset_path=adbench_data)


if __name__ == "__main__":
    main()
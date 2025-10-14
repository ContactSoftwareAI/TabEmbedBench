import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.utils import infer_categorical_features
from tabpfn_extensions.many_class import ManyClassClassifier

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device


class TabPFNEmbedding(AbstractEmbeddingGenerator):
    """Universal TabPFN-based embedding generator for tabular data.

    This class generates embeddings using TabPFN (Tabular Prior-data Fitted Networks)
    by treating each feature as a target and using the remaining features as inputs.
    It supports both classification and regression tasks automatically based on
    feature type detection.

    The embedding process works by:
    1. For each feature column, mask it as the target
    2. Use remaining features as inputs to predict the masked feature
    3. Extract embeddings from the trained TabPFN model
    4. Aggregate embeddings across features and estimators

    Attributes:
        num_estimators (int): Number of TabPFN estimators to use.
        estimator_agg (str): Aggregation method for multiple estimators.
        emb_agg (str): Aggregation method for feature embeddings.
        device (torch.device): Device for computation (CPU/GPU).
        tabpfn_dim (int): Dimensionality of TabPFN embeddings (192).
        categorical_indices (list[int]): Indices of categorical features.
        num_features (int): Number of features in the dataset.
    """

    def __init__(
        self,
        num_estimators: int = 1,
        estimator_agg: str = "mean",
        emb_agg: str = "mean",
    ) -> None:
        """Initialize the UniversalTabPFNEmbedding.

        Args:
            num_estimators (int, optional): Number of TabPFN estimators to use for
                ensemble predictions. Defaults to 1.
            estimator_agg (str, optional): Aggregation method for combining embeddings
                from multiple estimators. Options are "mean" or "first_element".
                Defaults to "mean".
            emb_agg (str, optional): Aggregation method for combining embeddings
                across features. Options are "concat" or "mean". Defaults to "mean".

        Raises:
            NotImplementedError: If unsupported aggregation methods are specified.
        """
        super().__init__(name="TabPFN")
        self.num_estimators = num_estimators

        self.device = get_device()

        self.tabpfn_dim = 192

        self._init_tabpfn_configs = {
            "device": self.device,
            "n_estimators": self.num_estimators,
            "ignore_pretraining_limits": True,
            "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
        }

        self.emb_agg = emb_agg
        self.estimator_agg = estimator_agg

        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.many_classifier = ManyClassClassifier(
            estimator=self.tabpfn_clf,
            alphabet_size=10,
            n_estimators=self.num_estimators,  # Number of subproblems to create
        )

        self._is_fitted = False

    def _preprocess_data(
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ):
        """Preprocess input data for TabPFN embedding generation.

        Converts numpy arrays to PyTorch tensors and moves them to the appropriate
        device (CPU/GPU) for computation.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features).
            train (bool, optional): Whether this is training data. Currently unused
                but kept for interface compatibility. Defaults to True.
            outlier (bool, optional): Whether this is outlier data. Currently unused
                but kept for interface compatibility. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Preprocessed data as a float tensor on the specified device.
        """
        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        categorical_indices: list[int] | None = None,
        **kwargs,
    ):
        """Fit the TabPFN embedding model to the preprocessed data.

        This method prepares the model for embedding generation by identifying
        categorical features and storing dataset metadata. No actual model training
        occurs here as TabPFN models are fitted during embedding computation.

        Args:
            X_preprocessed (torch.Tensor): Preprocessed input data of shape
                (n_samples, n_features).
            categorical_indices (list[int] | None, optional): List of indices
                indicating which features are categorical. If None, categorical
                features will be automatically inferred. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Note:
            Sets the internal state to fitted and stores feature information
            for subsequent embedding computation.
        """
        if categorical_indices is not None:
            self.categorical_indices = categorical_indices
        else:
            # Convert CUDA tensor to CPU numpy array for categorical inference
            self.categorical_indices = infer_categorical_features(X_preprocessed)

        self.num_features = X_preprocessed.shape[-1]

        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_preprocessed: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute embeddings using TabPFN models.

        For each feature column, this method:
        1. Masks the feature as the target variable
        2. Uses remaining features as inputs to train a TabPFN model
        3. Extracts embeddings from the trained model
        4. Aggregates embeddings according to the specified strategy

        The method automatically selects TabPFNClassifier for categorical features
        and TabPFNRegressor for continuous features.

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data of shape
                (n_samples, n_features).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Generated embeddings. Shape depends on aggregation method:
                - If emb_agg="concat": (n_samples, n_features * tabpfn_dim)
                - If emb_agg="mean": (n_samples, tabpfn_dim)

        Raises:
            NotImplementedError: If unsupported aggregation methods are used.

        Note:
            This method creates fresh TabPFN models for each feature to avoid
            interference between different prediction tasks.
        """
        num_samples = X_preprocessed.shape[0]
        tmp_embeddings = []

        for column_idx in range(X_preprocessed.shape[1]):
            # Create mask for the current column
            mask = np.zeros_like(X_preprocessed, dtype=bool)
            mask[:, column_idx] = True

            # Extract features (all columns except current) and target (current column)
            features = X_preprocessed[~mask].reshape(num_samples, -1)
            target = X_preprocessed[mask]

            is_categorical = column_idx in self.categorical_indices

            if is_categorical:
                n_classes = len(np.unique(target))
                if n_classes > 10:
                    model = self.tabpfn_reg
                else:
                    model = self.tabpfn_clf
            else:
                model = self.tabpfn_reg


            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_indices
                else self.tabpfn_reg
            )

            try:
                model.fit(features, target)
            except ValueError as e:
                # If a column is marked as categorical but has continuous values,
                # fall back to using the regression model
                if "Unknown label type: continuous" in str(e):
                    model = self.tabpfn_reg
                    model.fit(features, target)
                elif "Number of classes" in str(e) and ("exceeds the maximal "
                                                        "number" in str(e)):
                    model = self.many_classifier
                    model.fit(features, target)
                else:
                    raise

            estimator_embeddings = model.get_embeddings(features)

            if self.num_estimators > 1:
                if self.estimator_agg == "mean":
                    estimator_embeddings = np.mean(estimator_embeddings, axis=0)
                elif self.estimator_agg == "first_element":
                    estimator_embeddings = np.squeeze(estimator_embeddings[0, :])
                else:
                    raise NotImplementedError
            else:
                estimator_embeddings = np.squeeze(estimator_embeddings)

            tmp_embeddings += [estimator_embeddings]

        concat_embeddings = np.concatenate(tmp_embeddings, axis=1).reshape(
            tmp_embeddings[0].shape[0], -1
        )

        if self.emb_agg == "concat":
            return concat_embeddings
        elif self.emb_agg == "mean":
            reshaped_embeddings = concat_embeddings.reshape(
                *concat_embeddings.shape[:-1], self.num_features, self.tabpfn_dim
            )
            embeddings = np.mean(reshaped_embeddings, axis=-2)

            return embeddings
        else:
            raise NotImplementedError

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        This method reinitializes the TabPFN models and clears all fitted state,
        allowing the model to be used on new datasets. It's typically called
        between different datasets or experiments.

        Note:
            After calling this method, the model will need to be fitted again
            before generating embeddings.
        """
        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.num_features = None
        self.categorical_indices = None
        self._is_fitted = False

if __name__ == "__main__":
    import openml
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    from sklearn.preprocessing import LabelEncoder
    from tabicl.sklearn.preprocessing import TransformToNumerical

    from tabembedbench.evaluators.classifier import KNNClassifierEvaluator
    from tabembedbench.evaluators.outlier import (
        IsolationForestEvaluator,
        LocalOutlierFactorEvaluator,
        DeepSVDDEvaluator,
    )
    from tabembedbench.evaluators.mlp_evaluator import (
        MLPClassifierEvaluator,
        MLPRegressorEvaluator,
    )
    from tabembedbench.evaluators.regression import KNNRegressorEvaluator
    from sklearn.metrics import mean_absolute_percentage_error, roc_auc_score, log_loss

    # Configuration
    task_id = 363702
    result_dir = Path("result_tabpfn_embedding")
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define evaluator algorithms
    evaluator_algorithms = []

    for num_neighbors in range(5, 50, 5):
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

    # Add MLP evaluators
    evaluator_algorithms.extend([
        MLPClassifierEvaluator(),
        MLPRegressorEvaluator(),
    ])

    print(f"Loading OpenML task {task_id}...")
    
    # Load task and dataset from OpenML
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    
    print(f"Dataset: {dataset.name}")
    print(f"Task type: {task.task_type}")
    
    # Get data
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    
    categorical_indices = np.nonzero(categorical_indicator)[0].tolist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of categorical features: {len(categorical_indices)}")
    
    # Get train/test split (using first fold and repeat)
    train_indices, test_indices = task.get_train_test_split_indices(
        fold=0,
        repeat=0,
    )
    
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Preprocess data
    print("Preprocessing data...")
    numerical_transformer = TransformToNumerical()
    X_train_transformed = numerical_transformer.fit_transform(X_train)
    X_test_transformed = numerical_transformer.transform(X_test)
    
    # Encode labels for classification
    if task.task_type == "Supervised Classification":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_train_encoded = y_train.values
        y_test_encoded = y_test.values
    
    # Initialize TabPFN embedding model
    print("Initializing TabPFN embedding model...")
    embedding_model = TabPFNEmbedding(
        num_estimators=1,
        estimator_agg="mean",
        emb_agg="mean"
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    train_embeddings, train_time, test_embeddings, test_time = embedding_model.generate_embeddings(
        X_train_transformed,
        X_test=X_test_transformed,
        categorical_indices=categorical_indices
    )
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Train embedding time: {train_time:.2f}s")
    print(f"Test embedding time: {test_time:.2f}s")
    
    # Evaluate embeddings with each evaluator
    results = []
    
    print(f"\nEvaluating with {len(evaluator_algorithms)} evaluators...")
    for i, evaluator in enumerate(evaluator_algorithms, 1):
        # Check if evaluator is compatible with task type
        if task.task_type != evaluator.task_type:
            continue
        
        print(f"[{i}/{len(evaluator_algorithms)}] Evaluating with {evaluator._name}...")
        
        try:
            # Train evaluator
            prediction_train, _ = evaluator.get_prediction(
                train_embeddings,
                y_train_encoded,
                train=True,
            )
            
            # Get test predictions
            test_prediction, _ = evaluator.get_prediction(
                test_embeddings,
                train=False,
            )
            
            # Build result dictionary
            result_dict = {
                "task_id": task_id,
                "dataset_name": dataset.name,
                "dataset_size": X.shape[0],
                "num_features": X.shape[1],
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "embedding_model": embedding_model.name,
                "embed_dim": train_embeddings.shape[-1],
                "time_to_compute_train_embedding": train_time,
                "time_to_compute_test_embedding": test_time,
                "algorithm": evaluator._name,
                "task_type": task.task_type,
            }
            
            # Add evaluator parameters
            evaluator_params = evaluator.get_parameters()
            for key, value in evaluator_params.items():
                result_dict[f"algorithm_{key}"] = value
            
            # Compute task-specific metrics
            if task.task_type == "Supervised Regression":
                mape_score = mean_absolute_percentage_error(y_test_encoded, test_prediction)
                result_dict["mape_score"] = mape_score
                print(f"  MAPE: {mape_score:.4f}")
                
            elif task.task_type == "Supervised Classification":
                n_classes = test_prediction.shape[1]
                if n_classes == 2:
                    auc_score = roc_auc_score(y_test_encoded, test_prediction[:, 1])
                    result_dict["classification_type"] = "binary"
                    result_dict["auc_score"] = auc_score
                    print(f"  AUC: {auc_score:.4f}")
                else:
                    auc_score = roc_auc_score(y_test_encoded, test_prediction, multi_class="ovr")
                    log_loss_score = log_loss(y_test_encoded, test_prediction)
                    result_dict["classification_type"] = "multiclass"
                    result_dict["auc_score"] = auc_score
                    result_dict["log_loss_score"] = log_loss_score
                    print(f"  AUC: {auc_score:.4f}, Log Loss: {log_loss_score:.4f}")
            
            results.append(result_dict)
            
            # Reset evaluator
            evaluator.reset_evaluator()
            
        except Exception as e:
            print(f"  Error with {evaluator._name}: {e}")
            continue
    
    # Convert results to DataFrame
    print(f"\nCreating results DataFrame with {len(results)} rows...")
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = result_dir / f"tabpfn_embedding_task_{task_id}_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Task ID: {task_id}")
    print(f"Dataset: {dataset.name}")
    print(f"Task Type: {task.task_type}")
    print(f"Total evaluations: {len(results)}")
    
    if task.task_type == "Supervised Classification" and "auc_score" in results_df.columns:
        print(f"\nBest AUC: {results_df['auc_score'].max():.4f}")
        best_row = results_df.loc[results_df['auc_score'].idxmax()]
        print(f"Best algorithm: {best_row['algorithm']}")
    elif (task.task_type == "Supervised Classification" and "log_loss_score" in
         results_df.columns):
        print(f"\nBest AUC: {results_df['log_loss_score'].max():.4f}")
        best_row = results_df.loc[results_df['log_loss_score'].idxmax()]
        print(f"Best algorithm: {best_row['algorithm']}")
    elif task.task_type == "Supervised Regression" and "mape_score" in results_df.columns:
        print(f"\nBest MAPE: {results_df['mape_score'].min():.4f}")
        best_row = results_df.loc[results_df['mape_score'].idxmin()]
        print(f"Best algorithm: {best_row['algorithm']}")
    
    print("="*80)

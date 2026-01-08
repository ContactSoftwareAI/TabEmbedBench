"""Contract tests for AbstractEmbeddingGenerator implementations.

This module provides automatic discovery and testing of all concrete implementations
of AbstractEmbeddingGenerator. Tests are automatically run against every implementation
without requiring manual test class creation for each one.

When a new implementation is added to the embedding_models package, it will automatically
be discovered and tested on the next test run.
"""

import logging

import numpy as np
import pandas as pd
import pytest

# Import the embedding_models package to ensure all implementations are loaded into memory
# This is critical - __subclasses__() only returns classes that have been imported
import tabembedbench.embedding_models  # noqa: F401
from tabembedbench.embedding_models import AbstractEmbeddingGenerator

logger = logging.getLogger(__name__)


def get_all_concrete_subclasses(base_class: type) -> list[type]:
    """
    Recursively retrieves all concrete (non-abstract) subclasses of a given base class.

    This function traverses the inheritance hierarchy of the specified base class to
    identify and return all subclasses that do not define any abstract methods, meaning
    they are concrete implementations.

    Args:
        base_class (type): The base class from which the traversal begins.

    Returns:
        list[type]: A list of all concrete subclasses derived from the base class.
    """
    concrete_classes = []

    def _find_subclasses(cls: type) -> None:
        for subclass in cls.__subclasses__():
            abstract_methods = getattr(subclass, "__abstractmethods__", set())
            if not abstract_methods:
                concrete_classes.append(subclass)
            _find_subclasses(subclass)

    _find_subclasses(base_class)
    return concrete_classes


# Factory functions for instantiating embedding generators with varying initialization requirements
# Some implementations require specific parameters while others can use defaults
EMBEDDING_INIT_PARAMS: dict[str, dict] = {
    "SphereBasedEmbedding": {"embed_dim": 64},
    # Other implementations can be instantiated with no arguments or defaults
}


def create_embedding_instance(cls: type):
    """
    Factory function to create an instance of an embedding generator class.

    Handles varying initialization requirements by looking up class-specific
    parameters in EMBEDDING_INIT_PARAMS.

    Args:
        cls (type): The embedding generator class to instantiate.

    Returns:
        An instance of the embedding generator class.
    """
    init_params = EMBEDDING_INIT_PARAMS.get(cls.__name__, {})
    return cls(**init_params)


# Discover all concrete implementations of AbstractEmbeddingGenerator
CONCRETE_EMBEDDING_GENERATORS = get_all_concrete_subclasses(AbstractEmbeddingGenerator)


def get_target_embedding_classes(request) -> list[type]:
    """Helper to get embedding classes based on --embedding-class option.

    Returns:
        List containing either all embedding generators or a single specific one.

    Raises:
        ValueError: If the specified class name is not found.
    """
    embedding_class_name = request.config.getoption("--embedding-class", default=None)

    if not embedding_class_name:
        # No specific class provided, use all
        return CONCRETE_EMBEDDING_GENERATORS

    # Find the matching class
    matching_classes = [
        cls
        for cls in CONCRETE_EMBEDDING_GENERATORS
        if cls.__name__ == embedding_class_name
    ]

    if not matching_classes:
        available = [cls.__name__ for cls in CONCRETE_EMBEDDING_GENERATORS]
        raise ValueError(
            f"Embedding class '{embedding_class_name}' not found. "
            f"Available classes: {available}"
        )

    return matching_classes


def pytest_generate_tests(metafunc):
    """Dynamic parametrization to filter embedding generators before they reach the fixture."""
    if "embedding_generator_instance" in metafunc.fixturenames:
        embedding_class_name = metafunc.config.getoption(
            "--embedding-class", default=None
        )

        if embedding_class_name:
            matching_classes = [
                cls
                for cls in CONCRETE_EMBEDDING_GENERATORS
                if cls.__name__ == embedding_class_name
            ]
            if not matching_classes:
                available = [cls.__name__ for cls in CONCRETE_EMBEDDING_GENERATORS]
                pytest.exit(
                    f"Embedding class '{embedding_class_name}' not found. Available: {available}"
                )

            metafunc.parametrize(
                "embedding_generator_instance",
                matching_classes,
                ids=lambda cls: cls.__name__,
                indirect=True,
            )
        else:
            metafunc.parametrize(
                "embedding_generator_instance",
                CONCRETE_EMBEDDING_GENERATORS,
                ids=lambda cls: cls.__name__,
                indirect=True,
            )


@pytest.fixture
def embedding_generator_instance(request):
    """
    Fixture to provide an instance of an embedding generator for parameterized testing.

    This fixture allows testing various implementations of embedding generators.
    It uses the factory function to handle varying initialization requirements.

    Args:
        request: A pytest fixture that provides the parameterized embedding class
            to be tested.

    Returns:
        An instance of the parameterized embedding generator class.
    """
    embedding_class = request.param
    return create_embedding_instance(embedding_class)


@pytest.fixture
def sample_tabular_data():
    """Fixture providing sample tabular data for testing.

    Returns:
        Tuple of (X_train, X_test) as numpy arrays suitable for embedding generation.
    """
    rng = np.random.default_rng(42)
    n_train_samples = 50
    n_test_samples = 20
    n_features = 10

    X_train = rng.standard_normal(size=(n_train_samples, n_features))
    X_test = rng.standard_normal(size=(n_test_samples, n_features))

    return X_train, X_test


@pytest.fixture
def sample_dataframe_data():
    """Fixture providing sample DataFrame data for testing.

    Returns:
        Tuple of (X_train, X_test) as pandas DataFrames suitable for embedding generation.
    """
    rng = np.random.default_rng(42)
    n_train_samples = 50
    n_test_samples = 20
    n_features = 10

    X_train = pd.DataFrame(
        rng.standard_normal(size=(n_train_samples, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    X_test = pd.DataFrame(
        rng.standard_normal(size=(n_test_samples, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    return X_train, X_test


class TestAbstractEmbeddingGeneratorDiscovery:
    """Tests for the automatic discovery mechanism."""

    def test_concrete_implementations_discovered(self):
        """Verify that concrete implementations are discovered."""
        assert len(CONCRETE_EMBEDDING_GENERATORS) > 0, (
            "No concrete implementations of AbstractEmbeddingGenerator were discovered. "
            "Ensure implementation modules are imported in the embedding_models package."
        )

    def test_all_discovered_classes_are_concrete(self):
        """Verify that all discovered classes have no abstract methods."""
        for cls in CONCRETE_EMBEDDING_GENERATORS:
            abstract_methods = getattr(cls, "__abstractmethods__", set())
            assert not abstract_methods, (
                f"{cls.__name__} has abstract methods: {abstract_methods}"
            )

    def test_all_discovered_classes_inherit_from_abstract_embedding_generator(self):
        """Verify that all discovered classes inherit from AbstractEmbeddingGenerator."""
        for cls in CONCRETE_EMBEDDING_GENERATORS:
            assert issubclass(cls, AbstractEmbeddingGenerator), (
                f"{cls.__name__} does not inherit from AbstractEmbeddingGenerator"
            )

    def test_discovered_classes_list(self):
        """Print discovered classes for debugging purposes."""
        class_names = [cls.__name__ for cls in CONCRETE_EMBEDDING_GENERATORS]

        logger.info(f"Discovered embedding generators: {class_names}")
        assert len(class_names) > 0


class TestAbstractEmbeddingGeneratorContract:
    """Contract tests that verify behavioral requirements of AbstractEmbeddingGenerator.

    These tests run against every concrete implementation to ensure they fulfill
    the contract defined by the abstract base class.
    """

    rng = np.random.default_rng(42)

    def test_instantiation(self, embedding_generator_instance):
        """Test that the embedding generator can be instantiated."""
        assert embedding_generator_instance is not None
        assert isinstance(embedding_generator_instance, AbstractEmbeddingGenerator)

    def test_has_name(self, embedding_generator_instance):
        """Test that the embedding generator has a name property."""
        name = embedding_generator_instance.name
        assert name is not None
        assert isinstance(name, str)
        assert len(name) > 0

    def test_name_setter(self, embedding_generator_instance):
        """Test that the name property can be set."""
        original_name = embedding_generator_instance.name
        new_name = "TestName"
        embedding_generator_instance.name = new_name
        assert embedding_generator_instance.name == new_name
        # Restore original name
        embedding_generator_instance.name = original_name

    def test_has_is_end_to_end_model_property(self, embedding_generator_instance):
        """Test that the embedding generator has is_end_to_end_model property."""
        is_end_to_end = embedding_generator_instance.is_end_to_end_model
        assert isinstance(is_end_to_end, bool)

    def test_has_max_num_samples_attribute(self, embedding_generator_instance):
        """Test that the embedding generator has max_num_samples attribute."""
        max_samples = embedding_generator_instance.max_num_samples
        assert max_samples is not None
        assert isinstance(max_samples, int)
        assert max_samples > 0

    def test_has_max_num_features_attribute(self, embedding_generator_instance):
        """Test that the embedding generator has max_num_features attribute."""
        max_features = embedding_generator_instance.max_num_features
        assert max_features is not None
        assert isinstance(max_features, int)
        assert max_features > 0

    def test_check_dataset_constraints_valid(self, embedding_generator_instance):
        """Test check_dataset_constraints returns True for valid constraints."""
        # Use values well within typical limits
        result = embedding_generator_instance.check_dataset_constraints(
            num_samples=100, num_features=10
        )
        assert result is True

    def test_check_dataset_constraints_exceeds_samples(
        self, embedding_generator_instance
    ):
        """Test check_dataset_constraints returns False when samples exceed limit."""
        max_samples = embedding_generator_instance.max_num_samples
        result = embedding_generator_instance.check_dataset_constraints(
            num_samples=max_samples + 1, num_features=10
        )
        assert result is False

    def test_check_dataset_constraints_exceeds_features(
        self, embedding_generator_instance
    ):
        """Test check_dataset_constraints returns False when features exceed limit."""
        max_features = embedding_generator_instance.max_num_features
        result = embedding_generator_instance.check_dataset_constraints(
            num_samples=100, num_features=max_features + 1
        )
        assert result is False

    def test_check_emb_shape_valid(self, embedding_generator_instance):
        """Test _check_emb_shape returns True for valid 2D embeddings."""
        valid_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5)
        )
        result = AbstractEmbeddingGenerator._check_emb_shape(valid_embeddings)
        assert result is True

    def test_check_emb_shape_invalid_1d(self, embedding_generator_instance):
        """Test _check_emb_shape returns False for 1D array."""
        invalid_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10,)
        )
        result = AbstractEmbeddingGenerator._check_emb_shape(invalid_embeddings)
        assert result is False

    def test_check_emb_shape_invalid_3d(self, embedding_generator_instance):
        """Test _check_emb_shape returns False for 3D array."""
        invalid_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5, 3)
        )
        result = AbstractEmbeddingGenerator._check_emb_shape(invalid_embeddings)
        assert result is False

    def test_check_nan_no_nans(self, embedding_generator_instance):
        """Test _check_nan returns True when no NaN values present."""
        clean_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5)
        )
        result = AbstractEmbeddingGenerator._check_nan(clean_embeddings)
        assert result is True

    def test_check_nan_with_nans(self, embedding_generator_instance):
        """Test _check_nan returns False when NaN values present."""
        nan_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5)
        )
        nan_embeddings[0, 0] = np.nan
        result = AbstractEmbeddingGenerator._check_nan(nan_embeddings)
        assert result is False

    def test_validate_embeddings_valid(self, embedding_generator_instance):
        """Test _validate_embeddings returns True for valid embeddings."""
        valid_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5)
        )
        result = AbstractEmbeddingGenerator._validate_embeddings(valid_embeddings)
        assert result is True

    def test_validate_embeddings_invalid_shape(self, embedding_generator_instance):
        """Test _validate_embeddings returns False for invalid shape."""
        invalid_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10,)
        )
        result = AbstractEmbeddingGenerator._validate_embeddings(invalid_embeddings)
        assert result is False

    def test_validate_embeddings_with_nans(self, embedding_generator_instance):
        """Test _validate_embeddings returns False when NaN values present."""
        nan_embeddings = TestAbstractEmbeddingGeneratorContract.rng.standard_normal(
            size=(10, 5)
        )
        nan_embeddings[0, 0] = np.nan
        result = AbstractEmbeddingGenerator._validate_embeddings(nan_embeddings)
        assert result is False

    def test_end_to_end_compatible_tasks_property(self, embedding_generator_instance):
        """Test end_to_end_compatible_tasks property getter."""
        tasks = embedding_generator_instance.end_to_end_compatible_tasks
        # Can be None or a list
        assert tasks is None or isinstance(tasks, list)

    def test_end_to_end_compatible_tasks_setter(self, embedding_generator_instance):
        """Test end_to_end_compatible_tasks property setter."""
        original_tasks = embedding_generator_instance.end_to_end_compatible_tasks
        new_tasks = ["classification", "regression"]
        embedding_generator_instance.end_to_end_compatible_tasks = new_tasks
        assert embedding_generator_instance.end_to_end_compatible_tasks == new_tasks
        # Restore original
        embedding_generator_instance.end_to_end_compatible_tasks = original_tasks


class TestAbstractEmbeddingGeneratorAbstractMethods:
    """Tests that verify abstract methods are properly implemented.

    These tests check that the abstract methods exist and have the expected signatures.
    """

    def test_has_preprocess_data_method(self, embedding_generator_instance):
        """Test that _preprocess_data method exists."""
        assert hasattr(embedding_generator_instance, "_preprocess_data")
        assert callable(getattr(embedding_generator_instance, "_preprocess_data"))

    def test_has_fit_model_method(self, embedding_generator_instance):
        """Test that _fit_model method exists."""
        assert hasattr(embedding_generator_instance, "_fit_model")
        assert callable(getattr(embedding_generator_instance, "_fit_model"))

    def test_has_compute_embeddings_method(self, embedding_generator_instance):
        """Test that _compute_embeddings method exists."""
        assert hasattr(embedding_generator_instance, "_compute_embeddings")
        assert callable(getattr(embedding_generator_instance, "_compute_embeddings"))

    def test_has_reset_embedding_model_method(self, embedding_generator_instance):
        """Test that _reset_embedding_model method exists."""
        assert hasattr(embedding_generator_instance, "_reset_embedding_model")
        assert callable(getattr(embedding_generator_instance, "_reset_embedding_model"))

    def test_has_preprocess_data_public_method(self, embedding_generator_instance):
        """Test that preprocess_data public method exists."""
        assert hasattr(embedding_generator_instance, "preprocess_data")
        assert callable(getattr(embedding_generator_instance, "preprocess_data"))

    def test_has_generate_embeddings_method(self, embedding_generator_instance):
        """Test that generate_embeddings method exists."""
        assert hasattr(embedding_generator_instance, "generate_embeddings")
        assert callable(getattr(embedding_generator_instance, "generate_embeddings"))

    def test_has_get_end_to_end_prediction_method(self, embedding_generator_instance):
        """Test that get_end_to_end_prediction method exists."""
        assert hasattr(embedding_generator_instance, "get_end_to_end_prediction")
        assert callable(
            getattr(embedding_generator_instance, "get_end_to_end_prediction")
        )


class TestAbstractEmbeddingGeneratorEdgeCases:
    """Edge case tests for AbstractEmbeddingGenerator implementations."""

    rng = np.random.default_rng(42)

    def test_check_dataset_constraints_boundary_samples(
        self, embedding_generator_instance
    ):
        """Test check_dataset_constraints at exact sample boundary."""
        max_samples = embedding_generator_instance.max_num_samples
        result = embedding_generator_instance.check_dataset_constraints(
            num_samples=max_samples, num_features=10
        )
        assert result is True

    def test_check_dataset_constraints_boundary_features(
        self, embedding_generator_instance
    ):
        """Test check_dataset_constraints at exact feature boundary."""
        max_features = embedding_generator_instance.max_num_features
        result = embedding_generator_instance.check_dataset_constraints(
            num_samples=100, num_features=max_features
        )
        assert result is True

    def test_check_emb_shape_empty_array(self, embedding_generator_instance):
        """Test _check_emb_shape with empty 2D array."""
        empty_embeddings = np.array([]).reshape(0, 5)
        result = AbstractEmbeddingGenerator._check_emb_shape(empty_embeddings)
        assert result is True  # Shape is still 2D

    def test_check_nan_empty_array(self, embedding_generator_instance):
        """Test _check_nan with empty array."""
        empty_embeddings = np.array([]).reshape(0, 5)
        result = AbstractEmbeddingGenerator._check_nan(empty_embeddings)
        assert result is True  # No NaN in empty array

    def test_validate_embeddings_single_row(self, embedding_generator_instance):
        """Test _validate_embeddings with single row."""
        single_row = TestAbstractEmbeddingGeneratorEdgeCases.rng.standard_normal(
            size=(1, 5)
        )
        result = AbstractEmbeddingGenerator._validate_embeddings(single_row)
        assert result is True

    def test_validate_embeddings_single_column(self, embedding_generator_instance):
        """Test _validate_embeddings with single column."""
        single_col = TestAbstractEmbeddingGeneratorEdgeCases.rng.standard_normal(
            size=(10, 1)
        )
        result = AbstractEmbeddingGenerator._validate_embeddings(single_col)
        assert result is True

    def test_get_end_to_end_prediction_not_end_to_end_model(
        self, embedding_generator_instance
    ):
        """Test get_end_to_end_prediction raises error for non-end-to-end models."""
        if not embedding_generator_instance.is_end_to_end_model:
            rng = np.random.default_rng(42)
            X_train = rng.standard_normal(size=(50, 10))
            y_train = rng.integers(0, 2, size=50)
            X_test = rng.standard_normal(size=(20, 10))

            with pytest.raises(NotImplementedError):
                embedding_generator_instance.get_end_to_end_prediction(
                    X_train, y_train, X_test
                )

    def test_compute_embeddings_for_non_end_to_end_models(
        self, embedding_generator_instance
    ):
        """Test that compute_embeddings works correctly for non-end-to-end models.

        This test ensures that models which are not end-to-end can actually
        generate embeddings. This catches issues like TabPFNEmbeddingConstantVectorRegression
        which may throw errors during embedding computation.
        """
        if not embedding_generator_instance.is_end_to_end_model:
            rng = np.random.default_rng(42)
            X_train = rng.standard_normal(size=(50, 10))
            X_test = rng.standard_normal(size=(20, 10))

            # This should not raise any errors for properly implemented models
            train_embeddings, test_embeddings, embedding_metadata = (
                embedding_generator_instance.generate_embeddings(X_train, X_test)
            )

            # Verify embeddings are returned with correct shapes
            assert train_embeddings is not None
            assert test_embeddings is not None
            assert len(train_embeddings.shape) == 2
            assert isinstance(train_embeddings, np.ndarray)
            assert isinstance(test_embeddings, np.ndarray)
            assert train_embeddings.shape[0] == X_train.shape[0]
            assert test_embeddings.shape[0] == X_test.shape[0]
            assert isinstance(embedding_metadata, dict)

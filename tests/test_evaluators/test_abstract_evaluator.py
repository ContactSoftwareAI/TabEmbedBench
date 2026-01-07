"""Contract tests for AbstractHPOEvaluator implementations.

This module provides automatic discovery and testing of all concrete implementations
of AbstractHPOEvaluator. Tests are automatically run against every implementation
without requiring manual test class creation for each one.

When a new implementation is added to the evaluators package, it will automatically
be discovered and tested on the next test run.
"""

import numpy as np
import pytest

# Import the evaluators package to ensure all implementations are loaded into memory
# This is critical - __subclasses__() only returns classes that have been imported
import tabembedbench.evaluators  # noqa: F401
from tabembedbench.evaluators import AbstractHPOEvaluator


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


# Discover all concrete implementations of AbstractHPOEvaluator
CONCRETE_HPO_EVALUATORS = get_all_concrete_subclasses(AbstractHPOEvaluator)


def get_target_evaluator_classes(request) -> list[type]:
    """Helper to get evaluator classes based on --evaluator-class option.

    Returns:
        List containing either all evaluators or a single specific evaluator.

    Raises:
        ValueError: If the specified class name is not found.
    """
    evaluator_class_name = request.config.getoption("--evaluator-class", default=None)

    if not evaluator_class_name:
        # No specific class provided, use all
        return CONCRETE_HPO_EVALUATORS

    # Find the matching class
    matching_classes = [
        cls for cls in CONCRETE_HPO_EVALUATORS if cls.__name__ == evaluator_class_name
    ]

    if not matching_classes:
        available = [cls.__name__ for cls in CONCRETE_HPO_EVALUATORS]
        raise ValueError(
            f"Evaluator class '{evaluator_class_name}' not found. "
            f"Available classes: {available}"
        )

    return matching_classes


def pytest_generate_tests(metafunc):
    """Dynamic parametrization to filter evaluators before they reach the fixture."""
    if "hpo_evaluator_instance" in metafunc.fixturenames:
        evaluator_class_name = metafunc.config.getoption(
            "--evaluator-class", default=None
        )

        if evaluator_class_name:
            matching_classes = [
                cls
                for cls in CONCRETE_HPO_EVALUATORS
                if cls.__name__ == evaluator_class_name
            ]
            if not matching_classes:
                available = [cls.__name__ for cls in CONCRETE_HPO_EVALUATORS]
                pytest.exit(
                    f"Evaluator class '{evaluator_class_name}' not found. Available: {available}"
                )

            metafunc.parametrize(
                "hpo_evaluator_instance",
                matching_classes,
                ids=lambda cls: cls.__name__,
                indirect=True,
            )
        else:
            metafunc.parametrize(
                "hpo_evaluator_instance",
                CONCRETE_HPO_EVALUATORS,
                ids=lambda cls: cls.__name__,
                indirect=True,
            )


@pytest.fixture
def hpo_evaluator_instance(request):
    """
    Fixture to provide an instance of the Hyperparameter Optimizer (HPO) evaluator
    for parameterized testing. This fixture allows testing various implementations
    of HPO evaluators with minimal settings for efficient unit tests.

    Args:
        request: A pytest fixture that provides the parameterized evaluator class
            to be tested.

    Returns:
        An instance of the parameterized HPO evaluator class initialized with
        minimal required settings: 1 trial, 2 cross-validation folds, random state
        of 42 for reproducibility, and verbose output set to False.
    """
    evaluator_class = request.param
    # Instantiate with minimal settings for fast testing
    return evaluator_class(n_trials=1, cv_folds=2, random_state=42, verbose=False)


@pytest.fixture
def sample_classification_data():
    """Fixture providing sample classification data for testing.

    Returns:
        Tuple of (embeddings, labels) suitable for classification tasks.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    embeddings = np.random.randn(n_samples, n_features)
    # Binary classification labels
    labels = np.random.randint(0, 2, n_samples)
    return embeddings, labels


@pytest.fixture
def sample_regression_data():
    """Fixture providing sample regression data for testing.

    Returns:
        Tuple of (embeddings, targets) suitable for regression tasks.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    embeddings = np.random.randn(n_samples, n_features)
    # Continuous regression targets
    targets = np.random.randn(n_samples)
    return embeddings, targets


class TestAbstractHPOEvaluatorDiscovery:
    """Tests for the automatic discovery mechanism."""

    def test_concrete_implementations_discovered(self):
        """Verify that concrete implementations are discovered."""
        assert len(CONCRETE_HPO_EVALUATORS) > 0, (
            "No concrete implementations of AbstractHPOEvaluator were discovered. "
            "Ensure implementation modules are imported in the evaluators package."
        )

    def test_all_discovered_classes_are_concrete(self):
        """Verify that all discovered classes have no abstract methods."""
        for cls in CONCRETE_HPO_EVALUATORS:
            abstract_methods = getattr(cls, "__abstractmethods__", set())
            assert not abstract_methods, (
                f"{cls.__name__} has abstract methods: {abstract_methods}"
            )

    def test_all_discovered_classes_inherit_from_abstract_hpo_evaluator(self):
        """Verify that all discovered classes inherit from AbstractHPOEvaluator."""
        for cls in CONCRETE_HPO_EVALUATORS:
            assert issubclass(cls, AbstractHPOEvaluator), (
                f"{cls.__name__} does not inherit from AbstractHPOEvaluator"
            )


class TestAbstractHPOEvaluatorContract:
    """Contract tests that verify behavioral requirements of AbstractHPOEvaluator.

    These tests run against every concrete implementation to ensure they fulfill
    the contract defined by the abstract base class.
    """

    def test_instantiation(self, hpo_evaluator_instance):
        """Test that the evaluator can be instantiated."""
        assert hpo_evaluator_instance is not None
        assert isinstance(hpo_evaluator_instance, AbstractHPOEvaluator)

    def test_has_name(self, hpo_evaluator_instance):
        """Test that the evaluator has a name property."""
        name = hpo_evaluator_instance.name
        assert name is not None
        assert isinstance(name, str)
        assert len(name) > 0

    def test_has_task_type(self, hpo_evaluator_instance):
        """Test that the evaluator has a task_type attribute."""
        task_type = hpo_evaluator_instance.task_type
        assert task_type is not None
        assert isinstance(task_type, list)
        assert len(task_type) > 0

    def test_get_scoring_metric_returns_string(self, hpo_evaluator_instance):
        """Test that get_scoring_metric returns a valid sklearn scoring string."""
        metric = hpo_evaluator_instance.get_scoring_metric()
        assert metric is not None
        assert isinstance(metric, str)
        assert len(metric) > 0

    def test_get_search_space_returns_dict(self, hpo_evaluator_instance):
        """Test that _get_search_space returns a properly structured dictionary."""
        search_space = hpo_evaluator_instance._get_search_space()
        assert search_space is not None
        assert isinstance(search_space, dict)
        assert len(search_space) > 0

        # Verify each parameter in search space has required structure
        for param_name, config in search_space.items():
            assert isinstance(param_name, str), (
                f"Parameter name must be string: {param_name}"
            )
            assert isinstance(config, dict), f"Config for {param_name} must be dict"
            assert "type" in config, f"Config for {param_name} must have 'type' key"

    def test_get_parameters_returns_dict(self, hpo_evaluator_instance):
        """Test that get_parameters returns a dictionary."""
        params = hpo_evaluator_instance.get_parameters()
        assert params is not None
        assert isinstance(params, dict)

    def test_get_parameters_contains_expected_keys(self, hpo_evaluator_instance):
        """Test that get_parameters contains expected configuration keys."""
        params = hpo_evaluator_instance.get_parameters()
        expected_keys = ["n_trials", "cv_folds", "random_state", "sampler", "pruner"]
        for key in expected_keys:
            assert key in params, f"Expected key '{key}' not found in parameters"

    def test_initial_state_has_no_best_params(self, hpo_evaluator_instance):
        """Test that a fresh evaluator has no best parameters."""
        assert hpo_evaluator_instance.best_params is None
        assert hpo_evaluator_instance.best_score is None
        assert hpo_evaluator_instance.best_model is None
        assert hpo_evaluator_instance.study is None

    def test_reset_evaluator_clears_state(self, hpo_evaluator_instance):
        """Test that reset_evaluator clears all optimization state."""
        # Manually set some state
        hpo_evaluator_instance.best_params = {"test": 1}
        hpo_evaluator_instance.best_score = 0.9
        hpo_evaluator_instance.best_model = "dummy"
        hpo_evaluator_instance.study = "dummy_study"

        # Reset
        hpo_evaluator_instance.reset_evaluator()

        # Verify state is cleared
        assert hpo_evaluator_instance.best_params is None
        assert hpo_evaluator_instance.best_score is None
        assert hpo_evaluator_instance.best_model is None
        assert hpo_evaluator_instance.study is None

    def test_model_class_is_defined(self, hpo_evaluator_instance):
        """Test that the evaluator has a model_class defined."""
        # model_class should be defined either on the class or instance
        model_class = hpo_evaluator_instance.model_class
        assert model_class is not None, (
            f"{hpo_evaluator_instance.__class__.__name__} must define model_class"
        )

    def test_get_task_returns_task_type(self, hpo_evaluator_instance):
        """Test that get_task returns the task type."""
        task = hpo_evaluator_instance.get_task()
        assert task == hpo_evaluator_instance.task_type

    def test_check_task_type_with_valid_task(self, hpo_evaluator_instance):
        """Test check_task_type returns True for valid task."""
        valid_task = hpo_evaluator_instance.task_type[0]
        assert hpo_evaluator_instance.check_task_type(valid_task) is True

    def test_check_task_type_with_invalid_task(self, hpo_evaluator_instance):
        """Test check_task_type returns False for invalid task."""
        assert hpo_evaluator_instance.check_task_type("invalid_task_type_xyz") is False

    def test_get_prediction_requires_y_for_training(self, hpo_evaluator_instance):
        """Test that get_prediction raises ValueError when y is None during training."""
        embeddings = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="y must be provided"):
            hpo_evaluator_instance.get_prediction(embeddings, y=None, train=True)

    def test_get_prediction_requires_trained_model_for_inference(
        self, hpo_evaluator_instance
    ):
        """Test that get_prediction raises ValueError when no model is trained for inference."""
        embeddings = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="No trained model"):
            hpo_evaluator_instance.get_prediction(embeddings, y=None, train=False)

    def test_fit_best_model_requires_optimization(self, hpo_evaluator_instance):
        """Test that fit_best_model raises ValueError if optimization hasn't been run."""
        embeddings = np.random.randn(10, 5)
        labels = np.random.randint(0, 2, 10)
        with pytest.raises(ValueError, match="No best parameters"):
            hpo_evaluator_instance.fit_best_model(embeddings, labels)

    def test_get_optimization_history_returns_none_before_optimization(
        self, hpo_evaluator_instance
    ):
        """Test that get_optimization_history returns None before optimization."""
        history = hpo_evaluator_instance.get_optimization_history()
        assert history is None


class TestAbstractHPOEvaluatorIntegration:
    """Integration tests that verify the full HPO pipeline works correctly.

    These tests are more expensive as they run actual optimization, so they
    use minimal settings (n_trials=1, cv_folds=2).
    """

    def _get_appropriate_data(self, evaluator, classification_data, regression_data):
        """Get appropriate test data based on evaluator's task type."""
        task_types = evaluator.task_type
        # Check if any task type indicates regression
        is_regression = any("regression" in t.lower() for t in task_types)
        return regression_data if is_regression else classification_data

    def test_full_training_pipeline(
        self,
        hpo_evaluator_instance,
        sample_classification_data,
        sample_regression_data,
    ):
        """Test the complete training pipeline: optimize -> fit -> predict."""
        embeddings, labels = self._get_appropriate_data(
            hpo_evaluator_instance,
            sample_classification_data,
            sample_regression_data,
        )

        # Run the full pipeline via get_prediction with train=True
        predictions, additional_info = hpo_evaluator_instance.get_prediction(
            embeddings, y=labels, train=True
        )

        # Verify predictions
        assert predictions is not None
        assert len(predictions) == len(labels)

        # Verify additional info contains expected keys
        assert additional_info is not None
        assert "best_params" in additional_info
        assert "best_score" in additional_info
        assert "n_trials" in additional_info

        # Verify state is updated
        assert hpo_evaluator_instance.best_params is not None
        assert hpo_evaluator_instance.best_score is not None
        assert hpo_evaluator_instance.best_model is not None
        assert hpo_evaluator_instance.study is not None

    def test_inference_after_training(
        self,
        hpo_evaluator_instance,
        sample_classification_data,
        sample_regression_data,
    ):
        """Test that inference works after training."""
        embeddings, labels = self._get_appropriate_data(
            hpo_evaluator_instance,
            sample_classification_data,
            sample_regression_data,
        )

        # Train first
        hpo_evaluator_instance.get_prediction(embeddings, y=labels, train=True)

        # Now do inference
        predictions, additional_info = hpo_evaluator_instance.get_prediction(
            embeddings, y=None, train=False
        )

        assert predictions is not None
        assert len(predictions) == len(labels)
        assert additional_info is None  # No additional info for inference

    def test_optimization_history_after_training(
        self,
        hpo_evaluator_instance,
        sample_classification_data,
        sample_regression_data,
    ):
        """Test that optimization history is available after training."""
        embeddings, labels = self._get_appropriate_data(
            hpo_evaluator_instance,
            sample_classification_data,
            sample_regression_data,
        )

        # Train
        hpo_evaluator_instance.get_prediction(embeddings, y=labels, train=True)

        # Get history
        history = hpo_evaluator_instance.get_optimization_history()

        assert history is not None
        assert isinstance(history, list)
        assert len(history) > 0

        # Verify history structure
        for trial_info in history:
            assert "number" in trial_info
            assert "value" in trial_info
            assert "params" in trial_info
            assert "state" in trial_info

    def test_get_parameters_after_optimization(
        self,
        hpo_evaluator_instance,
        sample_classification_data,
        sample_regression_data,
    ):
        """Test that get_parameters includes best params after optimization."""
        embeddings, labels = self._get_appropriate_data(
            hpo_evaluator_instance,
            sample_classification_data,
            sample_regression_data,
        )

        # Train
        hpo_evaluator_instance.get_prediction(embeddings, y=labels, train=True)

        # Get parameters
        params = hpo_evaluator_instance.get_parameters()

        # Should include best_cv_score after optimization
        assert "best_cv_score" in params

    def test_reset_after_training(
        self,
        hpo_evaluator_instance,
        sample_classification_data,
        sample_regression_data,
    ):
        """Test that reset_evaluator properly clears state after training."""
        embeddings, labels = self._get_appropriate_data(
            hpo_evaluator_instance,
            sample_classification_data,
            sample_regression_data,
        )

        # Train
        hpo_evaluator_instance.get_prediction(embeddings, y=labels, train=True)

        # Verify state exists
        assert hpo_evaluator_instance.best_model is not None

        # Reset
        hpo_evaluator_instance.reset_evaluator()

        # Verify state is cleared
        assert hpo_evaluator_instance.best_params is None
        assert hpo_evaluator_instance.best_score is None
        assert hpo_evaluator_instance.best_model is None
        assert hpo_evaluator_instance.study is None

        # Verify inference fails after reset
        with pytest.raises(ValueError, match="No trained model"):
            hpo_evaluator_instance.get_prediction(embeddings, y=None, train=False)

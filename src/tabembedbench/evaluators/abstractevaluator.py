from abc import ABC, abstractmethod

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score


class AbstractEvaluator(ABC):
    """Abstract base class for model evaluators.

    This class defines the interface for evaluators that make predictions
    on embeddings and manage evaluation parameters.

    Attributes:
        _name (str): Name of the evaluator.
        task_type (str): Type of task (e.g., 'classification', 'regression').
    """

    def __init__(self, name: str, task_type: str | list[str]):
        """Initialize the AbstractEvaluator.

        Args:
            name (str): Name of the evaluator. If None, uses class name.
            task_type (str): Type of task for this evaluator.
        """
        self._name = name or self.__class__.__name__
        self.task_type = task_type if isinstance(task_type, list) else [task_type]

    @property
    def name(self) -> str:
        """Gets the name attribute value.

        This property retrieves the value of the `_name` attribute,
        which represents the name associated with the instance.

        Returns:
            str: The name associated with the instance.
        """
        return self._name

    @abstractmethod
    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ) -> tuple:
        """Get predictions from the evaluator.

        Args:
            embeddings (np.ndarray): Input embeddings for prediction.
            y (np.ndarray | None, optional): Target labels for training. Defaults to None.
            train (bool, optional): Whether to train the model. Defaults to True.

        Returns:
            tuple: Predictions and additional information.
        """
        pass

    @abstractmethod
    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        This method should clear any trained models, cached results, or
        optimization history.
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing evaluator parameters.
        """
        pass

    def get_task(self):
        """Get the task type of the evaluator.

        Returns:
            str: The task type (e.g., 'classification', 'regression').
        """
        return self.task_type

    def check_task_type(self, task: str):
        return task in self.task_type


class AbstractHPOEvaluator(AbstractEvaluator):
    """Abstract base class for evaluators with Optuna hyperparameter optimization.

    This class provides a framework for hyperparameter optimization using Optuna,
    including methods for defining search spaces, running optimization, and
    managing the best model found during optimization.

    Attributes:
        n_trials (int): Number of optimization trials to run.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print optimization progress.
        sampler (optuna.samplers.BaseSampler): Optuna sampler for hyperparameter search.
        pruner (optuna.pruners.BasePruner): Optuna pruner for early stopping.
        study (optuna.Study | None): Optuna study object containing optimization results.
        best_params (dict | None): Best hyperparameters found during optimization.
        best_score (float | None): Best cross-validation score achieved.
        best_model: Best model trained with optimal hyperparameters.
    """

    model_class = None

    def __init__(
        self,
        name: str,
        task_type: str | list[str],
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        optuna_sampler: optuna.samplers.BaseSampler | None = None,
        optuna_pruner: optuna.pruners.BasePruner | None = None,
        verbose: bool = False,
    ):
        """Initialize the AbstractHPOEvaluator.

        Args:
            name (str): Name of the evaluator.
            task_type (str): Type of task for this evaluator.
            n_trials (int, optional): Number of optimization trials. Defaults to 50.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            optuna_sampler (optuna.samplers.BaseSampler | None, optional): Custom Optuna sampler.
                If None, uses TPESampler. Defaults to None.
            optuna_pruner (optuna.pruners.BasePruner | None, optional): Custom Optuna pruner.
                If None, uses MedianPruner. Defaults to None.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.
        """
        super().__init__(name=name, task_type=task_type)

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        # Set default sampler and pruner if not provided
        self.sampler = optuna_sampler or optuna.samplers.TPESampler(seed=random_state)
        self.pruner = optuna_pruner or optuna.pruners.MedianPruner()

        # Store optimization results
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None

    # ========== Abstract Methods (Subclasses must implement) ==========
    @abstractmethod
    def get_scoring_metric(self) -> dict[str, str]:
        """Get the scoring metric for cross-validation.

        Returns:
            dict[str, str]: Dictionary mapping metric name to sklearn scoring string.
        """
        pass

    @abstractmethod
    def _get_search_space(self) -> dict[str, optuna.search_space]:
        """Get the search space for hyperparameter optimization.

        Returns:
            dict[str, optuna.search_space]: Dictionary
                mapping hyperparameter names to Optuna search spaces.
        """
        pass

    @abstractmethod
    def _get_model_predictions(self, model, embeddings: np.ndarray):
        """Get predictions from a trained model.

        Args:
            model: Trained model instance.
            embeddings (np.ndarray): Input embeddings for prediction.

        Returns:
            Model predictions on the embeddings.
        """
        pass

    # ========== Concrete Methods (Implemented in base class) ==========
    def create_model(self, trial: optuna.Trial, **kwargs):
        """Create a model with hyperparameters suggested by Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.

        Returns:
            Model instance with trial-suggested hyperparameters.
        """
        if self.model_class is None:
            raise NotImplementedError(
                "Subclasses must define model_class or override create_model."
            )

        search_space = self._get_search_space()
        params = kwargs.copy()

        suggestion_methods = {
            "int": lambda name, cfg: trial.suggest_int(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            ),
            "float": lambda name, cfg: trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            ),
            "categorical": lambda name, cfg: trial.suggest_categorical(
                name, cfg["choices"]
            ),
            "int_sequence": lambda name, cfg: tuple(
                trial.suggest_int(
                    f"{name}_{i}", cfg["low"], cfg["high"], log=cfg.get("log", False)
                )
                for i in range(params[cfg.get("length_param")])
            ),
        }

        for param_name in sorted(
            search_space.keys(),
            key=lambda x: search_space[x].get("type") == "int_sequence",
        ):
            config = search_space.get(param_name)
            param_type = config.get("type")
            if param_type not in suggestion_methods:
                raise TypeError(
                    f"Unsupported hyperparameter type '{param_type}' for '{param_name}'. "
                    f"Supported types: {list(suggestion_methods.keys())}"
                )

            params[param_name] = suggestion_methods[param_type](param_name, config)

        return self.model_class(**params)

    def objective(
        self, trial: optuna.Trial, embeddings: np.ndarray, y: np.ndarray
    ) -> float:
        """Optuna objective function for hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.
            embeddings (np.ndarray): Input embeddings for training.
            y (np.ndarray): Target labels.

        Returns:
            float: Mean cross-validation score.
        """
        model = self.create_model(trial)

        scoring = self.get_scoring_metric()

        scores = cross_val_score(
            model,
            embeddings,
            y,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=1,
        )

        return scores.mean()

    def optimize_hyperparameters(
        self,
        embeddings: np.ndarray,
        y: np.ndarray,
        study_name: str | None = None,
    ) -> dict:
        """Optimize hyperparameters using Optuna.

        Args:
            embeddings (np.ndarray): Input embeddings for training.
            y (np.ndarray): Target labels.
            study_name (str | None, optional): Name for the Optuna study.
                If None, uses evaluator name. Defaults to None.

        Returns:
            dict: Best hyperparameters found during optimization.
        """
        # Create study
        study_name = study_name or f"{self._name}_optimization"

        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # Run optimization
        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            lambda trial: self.objective(trial, embeddings, y),
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        if self.verbose:
            print(f"Best score: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")

        return self.best_params

    def fit_best_model(
        self, embeddings: np.ndarray, y: np.ndarray, **additional_parameters
    ):
        """Fit a model using the best hyperparameters found during optimization.

        Args:
            embeddings (np.ndarray): Input embeddings for training.
            y (np.ndarray): Target labels.

        Raises:
            ValueError: If no best parameters are available (optimization not run).
        """
        if self.best_params is None:
            raise ValueError(
                "No best parameters found. Run optimize_hyperparameters first."
            )
        best_model_params = self.best_params
        best_model_params.update(additional_parameters)
        # Create a trial with best parameters for model creation
        # We use a frozen trial to avoid suggesting new parameters
        frozen_trial = optuna.trial.FixedTrial(best_model_params)
        self.best_model = self.create_model(frozen_trial)

        # Fit the model
        self.best_model.fit(embeddings, y)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
        **kwargs,
    ) -> tuple:
        """Get predictions from the evaluator.

        If train=True, performs hyperparameter optimization, trains the best model,
        and returns predictions with optimization info. If train=False, uses the
        previously trained model for predictions.

        Args:
            embeddings (np.ndarray): Input embeddings for prediction.
            y (np.ndarray | None, optional): Target labels for training. Required if train=True.
                Defaults to None.
            train (bool, optional): Whether to train the model. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions: Model predictions on the embeddings.
                - additional_info (dict | None): Dictionary with optimization info if train=True,
                    None otherwise.

        Raises:
            ValueError: If train=True and y is None, or if train=False and no model is trained.
        """
        if train:
            if y is None:
                raise ValueError("y must be provided for training")

            # Optimize hyperparameters
            self.optimize_hyperparameters(embeddings, y)

            # Fit best model
            self.fit_best_model(embeddings, y, **kwargs)

            # Get predictions
            predictions = self._get_model_predictions(self.best_model, embeddings)

            # Return predictions and optimization info
            additional_info = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "n_trials": len(self.study.trials),
            }

            return predictions, additional_info
        if self.best_model is None:
            raise ValueError(
                "No trained model found. Set train=True to train the model first."
            )

        predictions = self._get_model_predictions(self.best_model, embeddings)
        return predictions, None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Clears the optimization study, best parameters, best score, and trained model.
        """
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None

    def get_parameters(self) -> dict:
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing evaluator configuration and optimization results.
                Always includes n_trials, cv_folds, random_state, sampler, and pruner.
                If optimization has been run, also includes the best hyperparameters
                (flattened with 'best_' prefix) and best_cv_score.
        """
        params = {
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
            "sampler": self.sampler.__class__.__name__,
            "pruner": self.pruner.__class__.__name__,
        }

        if self.best_params is not None:
            # Flatten best hyperparameters into individual parameters with 'best_' prefix
            # This avoids nested data structures in the result DataFrame
            for param_name, param_value in self.best_params.items():
                params[f"best_{param_name}"] = param_value
            params["best_cv_score"] = self.best_score

        return params

    def get_optimization_history(self) -> list[dict] | None:
        """Get the history of all optimization trials.

        Returns:
            list[dict] | None: List of dictionaries containing trial information
                (number, value, params, state) for each trial, or None if no
                optimization has been run.
        """
        if self.study is None:
            return None

        history = []
        for trial in self.study.trials:
            history.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )

        return history

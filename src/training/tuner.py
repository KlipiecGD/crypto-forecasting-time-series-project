import optuna
import pandas as pd
from typing import Optional

from mlflow.sklearn import autolog as mlflow_sklearn_autolog

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config.config import config
from src.logging_utils.loggers import training_logger as logger


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: Optional[int] = None,
) -> dict:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        n_trials (Optional[int]): Number of optimization trials. If None, uses default from config.

    Returns:
        dict: Best hyperparameters
    """
    n_trials = n_trials or config.get_pipeline_config.get("n_trials", 50)
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

    # Disable MLflow autologging for Optuna trials
    mlflow_sklearn_autolog(disable=True)

    def objective(trial) -> float:
        """Objective function for Optuna optimization."""
        # Define hyperparameter search space
        C = trial.suggest_float("C", 0.001, 5.0, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000, step=500)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])

        # Build pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver=solver,
                        class_weight="balanced",  
                        random_state=config.get_model_config.get("random_seed", 2137),
                    ),
                ),
            ]
        )

        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        f1_macro = f1_score(y_val, y_pred, average="macro")

        return float(f1_macro)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="lr_hyperopt",
    )

    # Optimize hyperparameters
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=-1)

    # Re-enable MLflow autologging after optimization
    mlflow_sklearn_autolog(silent=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val F1 macro: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")

    return study.best_params

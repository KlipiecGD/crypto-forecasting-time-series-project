import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from src.config.config import config
from src.logging_utils.loggers import training_logger as logger


def plot_learning_curve(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: Optional[int] = None
) -> str:
    """
    Generate and save learning curve using TimeSeriesSplit.

    Args:
        pipeline (Pipeline): Trained sklearn pipeline
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        cv_splits (int): Number of time series splits

    Returns:
        str: Path to saved plot
    """
    cv_splits = cv_splits or config.get_model_config.get("cv_splits", 5)
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    train_sizes = []
    train_scores = []
    val_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        # Clone and fit model on this fold
        model_clone = clone(pipeline)
        model_clone.fit(X_train_cv, y_train_cv)

        train_sizes.append(len(X_train_cv))
        train_scores.append(model_clone.score(X_train_cv, y_train_cv))
        val_scores.append(model_clone.score(X_val_cv, y_val_cv))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, "o-", label="Training score", linewidth=2)
    plt.plot(train_sizes, val_scores, "o-", label="Validation score", linewidth=2)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Learning Curve (TimeSeriesSplit)", fontsize=14)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = config.get_model_config.get("plot_path", "learning_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def build_pipeline(hyperparameters: dict) -> Pipeline:
    """
    Build the model training pipeline.

    Args:
        hyperparameters (dict): Hyperparameters for the model

    Returns:
        Pipeline: Scikit-learn pipeline with scaler and classifier
    """
    random_seed = config.get_model_config.get("random_seed", 2137)
    c_param = (
        hyperparameters["C"]
        if "C" in hyperparameters
        else config.get_hyperparameters_config.get("C", 0.1)
    )
    class_weight_param = (
        hyperparameters["class_weight"]
        if "class_weight" in hyperparameters
        else config.get_hyperparameters_config.get("class_weight", "balanced")
    )
    max_iter_param = (
        hyperparameters["max_iter"]
        if "max_iter" in hyperparameters
        else config.get_hyperparameters_config.get("max_iter", 1000)
    )
    solver_param = (
        hyperparameters["solver"]
        if "solver" in hyperparameters
        else config.get_hyperparameters_config.get("solver", "lbfgs")
    )

    logger.info(
        f"Building pipeline with hyperparameters: C={c_param}, class_weight={class_weight_param}, max_iter={max_iter_param}, solver={solver_param}"
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_param,
                    class_weight=class_weight_param,
                    max_iter=max_iter_param,
                    random_state=random_seed,
                    solver=solver_param,
                ),
            ),
        ]
    )

    return pipeline


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: dict
) -> Pipeline:
    """
    Train the volatility prediction model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        hyperparameters (dict): Hyperparameters for the model

    Returns:
        Pipeline: Trained pipeline
    """
    logger.info("Building training pipeline...")
    pipeline = build_pipeline(hyperparameters)

    # Log to mlflow
    # Most of the hyperparameters are automatically logged by mlflow_autolog in main pipeline, but we can log additional parameters
    mlflow.log_param("val_size", config.get_model_config.get("val_size"))
    mlflow.log_param("test_size", config.get_model_config.get("test_size"))

    logger.info("Fitting the model...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete")

    # Generate and log learning curve
    logger.info("Generating learning curve...")
    plot_path = plot_learning_curve(pipeline, X_train, y_train)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)  # Clean up local file

    return pipeline

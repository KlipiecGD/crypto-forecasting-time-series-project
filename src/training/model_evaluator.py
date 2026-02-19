import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.logging_utils.loggers import training_logger as logger
from src.config.config import config


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        dict: Dictionary of metric names and values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    class_names = config.get_model_config.get("class_names", ["Low", "Normal", "High"])
    for i, class_name in enumerate(class_names):
        metrics[f"f1_{class_name}"] = f1_per_class[i]
        metrics[f"precision_{class_name}"] = precision_per_class[i]
        metrics[f"recall_{class_name}"] = recall_per_class[i]

    return metrics


def display_metrics(metrics: dict[str, float], dataset_name: str = "Dataset") -> None:
    """
    Display metrics in a formatted way.

    Args:
        metrics (dict[str, float]): Dictionary of metrics
        dataset_name (str): Name of the dataset (e.g., "Train", "Test")
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{dataset_name} Metrics")
    logger.info(f"{'=' * 60}")

    # Overall metrics
    logger.info("Overall Metrics:")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision Macro:   {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall Macro:      {metrics['recall_macro']:.4f}")

    # Per-class metrics
    logger.info("\nPer-Class Metrics:")
    for class_name in config.get_model_config.get(
        "class_names", ["Low", "Normal", "High"]
    ):
        logger.info(f"  {class_name}:")
        logger.info(f"    F1:        {metrics[f'f1_{class_name}']:.4f}")
        logger.info(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
        logger.info(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}")

    logger.info(f"{'=' * 60}\n")


def display_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, dataset_name: str = "Dataset"
) -> None:
    """
    Display confusion matrix in a readable format.

    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        dataset_name (str): Name of the dataset
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = config.get_model_config.get("class_names", ["Low", "Normal", "High"])

    logger.info(f"\n{dataset_name} Confusion Matrix:")
    logger.info(f"{'':>12} " + " ".join([f"{name:>8}" for name in class_names]))
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i]:>12} " + " ".join([f"{val:>8}" for val in row]))
    logger.info("")


def log_confusion_matrix_mlflow(
    y_true: pd.Series, y_pred: np.ndarray, dataset_name: str
) -> None:
    """
    Log confusion matrix to MLflow as an artifact.

    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        dataset_name (str): Name of the dataset
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    class_names = config.get_model_config.get("class_names", ["Low", "Normal", "High"])

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Normalized Confusion Matrix - {dataset_name} Set")

    # Log figure to MLflow
    mlflow.log_figure(fig, f"{dataset_name.lower()}_confusion_matrix.png")

    # Close plot to free memory
    plt.close(fig)


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: Optional[bool] = None,
) -> dict[str, dict[str, float]]:
    """
    Evaluate the trained model on train and test sets.

    Args:
        pipeline (Pipeline): Trained pipeline
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        verbose (bool): Whether to print detailed metrics

    Returns:
        dict: Evaluation metrics for train, validation, and test sets
    """
    logger.info("Evaluating model performance...")

    verbose = (
        verbose
        if verbose is not None
        else config.get_pipeline_config.get("verbose", True)
    )

    # Make predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_val = pipeline.predict(X_val)
    y_pred_test = pipeline.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    # Log confusion matrices to MLflow. For training it is automatically logged by mlflow_autolog
    log_confusion_matrix_mlflow(y_val, y_pred_val, "Validation")
    log_confusion_matrix_mlflow(y_test, y_pred_test, "Test")

    # Log metrics to MLflow
    for metric_name, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", value)
    for metric_name, value in val_metrics.items():
        mlflow.log_metric(f"val_{metric_name}", value)
    for metric_name, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", value)

    if verbose:
        # Print training metrics
        display_metrics(train_metrics, "Training Set")
        display_confusion_matrix(y_train, y_pred_train, "Training Set")

        # Print validation metrics
        display_metrics(val_metrics, "Validation Set")
        display_confusion_matrix(y_val, y_pred_val, "Validation Set")

        # Print test metrics
        display_metrics(test_metrics, "Test Set")
        display_confusion_matrix(y_test, y_pred_test, "Test Set")
    else:
        # Just log key metrics
        logger.info(f"Training F1 Macro: {train_metrics['f1_macro']:.4f}")
        logger.info(f"Validation F1 Macro: {val_metrics['f1_macro']:.4f}")
        logger.info(f"Testing F1 Macro: {test_metrics['f1_macro']:.4f}")

    return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

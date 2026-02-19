import os
import mlflow
import pandas as pd
import numpy as np

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.config.config import config
from src.logging_utils.loggers import monitoring_logger as logger


def save_reference_data(
    X_train: pd.DataFrame, y_train: pd.Series, y_train_pred: np.ndarray
):
    """
    Save reference data for Evidently monitoring.
    Combines X_train, y_train, and y_train_pred into a single DataFrame and saves as CSV.

    Args:
        X_train: Training features
        y_train: True labels for training data
        y_train_pred: Model predictions for training data
    """
    ref_data = X_train.copy()
    ref_data["target"] = y_train.values
    ref_data["prediction"] = y_train_pred

    ref_path = config.get_monitoring_config.get(
        "reference_data_path", "data/reference_data.csv"
    )
    ref_data.to_csv(ref_path, index=False)
    logger.info(f"Reference data saved to {ref_path} for Evidently monitoring")


def load_reference_data() -> pd.DataFrame:
    """
    Load reference data saved during last training run.
    Contains X_train + y_train + y_train_pred.

    Returns:
        pd.DataFrame: Reference data for Evidently
    """
    ref_path = config.get_monitoring_config.get(
        "reference_data_path", "data/reference_data.csv"
    )
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference data not found at {ref_path}. "
            "Run training pipeline first to generate it."
        )
    return pd.read_csv(ref_path)


def run_evidently_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> tuple[bool, float]:
    """
    Run Evidently data drift and classification report.
    Saves HTML report as MLflow artifact.

    Args:
        reference_data: Training data with target and prediction columns
        current_data: Current evaluation features (X only)
        y_true: True labels for current data
        y_pred: Model predictions for current data

    Returns:
        tuple: (drift_detected, drift_share)
    """
    # Add real labels and predictions to current data
    current_with_preds = current_data.copy()
    current_with_preds["target"] = y_true.values
    current_with_preds["prediction"] = y_pred

    # Reference already has real target and prediction from training
    feature_cols = list(current_data.columns)

    # Define column mapping for Evidently
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=feature_cols,
    )

    # Create and run Evidently report
    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_with_preds,
        column_mapping=column_mapping,
    )

    # Extract drift results
    report_dict = report.as_dict()
    drift_result = report_dict["metrics"][0]["result"]
    drift_detected = drift_result["dataset_drift"]
    drift_share = drift_result["share_of_drifted_columns"]

    logger.info(f"Evidently drift share: {drift_share:.2%}")
    logger.info(f"Evidently dataset drift detected: {drift_detected}")

    # Save HTML report as MLflow artifact
    report_path = config.get_monitoring_config.get(
        "report_path", "evidently_report.html"
    )
    report.save_html(report_path)
    mlflow.log_artifact(report_path)

    # Clean up local report file
    os.remove(report_path)

    return drift_detected, drift_share

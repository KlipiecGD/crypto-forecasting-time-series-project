import pandas as pd
from datetime import datetime, timedelta
from typing import Any

from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model as mlflow_load_model

from src.fetch_data.live_data import fetch_live_data
from src.features.preprocess_data import preprocess_data
from src.features.data_transformer import DataTransformer

from src.config.config import config
from src.logging_utils.loggers import monitoring_logger as logger


def fetch_evaluation_data(days: int) -> pd.DataFrame:
    """
    Fetch recent data and calculate true volatility labels.

    Args:
        days (int): Number of recent days to evaluate

    Returns:
        pd.DataFrame: Features DataFrame
    """
    # Fetch extra days for feature engineering
    # At first get how many extra days we need for feature engineering
    buffer = config.get_live_data_config.get("historical_data_buffer", 61)
    # Get forecast horizon to ensure we have enough data for evaluation period
    forecast_horizon = config.get_forecast_config.get("default_horizon", 30)
    # Total days to fetch = evaluation period + buffer for features + forecast horizon for labels - 1 (because in buffer we already add 1 day for the first label)
    total_days = days + buffer + forecast_horizon - 1

    # Calculate date range for fetching data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_days)

    # Fetch raw data
    logger.info(
        f"Fetching live data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    raw_data = fetch_live_data(start_date=start_date, end_date=end_date, save=False)
    if raw_data is None:
        raise ValueError("Failed to fetch live data for monitoring")

    return raw_data


def prepare_evaluation_data(
    raw_data: pd.DataFrame, forecast_horizon: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare evaluation data by preprocessing and transforming it to get features and true labels.
    Args:
        raw_data (pd.DataFrame): Raw data fetched for evaluation
        forecast_horizon (int): Number of days ahead to forecast (used for label creation)
    Returns:
        tuple: (features DataFrame, true labels Series)
    """
    # Preprocess
    preprocessed = preprocess_data(raw_data)

    # Use DataTransformer in training mode to get features and labels needed for evaluation
    transformer = DataTransformer(training_mode=True, forecast_horizon=forecast_horizon)
    transformed = transformer.transform(preprocessed)

    # Separate features and labels
    X = transformed.drop(columns=["Target"])
    y = transformed["Target"].astype(int)

    logger.info(f"Prepared {len(X)} samples for evaluation")

    return X, y


def get_latest_model_from_mlflow() -> tuple[Any, str]:
    """
    Get the latest model from MLflow runs (newest by timestamp).

    Returns:
        tuple: (model, run_id)
    """
    client = MlflowClient()

    # Get the experiment
    experiment_name = config.get_mlflow_config.get(
        "experiment_name", "volatility_prediction"
    )
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    logger.info(
        f"Found experiment '{experiment_name}' with ID: {experiment.experiment_id}"
    )

    # Search for all successful runs in the experiment, sorted by start_time and get the latest one
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_type = 'production'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment 'volatility_prediction'")

    logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    logger.info(f"Latest run ID: {run_id}")

    # Load the model from the run
    model_name = config.get_mlflow_config.get("default_model_name", "model")
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow_load_model(model_uri)

    logger.info(f"Loaded latest model from run: {run_id}")
    logger.info(f"Run name: {latest_run.info.run_name}")
    logger.info(
        f"Run start time: {datetime.fromtimestamp(latest_run.info.start_time / 1000)}"
    )

    return model, run_id


def check_thresholds(metrics: dict) -> tuple[bool, list[str]]:
    """
    Check if metrics are below thresholds.

    Args:
        metrics (dict): Calculated metrics

    Returns:
        tuple: (degradation_detected, failed_metrics_list)
    """
    thresholds = config.get_monitoring_config.get("performance_thresholds", {})

    degradation_detected = False
    failed_metrics = []

    for metric_name, threshold in thresholds.items():
        metric_value = metrics.get(metric_name, None)
        if metric_value is None:
            logger.warning(f"{metric_name} not found in metrics")
            continue

        # Check if metric is below threshold
        if metric_value < threshold:
            degradation_detected = True
            failed_metrics.append(f"{metric_name}: {metric_value:.4f} < {threshold}")
            logger.warning(f"{metric_name}: {metric_value:.4f} < {threshold}")
        else:
            logger.info(f"{metric_name}: {metric_value:.4f} >= {threshold}")

    return degradation_detected, failed_metrics

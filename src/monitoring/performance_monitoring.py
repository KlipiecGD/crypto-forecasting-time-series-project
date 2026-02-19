import mlflow
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")

from src.monitoring.utils import get_latest_model_from_mlflow
from src.monitoring.utils import fetch_evaluation_data, prepare_evaluation_data
from src.monitoring.utils import check_thresholds
from src.monitoring.drift_detection import load_reference_data
from src.monitoring.drift_detection import run_evidently_report
from src.training.model_evaluator import calculate_metrics
from src.training.model_evaluator import log_confusion_matrix_mlflow
from src.fetch_data.historical_data import download_historical_data
from src.pipelines.training_pipeline import run_training_pipeline
from src.config.config import config
from src.logging_utils.loggers import monitoring_logger as logger


def analyze_performance(
    days: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    verbose: Optional[bool] = None,
) -> bool:
    """
    Analyzes the performance of the latest production model on recent data and checks for degradation.

    Args:
        days (int): Evaluation window in days
        forecast_horizon (int): Forecast horizon used in training (default from config)
        verbose (bool): Print detailed output

    Returns:
        bool: True if degradation detected, False otherwise
    """
    days = days or config.get_monitoring_config.get("evaluation_days", 360)
    forecast_horizon = forecast_horizon or config.get_forecast_config.get(
        "default_horizon", 30
    )
    verbose = (
        verbose
        if verbose is not None
        else config.get_monitoring_config.get("verbose", True)
    )
    experiment_name = config.get_monitoring_config.get(
        "experiment_name", "volatility_monitoring"
    )
    logger.info(
        f"Starting performance monitoring (last {days} days, forecast_horizon={forecast_horizon})"
    )

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"monitor_{datetime.now().strftime('-%Y-%m-%d_%H-%M-%S')}"
    ):
        try:
            # 1. Load model
            logger.info("Loading latest production model...")
            model, run_id = get_latest_model_from_mlflow()

            if not model:
                raise ValueError("No model found for monitoring")

            # 2. Fetch evaluation data
            logger.info("Fetching evaluation data...")
            eval_data = fetch_evaluation_data(days)

            # 3. Prepare evaluation data
            logger.info("Preparing evaluation data...")
            X, y_true = prepare_evaluation_data(eval_data, forecast_horizon)

            # 4. Make predictions
            logger.info("Making predictions...")
            y_pred = model.predict(X)

            # 5. Calculate metrics
            logger.info("Calculating metrics...")
            metrics = calculate_metrics(y_true, y_pred)

            # 6. Run Evidently report
            logger.info("Running Evidently drift and classification report...")
            evidently_drift_detected = False
            evidently_drift_share = 0.0
            try:
                reference_data = load_reference_data()
                evidently_drift_detected, evidently_drift_share = run_evidently_report(
                    reference_data=reference_data,
                    current_data=X,
                    y_true=y_true,
                    y_pred=y_pred,
                )

            except FileNotFoundError as e:
                logger.warning(f"Skipping Evidently report: {e}")

            # 7. Check thresholds
            logger.info("Checking thresholds...")
            degradation_detected, failed_metrics = check_thresholds(metrics)

            # 8. Log to MLflow
            mlflow.log_param("evaluation_days", days)
            mlflow.log_param("forecast_horizon", forecast_horizon)
            mlflow.log_param("data_points", len(X))
            mlflow.log_param("evaluation_date", datetime.now().strftime("%Y-%m-%d"))
            mlflow.log_param("model_run_id", run_id)
            mlflow.log_metric("evidently_drift_share", evidently_drift_share)
            mlflow.log_metric("evidently_drift_detected", int(evidently_drift_detected))
            mlflow.set_tag("evidently_drift", str(evidently_drift_detected))

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.set_tag("degradation_detected", str(degradation_detected))
            mlflow.set_tag("monitored_model_run_id", run_id)
            if failed_metrics:
                mlflow.set_tag("failed_metrics", ", ".join(failed_metrics))

            # Log confusion matrix
            log_confusion_matrix_mlflow(y_true, y_pred, "Monitoring")

            # 9. Print results
            if verbose:
                logger.info("MONITORING RESULTS")
                logger.info(f"Model Run ID: {run_id}")
                logger.info(f"Evaluation Period: Last {days} days")
                logger.info(f"Forecast Horizon: {forecast_horizon} days")
                logger.info(f"Data Points: {len(X)}")
                logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
                logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
                logger.info(f"Evidently Drift Detected: {evidently_drift_detected}")
                logger.info(f"Evidently Drift Share: {evidently_drift_share:.2%}")

                if degradation_detected:
                    logger.warning("\nMODEL DEGRADATION DETECTED")
                    logger.warning("Failed metrics:")
                    for failed in failed_metrics:
                        logger.warning(f"  - {failed}")
                else:
                    logger.info("\nModel performing above thresholds")

        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            raise

    return degradation_detected


def run_monitoring(
    days: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    verbose: Optional[bool] = None,
    invoke_retraining: Optional[bool] = None,
    ignore_no_new_data: Optional[bool] = None,
    deploy_after_retraining: Optional[bool] = None,
) -> bool:
    """
    Runs the performance monitoring and optionally invokes retraining if degradation is detected.

    Args:
        days (Optional[int]): Evaluation window in days
        forecast_horizon (Optional[int]): Forecast horizon in days
        verbose (Optional[bool]): Whether to print detailed logs
        invoke_retraining (Optional[bool]): Whether to automatically invoke retraining if degradation detected
        ignore_no_new_data (Optional[bool]): If True, retrain even without new Kaggle data
        deploy_after_retraining (Optional[bool]): Whether to deploy after retraining completes

    Returns:
        bool: True if degradation detected, False otherwise
    """
    # Load config with defaults
    params = _load_monitoring_params(
        days, forecast_horizon, verbose, invoke_retraining,
        ignore_no_new_data, deploy_after_retraining
    )

    # Run performance analysis
    degradation_detected = analyze_performance(
        days=params["days"],
        forecast_horizon=params["forecast_horizon"],
        verbose=params["verbose"],
    )

    # Handle degradation
    if degradation_detected:
        logger.warning("Performance degradation detected during monitoring.")
        if params["invoke_retraining"]:
            _handle_retraining(params)
        else:
            logger.info("Retraining disabled in config. Skipping retraining.")
    else:
        logger.info("No performance degradation detected during monitoring.")

    return degradation_detected


def _load_monitoring_params(
    days: Optional[int],
    forecast_horizon: Optional[int],
    verbose: Optional[bool],
    invoke_retraining: Optional[bool],
    ignore_no_new_data: Optional[bool],
    deploy_after_retraining: Optional[bool],
) -> dict:
    """
    Load monitoring parameters with config fallbacks.
    Args:
        days (Optional[int]): Evaluation window in days
        forecast_horizon (Optional[int]): Forecast horizon in days
        verbose (Optional[bool]): Whether to print detailed logs
        invoke_retraining (Optional[bool]): Whether to automatically invoke retraining if degradation detected
        ignore_no_new_data (Optional[bool]): If True, retrain even without new Kaggle data
        deploy_after_retraining (Optional[bool]): Whether to deploy after retraining completes
    Returns:
        dict: Parameters for monitoring and retraining logic
    """
    return {
        "days": days or config.get_monitoring_config.get("evaluation_days", 360),
        "forecast_horizon": forecast_horizon or config.get_forecast_config.get("default_horizon", 30),
        "verbose": verbose if verbose is not None else config.get_monitoring_config.get("verbose", True),
        "invoke_retraining": invoke_retraining if invoke_retraining is not None else config.get_monitoring_config.get("invoke_retraining", True),
        "ignore_no_new_data": ignore_no_new_data if ignore_no_new_data is not None else config.get_monitoring_config.get("ignore_no_new_data", False),
        "deploy_after_retraining": deploy_after_retraining if deploy_after_retraining is not None else config.get_pipeline_config.get("deploy_after_training", False),
    }


def _handle_retraining(params: dict) -> None:
    """
    Handle retraining logic when degradation is detected.
    Args:
        params (dict): Parameters for retraining logic
    """
    logger.info("Invoking retraining pipeline due to detected degradation...")
    logger.info("Downloading new data from Kaggle for retraining...")
    
    has_new_data, path = download_historical_data()
    
    if has_new_data:
        logger.info("New data available. Starting retraining...")
        run_training_pipeline(
            data_path=path,
            deploy=params["deploy_after_retraining"],
            verbose=params["verbose"]
        )
    elif params["ignore_no_new_data"]:
        logger.warning("No new data available, but ignore_no_new_data=True. Proceeding with retraining.")
        run_training_pipeline(
            data_path=path,
            deploy=params["deploy_after_retraining"],
            verbose=params["verbose"]
        )
    else:
        logger.info("No new data available and ignore_no_new_data=False. Skipping retraining.")


if __name__ == "__main__":
    degradation = run_monitoring(
        days=180,
        invoke_retraining=True,
        ignore_no_new_data=True,
        verbose=True,
        deploy_after_retraining=True,
    )  # Will always run
    if degradation:
        logger.warning(
            "Performance degradation detected! Model retrained and redeployed if configured to do so."
        )
    else:
        logger.info("No performance degradation detected.")

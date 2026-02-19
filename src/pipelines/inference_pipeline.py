from datetime import datetime, timedelta
from typing import Optional

from src.inference.model_loader import load_inference_model
from src.inference.predictor import make_prediction
from src.inference.result_formatter import format_prediction_output
from src.inference.data_loader import load_data_for_inference

from src.logging_utils.loggers import inference_logger as logger
from src.config.config import config


def run_inference(
    forecast_horizon: Optional[int] = None,
    verbose: Optional[bool] = None,
) -> tuple[str, list[float], datetime] | None:
    """
    Runs the inference pipeline to predict the volatility class for a specified forecast horizon.

    Args:
        forecast_horizon (int): Number of days ahead to predict volatility for (1-30). Default is 30.
        verbose (bool): Whether to print detailed prediction results. Default is True.

    Returns:
        tuple[str, list[float]]: Predicted volatility class and confidence scores for each class.
        Returns None if there was an error during inference (e.g., data fetching failure).
    """
    logger.info("Starting inference pipeline")

    # 1. Validate arguments
    max_horizon = config.get_forecast_config.get("max_horizon", 30)
    min_horizon = config.get_forecast_config.get("min_horizon", 1)
    forecast_horizon = forecast_horizon or config.get_forecast_config.get(
        "default_horizon", 30
    )
    if not (min_horizon <= forecast_horizon <= max_horizon):
        logger.error(
            f"Invalid forecast_horizon: {forecast_horizon}. Must be between {min_horizon} and {max_horizon}."
        )
        return None

    verbose = (
        verbose
        if verbose is not None
        else config.get_pipeline_config.get("verbose", True)
    )

    # 2. Load model
    pipeline = load_inference_model()

    # 3. Load and prepare data for inference
    features = load_data_for_inference(forecast_horizon)
    if features is None:
        logger.error("Inference aborted due to data loading failure.")
        return None

    # 4. Predict
    predicted_class, probabilities = make_prediction(pipeline, features)

    # 5. Calculate actual target date
    target_date = datetime.now() + timedelta(days=forecast_horizon)

    # 6. Format output
    if verbose:
        format_prediction_output(predicted_class, probabilities)

    logger.info("Inference pipeline completed successfully")

    return predicted_class, probabilities, target_date


if __name__ == "__main__":
    run_inference()

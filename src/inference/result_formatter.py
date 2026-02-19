import pandas as pd
from typing import Optional

from src.config.config import config
from src.logging_utils.loggers import inference_logger as logger


def format_prediction_output(
    predicted_class: str,
    probabilities: list[float],
    forecast_horizon: Optional[int] = None,
) -> None:
    """
    Print formatted prediction results.

    Args:
        predicted_class (str): Predicted volatility class
        probabilities (list[float]): Class probabilities
        forecast_horizon (Optional[int]): Number of days ahead for the forecast (default is 30)
    """
    # Get default forecast horizon from config if not provided
    forecast_horizon = forecast_horizon or config.get_forecast_config.get(
        "default_horizon", 30
    )

    # Get mapping from config to ensure class names are consistent
    mapping = config.get_model_config.get("mapping", {"Low": 0, "Normal": 1, "High": 2})

    logger.info(f"--- Prediction for {forecast_horizon} Days Ahead ---")
    logger.info(f"Today is: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    logger.info(
        f"Predicted Volatility Class for {forecast_horizon} Days Ahead ({pd.Timestamp.now() + pd.Timedelta(days=forecast_horizon)}): {predicted_class.upper()}"
    )
    logger.info(f"Confidence Scores:")

    for class_name, prob in zip(mapping.keys(), probabilities):
        logger.info(f"  - {class_name}: {prob:.2%}")


def format_as_dict(predicted_class: str, probabilities: list[float]) -> dict:
    """
    Format prediction as dictionary (useful for APIs).

    Args:
        predicted_class (str): Predicted volatility class
        probabilities (list[float]): Class probabilities

    Returns:
        dict: Structured prediction result
    """
    mapping = config.get_model_config.get("mapping", {"Low": 0, "Normal": 1, "High": 2})

    return {
        "predicted_class": predicted_class,
        "confidence": probabilities[mapping[predicted_class]],
        "probabilities": {
            class_name: prob for class_name, prob in zip(mapping.keys(), probabilities)
        },
    }

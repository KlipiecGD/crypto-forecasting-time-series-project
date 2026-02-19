import joblib
from typing import Optional
from sklearn.pipeline import Pipeline

from src.logging_utils.loggers import inference_logger as logger
from src.config.config import config


def load_inference_model(
    model_path: Optional[str] = None,
) -> Pipeline:
    """
    Load the trained model for inference.

    Args:
        model_path (Optional[str]): Model path

    Returns:
        Pipeline: Loaded inference pipeline
    """
    model_path = model_path or config.get_model_config.get(
        "model_save_path", "models/volatility_model.joblib"
    )

    logger.info(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    logger.info("Model loaded successfully")

    return pipeline

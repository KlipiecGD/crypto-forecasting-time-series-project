import os
import joblib
from typing import Optional
from sklearn.pipeline import Pipeline

from src.config.config import config
from src.logging_utils.loggers import training_logger as logger


def save_model(
    pipeline: Pipeline,
    model_path: Optional[str] = None,
) -> str:
    """
    Save the trained model pipeline to disk.

    Args:
        pipeline (Pipeline): Trained pipeline to save
        model_path (Optional[str]): Path to save the model (default: "models/volatility_model.joblib")

    Returns:
        str: Path where model was saved
    """
    model_path = model_path or config.get_model_config.get(
        "model_save_path", "models/volatility_model.joblib"
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(pipeline, model_path)
    logger.info(f"Pipeline successfully saved to {model_path}")

    return model_path

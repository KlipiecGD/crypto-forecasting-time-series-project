import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.config import config
from src.logging_utils.loggers import inference_logger as logger


def make_prediction(
    pipeline: Pipeline, features: pd.DataFrame
) -> tuple[str, list[float]]:
    """
    Make volatility prediction using the trained pipeline.

    Args:
        pipeline (Pipeline): Trained model pipeline
        features (pd.DataFrame): Engineered features

    Returns:
        tuple: (predicted_class, probabilities)
    """
    logger.info("Making prediction...")

    # Get numeric prediction and probabilities
    prediction = pipeline.predict(features)[0]
    probabilities = pipeline.predict_proba(features)[0]

    # Decode to class name
    predicted_class = decode_prediction(prediction)

    logger.info(f"Prediction complete\nPredicted class: {predicted_class}")

    return predicted_class, probabilities.tolist()


def decode_prediction(prediction: int) -> str:
    """
    Convert numeric prediction to class name.

    Args:
        prediction (int): Numeric class (0, 1, or 2)

    Returns:
        str: Class name ('Low', 'Normal', or 'High')
    """
    mapping = config.get_model_config.get("mapping", {"Low": 0, "Normal": 1, "High": 2})

    inv_mapping = {v: k for k, v in mapping.items()}
    return inv_mapping[prediction]

import pandas as pd
from datetime import datetime, timedelta

from src.fetch_data.live_data import fetch_live_data
from src.features.preprocess_data import preprocess_data
from src.features.data_transformer import DataTransformer

from src.logging_utils.loggers import inference_logger as logger
from src.config.config import config


def load_data_for_inference(
    forecast_horizon: int,
) -> pd.DataFrame | None:
    """
    Load and prepare data for inference.

    Returns:
        pd.DataFrame: Prepared features for inference. In case of failure, returns None.
    """
    # 1. Fetch raw data

    # Calculate date range to fetch based on forecast horizon and historical buffer
    date_shift = config.get_forecast_config.get("max_horizon", 30) - forecast_horizon
    end_date = datetime.now() - timedelta(days=date_shift)
    start_date = end_date - timedelta(
        days=config.get_live_data_config.get("historical_data_buffer", 61)
    )

    raw_data = fetch_live_data(start_date=start_date, end_date=end_date)
    if raw_data is None:
        logger.error("Failed to fetch live data. Aborting inference.")
        return None

    # 2. Preprocess
    preprocessed_data = preprocess_data(raw_data)
    if preprocessed_data is None or preprocessed_data.empty:
        logger.error(
            "Preprocessing failed or resulted in empty data. Aborting inference."
        )
        return None

    # 3. Transform to features
    transformer = DataTransformer(training_mode=False)
    features = transformer.transform(preprocessed_data)

    return features

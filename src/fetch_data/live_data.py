import requests
import pandas as pd
from datetime import datetime
from typing import Optional

from src.config.config import config
from src.logging_utils.loggers import data_logger as logger


def fetch_live_data(
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save: bool = False,
    path: Optional[str] = None,
) -> pd.DataFrame | None:
    """
    Fetches live cryptocurrency data from Binance API.

    Args:
        limit (Optional[int]): Number of data points (used if start_date/end_date not provided)
        start_date (Optional[datetime]): Start date for data fetch
        end_date (Optional[datetime]): End date for data fetch
        save (bool): Whether to save data
        path (Optional[str]): Save path
    """
    url = config.get_live_data_config.get(
        "api_url", "https://api.binance.com/api/v3/klines"
    )
    symbol = config.get_live_data_config.get("symbol", "BTCUSDT")
    interval = config.get_live_data_config.get("interval", "1d")

    params = {"symbol": symbol, "interval": interval}

    # Use dates if provided, otherwise fall back to limit
    if start_date and end_date:
        params["startTime"] = int(start_date.timestamp() * 1000)
        params["endTime"] = int(end_date.timestamp() * 1000)
    else:
        limit = limit or config.get_live_data_config.get("limit", 91)
        params["limit"] = limit

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        if not data:
            logger.warning("No data received from API.")
            return None

        # Save to DataFrame
        df = pd.DataFrame(data)

        if save:
            if path is None:
                logger.warning(
                    "Save path not provided. Using default path from config."
                )
                path = config.get_live_data_config.get(
                    "fetched_data_save_path", "data/live_data.csv"
                )
            df.to_csv(path, index=False)
            logger.info(f"Live data saved to {path}")

        return df

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

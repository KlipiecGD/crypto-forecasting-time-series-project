import pandas as pd
import numpy as np

from src.config.config import config
from src.fetch_data.live_data import fetch_live_data
from src.features.preprocess_data import preprocess_data


def load_historical_data() -> pd.DataFrame | None:
    """
    Loads historical volatility data from CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns: ['date', 'volatility', 'class']
    """
    try:
        # Load the historical data
        data_path = config.get_historical_data_config.get(
            "file_path", "data/Bitcoin_history_data.csv"
        )
        df_raw = pd.read_csv(data_path)

        # Prepare dataframe
        df_raw_copy = df_raw.copy()
        df_raw_copy["Date"] = pd.to_datetime(df_raw_copy["Date"])
        df_raw_copy = df_raw_copy.sort_values("Date").reset_index(drop=True)

        # Get volatility ranges
        volatility_ranges = {
            "Low": config.get_thresholds_config.get("Low", 0.40),
            "Normal": config.get_thresholds_config.get("Normal", 0.65),
            "High": np.inf,
        }

        # Calculate volatility (same as in DataTransformer)
        log_returns = pd.Series(
            np.log(df_raw_copy["Close"] / df_raw_copy["Close"].shift(1))
        )
        volatility = log_returns.rolling(
            window=config.get_volatility_transformations_config.get("main_window", 30)
        ).std() * np.sqrt(365)

        # Classify volatility
        def classify_volatility(vol):
            if pd.isna(vol):
                return None
            if vol <= volatility_ranges["Low"]:
                return "Low"
            elif vol <= volatility_ranges["Normal"]:
                return "Normal"
            else:
                return "High"

        # Create result dataframe
        result_df = pd.DataFrame(
            {
                "date": df_raw_copy["Date"],
                "volatility": volatility,
                "class": volatility.apply(classify_volatility),
            }
        )

        # Drop NaN values
        result_df = result_df.dropna()

        return result_df

    except Exception as e:
        return None


def fetch_current_market_data(limit: int) -> pd.DataFrame | None:
    """
    Fetch and preprocess current market data.

    Args:
        limit (int): Number of data points to fetch

    Returns:
        pd.DataFrame | None: Preprocessed market data or None if fetch fails
    """
    try:
        raw_data = fetch_live_data(limit=limit, save=False)
        required_columns = config.get_pipeline_config.get(
            "required_columns", ["Date", "Open", "High", "Low", "Close"]
        )
        if "Volume" not in required_columns:
            required_columns.append("Volume")

        if raw_data is not None:
            return preprocess_data(raw_data, required_columns=required_columns)
        return None
    except Exception:
        return None

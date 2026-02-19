import pandas as pd
from typing import Optional

from src.config.config import config


def preprocess_data(
    df: pd.DataFrame, required_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Preprocesses the raw data fetched from Binance API.
    - Selects relevant columns
    - Converts timestamp to datetime and prices to float
    Args:
        df (pd.DataFrame): Raw DataFrame fetched from Binance API
        columns (Optional[list[str]]): List of required columns to select and rename. If None, defaults to ["Date", "Open", "High", "Low", "Close"].
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for feature transformation
    """
    # Select the first 5 columns and rename them
    required_columns = required_columns or config.get_pipeline_config.get(
        "required_columns", ["Date", "Open", "High", "Low", "Close"]
    )
    # Check whether the data is empty before selecting columns
    if df.empty:
        return pd.DataFrame(columns=required_columns)

    preprocessed_data = df.iloc[:, : len(required_columns)].copy()
    preprocessed_data.columns = required_columns

    # Convert timestamp to datetime and prices to float
    preprocessed_data["Date"] = pd.to_datetime(preprocessed_data["Date"], unit="ms")

    prices_columns = [col for col in required_columns if col != "Date"]
    for col in prices_columns:
        preprocessed_data[col] = pd.to_numeric(preprocessed_data[col], errors="coerce")

    return preprocessed_data

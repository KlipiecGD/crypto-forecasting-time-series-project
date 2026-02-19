import pandas as pd
import numpy as np
from datetime import datetime


def generate_sample_ohlc_data(n_days: int = 100, seed: int = 2147) -> pd.DataFrame:
    """
    Generate sample OHLC data for testing.

    Args:
        n_days (int): Number of days of data to generate
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: OHLC data with Date, Open, High, Low, Close columns
    """
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    np.random.seed(seed)

    data = {
        "Date": dates,
        "Open": 50000 + np.random.randn(n_days) * 1000,
        "High": 51000 + np.random.randn(n_days) * 1000,
        "Low": 49000 + np.random.randn(n_days) * 1000,
        "Close": 50000 + np.random.randn(n_days) * 1000,
    }

    # Ensure High is highest and Low is lowest
    df = pd.DataFrame(data)
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

    return df


def generate_raw_binance_data(n_days: int = 10) -> pd.DataFrame:
    """
    Generate sample raw Binance API data.

    Args:
        n_days (int): Number of days of data to generate

    Returns:
        pd.DataFrame: Raw Binance format data (12 columns)
    """
    timestamps = [
        int(datetime(2024, 1, i).timestamp() * 1000) for i in range(1, n_days + 1)
    ]

    data = []
    for ts in timestamps:
        row = [
            ts,  # Open time
            "50000.5",  # Open
            "51000.2",  # High
            "49000.1",  # Low
            "50500.3",  # Close
            "1000.5",  # Volume
            ts + 86400000,  # Close time
            "50000000",  # Quote asset volume
            100,  # Number of trades
            "500.2",  # Taker buy base
            "25000000",  # Taker buy quote
            "0",  # Ignore
        ]
        data.append(row)

    return pd.DataFrame(data)


def generate_csv_data(start_date: str, n_days: int) -> pd.DataFrame:
    """
    Generate CSV data for file comparison tests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        n_days: Number of days of data

    Returns:
        pd.DataFrame: Simple CSV with Date and Close columns
    """
    return pd.DataFrame(
        {
            "Date": pd.date_range(start=start_date, periods=n_days),
            "Close": [50000] * n_days,
        }
    )


def generate_minimal_ohlc_data(n_days: int = 10) -> pd.DataFrame:
    """
    Generate minimal OHLC data (constant prices for edge case testing).

    Args:
        n_days: Number of days of data

    Returns:
        pd.DataFrame: Minimal OHLC data
    """
    return pd.DataFrame(
        {
            "Date": pd.date_range(start="2024-01-01", periods=n_days),
            "Open": [50000] * n_days,
            "High": [51000] * n_days,
            "Low": [49000] * n_days,
            "Close": [50000] * n_days,
        }
    )


def generate_empty_ohlc_data() -> pd.DataFrame:
    """
    Generate empty OHLC DataFrame with correct columns.

    Returns:
        pd.DataFrame: Empty OHLC DataFrame
    """
    return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])


def generate_invalid_price_data() -> pd.DataFrame:
    """
    Generate data with invalid price strings for error handling tests.

    Returns:
        pd.DataFrame: Data with some invalid prices
    """
    return pd.DataFrame(
        [
            [1704067200000, "invalid", "51000", "49000", "50000"],
            [1704153600000, "50000", "51000", "49000", "50000"],
        ]
    )


def generate_single_row_binance_data() -> pd.DataFrame:
    """
    Generate single row of Binance data.

    Returns:
        pd.DataFrame: Single row of Binance format data
    """
    return pd.DataFrame([[1704067200000, "50000", "51000", "49000", "50500", "1000"]])


def generate_corrupted_csv_content() -> str:
    """
    Generate corrupted CSV content for error handling tests.

    Returns:
        str: Corrupted CSV content
    """
    return "not,valid,csv,data\n@#$%^&*()"


def generate_no_date_column_data() -> pd.DataFrame:
    """
    Generate DataFrame without Date column for error testing.

    Returns:
        pd.DataFrame: Data without Date column
    """
    return pd.DataFrame({"Close": [50000] * 10})


def generate_string_dates_data() -> pd.DataFrame:
    """
    Generate data with string-formatted dates.

    Returns:
        pd.DataFrame: Data with string dates
    """
    return pd.DataFrame(
        {
            "Date": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "Close": [50000] * 3,
        }
    )

import pandas as pd
from pathlib import Path

from src.logging_utils.loggers import data_logger as logger


def compare_csv_files(existing_path: str, new_path: str) -> bool:
    """
    Compares the max date in the existing CSV vs the new CSV.
    Returns True if the new CSV contains newer data, False otherwise.
    Args:
        existing_path (str): Path to the existing CSV file
        new_path (str): Path to the newly downloaded CSV file
    Returns:
        bool: True if new data is detected, False if no new data is found or if an error occurs during comparison
    """
    if not Path(existing_path).exists():
        logger.info(f"Existing file not found at {existing_path}. Accepting new data.")
        return True

    try:
        # Load only the 'Date' column to optimize performance
        df_existing = pd.read_csv(existing_path, usecols=["Date"])
        df_new = pd.read_csv(new_path, usecols=["Date"])

        # Check whether existing file has any data
        if df_existing.empty:
            logger.info(
                f"Existing file at {existing_path} is empty. Accepting new data."
            )
            return True

        # Ensure Date column is datetime
        existing_max_date = pd.to_datetime(df_existing["Date"]).max()
        new_max_date = pd.to_datetime(df_new["Date"]).max()

        if new_max_date > existing_max_date:
            logger.info(
                f"New data found. Current max: {existing_max_date}, New max: {new_max_date}"
            )
            return True
        else:
            logger.info(
                f"No new data detected. Current max: {existing_max_date} >= New max: {new_max_date}"
            )
            return False

    except Exception as e:
        logger.warning(f"Error comparing CSV files ({e}). Defaulting to update.")
        return True

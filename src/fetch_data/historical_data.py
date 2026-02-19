import kagglehub
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from src.fetch_data.compare_files import compare_csv_files
from src.config.config import config
from src.logging_utils.loggers import data_logger as logger


def download_historical_data(save_path: Optional[str] = None) -> tuple[bool, str]:
    """
    Downloads historical Bitcoin data from Kaggle to a temporary folder,
    then copies only the CSV file to the specified save_path.

    Args:
        save_path (Optional[str]): Local directory path where the CSV will be saved.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating wheter the downloaded CSV file is different from the existing one (if exists)
                          and the path to the saved CSV file (if successful).
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Downloading historical data to temporary folder: {temp_dir}...")

    try:
        # Download to temporary directory
        path = kagglehub.dataset_download(
            config.get_historical_data_config.get(
                "kaggle_dataset", "adilshamim8/bitcoin-historical-data"
            ),
            output_dir=temp_dir,
            force_download=True,
        )
        logger.info(f"Historical data downloaded to temporary location: {path}")

        # Find CSV file in downloaded data
        temp_path = Path(path)
        csv_files = list(temp_path.glob("**/*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset")

        # Get the first CSV file - In this dataset, there should only be one CSV file
        csv_file = csv_files[0]
        logger.info(f"Found CSV file: {csv_file}")

        # Check whether the new CSV file is different from the existing one (if exists)
        existing_csv_path = config.get_historical_data_config.get(
            "file_path", "data/Bitcoin_history_data.csv"
        )
        has_new_data = compare_csv_files(existing_csv_path, str(csv_file))

        if has_new_data:
            logger.info("New data detected. Saving the new CSV file...")
            # Determine final save location
            final_dir = save_path or config.get_historical_data_config.get(
                "download_dir", "data/"
            )
            final_path = Path(final_dir)
            final_path.mkdir(parents=True, exist_ok=True)

            # Copy CSV to final destination
            destination = final_path / csv_file.name
            shutil.copy2(csv_file, destination)
            logger.info(f"New CSV file saved to {destination}")

            return True, str(destination)
        else:
            logger.info("No new data detected. Existing CSV file is up to date.")
            return False, existing_csv_path

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Temporary folder cleaned up: {temp_dir}")


if __name__ == "__main__":
    download_historical_data()

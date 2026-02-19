import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split

from src.features.data_transformer import DataTransformer
from src.config.config import config
from src.logging_utils.loggers import training_logger as logger


def prepare_training_data(
    data_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepares training data with train/val/test split.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Loading training data...")

    # Load raw data
    data_path = data_path or config.get_historical_data_config.get(
        "file_path", "data/Bitcoin_history_data.csv"
    )
    df_raw = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df_raw)} records from {data_path}")

    # Transform data (creates Target column)
    transformer = DataTransformer(training_mode=True)
    df_transformed = transformer.transform(df_raw)
    logger.info(f"Transformed data shape: {df_transformed.shape}")

    # Separate features and target
    X = df_transformed.drop(columns=["Target"])
    y = df_transformed["Target"].astype(int)

    # First seperate out training set
    test_size = config.get_model_config.get("test_size", 0.15)
    val_size = config.get_model_config.get("val_size", 0.15)
    random_seed = config.get_model_config.get("random_seed", 2137)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_seed, shuffle=False
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (test_size + val_size),
        random_state=random_seed,
        shuffle=False,
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

import pandas as pd
import numpy as np
import lightgbm as lgb

from notebooks.utils.common_utils import calculate_evaluation_metrics


def back_transform_returns_to_price(
    val_df: pd.DataFrame,
    return_forecasts: np.ndarray,
    price_col: str = "Close",
    target_col: str = "Next_Close",
    log: bool = False,
) -> tuple[np.ndarray, pd.Series]:
    """
    Transform predicted returns back into absolute price levels.

    Args:
        val_df (pd.DataFrame): Validation DataFrame containing actual prices and target column.
        return_forecasts (np.ndarray): Predicted returns from the model.
        price_col (str): Column name for the price used as the base for transformation.
            Defaults to "Close".
        target_col (str): Column name for the actual target prices. Defaults to "Next_Close".
        log (bool): Whether the returns are logarithmic. If True, uses exp transformation.
            Defaults to False.

    Returns:
        tuple[np.ndarray, pd.Series]: Tuple containing actual prices (numpy array)
            and predicted prices (pandas Series).
    """

    # 1. Convert to numpy for vectorization
    forecasts = np.array(return_forecasts)
    n = len(forecasts)

    # 2. Predicted: Back-transform returns to prices
    base_prices = val_df[price_col][:n]

    if log:
        y_pred_prices = base_prices * np.exp(forecasts)
    else:
        y_pred_prices = base_prices * (1 + forecasts)

    # 3. Actual: Extract actual prices from target column
    y_actual_prices = np.array(val_df[target_col][:n])

    return y_actual_prices, y_pred_prices


def evaluate_feature_set(
    features: list,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    log=False,
    random_seed=42,
) -> tuple:
    """
    Evaluate a feature set using default LightGBM.

    Args:
        features (list): List of feature names to use for training.
        train_df (pd.DataFrame): Training DataFrame (used for back-transformation).
        val_df (pd.DataFrame): Validation DataFrame (used for back-transformation).
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
        X_val (pd.DataFrame): Validation feature set.
        log (bool): Whether to back-transform using log returns. Defaults to False.
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Tuple containing trained LightGBM model, training metrics dict,
            and validation metrics dict.
    """
    model = lgb.LGBMRegressor(random_state=random_seed, verbose=-1)
    model.fit(X_train[features], y_train)
    train_preds = model.predict(X_train[features])
    val_preds = model.predict(X_val[features])

    y_train_actual, y_train_pred = back_transform_returns_to_price(
        train_df, train_preds, "Close", "Next_Close", log=log
    )
    y_val_actual, y_val_pred = back_transform_returns_to_price(
        val_df, val_preds, "Close", "Next_Close", log=log
    )
    train_metrics = calculate_evaluation_metrics(y_train_actual, y_train_pred)
    val_metrics = calculate_evaluation_metrics(y_val_actual, y_val_pred)

    return model, train_metrics, val_metrics

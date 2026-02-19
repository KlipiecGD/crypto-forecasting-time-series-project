import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.config import config
from src.logging_utils.loggers import data_logger as logger


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for data preprocessing in the pipeline.
    Prepares the data for volatility prediction by adding features such as lagged returns, rolling statistics, and volatility measures.
    After transformations dataset will have all the necessary features for training the volatility prediction model.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        training_mode: bool = False,
        forecast_horizon: Optional[int] = None,
    ) -> None:
        """
        Initializes the DataTransformer with specified parameters.

        Args:
            window_size (Optional[int]): The size of the rolling window for feature engineering.
            training_mode (bool): Whether the transformer is being used in training mode (True)
                or inference mode (False). In training mode, the target variable will be created
                by shifting the volatility class number by the forecast horizon. In inference mode,
                only the most recent feature row will be returned.
            forecast_horizon (Optional[int]): The number of days ahead for which the volatility class will
                be predicted. This is used to create the target variable in training mode by
                shifting the volatility class number accordingly.
        """
        self.window_size = (
            window_size
            or config.get_volatility_transformations_config.get("main_window", 30)
        )
        self.training_mode = training_mode
        self.forecast_horizon = forecast_horizon or config.get_forecast_config.get(
            "default_horizon", 30
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataTransformer":
        """Nothings to fit, so we just return self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding engineered features for volatility prediction.
        Args:
            X (pd.DataFrame): Input DataFrame with historical price data.
        Returns:
            pd.DataFrame: Transformed DataFrame with additional features.
        """
        logger.info("Transforming data...")

        # Step 0: Check if required columns are present
        self._check_required_columns(X)

        # Step 1: Set index to datetime and sort
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        X = X.sort_values("Date").reset_index(drop=True)

        # Step 2: Calculate historical volatility
        X = self._calculate_volatility(X)

        # Step 3: Calculate volatility class number
        X = self._calculate_volatility_class_number(
            X,
            low=config.get_thresholds_config.get("Low", 0.40),
            normal=config.get_thresholds_config.get("Normal", 0.65),
            mapping=config.get_model_config.get(
                "mapping", {"Low": 0, "Normal": 1, "High": 2}
            ),
        )

        # If we are in training mode we need to add the target variable for the model, which is the volatility class number shifted by the forecast horizon
        if self.training_mode:
            X["Target"] = X["Volatility_Class_Num"].shift(-self.forecast_horizon)

        # Step 4: Calculate Parkinson's, Garman-Klass, and Roger-Satchel volatility estimators
        X = self._calculate_parkinson_volatility(X)
        X = self._calculate_garman_klass_volatility(X)
        X = self._calculate_roger_satchel_volatility(X)

        # Step 5: Add lagged volatility features
        X = self._add_lagged_volatility(
            X,
            lags=config.get_volatility_transformations_config.get(
                "lagged_features", [30]
            ),
        )

        # Step 6: Add rolling statistics for volatility
        X = self._calculate_rolling_statistics(
            X,
            windows=config.get_volatility_transformations_config.get(
                "rolling_windows", [30]
            ),
        )
        logger.debug(f"DF shape after rolling statistics features: {X.shape}")

        # Step 7: Add volatility change features
        X = self._calculate_volatility_change(
            X,
            window=config.get_volatility_transformations_config.get(
                "change_windows", [30]
            ),
        )

        # Step 8: Add volatility momentum feature
        X = self._calculate_volatility_momentum(X)

        # Step 9: Select only relevant features for the model
        features_list = config.get_transformations_config.get("features", []).copy()
        if self.training_mode:
            features_list.append("Target")
            logger.info("Training mode: Added 'Target' to features list.")

        logger.info(f"Selecting features: {features_list}")

        X = self._select_features(X, features_list=features_list)

        # Step 10: Final clean up
        # For inference we keep only the last row of the dataset.
        # For training we drop NaNs created by lags/shifts
        if self.training_mode:
            X = X.dropna().reset_index(drop=True)
        else:
            X = X.tail(1).reset_index(drop=True)

        logger.info(f"DF shape after final clean up: {X.shape}")
        return X

    # Checks if required columns are present in the DataFrame
    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """
        Checks if the required columns are present in the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame to check.
        Raises:
            ValueError: If any of the required columns are missing.
        """
        required_columns = config.get_pipeline_config.get(
            "required_columns", ["Date", "Open", "High", "Low", "Close"]
        )
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

    # Historical Volatility Calculation using Log Returns
    def _calculate_volatility(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculates historical volatility using log returns and rolling standard deviation.
        Args:
            df (pd.DataFrame): Input DataFrame with a 'Close' price column.
        Returns:
            pd.Series: A Series containing the calculated volatility values.
        """
        window_size = self.window_size
        log_returns = pd.Series(np.log(df["Close"] / df["Close"].shift(1)))
        volatility = log_returns.rolling(window=window_size).std() * np.sqrt(365)
        df["Volatility"] = volatility

        return df

    def _assign_volatility_class(
        self, volatility: float, low: float, normal: float
    ) -> str:
        """
        Assigns a volatility class label based on the volatility value.
        Args:
            volatility (float): The volatility value to classify.
            low (float): The threshold for low volatility.
            normal (float): The threshold for normal volatility.
        Returns:
            str: The assigned volatility class label ('Low', 'Normal', 'High').
        """
        if pd.isna(volatility):
            return "Unknown"
        if volatility <= low:
            return "Low"
        elif volatility <= normal:
            return "Normal"
        else:
            return "High"

    def _calculate_volatility_class_number(
        self, df: pd.DataFrame, low: float, normal: float, mapping: dict
    ) -> pd.DataFrame:
        """
        Calculate volatility class number based on quantiles.
        Args:
            df (pd.DataFrame): Input DataFrame with a 'Volatility' column.
        Returns:
            pd.DataFrame: DataFrame with an added 'Volatility_Class_Num' column.
        """
        df["Volatility_Class_Num"] = (
            df["Volatility"]
            .apply(lambda x: self._assign_volatility_class(x, low, normal))
            .map(mapping)
        )
        return df

    # Parkinson's Volatility Estimator
    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parkinson's volatility estimator.
        Args:
            df (pd.DataFrame): DataFrame containing 'High' and 'Low' price columns.
        Returns:
            pd.DataFrame: DataFrame with an added 'Parkinson_Volatility' column containing the Parkinson's volatility estimates.
        """
        window = self.window_size
        term_1 = 1 / (4 * window * np.log(2))
        term_2 = pd.Series(np.log(df["High"] / df["Low"])) ** 2
        parkinson_vol = np.sqrt(term_1 * term_2.rolling(window=window).sum()) * np.sqrt(
            365
        )
        df["Parkinson_Volatility"] = parkinson_vol

        return df

    # Garman-Klass Volatility Estimator
    def _calculate_garman_klass_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Garman-Klass volatility estimator.
        Args:
            df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' price columns.
        Returns:
            pd.DataFrame: DataFrame with an added 'Garman_Klass_Volatility' column containing the Garman-Klass volatility estimates.
        """
        window = self.window_size
        term_1 = 0.5 * (np.log(df["High"] / df["Low"])) ** 2
        term_2 = (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"])) ** 2
        gk_vol = np.sqrt(
            (1 / window) * (term_1 - term_2).rolling(window=window).sum()
        ) * np.sqrt(365)

        df["Garman_Klass_Volatility"] = gk_vol
        return df

    # Roger-Satchel Volatility Estimator
    def _calculate_roger_satchel_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Roger-Satchel volatility estimator.
        Args:
            df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' price columns.
        Returns:
            pd.DataFrame: DataFrame with an added 'Roger_Satchel_Volatility' column containing the Roger-Satchel volatility estimates.
        """
        window = self.window_size
        term1 = pd.Series(np.log(df["High"] / df["Close"])) * pd.Series(
            np.log(df["High"] / df["Open"])
        )
        term2 = pd.Series(np.log(df["Low"] / df["Close"])) * pd.Series(
            np.log(df["Low"] / df["Open"])
        )
        rs_var = (term1 + term2).rolling(window=window).mean()
        df["Roger_Satchel_Volatility"] = np.sqrt(rs_var) * np.sqrt(365)
        return df

    # Adding Lagged Volatility Features
    def _add_lagged_volatility(self, df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
        """
        Adds lagged volatility features to the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
            lags (list[int]): List of lag periods to create features for.
        Returns:
            pd.DataFrame: DataFrame with added lagged volatility features.
        """
        for lag in lags:
            df[f"Volatility_Lag_{lag}"] = df["Volatility"].shift(lag)
        return df

    # Adding Rolling Statistics for Volatility
    def _calculate_rolling_statistics(
        self, df: pd.DataFrame, windows: list[int]
    ) -> pd.DataFrame:
        """
        Calculates rolling statistics (mean and standard deviation) for the volatility feature.
        Args:
            df (pd.DataFrame): Input DataFrame with a 'Volatility' column.
            windows (list[int]): List of window sizes to calculate rolling statistics for.
        Returns:
            pd.DataFrame: DataFrame with added rolling mean and standard deviation features for volatility.
        """
        for window in windows:
            df[f"Volatility_MA_{window}"] = (
                df["Volatility"].rolling(window=window).mean()
            )
            df[f"Volatility_STD_{window}"] = (
                df["Volatility"].rolling(window=window).std()
            )
        return df

    # Adding Volatility Change Features
    def _calculate_volatility_change(
        self, df: pd.DataFrame, window: list[int]
    ) -> pd.DataFrame:
        """
        Calculates the change in volatility and adds it as a new feature.
        Args:
            df (pd.DataFrame): Input DataFrame with a 'Volatility' column.
            window (list[int]): List of periods to calculate volatility change for.
        Returns:
            pd.DataFrame: DataFrame with an added 'Volatility_Change' column representing the change in volatility.
        """
        for w in window:
            df[f"Volatility_Change_{w}"] = df["Volatility"].diff(periods=w)
        return df

    # Adding Volatility Momentum Feature
    def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the momentum of volatility and adds it as a new feature.
        Args:
            df (pd.DataFrame): Input DataFrame with a 'Volatility' column.
        Returns:
            pd.DataFrame: DataFrame with an added 'Volatility_Momentum' column representing the momentum of volatility.
        """
        df["Volatility_Momentum"] = df["Volatility"] - df["Volatility_MA_30"]
        return df

    # Select only relevant features for the model
    def _select_features(
        self, df: pd.DataFrame, features_list: list[str]
    ) -> pd.DataFrame:
        """
        Selects only the relevant features from the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
            features_list (list[str]): List of feature names to select.
        Returns:
            pd.DataFrame: DataFrame with only the selected features.
        """
        return df[features_list]

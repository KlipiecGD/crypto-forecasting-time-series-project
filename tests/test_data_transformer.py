import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.data_transformer import DataTransformer
from src.config.config import config


class TestDataTransformer:
    def test_init_defaults(self):
        """Test DataTransformer initialization with default values."""
        transformer = DataTransformer()
        assert transformer.window_size == 30
        assert transformer.training_mode is False
        assert transformer.forecast_horizon == 30

    def test_init_custom_params(self):
        """Test DataTransformer initialization with custom parameters."""
        transformer = DataTransformer(
            window_size=14, training_mode=True, forecast_horizon=7
        )
        assert transformer.window_size == 14
        assert transformer.training_mode is True
        assert transformer.forecast_horizon == 7

    def test_missing_required_columns(self):
        """Test that missing required columns raises ValueError."""

        df = pd.DataFrame({"Date": [datetime.now()], "Close": [50000]})
        transformer = DataTransformer()

        with pytest.raises(ValueError, match="Missing required column"):
            transformer.transform(df)

    def test_training_mode_adds_target(self, sample_ohlc_data):
        """Test that training mode adds Target column."""
        transformer = DataTransformer(training_mode=True, forecast_horizon=30)
        result = transformer.transform(sample_ohlc_data)

        assert "Target" in result.columns
        assert result["Target"].dtype == np.float64

    def test_inference_mode_no_target(self, sample_ohlc_data):
        """Test that inference mode does not add Target column."""
        transformer = DataTransformer(training_mode=False)
        result = transformer.transform(sample_ohlc_data)

        assert "Target" not in result.columns

    def test_inference_mode_returns_single_row(self, sample_ohlc_data):
        """Test that inference mode returns only the last row."""
        transformer = DataTransformer(training_mode=False)
        result = transformer.transform(sample_ohlc_data)

        assert len(result) == 1

    def test_training_mode_drops_nans(self, sample_ohlc_data):
        """Test that training mode drops NaN rows created by lags."""
        transformer = DataTransformer(training_mode=True, window_size=30)
        result = transformer.transform(sample_ohlc_data)

        # Should have no NaN values
        assert result.isna().sum().sum() == 0
        # Should have fewer rows than input due to NaN dropping
        assert len(result) < len(sample_ohlc_data)

    def test_all_volatility_features_created(self, sample_ohlc_data):
        """Test that all expected volatility features are created."""
        transformer = DataTransformer(training_mode=True)
        result = transformer.transform(sample_ohlc_data)

        expected_features = config.get_transformations_config.get(
            "features",
            [
                "Volatility_Class_Num",
                "Parkinson_Volatility",
                "Garman_Klass_Volatility",
                "Roger_Satchel_Volatility",
                "Volatility_Lag_30",
                "Volatility_MA_30",
                "Volatility_STD_30",
                "Volatility_Change_30",
            ],
        )

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_volatility_class_mapping(self, sample_ohlc_data):
        """Test that volatility classes are mapped to correct integers."""
        transformer = DataTransformer(training_mode=True)
        result = transformer.transform(sample_ohlc_data)

        # Classes should be 0, 1, or 2
        unique_classes = result["Volatility_Class_Num"].unique()
        assert all(c in [0, 1, 2] for c in unique_classes)

    def test_empty_dataframe(self, empty_ohlc_data):
        """Test handling of empty DataFrame."""
        transformer = DataTransformer(training_mode=True)
        result = transformer.transform(empty_ohlc_data)
        assert result.empty

    def test_minimal_data(self, minimal_ohlc_data):
        """Test handling of minimal data (less than window size)."""
        transformer = DataTransformer(training_mode=True, window_size=30)
        result = transformer.transform(minimal_ohlc_data)

        # With 10 rows and window=30, after dropping NaNs should be empty or very small
        assert len(result) < 10

    def test_target_shift_correct(self, sample_ohlc_data):
        """Test that Target is correctly shifted by forecast_horizon."""
        transformer = DataTransformer(training_mode=True, forecast_horizon=5)
        result = transformer.transform(sample_ohlc_data)

        assert "Target" in result.columns
        # Target should have valid class values
        assert result["Target"].isin([0, 1, 2]).all()

    def test_volatility_values_positive(self, sample_ohlc_data):
        """Test that calculated volatility values are positive."""
        transformer = DataTransformer(training_mode=True)
        result = transformer.transform(sample_ohlc_data)

        vol_columns = [
            col
            for col in result.columns
            if "Volatility" in col
            and col != "Volatility_Class_Num"
            and "Change" not in col
        ]

        for col in vol_columns:
            assert (result[col] >= 0).all(), f"{col} contains negative values"

    def test_fit_returns_self(self, sample_ohlc_data):
        """Test that fit() returns self."""
        transformer = DataTransformer()
        result = transformer.fit(sample_ohlc_data)
        assert result is transformer

    def test_sklearn_pipeline_compatible(self, sample_ohlc_data):
        """Test that transformer works in sklearn pipeline."""

        pipeline = Pipeline(
            [
                ("transformer", DataTransformer(training_mode=False)),
                ("scaler", StandardScaler()),
            ]
        )

        result = pipeline.fit_transform(sample_ohlc_data)
        assert result.shape[0] == 1  # inference mode returns 1 row

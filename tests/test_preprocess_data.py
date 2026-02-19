import pytest
import pandas as pd

from src.features.preprocess_data import preprocess_data
from src.config.config import config

expected_columns = config.get_pipeline_config.get(
    "required_columns", ["Date", "Open", "High", "Low", "Close"]
)


class TestPreprocessData:
    def test_default_columns(self, raw_binance_data):
        """Test preprocessing with default columns."""
        result = preprocess_data(raw_binance_data)

        assert list(result.columns) == expected_columns

    def test_custom_columns(self, raw_binance_data):
        """Test preprocessing with custom columns including Volume."""
        result = preprocess_data(
            raw_binance_data, required_columns=expected_columns + ["Volume"]
        )

        assert "Volume" in result.columns
        assert len(result.columns) == 6

    def test_timestamp_conversion(self, raw_binance_data):
        """Test that timestamps are converted to datetime."""
        result = preprocess_data(raw_binance_data)

        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_price_conversion_to_float(self, raw_binance_data):
        """Test that price strings are converted to float."""
        result = preprocess_data(raw_binance_data)

        for col in ["Open", "High", "Low", "Close"]:
            assert pd.api.types.is_float_dtype(result[col])
            assert not result[col].isna().any()

    def test_correct_number_of_rows(self, raw_binance_data):
        """Test that row count is preserved."""
        result = preprocess_data(raw_binance_data)
        assert len(result) == len(raw_binance_data)

    def test_handles_invalid_prices(self, invalid_price_data):
        """Test handling of invalid price strings."""
        result = preprocess_data(invalid_price_data)

        # Invalid price should be NaN
        assert result["Open"].isna().iloc[0]
        assert not result["Open"].isna().iloc[1]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = preprocess_data(df)

        assert result.empty
        assert list(result.columns) == expected_columns

    def test_single_row(self, single_row_binance_data):
        """Test processing single row of data."""
        result = preprocess_data(single_row_binance_data)

        assert len(result) == 1
        assert result["Close"].iloc[0] == 50500.0

    def test_selects_only_required_columns(self, raw_binance_data):
        """Test that only required number of columns are selected."""
        result = preprocess_data(raw_binance_data)

        assert len(result.columns) == 5
        assert "Date" in result.columns

import pytest
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures.sample_data import (
    generate_sample_ohlc_data,
    generate_raw_binance_data,
    generate_csv_data,
    generate_minimal_ohlc_data,
    generate_empty_ohlc_data,
    generate_invalid_price_data,
    generate_single_row_binance_data,
    generate_corrupted_csv_content,
    generate_no_date_column_data,
    generate_string_dates_data,
)


# Data Transformer Fixtures


@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Standard OHLC data for transformer tests."""
    return generate_sample_ohlc_data(n_days=100)


@pytest.fixture
def minimal_ohlc_data() -> pd.DataFrame:
    """Minimal OHLC data (less than window size)."""
    return generate_minimal_ohlc_data(n_days=10)


@pytest.fixture
def empty_ohlc_data() -> pd.DataFrame:
    """Empty OHLC DataFrame."""
    return generate_empty_ohlc_data()


# Preprocess Data Fixtures


@pytest.fixture
def raw_binance_data() -> pd.DataFrame:
    """Raw Binance API format data."""
    return generate_raw_binance_data(n_days=10)


@pytest.fixture
def invalid_price_data() -> pd.DataFrame:
    """Binance data with invalid price strings."""
    return generate_invalid_price_data()


@pytest.fixture
def single_row_binance_data() -> pd.DataFrame:
    """Single row of Binance data."""
    return generate_single_row_binance_data()


# Compare Files Fixtures ───


@pytest.fixture
def temp_csv_files() -> dict:
    """Create temporary CSV files for comparison testing."""
    temp_dir = tempfile.mkdtemp()

    # Existing file with older data
    existing_data = generate_csv_data(start_date="2024-01-01", n_days=10)
    existing_path = Path(temp_dir) / "existing.csv"
    existing_data.to_csv(existing_path, index=False)

    # New file with newer data
    new_data_newer = generate_csv_data(start_date="2024-01-01", n_days=15)
    new_path_newer = Path(temp_dir) / "new_newer.csv"
    new_data_newer.to_csv(new_path_newer, index=False)

    # New file with same max date
    new_data_same = generate_csv_data(start_date="2024-01-01", n_days=10)
    new_path_same = Path(temp_dir) / "new_same.csv"
    new_data_same.to_csv(new_path_same, index=False)

    # New file with older data
    new_data_older = generate_csv_data(start_date="2024-01-01", n_days=5)
    new_path_older = Path(temp_dir) / "new_older.csv"
    new_data_older.to_csv(new_path_older, index=False)

    return {
        "temp_dir": temp_dir,
        "existing": str(existing_path),
        "new_newer": str(new_path_newer),
        "new_same": str(new_path_same),
        "new_older": str(new_path_older),
    }


@pytest.fixture
def corrupted_csv_file(temp_csv_files) -> str:
    """Create corrupted CSV file."""
    temp_dir = Path(temp_csv_files["existing"]).parent
    corrupted_path = temp_dir / "corrupted.csv"
    with open(corrupted_path, "w") as f:
        f.write(generate_corrupted_csv_content())
    return str(corrupted_path)


@pytest.fixture
def no_date_csv_file(temp_csv_files) -> str:
    """Create CSV file without Date column."""
    temp_dir = Path(temp_csv_files["existing"]).parent
    no_date_path = temp_dir / "no_date.csv"

    df = generate_no_date_column_data()
    df.to_csv(no_date_path, index=False)
    return str(no_date_path)


@pytest.fixture
def string_dates_csv_file(temp_csv_files) -> str:
    """Create CSV file with string-formatted dates."""
    temp_dir = Path(temp_csv_files["existing"]).parent
    string_dates_path = temp_dir / "string_dates.csv"

    df = generate_string_dates_data()
    df.to_csv(string_dates_path, index=False)
    return str(string_dates_path)


@pytest.fixture
def empty_csv_file(temp_csv_files) -> str:
    """Create empty CSV file with header only."""
    temp_dir = Path(temp_csv_files["existing"]).parent
    empty_path = temp_dir / "empty.csv"

    pd.DataFrame(columns=["Date"]).to_csv(empty_path, index=False)
    return str(empty_path)

import pytest
from pathlib import Path

from src.fetch_data.compare_files import compare_csv_files


class TestCompareFiles:
    def test_existing_file_not_found(self, temp_csv_files):
        """Test that missing existing file returns True."""
        result = compare_csv_files("nonexistent.csv", temp_csv_files["new_newer"])
        assert result is True

    def test_new_data_detected(self, temp_csv_files):
        """Test that newer data is detected correctly."""
        result = compare_csv_files(
            temp_csv_files["existing"], temp_csv_files["new_newer"]
        )
        assert result is True

    def test_same_max_date(self, temp_csv_files):
        """Test that same max date returns False."""
        result = compare_csv_files(
            temp_csv_files["existing"], temp_csv_files["new_same"]
        )
        assert result is False

    def test_older_data(self, temp_csv_files):
        """Test that older data returns False."""
        result = compare_csv_files(
            temp_csv_files["existing"], temp_csv_files["new_older"]
        )
        assert result is False

    def test_corrupted_existing_file(self, corrupted_csv_file, temp_csv_files):
        """Test handling of corrupted existing file."""
        result = compare_csv_files(corrupted_csv_file, temp_csv_files["new_newer"])
        # Should default to True on error
        assert result is True

    def test_corrupted_new_file(self, temp_csv_files):
        """Test handling of corrupted new file."""
        temp_dir = Path(temp_csv_files["temp_dir"])
        corrupted_path = temp_dir / "corrupted_new.csv"
        with open(corrupted_path, "w") as f:
            f.write("garbage data")

        result = compare_csv_files(temp_csv_files["existing"], str(corrupted_path))
        # Should default to True on error
        assert result is True

    def test_missing_date_column_existing(self, no_date_csv_file, temp_csv_files):
        """Test handling when existing file has no Date column."""
        result = compare_csv_files(no_date_csv_file, temp_csv_files["new_newer"])
        # Should default to True on error
        assert result is True

    def test_empty_existing_file(self, empty_csv_file, temp_csv_files):
        """Test handling of empty existing file."""
        result = compare_csv_files(empty_csv_file, temp_csv_files["new_newer"])
        # Should default to True on error
        assert result is True

    def test_date_format_compatibility(self, string_dates_csv_file, temp_csv_files):
        """Test that different date formats are handled correctly."""
        result = compare_csv_files(temp_csv_files["existing"], string_dates_csv_file)
        # String dates "2024-01-17" > existing max "2024-01-10"
        assert result is True

"""
Tests for cricket data loader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from Ball_By_Ball.data_loader import CricketDataLoader


class TestCricketDataLoader:
    """Test cases for CricketDataLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = CricketDataLoader(data_dir="test_data")

    def test_initialization(self):
        """Test data loader initialization."""
        assert self.loader.data_dir == Path("test_data")
        assert len(self.loader.required_columns) > 0

    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_load_historical_data_success(self, mock_exists, mock_read_csv):
        """Test successful data loading."""
        mock_exists.return_value = True

        # Create mock data
        mock_data = pd.DataFrame({
            'match_id': [1, 2, 3],
            'runs_off_bat': [0, 1, 4],
            'batting_team': ['A', 'B', 'A']
        })
        mock_read_csv.return_value = mock_data

        result = self.loader.load_historical_data("test.csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_read_csv.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_load_historical_data_file_not_found(self, mock_exists):
        """Test file not found error."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            self.loader.load_historical_data("nonexistent.csv")

    def test_get_data_info(self):
        """Test data info extraction."""
        test_data = pd.DataFrame({
            'match_id': [1, 2, 3],
            'runs_off_bat': [0, 1, 4],
            'batting_team': ['A', 'B', 'A']
        })

        info = self.loader.get_data_info(test_data)

        assert info['shape'] == (3, 3)
        assert 'runs_off_bat' in info['columns']
        assert info['target_distribution'] is not None

    def test_validate_data_quality(self):
        """Test data quality validation."""
        test_data = pd.DataFrame({
            'match_id': [1, 2, 3],
            'runs_off_bat': [0, 1, 4],
            'batting_team': ['A', 'B', 'A']
        })

        quality = self.loader.validate_data_quality(test_data)

        assert quality['total_rows'] == 3
        assert quality['duplicate_rows'] == 0
        assert 'missing_percentage' in quality
"""
Pytest configuration and fixtures for cricket prediction tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_cricket_data():
    """Create sample cricket data for testing."""
    np.random.seed(42)

    n_samples = 1000

    data = {
        'match_id': np.random.randint(1, 100, n_samples),
        'start_date': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples],
        'venue': np.random.choice(['Melbourne', 'Sydney', 'Brisbane', 'Perth'], n_samples),
        'innings': np.random.choice([1, 2], n_samples),
        'batting_team': np.random.choice(['Australia', 'India', 'England', 'South Africa'], n_samples),
        'bowling_team': np.random.choice(['Australia', 'India', 'England', 'South Africa'], n_samples),
        'completed_over': np.random.randint(1, 21, n_samples),
        'ball_no': np.random.randint(1, 7, n_samples),
        'striker': np.random.choice(['Batsman_A', 'Batsman_B', 'Batsman_C'], n_samples),
        'non_striker': np.random.choice(['Batsman_A', 'Batsman_B', 'Batsman_C'], n_samples),
        'bowler': np.random.choice(['Bowler_X', 'Bowler_Y', 'Bowler_Z'], n_samples),
        'runs_off_bat': np.random.choice([0, 1, 2, 3, 4, 6], n_samples, p=[0.6, 0.25, 0.08, 0.03, 0.03, 0.01]),
        'extras': np.random.choice([0, 1, 2, 4], n_samples, p=[0.9, 0.05, 0.03, 0.02]),
        'wickets_remaining': np.random.randint(1, 11, n_samples),
        'balls_remaining': np.random.randint(1, 121, n_samples),
        'CRR': np.random.uniform(4, 12, n_samples),
        'RRR': np.random.uniform(4, 12, n_samples),
        'batter_sr_l10': np.random.uniform(50, 200, n_samples),
        'batter_avg_l10': np.random.uniform(10, 50, n_samples),
        'bowler_eco_l10': np.random.uniform(4, 12, n_samples),
        'bowler_sr_l10': np.random.uniform(10, 50, n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(sample_cricket_data):
    """Create temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir) / "historical_data.csv"
        sample_cricket_data.to_csv(data_path, index=False)

        yield temp_dir


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def sample_ball_data():
    """Sample ball data for prediction testing."""
    return {
        'match_id': 1,
        'venue': 'Melbourne',
        'innings': 1,
        'batting_team': 'Australia',
        'bowling_team': 'India',
        'completed_over': 10,
        'ball_no': 3,
        'striker': 'Batsman_A',
        'non_striker': 'Batsman_B',
        'bowler': 'Bowler_X',
        'wickets_remaining': 7,
        'balls_remaining': 60,
        'CRR': 6.5,
        'RRR': 8.2,
        'batter_sr_l10': 120.5,
        'batter_avg_l10': 25.3,
        'bowler_eco_l10': 7.2,
        'bowler_sr_l10': 25.8
    }
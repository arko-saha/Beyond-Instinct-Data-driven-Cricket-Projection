"""
Data loading module for cricket analytics.
Handles loading and basic validation of cricket datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CricketDataLoader:
    """Handles loading and basic validation of cricket datasets."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.required_columns = [
            'match_id', 'start_date', 'venue', 'innings', 'batting_team',
            'bowling_team', 'completed_over', 'ball_no', 'striker', 'non_striker',
            'bowler', 'runs_off_bat', 'extras', 'total_runs', 'cumulative_runs',
            'wickets_remaining', 'balls_remaining', 'CRR', 'RRR'
        ]

    def load_historical_data(self, filename: str = "historical_data.csv") -> pd.DataFrame:
        """
        Load historical cricket data from CSV file.

        Args:
            filename: Name of the CSV file to load

        Returns:
            DataFrame containing the cricket data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Validate required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = np.nan

        return df

    def load_player_stats(self, filename: str = "player_stats.csv") -> Optional[pd.DataFrame]:
        """
        Load player statistics data if available.

        Args:
            filename: Name of the player stats file

        Returns:
            DataFrame with player statistics or None if file doesn't exist
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            logger.warning(f"Player stats file not found: {file_path}")
            return None

        logger.info(f"Loading player stats from {file_path}")
        return pd.read_csv(file_path)

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'target_distribution': df['runs_off_bat'].value_counts().to_dict() if 'runs_off_bat' in df.columns else None
        }

        return info

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with quality check results
        """
        quality_checks = {
            'total_rows': len(df),
            'duplicate_rows': df.duplicated().sum(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'target_range': {
                'min': df['runs_off_bat'].min(),
                'max': df['runs_off_bat'].max(),
                'unique_values': sorted(df['runs_off_bat'].unique())
            } if 'runs_off_bat' in df.columns else None
        }

        return quality_checks
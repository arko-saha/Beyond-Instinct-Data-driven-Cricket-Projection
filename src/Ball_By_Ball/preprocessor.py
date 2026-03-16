"""
Data preprocessing module for cricket analytics.
Handles data cleaning, type conversion, and basic preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CricketDataPreprocessor:
    """Handles data preprocessing for cricket analytics."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.categorical_columns = [
            'match_id', 'venue', 'innings', 'batting_team', 'bowling_team',
            'striker', 'non_striker', 'bowler', 'completed_over', 'ball_no',
            'fall_of_wicket', 'player_dismissed'
        ]

        self.numeric_columns = [
            'runs_off_bat', 'extras', 'total_runs', 'cumulative_runs',
            'wickets_remaining', 'balls_remaining', 'CRR', 'RRR',
            'batter_sr_l10', 'batter_avg_l10', 'bowler_eco_l10', 'bowler_sr_l10'
        ]

        self.datetime_columns = ['start_date']

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            df: Raw cricket data DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")

        # Create a copy to avoid modifying original
        df_processed = df.copy()

        # Rename columns for consistency
        df_processed = self._rename_columns(df_processed)

        # Add missing columns
        df_processed = self._add_missing_columns(df_processed)

        # Convert data types
        df_processed = self._convert_data_types(df_processed)

        # Create derived features
        df_processed = self._create_derived_features(df_processed)

        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)

        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        return df_processed

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns for consistency."""
        rename_dict = {
            'batter': 'striker',
            'player_out': 'player_dismissed',
            'wicket_type': 'fall_of_wicket'
        }

        df_renamed = df.rename(columns=rename_dict)
        logger.info(f"Renamed columns: {rename_dict}")
        return df_renamed

    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing columns with default values."""
        missing_cols = ['other_wicket_type', 'other_player_dismissed']

        for col in missing_cols:
            if col not in df.columns:
                df[col] = np.nan
                logger.info(f"Added missing column: {col}")

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types for memory efficiency."""
        df_converted = df.copy()

        # Convert categorical columns
        for col in self.categorical_columns:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype('category')

        # Convert datetime columns
        for col in self.datetime_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_datetime(df_converted[col])

        # Convert numeric columns with downcasting
        for col in self.numeric_columns:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], downcast='integer', errors='coerce')

        logger.info("Data type conversion completed")
        return df_converted

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        df_derived = df.copy()

        # Create fall_of_wicket binary indicator
        if 'fall_of_wicket' in df_derived.columns:
            df_derived['fall_of_wicket'] = df_derived['fall_of_wicket'].notna().astype(int)
        else:
            df_derived['fall_of_wicket'] = 0

        # Create wickets_remaining if not present
        if 'wickets_remaining' not in df_derived.columns and 'cumulative_wickets' in df_derived.columns:
            df_derived['wickets_remaining'] = 10 - df_derived['cumulative_wickets']

        # Create balls_remaining if not present
        if 'balls_remaining' not in df_derived.columns and 'completed_over' in df_derived.columns:
            df_derived['balls_remaining'] = (19 - df_derived['completed_over']) * 6 + (6 - df_derived['ball_no'])

        logger.info("Derived features created")
        return df_derived

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()

        # Fill numeric missing values with median
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")

        # Fill categorical missing values with mode
        categorical_cols = df_cleaned.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val[0]}")

        return df_cleaned

    def get_preprocessing_summary(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict:
        """Get summary of preprocessing changes."""
        summary = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'columns_added': len(df_processed.columns) - len(df_original.columns),
            'missing_values_original': df_original.isnull().sum().sum(),
            'missing_values_processed': df_processed.isnull().sum().sum(),
            'memory_usage_original': df_original.memory_usage(deep=True).sum(),
            'memory_usage_processed': df_processed.memory_usage(deep=True).sum()
        }

        return summary
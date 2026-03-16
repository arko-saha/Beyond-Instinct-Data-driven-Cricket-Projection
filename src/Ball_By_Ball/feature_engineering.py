"""
Feature engineering module for cricket analytics.
Creates advanced features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class CricketFeatureEngineer:
    """Handles advanced feature engineering for cricket data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.label_encoders = {}
        self.feature_columns = []

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for cricket prediction.

        Args:
            df: Preprocessed cricket data

        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Creating advanced features")
        df_featured = df.copy()

        # Performance ratio features
        df_featured = self._create_performance_ratios(df_featured)

        # Match context features
        df_featured = self._create_match_context_features(df_featured)

        # Pressure indicators
        df_featured = self._create_pressure_indicators(df_featured)

        # Team performance indicators
        df_featured = self._create_team_indicators(df_featured)

        # Batting position features
        df_featured = self._create_batting_position_features(df_featured)

        # Interaction features
        df_featured = self._create_interaction_features(df_featured)

        logger.info(f"Created {len(df_featured.columns) - len(df.columns)} additional features")
        return df_featured

    def _create_performance_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance ratio features."""
        df_ratios = df.copy()

        # Batter performance ratio
        if 'batter_sr_l10' in df_ratios.columns and 'batter_avg_l10' in df_ratios.columns:
            df_ratios['batter_performance_ratio'] = df_ratios['batter_sr_l10'] / (df_ratios['batter_avg_l10'] + 1)

        # Bowler efficiency ratio
        if 'bowler_sr_l10' in df_ratios.columns and 'bowler_eco_l10' in df_ratios.columns:
            df_ratios['bowler_efficiency_ratio'] = df_ratios['bowler_sr_l10'] / (df_ratios['bowler_eco_l10'] + 1)

        return df_ratios

    def _create_match_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match context features."""
        df_context = df.copy()

        # Powerplay indicator
        if 'completed_over' in df_context.columns:
            df_context['is_powerplay'] = (df_context['completed_over'] <= 5).astype(int)

        # Death overs indicator
        if 'completed_over' in df_context.columns:
            df_context['is_death_overs'] = (df_context['completed_over'] >= 16).astype(int)

        # Middle overs indicator
        if 'completed_over' in df_context.columns:
            df_context['is_middle_overs'] = ((df_context['completed_over'] >= 6) &
                                           (df_context['completed_over'] <= 15)).astype(int)

        # Balls remaining percentage
        if 'balls_remaining' in df_context.columns:
            df_context['balls_remaining_pct'] = df_context['balls_remaining'] / 120

        # Run rate features
        if 'CRR' in df_context.columns:
            df_context['run_rate_current'] = df_context['CRR']

        if 'RRR' in df_context.columns:
            df_context['run_rate_required'] = df_context['RRR']

        return df_context

    def _create_pressure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pressure situation indicators."""
        df_pressure = df.copy()

        # High pressure situations (last 5 overs, high required rate)
        if 'completed_over' in df_pressure.columns and 'RRR' in df_pressure.columns:
            df_pressure['high_pressure'] = ((df_pressure['completed_over'] >= 15) &
                                          (df_pressure['RRR'] > 8)).astype(int)

        # Chase pressure (team batting last)
        if 'innings' in df_pressure.columns:
            df_pressure['is_chasing'] = (df_pressure['innings'] == 2).astype(int)

        # Target known situations
        if 'RRR' in df_pressure.columns:
            df_pressure['target_known'] = (~df_pressure['RRR'].isnull()).astype(int)

        return df_pressure

    def _create_team_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team performance indicators."""
        df_team = df.copy()

        # Wickets remaining percentage
        if 'wickets_remaining' in df_team.columns:
            df_team['wickets_remaining_pct'] = df_team['wickets_remaining'] / 10

        # Partnership indicators
        if 'cumulative_wickets' in df_team.columns:
            df_team['partnership_size'] = df_team.groupby(['match_id', 'innings', 'cumulative_wickets'])['cumulative_runs'].transform('count')

        return df_team

    def _create_batting_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create batting position-based features."""
        df_position = df.copy()

        # Opening batsman indicator (based on common opening pairs)
        opening_batsmen = ['AC Gilchrist', 'RT Ponting', 'DR Martyn', 'SM Katich', 'ML Hayden']
        if 'striker' in df_position.columns:
            df_position['is_opening_batsman'] = df_position['striker'].isin(opening_batsmen).astype(int)

        # Middle order batsman
        if 'cumulative_wickets' in df_position.columns:
            df_position['is_middle_order'] = ((df_position['cumulative_wickets'] >= 2) &
                                            (df_position['cumulative_wickets'] <= 7)).astype(int)

        # Closing batsman
        if 'cumulative_wickets' in df_position.columns:
            df_position['is_closing_batsman'] = (df_position['cumulative_wickets'] >= 8).astype(int)

        return df_position

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        df_interaction = df.copy()

        # Venue and team interaction
        if 'venue' in df_interaction.columns and 'batting_team' in df_interaction.columns:
            df_interaction['venue_team_interaction'] = (df_interaction['venue'].astype(str) + '_' +
                                                      df_interaction['batting_team'].astype(str))

        # Bowler vs batsman historical performance
        if 'bowler' in df_interaction.columns and 'striker' in df_interaction.columns:
            df_interaction['bowler_batsman_pair'] = (df_interaction['bowler'].astype(str) + '_vs_' +
                                                   df_interaction['striker'].astype(str))

        return df_interaction

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.

        Args:
            df: DataFrame with categorical features
            fit: Whether to fit new encoders or use existing ones

        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()

        categorical_features = [
            'match_id', 'venue', 'innings', 'batting_team', 'bowling_team',
            'striker', 'non_striker', 'bowler', 'completed_over', 'ball_no',
            'fall_of_wicket', 'player_dismissed', 'venue_team_interaction',
            'bowler_batsman_pair'
        ]

        if fit:
            self.label_encoders = {}

        for feature in categorical_features:
            if feature in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                    self.label_encoders[feature] = le
                else:
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Handle unknown categories
                        df_encoded[feature] = df_encoded[feature].astype(str)
                        known_categories = set(le.classes_)
                        df_encoded[feature] = df_encoded[feature].apply(
                            lambda x: x if x in known_categories else 'unknown'
                        )
                        # Add 'unknown' to classes if not present
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                        df_encoded[feature] = le.transform(df_encoded[feature])

        logger.info(f"Encoded {len(categorical_features)} categorical features")
        return df_encoded

    def select_features(self, df: pd.DataFrame, target_column: str = 'runs_off_bat',
                       method: str = 'importance', n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most important features for modeling.

        Args:
            df: Feature DataFrame
            target_column: Target column name
            method: Feature selection method ('importance', 'correlation', 'variance')
            n_features: Number of features to select

        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectKBest, f_regression

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]

        if method == 'importance':
            # Use Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_

            # Get top features
            indices = np.argsort(importances)[::-1][:n_features]
            selected_features = [feature_cols[i] for i in indices]

        elif method == 'correlation':
            # Use correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(n_features).index.tolist()

        elif method == 'variance':
            # Use SelectKBest with f_regression
            selector = SelectKBest(score_func=f_regression, k=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Create selected features DataFrame
        X_selected = X[selected_features]

        logger.info(f"Selected {len(selected_features)} features using {method} method")
        logger.info(f"Top features: {selected_features[:5]}")

        return X_selected, selected_features

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of engineered features."""
        summary = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['category', 'object']).columns),
            'encoded_features': len(self.label_encoders),
            'feature_types': {
                'performance_ratios': [col for col in df.columns if 'ratio' in col.lower()],
                'match_context': [col for col in df.columns if any(x in col.lower() for x in ['powerplay', 'death', 'middle', 'remaining_pct'])],
                'pressure_indicators': [col for col in df.columns if any(x in col.lower() for x in ['pressure', 'chasing', 'target_known'])],
                'team_indicators': [col for col in df.columns if any(x in col.lower() for x in ['wickets_remaining_pct', 'partnership'])],
                'position_features': [col for col in df.columns if any(x in col.lower() for x in ['opening', 'middle_order', 'closing'])]
            }
        }

        return summary
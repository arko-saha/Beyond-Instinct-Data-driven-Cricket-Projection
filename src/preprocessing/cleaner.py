"""
Data-cleaning utilities for cricket ball-by-ball data.

Applies the cleaning steps originally found in the legacy
``Data_preprocessing.ipynb`` notebook:

* Select and reorder the canonical column subset.
* Remove trailing all-null rows.
* Filter out super-over innings.
* Create a binary ``fall_of_wicket`` indicator.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column order used throughout the project.
# ---------------------------------------------------------------------------
_COLUMN_ORDER: list[str] = [
    "match_id",
    "start_date",
    "venue",
    "innings",
    "batting_team",
    "bowling_team",
    "completed_over",
    "ball_no",
    "striker",
    "non_striker",
    "bowler",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "wicket_type",
    "player_dismissed",
    "other_wicket_type",
    "other_player_dismissed",
    "total_runs",
]

# Innings values that represent super overs or erroneous data.
_INVALID_INNINGS = {"3", "4", "5", "7", 3, 4, 5, 7}


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to the canonical project order.

    Columns present in *df* but absent from the canonical list are appended
    at the end.  Missing canonical columns are silently skipped.
    """
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def remove_trailing_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the last row if it is entirely null."""
    if df.iloc[-1].isnull().all():
        return df.iloc[:-1].reset_index(drop=True)
    return df


def remove_super_overs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where ``innings`` is 3, 4, 5, or 7 (super overs / errors)."""
    mask = df["innings"].isin(_INVALID_INNINGS)
    return df[~mask].reset_index(drop=True)


def create_wicket_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``fall_of_wicket`` column (1 if a wicket fell, 0 otherwise)."""
    df["fall_of_wicket"] = df["wicket_type"].notna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Steps (in order):
    1. Reorder columns.
    2. Remove trailing null rows.
    3. Remove super-over innings.
    4. Create ``fall_of_wicket`` indicator.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that has already been loaded and had ball numbers parsed.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = reorder_columns(df)
    df = remove_trailing_nulls(df)
    df = remove_super_overs(df)
    df = create_wicket_indicator(df)
    return df

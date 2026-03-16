"""
Data loader for Cricsheet ball-by-ball CSV files.

Handles loading raw CSV data and standardizing column names to match the
project's canonical schema (e.g. ``batter`` → ``striker``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column mapping: raw Cricsheet names → internal canonical names
# ---------------------------------------------------------------------------
_COLUMN_RENAMES: dict[str, str] = {
    "batter": "striker",
    "player_out": "player_dismissed",
}

# Columns that may be absent in the raw CSV; added with NaN if missing.
_OPTIONAL_COLUMNS: list[str] = [
    "other_wicket_type",
    "other_player_dismissed",
]


def load_match_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a Cricsheet ball-by-ball CSV and standardize column names.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw ``merged.csv`` (or equivalent) file produced by the
        Cricsheet data extraction pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns (``striker``, ``player_dismissed``) and
        any missing optional columns filled with ``NaN``.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename columns that differ between the raw Cricsheet schema and the
    # internal schema used throughout the project.
    rename_map = {k: v for k, v in _COLUMN_RENAMES.items() if k in df.columns}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure optional columns exist (filled with NaN).
    for col in _OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df

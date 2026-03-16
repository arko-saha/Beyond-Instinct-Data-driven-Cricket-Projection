"""
Preprocessing package for cricket ball-by-ball data.

Transforms raw Cricsheet CSV data into clean, feature-enriched DataFrames
ready for predictive modeling. This module was refactored from the legacy
``Data_preprocessing.ipynb`` notebook.

Modules
-------
loader
    Load raw CSV and standardize column names.
ball_parser
    Separate the ``ball`` column into ``completed_over`` and ``ball_no``.
cleaner
    Select columns, remove super overs, create wicket indicators.
features
    Engineer cumulative, rate-based, and rolling player-form features.
pipeline
    CLI entry point that orchestrates the full pipeline.
"""

from .loader import load_match_data
from .ball_parser import separate_ball_numbers
from .cleaner import clean
from .features import engineer_features

__all__ = [
    "load_match_data",
    "separate_ball_numbers",
    "clean",
    "engineer_features",
]

"""
forecast/config.py
==================
Centralized configuration constants for the Beyond Instinct Forecast Engine.

All hardcoded magic numbers that were previously scattered across notebook
cells are defined here. Change a value in ONE place; every downstream module
picks it up automatically.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (resolved relative to this file's location in src/forecast/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from src/forecast/

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BALL_BY_BALL_PATH = DATA_DIR / "consolidated_t20_data.csv"
MATCH_METADATA_PATH = DATA_DIR / "match_metadata.csv"
BATTERS_CLEAN_PATH = DATA_DIR / "batters_clean.csv"
BOWLERS_CLEAN_PATH = DATA_DIR / "bowlers_clean.csv"

# ---------------------------------------------------------------------------
# Data split — chronological, not random
# ---------------------------------------------------------------------------
# All matches with start_date < CALIBRATION_CUTOFF are used to build lookups
# and compute player XP. Matches on or after the cutoff form the evaluation
# set and must NEVER appear in any lookup table.
CALIBRATION_CUTOFF_DATE = "2024-01-01"  # train on pre-2024, evaluate on 2024+

# ---------------------------------------------------------------------------
# Empirical lookup quality thresholds
# ---------------------------------------------------------------------------
# A lookup cell must have at least this many observations to be considered
# statistically reliable. Below this threshold, we fall back to a coarser
# aggregation level. This was previously hardcoded as `>= 30` inline.
MIN_SAMPLE_SIZE: int = 30

# ---------------------------------------------------------------------------
# Match state constants
# ---------------------------------------------------------------------------
MAX_LEGAL_BALLS_PER_INNINGS: int = 120   # 20 overs × 6 legal balls
MAX_WICKETS: int = 10

# ---------------------------------------------------------------------------
# Rate bin breakpoints (RPB thresholds used by calculate_rate_bin)
# ---------------------------------------------------------------------------
RATE_BINS = {
    "very_low":  (None, 0.80),
    "low":       (0.80, 1.10),
    "medium":    (1.10, 1.40),
    "high":      (1.40, 1.70),
    "very_high": (1.70, None),
}

# ---------------------------------------------------------------------------
# Monte Carlo simulation parameters (Phase 3)
# ---------------------------------------------------------------------------
MC_N_SIMULATIONS: int = 10_000
MC_RANDOM_SEED: int = 42          # set to None for fully stochastic runs

# ---------------------------------------------------------------------------
# Backtesting parameters (Phase 4)
# ---------------------------------------------------------------------------
BACKTEST_FREEZE_OVER: int = 10    # freeze match state at end of this over
BACKTEST_COVERAGE_TARGET: float = 0.80   # ≥80% of actuals inside P10–P90
BACKTEST_MAE_TARGET: float = 12.0        # runs — acceptable median abs error

# ---------------------------------------------------------------------------
# Player XP eligibility thresholds
# ---------------------------------------------------------------------------
# Minimum balls faced/bowled for a player to have a statistically meaningful XP.
# Players below this threshold are treated as "unknown" (XP delta = 0.0).
MIN_BALLS_BATTER: int = 120
MIN_BALLS_BOWLER: int = 120

"""
Feature-engineering module for cricket ball-by-ball data.

Transforms a cleaned ball-by-ball DataFrame into a feature-rich dataset
suitable for predictive modeling.  Mirrors—and improves upon—the logic from
the legacy ``Data_preprocessing.ipynb`` notebook.

Feature groups
--------------
1. **Cumulative in-innings stats** — runs, wickets, wickets in hand.
2. **Ball tracking** — total balls bowled, balls remaining, current run rate.
3. **Required run rate (RRR)** — 2nd-innings chase dynamics.
4. **Rolling batter form** — strike rate & average over last 10 matches.
5. **Rolling bowler form** — economy & strike rate over last 10 matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BALLS_PER_INNINGS = 120  # T20: 20 overs × 6 balls
_ROLLING_WINDOW = 10
_RRR_CAP = 36.0  # 6 runs per ball = theoretical max

# Default fallback values when a player has no prior history.
_DEFAULT_BATTER_SR = 120.0
_DEFAULT_BATTER_AVG = 25.0
_DEFAULT_BOWLER_ECO = 8.0
_DEFAULT_BOWLER_SR = 20.0


# ---------------------------------------------------------------------------
# 1. Cumulative in-innings statistics
# ---------------------------------------------------------------------------

def add_cumulative_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative runs, wicket counts, and wickets-in-hand columns.

    New columns: ``cumulative_runs``, ``is_wicket``, ``cumulative_wickets``,
    ``wickets_in_hand``.
    """
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0)
    df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0)
    df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce").fillna(0)

    df["cumulative_runs"] = df.groupby(["match_id", "innings"])["total_runs"].cumsum()

    df["is_wicket"] = df["player_dismissed"].notna().astype(int)
    df["cumulative_wickets"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["wickets_in_hand"] = 10 - df["cumulative_wickets"]

    return df


# ---------------------------------------------------------------------------
# 2. Ball tracking & current run rate
# ---------------------------------------------------------------------------

def add_ball_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Add total balls bowled, balls remaining, and current run rate (CRR).

    New columns: ``total_balls_bowled``, ``balls_remaining``, ``CRR``.
    """
    df["total_balls_bowled"] = df.groupby(["match_id", "innings"]).cumcount() + 1
    df["balls_remaining"] = (_BALLS_PER_INNINGS - df["total_balls_bowled"]).clip(lower=0)

    # CRR = (cumulative_runs / balls bowled) × 6  (runs per over).
    df["CRR"] = (df["cumulative_runs"] / df["total_balls_bowled"]) * 6

    return df


# ---------------------------------------------------------------------------
# 3. Required run rate (2nd innings only)
# ---------------------------------------------------------------------------

def add_required_run_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target runs and required run rate for 2nd-innings chases.

    New columns: ``target_runs``, ``RRR``.
    """
    innings_col = df["innings"].astype(str)

    # Compute 1st-innings totals.
    first_mask = innings_col == "1"
    first_totals = (
        df.loc[first_mask]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target_runs"})
    )
    first_totals["target_runs"] += 1  # target = 1st-innings total + 1

    df = df.merge(first_totals, on="match_id", how="left")
    df["target_runs"] = df["target_runs"].fillna(-1)

    # RRR only applies to the 2nd innings.
    innings_str = df["innings"].astype(str)
    safe_balls = df["balls_remaining"].replace(0, 1)
    df["RRR"] = np.where(
        innings_str == "2",
        ((df["target_runs"] - df["cumulative_runs"]) / safe_balls) * 6,
        0,
    )
    df["RRR"] = df["RRR"].clip(upper=_RRR_CAP)

    return df


# ---------------------------------------------------------------------------
# 4. Rolling batter form (last N matches)
# ---------------------------------------------------------------------------

def add_rolling_batter_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-10-match rolling strike rate and batting average.

    New columns: ``batter_sr_l10``, ``batter_avg_l10``.
    """
    batter_match = (
        df.groupby(["striker", "match_id", "start_date"])
        .agg(
            runs_scored=("runs_off_bat", "sum"),
            balls_faced=("ball_no", "count"),
            dismissed=("is_wicket", "sum"),
        )
        .reset_index()
        .sort_values(by=["striker", "start_date"])
    )

    for col in ("runs_scored", "balls_faced", "dismissed"):
        batter_match[f"{col}_l{_ROLLING_WINDOW}"] = (
            batter_match
            .groupby("striker")[col]
            .transform(
                lambda s: s.shift(1).rolling(_ROLLING_WINDOW, min_periods=1).sum()
            )
        )

    batter_match["batter_sr_l10"] = (
        batter_match[f"runs_scored_l{_ROLLING_WINDOW}"]
        / batter_match[f"balls_faced_l{_ROLLING_WINDOW}"]
    ) * 100

    batter_match["batter_avg_l10"] = (
        batter_match[f"runs_scored_l{_ROLLING_WINDOW}"]
        / batter_match[f"dismissed_l{_ROLLING_WINDOW}"].replace(0, 1)
    )

    df = df.merge(
        batter_match[["match_id", "striker", "batter_sr_l10", "batter_avg_l10"]],
        on=["match_id", "striker"],
        how="left",
    )
    df["batter_sr_l10"] = df["batter_sr_l10"].fillna(_DEFAULT_BATTER_SR)
    df["batter_avg_l10"] = df["batter_avg_l10"].fillna(_DEFAULT_BATTER_AVG)

    return df


# ---------------------------------------------------------------------------
# 5. Rolling bowler form (last N matches)
# ---------------------------------------------------------------------------

def add_rolling_bowler_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-10-match rolling economy and bowling strike rate.

    New columns: ``bowler_eco_l10``, ``bowler_sr_l10``.
    """
    bowler_match = (
        df.groupby(["bowler", "match_id", "start_date"])
        .agg(
            runs_conceded=("total_runs", "sum"),
            balls_bowled=("ball_no", "count"),
            wickets_taken=("is_wicket", "sum"),
        )
        .reset_index()
        .sort_values(by=["bowler", "start_date"])
    )

    for col in ("runs_conceded", "balls_bowled", "wickets_taken"):
        bowler_match[f"{col}_l{_ROLLING_WINDOW}"] = (
            bowler_match
            .groupby("bowler")[col]
            .transform(
                lambda s: s.shift(1).rolling(_ROLLING_WINDOW, min_periods=1).sum()
            )
        )

    bowler_match["bowler_eco_l10"] = (
        bowler_match[f"runs_conceded_l{_ROLLING_WINDOW}"]
        / (bowler_match[f"balls_bowled_l{_ROLLING_WINDOW}"] / 6)
    )

    bowler_match["bowler_sr_l10"] = (
        bowler_match[f"balls_bowled_l{_ROLLING_WINDOW}"]
        / bowler_match[f"wickets_taken_l{_ROLLING_WINDOW}"].replace(0, 1)
    )

    df = df.merge(
        bowler_match[["match_id", "bowler", "bowler_eco_l10", "bowler_sr_l10"]],
        on=["match_id", "bowler"],
        how="left",
    )
    df["bowler_eco_l10"] = df["bowler_eco_l10"].fillna(_DEFAULT_BOWLER_ECO)
    df["bowler_sr_l10"] = df["bowler_sr_l10"].fillna(_DEFAULT_BOWLER_SR)

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Must be called on a **cleaned** DataFrame (output of :func:`cleaner.clean`),
    sorted chronologically.

    Steps (in order):
    1. Sort rows chronologically.
    2. Add cumulative runs / wickets.
    3. Add ball tracking and current run rate.
    4. Add required run rate.
    5. Add rolling batter form.
    6. Add rolling bowler form.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned ball-by-ball DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    df = df.sort_values(
        by=["start_date", "match_id", "innings", "completed_over", "ball_no"]
    ).reset_index(drop=True)

    df = add_cumulative_stats(df)
    df = add_ball_tracking(df)
    df = add_required_run_rate(df)

    print("Calculating rolling batter form features…")
    df = add_rolling_batter_form(df)

    print("Calculating rolling bowler form features…")
    df = add_rolling_bowler_form(df)

    print("Feature engineering complete.")
    return df

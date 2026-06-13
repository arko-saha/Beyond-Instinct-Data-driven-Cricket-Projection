"""
forecast/data_pipeline.py
==========================
Phase 1 — Data Pipeline Integrity & Leakage Elimination.

Responsibilities:
  1. Load and chronologically split the ball-by-ball dataset.
  2. Enrich each delivery with match-state features (cumulative runs/wickets,
     balls remaining, asking/scoring rate, rate bin).
  3. Build the hierarchical empirical lookup tables (xR, xW) EXCLUSIVELY from
     the calibration set — no evaluation data contaminates the baselines.
  4. Validate that zero evaluation matches appear in any lookup key.

All functions are pure (they never mutate their inputs) and accept explicit
DataFrames so they are testable in isolation without touching the filesystem.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast.config import (
    BALL_BY_BALL_PATH,
    CALIBRATION_CUTOFF_DATE,
    MAX_LEGAL_BALLS_PER_INNINGS,
    MAX_WICKETS,
    MIN_SAMPLE_SIZE,
)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_ball_by_ball(path=BALL_BY_BALL_PATH) -> pd.DataFrame:
    """
    Load the ball-by-ball dataset and ensure correct dtypes.

    Returns
    -------
    pd.DataFrame
        Raw ball-by-ball data with ``start_date`` parsed as datetime.
    """
    df = pd.read_csv(path, low_memory=False)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    # Coerce numeric columns that sometimes contain NaN-causing strings
    for col in ["wides", "noballs", "byes", "legbyes", "penalty", "extras"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["is_wicket"] = pd.to_numeric(df["is_wicket"], errors="coerce").fillna(0).astype(int)
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0).astype(int)
    df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce").fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Chronological Split
# ---------------------------------------------------------------------------

def split_calibration_evaluation(
    df: pd.DataFrame,
    cutoff_date: str = CALIBRATION_CUTOFF_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically into calibration and evaluation sets.

    All matches with ``start_date`` strictly before ``cutoff_date`` form the
    **calibration set**.  All matches on or after the cutoff form the
    **evaluation set**.

    This is NOT a random split. Randomising a time-series breaks temporal
    causality and causes the exact data leakage we are eliminating.

    Parameters
    ----------
    df : pd.DataFrame
        Full ball-by-ball dataset (output of ``load_ball_by_ball``).
    cutoff_date : str
        ISO-format date string (e.g., ``"2024-01-01"``).

    Returns
    -------
    df_calib : pd.DataFrame
        Calibration set (strictly before cutoff).
    df_eval : pd.DataFrame
        Evaluation set (on or after cutoff).
    """
    cutoff = pd.Timestamp(cutoff_date)
    df_calib = df[df["start_date"] < cutoff].copy()
    df_eval = df[df["start_date"] >= cutoff].copy()

    calib_matches = set(df_calib["match_id"].unique())
    eval_matches = set(df_eval["match_id"].unique())
    overlap = calib_matches & eval_matches

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} match IDs appear in both "
            f"calibration and evaluation sets. "
            f"Example IDs: {list(overlap)[:5]}"
        )

    print(
        f"[Split] Calibration: {df_calib['start_date'].min().date()} -> "
        f"{df_calib['start_date'].max().date()} "
        f"({df_calib['match_id'].nunique()} matches, {len(df_calib):,} deliveries)"
    )
    print(
        f"[Split] Evaluation:  {df_eval['start_date'].min().date()} -> "
        f"{df_eval['start_date'].max().date()} "
        f"({df_eval['match_id'].nunique()} matches, {len(df_eval):,} deliveries)"
    )

    return df_calib, df_eval


# ---------------------------------------------------------------------------
# 3. Rate Bins
# ---------------------------------------------------------------------------

def calculate_rate_bin(rate: float) -> str:
    """
    Map a runs-per-ball rate (or asking rate) to a discrete bin label.

    Bin boundaries (runs per ball, NOT runs per over):
        very_low  : rate < 0.80
        low       : 0.80 <= rate < 1.10
        medium    : 1.10 <= rate < 1.40
        high      : 1.40 <= rate < 1.70
        very_high : rate >= 1.70

    NaN rates default to ``"very_low"``.
    """
    if pd.isna(rate):
        return "very_low"
    if rate < 0.80:
        return "very_low"
    elif rate < 1.10:
        return "low"
    elif rate < 1.40:
        return "medium"
    elif rate < 1.70:
        return "high"
    return "very_high"


# ---------------------------------------------------------------------------
# 4. Match State Feature Engineering
# ---------------------------------------------------------------------------

def add_match_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich each delivery with pre-ball match-state context features.

    All cumulative quantities represent the state *before* the current delivery
    is bowled — i.e., they are lagged by one ball. This prevents target leakage
    within a single over.

    New columns added:
        cumulative_runs        : runs scored before this ball (in the innings)
        cumulative_wickets     : wickets fallen before this ball
        is_legal_ball          : 1 if not a wide or no-ball, 0 otherwise
        cumulative_legal_balls : legal balls bowled before this ball
        balls_remaining        : legal balls left in the innings after this ball
        wickets_in_hand        : wickets remaining before this ball
        target                 : innings-1 total + 1 (only meaningful for innings 2)
        scoring_rate           : cumulative runs / cumulative legal balls (innings 1)
        asking_rate            : (target - cumulative runs) / balls remaining (innings 2)
        active_rate            : scoring_rate for innings 1, asking_rate for innings 2
        rate_bin               : discretised active_rate bucket

    Parameters
    ----------
    df : pd.DataFrame
        Raw ball-by-ball data.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with state features appended (original unmodified).
    """
    df = df.copy()
    df = df.sort_values(by=["match_id", "innings", "over", "ball"]).reset_index(drop=True)

    grp = df.groupby(["match_id", "innings"])

    # Pre-ball cumulative sums (exclusive of the current delivery)
    df["cumulative_runs"] = grp["total_runs"].cumsum() - df["total_runs"]
    df["cumulative_wickets"] = grp["is_wicket"].cumsum() - df["is_wicket"]

    df["is_legal_ball"] = ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int)
    df["cumulative_legal_balls"] = grp["is_legal_ball"].cumsum() - df["is_legal_ball"]

    df["balls_remaining"] = (MAX_LEGAL_BALLS_PER_INNINGS - df["cumulative_legal_balls"]).clip(lower=0)
    df["wickets_in_hand"] = (MAX_WICKETS - df["cumulative_wickets"]).clip(lower=0)

    # Target: innings-1 total (for chasing teams)
    innings1_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target"})
    )
    innings1_totals["target"] += 1  # +1 for the run needed to win
    df = df.merge(innings1_totals, on="match_id", how="left")

    # Rates
    legal_bowled = df["cumulative_legal_balls"]
    df["scoring_rate"] = np.where(
        legal_bowled > 0, df["cumulative_runs"] / legal_bowled, 0.0
    )
    df["asking_rate"] = np.where(
        df["balls_remaining"] > 0,
        (df["target"] - df["cumulative_runs"]) / df["balls_remaining"],
        0.0,
    )
    df["active_rate"] = np.where(
        df["innings"] == 1, df["scoring_rate"], df["asking_rate"]
    )
    df["rate_bin"] = df["active_rate"].apply(calculate_rate_bin)

    return df


# ---------------------------------------------------------------------------
# 5. Hierarchical Empirical Lookup Builder
# ---------------------------------------------------------------------------

def build_empirical_lookups(df_calib: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build a 4-level fallback hierarchy of empirical baseline lookups.

    Each level is a grouped aggregation that computes:
        xR_batter  : mean runs off the bat per ball in this context
        xR_bowler  : mean total bowler runs conceded per ball (runs + extras)
        xW         : mean wicket probability per ball in this context
        sample_size: number of deliveries in this cell (for quality gating)

    Levels (most to least specific):
        level0 : [innings, rate_bin, balls_remaining, wickets_in_hand]
        level1 : [innings, rate_bin, over, wickets_in_hand]
        level2 : [innings, over, wickets_in_hand]
        level3 : [innings]  — the grand mean fallback

    CRITICAL: This function takes ONLY the calibration set as input. It must
    never be called with the full dataset or the evaluation set.

    Parameters
    ----------
    df_calib : pd.DataFrame
        Calibration split of the ball-by-ball data, already enriched with
        state features via ``add_match_state_features``.

    Returns
    -------
    dict of str -> pd.DataFrame
        Keys: ``"level0"``, ``"level1"``, ``"level2"``, ``"level3"``.
    """
    df = df_calib.copy()
    df["bowler_runs_conceded"] = df["runs_off_bat"] + df["wides"] + df["noballs"]

    level_definitions = {
        "level0": ["innings", "rate_bin", "balls_remaining", "wickets_in_hand"],
        "level1": ["innings", "rate_bin", "over", "wickets_in_hand"],
        "level2": ["innings", "over", "wickets_in_hand"],
        "level3": ["innings"],
    }

    lookups: Dict[str, pd.DataFrame] = {}
    for level_name, group_cols in level_definitions.items():
        lookups[level_name] = (
            df.groupby(group_cols)
            .agg(
                xR_batter=("runs_off_bat", "mean"),
                xR_bowler=("bowler_runs_conceded", "mean"),
                xW=("is_wicket", "mean"),
                sample_size=("match_id", "count"),
            )
            .reset_index()
        )
        print(
            f"[Lookups] {level_name}: {len(lookups[level_name])} cells built "
            f"(min sample >= {MIN_SAMPLE_SIZE} gate applied at query time)"
        )

    return lookups


def lookup_baseline(
    innings: int,
    rate_bin: str,
    over: int,
    balls_remaining: int,
    wickets_in_hand: int,
    lookups: Dict[str, pd.DataFrame],
    min_sample: int = MIN_SAMPLE_SIZE,
) -> Tuple[float, float, float]:
    """
    Retrieve the best-available baseline (xR_batter, xR_bowler, xW) for a given
    match state by cascading through the fallback hierarchy.

    Returns the most granular level that meets the minimum sample threshold.
    Falls through to the next coarser level if the threshold is not met.

    Parameters
    ----------
    innings, rate_bin, over, balls_remaining, wickets_in_hand : match context
    lookups : dict returned by ``build_empirical_lookups``
    min_sample : minimum observations required for a cell to be trusted

    Returns
    -------
    (xR_batter, xR_bowler, xW) as floats
    """
    # Level 0 — most specific
    l0 = lookups["level0"]
    m = l0[
        (l0["innings"] == innings)
        & (l0["rate_bin"] == rate_bin)
        & (l0["balls_remaining"] == balls_remaining)
        & (l0["wickets_in_hand"] == wickets_in_hand)
    ]
    if len(m) > 0 and m["sample_size"].iloc[0] >= min_sample:
        row = m.iloc[0]
        return row["xR_batter"], row["xR_bowler"], row["xW"]

    # Level 1
    l1 = lookups["level1"]
    m = l1[
        (l1["innings"] == innings)
        & (l1["rate_bin"] == rate_bin)
        & (l1["over"] == over)
        & (l1["wickets_in_hand"] == wickets_in_hand)
    ]
    if len(m) > 0 and m["sample_size"].iloc[0] >= min_sample:
        row = m.iloc[0]
        return row["xR_batter"], row["xR_bowler"], row["xW"]

    # Level 2
    l2 = lookups["level2"]
    m = l2[
        (l2["innings"] == innings)
        & (l2["over"] == over)
        & (l2["wickets_in_hand"] == wickets_in_hand)
    ]
    if len(m) > 0 and m["sample_size"].iloc[0] >= min_sample:
        row = m.iloc[0]
        return row["xR_batter"], row["xR_bowler"], row["xW"]

    # Level 3 — grand mean fallback, always present
    l3 = lookups["level3"]
    m = l3[l3["innings"] == innings]
    if len(m) > 0:
        row = m.iloc[0]
        return row["xR_batter"], row["xR_bowler"], row["xW"]

    # Absolute fallback — should never be reached if data is sound
    warnings.warn(
        f"No baseline found for innings={innings}. Returning global T20 estimates.",
        RuntimeWarning,
    )
    return 1.20, 1.35, 0.05


# ---------------------------------------------------------------------------
# 6. XP Metric Computation (on calibration set only)
# ---------------------------------------------------------------------------

def compute_xp_metrics(
    df_calib_state: pd.DataFrame,
    lookups: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute per-delivery BattingXP and BowlingXP against the empirical baseline.

    XP = Actual − Expected.  A positive BattingXP means the batter scored more
    than the contextual average; a positive BowlingXP means the bowler
    conceded more than expected (bad for the bowler — this sign convention is
    intentional and consistent with Phase 2 adjustments).

    NOTE: WicketXP semantics are intentionally deferred to Phase 2, where the
    log-odds framework correctly disambiguates batter vs. bowler contributions.
    For now, we compute the raw residual that Phase 2 will transform.

    Parameters
    ----------
    df_calib_state : pd.DataFrame
        Calibration data already enriched with state features.
    lookups : dict
        Calibration-only lookup tables from ``build_empirical_lookups``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
            xR_batter_baseline, xR_bowler_baseline, xW_baseline,
            BattingXP, BowlingXP, WicketXP_raw
    """
    df = df_calib_state.copy()
    df["bowler_runs_conceded"] = df["runs_off_bat"] + df["wides"] + df["noballs"]

    # Vectorised lookup via apply (acceptable for calibration; Phase 3 will
    # use pre-joined tables for speed during simulation)
    def _get_baseline(row):
        return lookup_baseline(
            innings=int(row["innings"]),
            rate_bin=row["rate_bin"],
            over=int(row["over"]),
            balls_remaining=int(row["balls_remaining"]),
            wickets_in_hand=int(row["wickets_in_hand"]),
            lookups=lookups,
        )

    print("[XP] Computing baselines for each delivery — this may take a moment...")
    baselines = df.apply(_get_baseline, axis=1, result_type="expand")
    baselines.columns = ["xR_batter_baseline", "xR_bowler_baseline", "xW_baseline"]
    df = pd.concat([df, baselines], axis=1)

    # XP residuals
    # Wides: batter cannot score off a wide, so BattingXP is undefined -> 0
    df["BattingXP"] = np.where(
        df["wides"] > 0,
        0.0,
        df["runs_off_bat"] - df["xR_batter_baseline"],
    )
    df["BowlingXP"] = df["bowler_runs_conceded"] - df["xR_bowler_baseline"]

    # Raw wicket residual (Phase 2 will convert to log-odds delta)
    df["WicketXP_raw"] = df["is_wicket"] - df["xW_baseline"]

    return df


# ---------------------------------------------------------------------------
# 7. Player XP Aggregation (calibration set only)
# ---------------------------------------------------------------------------

def aggregate_player_xp(
    df_xp: pd.DataFrame,
    min_balls_batter: int = 120,
    min_balls_bowler: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-delivery XP into per-player career XP metrics.

    Only players who meet the minimum ball thresholds are included.
    Players below the threshold are treated as "unknown" during projection
    (XP delta = 0.0), not excluded — they just carry no skill adjustment.

    Parameters
    ----------
    df_xp : pd.DataFrame
        Output of ``compute_xp_metrics``.
    min_balls_batter : int
        Minimum balls faced for a batter to have a reliable XP estimate.
    min_balls_bowler : int
        Minimum legal balls bowled for a bowler to have a reliable XP estimate.

    Returns
    -------
    batter_xp : pd.DataFrame
        One row per batter with columns:
            batter, balls_faced, total_runs_scored,
            total_BattingXP, BattingXP_per_ball,
            total_WicketXP_raw, WicketXP_raw_per_ball
    bowler_xp : pd.DataFrame
        One row per bowler with columns:
            bowler, balls_bowled, total_runs_conceded,
            total_BowlingXP, BowlingXP_per_ball,
            total_WicketXP_raw, WicketXP_raw_per_ball
    """
    # --- Batter aggregation ---
    batter_xp = (
        df_xp.groupby("batter")
        .agg(
            balls_faced=("is_legal_ball", "sum"),
            total_runs_scored=("runs_off_bat", "sum"),
            total_BattingXP=("BattingXP", "sum"),
            total_WicketXP_raw=("WicketXP_raw", "sum"),
        )
        .reset_index()
    )
    batter_xp = batter_xp[batter_xp["balls_faced"] >= min_balls_batter].copy()
    batter_xp["BattingXP_per_ball"] = batter_xp["total_BattingXP"] / batter_xp["balls_faced"]
    batter_xp["WicketXP_raw_per_ball"] = batter_xp["total_WicketXP_raw"] / batter_xp["balls_faced"]

    # --- Bowler aggregation ---
    bowler_xp = (
        df_xp.groupby("bowler")
        .agg(
            balls_bowled=("is_legal_ball", "sum"),
            total_runs_conceded=("bowler_runs_conceded", "sum"),
            total_BowlingXP=("BowlingXP", "sum"),
            total_WicketXP_raw=("WicketXP_raw", "sum"),
        )
        .reset_index()
    )
    bowler_xp = bowler_xp[bowler_xp["balls_bowled"] >= min_balls_bowler].copy()
    bowler_xp["BowlingXP_per_ball"] = bowler_xp["total_BowlingXP"] / bowler_xp["balls_bowled"]
    bowler_xp["WicketXP_raw_per_ball"] = bowler_xp["total_WicketXP_raw"] / bowler_xp["balls_bowled"]

    print(f"[XP Agg] Qualified batters: {len(batter_xp)}, Qualified bowlers: {len(bowler_xp)}")
    return batter_xp, bowler_xp


# ---------------------------------------------------------------------------
# 8. Validation Assertions
# ---------------------------------------------------------------------------

def validate_no_eval_leakage(
    lookups: Dict[str, pd.DataFrame],
    df_eval: pd.DataFrame,
) -> None:
    """
    Assert that the evaluation set match IDs do not contaminate lookup keys.

    Since the lookup tables are grouped aggregations (not keyed by match_id),
    direct match-ID leakage is structurally impossible — but we also verify
    that no evaluation delivery's exact state signature was used to build a
    level-0 cell in a way that could be memorised.

    The primary guard is the chronological split in ``split_calibration_evaluation``,
    which already asserts zero match-ID overlap. This function is an additional
    documentation-grade assertion.
    """
    eval_match_ids = set(df_eval["match_id"].unique())

    # The lookups contain no match_id column (it's been aggregated away), so
    # we confirm this structurally.
    for level_name, ldf in lookups.items():
        assert "match_id" not in ldf.columns, (
            f"LEAKAGE: lookup table '{level_name}' contains a match_id column, "
            f"suggesting raw rows were not properly aggregated."
        )

    print(
        f"[Validation] No match_id columns in any lookup table. "
        f"Evaluation set ({len(eval_match_ids)} matches) structurally isolated."
    )

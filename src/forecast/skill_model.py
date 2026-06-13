"""
forecast/skill_model.py
========================
Phase 2 — Log-Odds Skill Interaction Model.

Replaces the original linear-additive player skill adjustment with a
mathematically sound, probability-bounded framework.

## The Problem With Linear Additivity
The original code did:
    adj_xw = max(0, min(1, base_xw + bat_wxp_pb + bowl_wxp_pb))

This has two critical flaws:
  1. The clip(0, 1) is a *hack*, not a model. It masks the fact that the
     underlying arithmetic is wrong — probabilities do not add linearly.
  2. The adjustment scale is wrong. A 5% base wicket probability behaves
     very differently near 0 than near 0.5. Raw probability deltas ignore this.

## The Fix: Log-Odds (Logit) Space
Probabilities are naturally transformed to log-odds space before adjustment:

    log_odds = log(p / (1 - p))       # logit transformation
    p        = 1 / (1 + exp(-x))      # sigmoid (inverse logit)

Player skill deltas are expressed as LOG-ODDS DELTAS, which:
  - Are unbounded (-inf, +inf) — no clipping needed
  - Are additive by construction (this is how logistic regression works)
  - Correctly model the interaction: a small delta at p=0.05 has a different
    absolute effect on p than the same delta at p=0.5

## Adjusted Wicket Probability
    adj_xw = sigmoid(logit(base_xw) + bowl_wkt_logodds + bat_wkt_logodds)

Where:
  bowl_wkt_logodds > 0 : bowler takes MORE wickets than context expects (good bowler)
  bowl_wkt_logodds < 0 : bowler takes FEWER wickets than context expects (poor bowler)
  bat_wkt_logodds  < 0 : batter gets out LESS than context expects (good batter)
  bat_wkt_logodds  > 0 : batter gets out MORE than context expects (poor batter)

Both use the SAME sign convention: positive = more wickets in this matchup.
A good bowler contributes positive log-odds; a good batter contributes negative
log-odds. They are added together — no sign flip needed.

## Adjusted Run Expectation (Multiplicative)
Run expectation is a continuous positive quantity, not a probability.
A multiplicative (log-space additive) model prevents negative run expectations:

    adj_xr = base_xr * exp(bat_run_logfactor + bowl_run_logfactor)

Where:
  bat_run_logfactor  = log(batter's actual RPB / batter's contextual expected RPB)
  bowl_run_logfactor = log(bowler's actual concede RPB / bowler's contextual expected RPB)

  bat_run_logfactor  > 0 : batter scores MORE than expected (increases adj_xr)
  bowl_run_logfactor < 0 : bowler concedes LESS than expected (decreases adj_xr)

This guarantees adj_xr > 0 for all finite inputs, and is equivalent to a
Poisson log-link regression in structure.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast.config import MIN_BALLS_BATTER, MIN_BALLS_BOWLER, MIN_SAMPLE_SIZE


# ---------------------------------------------------------------------------
# Mathematical primitives
# ---------------------------------------------------------------------------

_EPS = 1e-6  # probability clipping floor/ceiling to avoid log(0)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """
    Logit (log-odds) transformation. Clips p to (_EPS, 1-_EPS) to avoid
    log(0) or log(inf).
    """
    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


def safe_log_ratio(numerator: float, denominator: float) -> float:
    """
    Compute log(numerator / denominator) safely.
    Returns 0.0 if either value is non-positive (unknown player = no adjustment).
    """
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return float(np.log(numerator / denominator))


# ---------------------------------------------------------------------------
# Phase 2 core: build player skill profiles from calibration XP data
# ---------------------------------------------------------------------------

def build_player_skill_profiles(
    df_calib_xp: pd.DataFrame,
    min_balls_batter: int = MIN_BALLS_BATTER,
    min_balls_bowler: int = MIN_BALLS_BOWLER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-player skill profiles in log-odds / log-ratio space from the
    calibration XP dataset.

    This replaces the raw ``WicketXP_raw_per_ball`` from Phase 1 with properly
    scaled log-odds deltas, and replaces the raw ``BattingXP_per_ball`` /
    ``BowlingXP_per_ball`` with multiplicative log-factors.

    Parameters
    ----------
    df_calib_xp : pd.DataFrame
        Output of ``compute_xp_metrics()`` from Phase 1. Must contain columns:
        batter, bowler, is_wicket, runs_off_bat, bowler_runs_conceded,
        xR_batter_baseline, xR_bowler_baseline, xW_baseline, is_legal_ball, wides.

    min_balls_batter : int
        Minimum legal balls faced for a batter to have a qualified skill profile.
        Below this, the batter is treated as "unknown" (all deltas = 0.0).

    min_balls_bowler : int
        Minimum legal balls bowled for a bowler to have a qualified skill profile.

    Returns
    -------
    batter_skill : pd.DataFrame
        One row per qualified batter with columns:
            batter
            balls_faced            : total legal balls faced
            bat_actual_wkt_rate    : actual dismissal rate (wickets / balls)
            bat_context_wkt_rate   : mean xW_baseline across their deliveries
            bat_wkt_logodds        : log-odds delta for wicket probability
                                     (negative = good batter, survives more)
            bat_actual_rpb         : actual runs per (legal) ball
            bat_context_rpb        : context-expected runs per ball
            bat_run_logfactor      : log(actual / expected) for run scoring
                                     (positive = scores more than expected)

    bowler_skill : pd.DataFrame
        One row per qualified bowler with columns:
            bowler
            balls_bowled           : total legal balls bowled
            bowl_actual_wkt_rate   : actual wicket-taking rate
            bowl_context_wkt_rate  : mean xW_baseline across their deliveries
            bowl_wkt_logodds       : log-odds delta for wicket probability
                                     (positive = good bowler, takes more wickets)
            bowl_actual_rpb        : actual runs conceded per ball
            bowl_context_rpb       : context-expected runs conceded per ball
            bowl_run_logfactor     : log(actual / expected) for runs conceded
                                     (negative = good bowler, concedes less)
    """
    df = df_calib_xp.copy()

    # Only consider legal deliveries for rate computations
    # (wides don't count as a batter facing a ball, but DO count as bowler ball)
    df_legal_batter = df[df["wides"] == 0].copy()  # balls batter actually faced

    # -----------------------------------------------------------------------
    # Batter skill profile
    # -----------------------------------------------------------------------
    bat_grp = df_legal_batter.groupby("batter")
    batter_skill = bat_grp.agg(
        balls_faced=("is_legal_ball", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_runs_scored=("runs_off_bat", "sum"),
        sum_xW_baseline=("xW_baseline", "sum"),
        sum_xR_batter_baseline=("xR_batter_baseline", "sum"),
    ).reset_index()

    # Filter minimum sample
    batter_skill = batter_skill[batter_skill["balls_faced"] >= min_balls_batter].copy()

    # Compute rates
    batter_skill["bat_actual_wkt_rate"] = (
        batter_skill["total_wickets"] / batter_skill["balls_faced"]
    )
    batter_skill["bat_context_wkt_rate"] = (
        batter_skill["sum_xW_baseline"] / batter_skill["balls_faced"]
    )
    batter_skill["bat_actual_rpb"] = (
        batter_skill["total_runs_scored"] / batter_skill["balls_faced"]
    )
    batter_skill["bat_context_rpb"] = (
        batter_skill["sum_xR_batter_baseline"] / batter_skill["balls_faced"]
    )

    # Log-odds delta for wicket probability
    # Positive = batter gets out MORE than context (poor batter)
    # Negative = batter gets out LESS than context (good batter, reduces adj_xw)
    batter_skill["bat_wkt_logodds"] = (
        logit(batter_skill["bat_actual_wkt_rate"].values)
        - logit(batter_skill["bat_context_wkt_rate"].values)
    )

    # Log-factor for run scoring (multiplicative model)
    # Positive = batter scores MORE than context expects (good batter, increases adj_xr)
    batter_skill["bat_run_logfactor"] = batter_skill.apply(
        lambda r: safe_log_ratio(r["bat_actual_rpb"], r["bat_context_rpb"]), axis=1
    )

    batter_skill = batter_skill[
        ["batter", "balls_faced", "bat_actual_wkt_rate", "bat_context_wkt_rate",
         "bat_wkt_logodds", "bat_actual_rpb", "bat_context_rpb", "bat_run_logfactor"]
    ]

    # -----------------------------------------------------------------------
    # Bowler skill profile
    # -----------------------------------------------------------------------
    bowl_grp = df.groupby("bowler")  # include all deliveries (wides count for bowlers)
    bowler_skill = bowl_grp.agg(
        balls_bowled=("is_legal_ball", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_runs_conceded=("bowler_runs_conceded", "sum"),
        sum_xW_baseline=("xW_baseline", "sum"),
        sum_xR_bowler_baseline=("xR_bowler_baseline", "sum"),
    ).reset_index()

    bowler_skill = bowler_skill[bowler_skill["balls_bowled"] >= min_balls_bowler].copy()

    bowler_skill["bowl_actual_wkt_rate"] = (
        bowler_skill["total_wickets"] / bowler_skill["balls_bowled"]
    )
    bowler_skill["bowl_context_wkt_rate"] = (
        bowler_skill["sum_xW_baseline"] / bowler_skill["balls_bowled"]
    )
    bowler_skill["bowl_actual_rpb"] = (
        bowler_skill["total_runs_conceded"] / bowler_skill["balls_bowled"]
    )
    bowler_skill["bowl_context_rpb"] = (
        bowler_skill["sum_xR_bowler_baseline"] / bowler_skill["balls_bowled"]
    )

    # Log-odds delta for wicket probability
    # Positive = bowler takes MORE wickets than context (good bowler, increases adj_xw)
    # Negative = bowler takes FEWER wickets than context (poor bowler, decreases adj_xw)
    bowler_skill["bowl_wkt_logodds"] = (
        logit(bowler_skill["bowl_actual_wkt_rate"].values)
        - logit(bowler_skill["bowl_context_wkt_rate"].values)
    )

    # Log-factor for runs conceded
    # Negative = bowler concedes LESS than context (good economy, decreases adj_xr)
    bowler_skill["bowl_run_logfactor"] = bowler_skill.apply(
        lambda r: safe_log_ratio(r["bowl_actual_rpb"], r["bowl_context_rpb"]), axis=1
    )

    bowler_skill = bowler_skill[
        ["bowler", "balls_bowled", "bowl_actual_wkt_rate", "bowl_context_wkt_rate",
         "bowl_wkt_logodds", "bowl_actual_rpb", "bowl_context_rpb", "bowl_run_logfactor"]
    ]

    print(
        f"[SkillModel] Qualified batters: {len(batter_skill)}, "
        f"Qualified bowlers: {len(bowler_skill)}"
    )
    return batter_skill, bowler_skill


# ---------------------------------------------------------------------------
# Lookup helpers for individual player deltas
# ---------------------------------------------------------------------------

_UNKNOWN_BATTER = {
    "bat_wkt_logodds": 0.0,
    "bat_run_logfactor": 0.0,
}

_UNKNOWN_BOWLER = {
    "bowl_wkt_logodds": 0.0,
    "bowl_run_logfactor": 0.0,
}


def get_batter_deltas(
    batter_name: str,
    batter_skill: pd.DataFrame,
) -> Dict[str, float]:
    """
    Retrieve the skill deltas for a named batter.

    Returns zero deltas (neutral adjustment) for unknown or unqualified batters,
    meaning they receive exactly the contextual baseline — not zero runs/wickets.

    Parameters
    ----------
    batter_name : str
    batter_skill : pd.DataFrame  (output of ``build_player_skill_profiles``)

    Returns
    -------
    dict with keys: bat_wkt_logodds, bat_run_logfactor
    """
    rows = batter_skill[batter_skill["batter"] == batter_name]
    if len(rows) == 0:
        return _UNKNOWN_BATTER.copy()
    row = rows.iloc[0]
    return {
        "bat_wkt_logodds": float(row["bat_wkt_logodds"]),
        "bat_run_logfactor": float(row["bat_run_logfactor"]),
    }


def get_bowler_deltas(
    bowler_name: str,
    bowler_skill: pd.DataFrame,
) -> Dict[str, float]:
    """
    Retrieve the skill deltas for a named bowler.

    Returns zero deltas (neutral adjustment) for unknown or unqualified bowlers.
    """
    rows = bowler_skill[bowler_skill["bowler"] == bowler_name]
    if len(rows) == 0:
        return _UNKNOWN_BOWLER.copy()
    row = rows.iloc[0]
    return {
        "bowl_wkt_logodds": float(row["bowl_wkt_logodds"]),
        "bowl_run_logfactor": float(row["bowl_run_logfactor"]),
    }


# ---------------------------------------------------------------------------
# Adjusted probability / expectation for a single delivery
# ---------------------------------------------------------------------------

def adjusted_wicket_prob(
    base_xw: float,
    bowl_wkt_logodds: float,
    bat_wkt_logodds: float,
) -> float:
    """
    Compute the matchup-adjusted wicket probability for a single delivery.

    Formula:
        adj_xw = sigmoid(logit(base_xw) + bowl_wkt_logodds + bat_wkt_logodds)

    This is guaranteed to return a value in (0, 1) for all finite inputs —
    no manual clipping required.

    Parameters
    ----------
    base_xw : float
        Contextual baseline wicket probability from the empirical lookup (Phase 1).
        Must be in (0, 1); will be clipped to (_EPS, 1-_EPS) internally.
    bowl_wkt_logodds : float
        Bowler's log-odds skill delta (positive = good bowler, increases prob).
    bat_wkt_logodds : float
        Batter's log-odds skill delta (negative = good batter, decreases prob).

    Returns
    -------
    float in (0, 1)
    """
    return float(sigmoid(logit(base_xw) + bowl_wkt_logodds + bat_wkt_logodds))


def adjusted_run_expectation(
    base_xr: float,
    bat_run_logfactor: float,
    bowl_run_logfactor: float,
) -> float:
    """
    Compute the matchup-adjusted expected runs off the bat for a single delivery.

    Formula:
        adj_xr = base_xr * exp(bat_run_logfactor + bowl_run_logfactor)

    Guaranteed to be strictly positive for all finite inputs, since
    base_xr > 0 and exp() > 0 always.

    Parameters
    ----------
    base_xr : float
        Contextual baseline expected runs-per-ball (from empirical lookup).
    bat_run_logfactor : float
        Batter's log-scale run factor (positive = scores more than context).
    bowl_run_logfactor : float
        Bowler's log-scale run factor (negative = concedes less than context).

    Returns
    -------
    float > 0
    """
    return float(max(_EPS, base_xr * np.exp(bat_run_logfactor + bowl_run_logfactor)))


# ---------------------------------------------------------------------------
# Convenience: apply both adjustments for a matchup
# ---------------------------------------------------------------------------

def compute_matchup_adjustments(
    base_xr: float,
    base_xw: float,
    batter_name: str,
    bowler_name: str,
    batter_skill: pd.DataFrame,
    bowler_skill: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute all adjusted probabilities / expectations for a single batter-bowler
    matchup, given pre-computed contextual baselines.

    This is the primary interface used by the Phase 3 Monte Carlo simulation
    to resolve each simulated delivery.

    Parameters
    ----------
    base_xr : float    Contextual expected runs off bat from lookup.
    base_xw : float    Contextual expected wicket probability from lookup.
    batter_name : str
    bowler_name : str
    batter_skill : pd.DataFrame   Output of ``build_player_skill_profiles``.
    bowler_skill : pd.DataFrame

    Returns
    -------
    dict with keys:
        adj_xw   : adjusted wicket probability, in (0, 1)
        adj_xr   : adjusted expected runs off bat, > 0
        bat_wkt_logodds   : the batter's raw log-odds delta (for diagnostics)
        bowl_wkt_logodds  : the bowler's raw log-odds delta (for diagnostics)
        bat_run_logfactor : the batter's raw run log-factor (for diagnostics)
        bowl_run_logfactor: the bowler's raw run log-factor (for diagnostics)
    """
    bat = get_batter_deltas(batter_name, batter_skill)
    bowl = get_bowler_deltas(bowler_name, bowler_skill)

    adj_xw = adjusted_wicket_prob(base_xw, bowl["bowl_wkt_logodds"], bat["bat_wkt_logodds"])
    adj_xr = adjusted_run_expectation(base_xr, bat["bat_run_logfactor"], bowl["bowl_run_logfactor"])

    return {
        "adj_xw": adj_xw,
        "adj_xr": adj_xr,
        "bat_wkt_logodds": bat["bat_wkt_logodds"],
        "bowl_wkt_logodds": bowl["bowl_wkt_logodds"],
        "bat_run_logfactor": bat["bat_run_logfactor"],
        "bowl_run_logfactor": bowl["bowl_run_logfactor"],
    }

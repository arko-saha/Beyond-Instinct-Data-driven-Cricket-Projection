"""
forecast/calibration.py
========================
Phase 4 -- Calibration diagnostics and interval recalibration.

## Why Coverage May Be Below 80%

The Phase 3 simulator's P10-P90 interval might under-cover because:
1. The empirical scoring distribution was built on only 40 (innings, over) cells
   from 100k calibration rows -- many cells use the global T20 default, which
   underestimates variance in extreme contexts (e.g., death overs, low-wicket).
2. The Phase 2 skill adjustments shift the MEAN but not the SPREAD -- if a match
   has an unusual batter/bowler combo, the distribution shape is unchanged.
3. Systematic bias: the simulator may consistently under-predict high totals
   (right-tail underrepresentation) from T20 power hitting.

## Recalibration Strategy

This module implements two tools:

### 1. Conformalized Interval Scaling (non-parametric)
Also known as "conformal prediction" recalibration:
- On a held-out calibration set, compute the empirical quantile of
  (actual_score - pred_median) / (pred_p90 - pred_p10) * 2
  i.e., normalised residuals.
- Find the scaling factor `alpha` such that P(|normalised_residual| <= alpha) = 0.80.
- At inference time: inflate the predicted interval by `alpha`.
  New P10 = pred_median - alpha * half_width
  New P90 = pred_median + alpha * half_width

### 2. Seasonal Cross-Validation
For seasons S in {2016..2023}, train on S and earlier, test on S+1.
Report per-season coverage, MAE, and Brier score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_conformalized_scale(
    backtest_df: pd.DataFrame,
    target_coverage: float = 0.80,
) -> float:
    """
    Compute the conformalized interval scaling factor `alpha` such that,
    on the backtest calibration set, P(actual in [scaled P10, scaled P90]) = target_coverage.

    The scaling is symmetric around the median:
        new_P10 = pred_median - alpha * (pred_median - pred_p10)
        new_P90 = pred_median + alpha * (pred_p90 - pred_median)

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``Backtester.run()`` with columns pred_p10, pred_median, pred_p90, actual_score.
    target_coverage : float
        Target coverage rate (default 0.80 for P10-P90 band).

    Returns
    -------
    float : scaling factor alpha (>= 1.0 means the interval needs to be widened)
    """
    df = backtest_df.copy()

    # Half-width of the predicted interval from median
    half_width_lo = (df["pred_median"] - df["pred_p10"]).clip(lower=1.0)
    half_width_hi = (df["pred_p90"] - df["pred_median"]).clip(lower=1.0)

    # Normalized distance from median to actual
    # Positive: actual is above median (right tail), Negative: below median
    dist_from_median = df["actual_score"] - df["pred_median"]
    norm_dist = np.where(
        dist_from_median >= 0,
        dist_from_median / half_width_hi,
        -dist_from_median / half_width_lo,
    )

    # alpha such that P(norm_dist <= alpha) = target_coverage
    alpha = np.quantile(np.abs(norm_dist), target_coverage)
    return float(alpha)


def apply_conformalized_intervals(
    backtest_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Apply the conformalized scaling factor to widen predicted intervals.

    Returns the input DataFrame with additional columns:
        cal_p10  : recalibrated lower bound
        cal_p90  : recalibrated upper bound
        cal_in_band : actual in [cal_p10, cal_p90]
        cal_interval_width
    """
    df = backtest_df.copy()
    half_lo = (df["pred_median"] - df["pred_p10"]).clip(lower=1.0)
    half_hi = (df["pred_p90"] - df["pred_median"]).clip(lower=1.0)

    df["cal_p10"] = (df["pred_median"] - alpha * half_lo).round()
    df["cal_p90"] = (df["pred_median"] + alpha * half_hi).round()
    df["cal_in_band"] = (df["actual_score"] >= df["cal_p10"]) & (df["actual_score"] <= df["cal_p90"])
    df["cal_interval_width"] = df["cal_p90"] - df["cal_p10"]
    return df


def seasonal_cross_validation(
    df_full: pd.DataFrame,
    lookups_fn,
    skill_fn,
    scoring_dist_fn,
    simulator_cls,
    freeze_over: int = 10,
    n_sim: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run seasonal cross-validation: for each season in the dataset,
    train on all prior seasons and evaluate on that season.

    Returns DataFrame with one row per season.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full ball-by-ball dataset with start_date column parsed.
    lookups_fn : callable
        Function that takes df_calibration and returns lookups dict.
    skill_fn : callable
        Function that takes df_calibration and returns (batter_skill, bowler_skill).
    scoring_dist_fn : callable
        Function that takes df_calibration_xp and returns scoring_dist dict.
    simulator_cls : class
        InningsSimulator class.
    """
    from src.forecast.backtester import extract_match_states, Backtester, compute_metrics

    df = df_full[df_full["innings"].isin([1, 2])].copy()
    df["year"] = df["start_date"].dt.year

    # Use seasons from 2019 onward (need >= 1 prior year of training)
    seasons = sorted(df["year"].dropna().unique().astype(int))
    seasons = [s for s in seasons if s >= 2019]

    results_rows = []

    for test_season in seasons:
        train_years = [s for s in seasons if s < test_season]
        if len(train_years) < 2:
            continue

        df_train = df[df["year"] < test_season]
        df_test  = df[df["year"] == test_season]

        if df_test["match_id"].nunique() < 5:
            continue

        if verbose:
            print(f"Season {test_season}: train={df_train['match_id'].nunique()} matches, "
                  f"test={df_test['match_id'].nunique()} matches", flush=True)

        # Build artefacts on training data
        try:
            lookups = lookups_fn(df_train)
            batter_skill, bowler_skill = skill_fn(df_train)
            scoring_dist = scoring_dist_fn(df_train)
            sim = simulator_cls(lookups, batter_skill, bowler_skill, scoring_dist)

            states = extract_match_states(df_test, freeze_overs=(freeze_over,))
            if len(states) == 0:
                continue

            bt = Backtester(sim, n_sim=n_sim, seed=seed)
            bt_results = bt.run(states, max_matches=80, verbose=False)

            if len(bt_results) == 0:
                continue

            m = compute_metrics(bt_results)
            m["season"] = test_season
            m["n_train_matches"] = df_train["match_id"].nunique()
            m["n_test_matches"] = df_test["match_id"].nunique()
            results_rows.append(m)

        except Exception as e:
            if verbose:
                print(f"  ERROR for season {test_season}: {e}")
            continue

    return pd.DataFrame(results_rows)


def bias_decomposition(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose prediction error by match context.

    Groups errors by:
    - innings (1 vs 2)
    - runs_at_freeze quintile (how many runs were on board at freeze point)
    - wickets_at_freeze bucket

    Returns a DataFrame showing MAE, signed_error (bias direction), and coverage
    per context bucket. Used to identify where the model is systematically wrong.
    """
    df = backtest_df.copy()

    df["score_quintile"] = pd.qcut(
        df["runs_at_freeze"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    df["wickets_bucket"] = pd.cut(
        df["wickets_at_freeze"],
        bins=[-1, 2, 5, 10],
        labels=["0-2 wkts", "3-5 wkts", "6-9 wkts"],
    )

    rows = []

    for groupby_col in ["innings", "score_quintile", "wickets_bucket"]:
        for val, grp in df.groupby(groupby_col, observed=True):
            if len(grp) < 5:
                continue
            rows.append({
                "dimension": groupby_col,
                "value": str(val),
                "n": len(grp),
                "coverage": round(grp["in_band"].mean(), 3),
                "mae": round(grp["abs_error"].mean(), 1),
                "bias": round(grp["signed_error"].mean(), 1),   # + = over-predict
                "band_width": round(grp["interval_width"].median(), 0),
            })

    return pd.DataFrame(rows)

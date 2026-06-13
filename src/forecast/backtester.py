"""
forecast/backtester.py
=======================
Phase 4 -- Backtesting & Calibration Validation.

Runs the Phase 3 Monte Carlo simulator against held-out evaluation matches
to measure how well the model's predicted distributions match actual outcomes.

## Methodology

### Freeze-Point Protocol
For each evaluation match-innings:
    1. Freeze at a specified over (e.g., over 10 = after 60 balls).
    2. Extract the actual match state at that point: runs, wickets, remaining
       lineup, actual bowling plan.
    3. Run N=5,000 Monte Carlo simulations from that state.
    4. Record: predicted P10, P50 (median), P90, actual final score.
    5. For innings 2: record predicted P(win) and actual result.

Using the ACTUAL bowling plan from the match (oracle knowledge) isolates
the score-prediction question from the planning question. Phase 5 handles
bowling plan optimisation.

### Metrics
Coverage rate   : P(actual_score in [P10, P90]) -- target ~80% for good calibration
MAE             : |median_prediction - actual_score| -- point accuracy
Interval width  : P90 - P10 -- sharpness (narrower = more decisive)
Brier score     : mean((P_win - actual_win)^2) -- win-prob calibration for innings 2
Calibration curve : bin P(win) into deciles, plot actual win rate per bin

### Baseline Comparison
Run the same backtest with a neutral simulator (all player skill adjustments
zeroed out). Compare coverage, MAE, and Brier score to the full model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast.simulator import InningsSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Match state extraction from ball-by-ball data
# ---------------------------------------------------------------------------

def extract_match_states(
    df_eval: pd.DataFrame,
    freeze_overs: Tuple[int, ...] = (10,),
    min_balls_in_over: int = 4,
) -> pd.DataFrame:
    """
    For each evaluation match-innings, extract the match state at each
    specified freeze point and the actual final score.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation ball-by-ball data. Must have columns:
        match_id, innings, over, batter, bowler,
        runs_off_bat, total_runs, is_wicket, start_date.
    freeze_overs : tuple of int
        Over numbers at which to freeze and predict. E.g., (10, 14) freezes
        after 10 overs have been bowled (before ball 1 of over 11).
    min_balls_in_over : int
        Minimum balls in an over to count it as complete. Avoids edge cases
        with interrupted innings.

    Returns
    -------
    pd.DataFrame with columns:
        match_id, innings, freeze_over,
        actual_score, runs_at_freeze, wickets_at_freeze,
        remaining_lineup,  -- list of str (JSON-serialised)
        bowling_plan,      -- dict {over: bowler} (JSON-serialised)
        has_result,        -- bool: True if match result is resolvable
        batting_won        -- int: 1 if batting team won innings 2, 0 otherwise (NaN for innings 1)
    """
    df = df_eval[(df_eval["innings"].isin([1, 2]))].copy()
    df = df.sort_values(["match_id", "innings", "over", "ball"])

    records = []

    for (mid, inn), grp in df.groupby(["match_id", "innings"]):
        grp = grp.sort_values(["over", "ball"])

        # Final score of the innings
        actual_score = int(grp["total_runs"].sum())

        # Actual batting won (innings 2 only): score >= first innings score
        # We compute this later when we have both innings data
        # For now, store raw score
        for freeze_over in freeze_overs:
            # Balls before the freeze point (overs 0..freeze_over-1)
            before = grp[grp["over"] < freeze_over]
            after  = grp[grp["over"] >= freeze_over]

            # Need at least freeze_over complete overs of data
            overs_before = before["over"].nunique()
            if overs_before < freeze_over:
                continue
            # Need meaningful remaining data
            if len(after) < min_balls_in_over:
                continue

            runs_at_freeze    = int(before["total_runs"].sum())
            wickets_at_freeze = int(before["is_wicket"].sum())

            if wickets_at_freeze >= 10:
                continue

            # Remaining lineup: batters who haven't yet been dismissed before the freeze
            # Approach: find all batters in this innings, ordered by first appearance
            all_batters_ordered = (
                grp.sort_values(["over", "ball"])
                .drop_duplicates("batter", keep="first")["batter"]
                .tolist()
            )
            # Dismissed before freeze
            dismissed_before = set(
                before[before["is_wicket"] == 1]["batter"].unique()
            )
            # Current batter: appeared before freeze AND not dismissed
            current_batter = before["batter"].iloc[-1] if len(before) > 0 else None

            # Remaining lineup = current batter + those not yet appeared + not yet dismissed
            remaining = []
            appeared_before = set(before["batter"].unique())
            for b in all_batters_ordered:
                if b in dismissed_before:
                    continue
                if b in appeared_before or b == current_batter:
                    remaining.append(b)
                else:
                    remaining.append(b)  # hasn't batted yet

            # Deduplicate while preserving order
            seen = set()
            remaining_deduped = []
            for b in remaining:
                if b not in seen:
                    seen.add(b)
                    remaining_deduped.append(b)

            # Put current batter first
            if current_batter and current_batter in remaining_deduped:
                remaining_deduped.remove(current_batter)
                remaining_deduped.insert(0, current_batter)

            # Bowling plan for remaining overs: {str(over): bowler}
            # Keys stored as str so the DataFrame is parquet-serialisable
            # (pyarrow rejects dicts with int keys).
            bowling_plan = {
                str(k): v
                for k, v in (
                    after.drop_duplicates("over", keep="first")
                    .set_index("over")["bowler"]
                    .to_dict()
                ).items()
            }

            records.append({
                "match_id": mid,
                "innings": inn,
                "freeze_over": freeze_over,
                "actual_score": actual_score,
                "runs_at_freeze": runs_at_freeze,
                "wickets_at_freeze": wickets_at_freeze,
                "remaining_lineup": remaining_deduped,
                "bowling_plan": bowling_plan,
                "start_date": grp["start_date"].iloc[0],
            })

    states_df = pd.DataFrame(records)

    # Compute innings 2 result: did the batting team win?
    # Compute per-match, innings 1 score
    if len(states_df) > 0:
        inn1_scores = (
            states_df[states_df["innings"] == 1]
            .groupby("match_id")["actual_score"].first()
            .rename("inn1_score")
        )
        states_df = states_df.join(inn1_scores, on="match_id")
        states_df["batting_won"] = np.where(
            states_df["innings"] == 2,
            (states_df["actual_score"] > states_df["inn1_score"]).astype(int),
            np.nan,
        )
        states_df["target"] = np.where(
            states_df["innings"] == 2,
            states_df["inn1_score"] + 1,
            np.nan,
        )

    return states_df


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Runs the Phase 3 simulator against held-out evaluation matches and
    computes calibration metrics.

    Parameters
    ----------
    simulator : InningsSimulator
        The Phase 3 simulator (with Phase 2 skill adjustments).
    n_sim : int
        Number of Monte Carlo paths per evaluation point. Default 5,000
        (faster than the 10,000 used in production; sufficient for metrics).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        simulator: InningsSimulator,
        n_sim: int = 5000,
        seed: int = 42,
    ):
        self._sim = simulator
        self._n_sim = n_sim
        self._seed = seed

    def run(
        self,
        states_df: pd.DataFrame,
        max_matches: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the backtest over all (match, innings, freeze_over) rows in states_df.

        Parameters
        ----------
        states_df : pd.DataFrame
            Output of ``extract_match_states()``.
        max_matches : int, optional
            Limit backtest to first N matches (for speed during development).
        verbose : bool
            Print progress every 50 rows.

        Returns
        -------
        pd.DataFrame -- one row per (match, innings, freeze_over) with columns:
            match_id, innings, freeze_over, actual_score, runs_at_freeze,
            wickets_at_freeze, start_date,
            pred_p10, pred_median, pred_p90, pred_mean,
            in_band,           -- bool: actual in [P10, P90]
            abs_error,         -- |pred_median - actual|
            signed_error,      -- pred_median - actual (positive = over-predicted)
            interval_width,    -- P90 - P10
            pred_p_win,        -- innings 2 only
            actual_win,        -- innings 2 only
            brier_sq_error,    -- (pred_p_win - actual_win)^2
        """
        df = states_df.copy()
        if max_matches is not None:
            match_ids = df["match_id"].unique()[:max_matches]
            df = df[df["match_id"].isin(match_ids)]

        results = []
        n_total = len(df)

        for i, row in enumerate(df.itertuples(index=False)):
            if verbose and i % 50 == 0:
                print(f"  [{i:4d}/{n_total}] match {row.match_id}, inn{row.innings}, over{row.freeze_over}")

            try:
                # Clamp lineup to at least 1 entry
                lineup = row.remaining_lineup if row.remaining_lineup else ["Unknown"]
                # Convert str keys back to int (stored as str for parquet compatibility)
                plan = {int(k): v for k, v in row.bowling_plan.items()} if row.bowling_plan else {}

                # Determine target for innings 2
                target = None
                if row.innings == 2 and hasattr(row, "target") and not pd.isna(row.target):
                    target = int(row.target)

                result = self._sim.simulate(
                    innings=int(row.innings),
                    current_over=int(row.freeze_over),
                    current_runs=int(row.runs_at_freeze),
                    current_wickets=int(row.wickets_at_freeze),
                    batting_lineup=lineup,
                    bowling_plan=plan,
                    target=target,
                    n_simulations=self._n_sim,
                    seed=self._seed,
                )

                pred_p10    = result.score_p10
                pred_median = result.score_median
                pred_p90    = result.score_p90
                actual      = int(row.actual_score)

                rec = {
                    "match_id":          row.match_id,
                    "innings":           int(row.innings),
                    "freeze_over":       int(row.freeze_over),
                    "start_date":        row.start_date,
                    "actual_score":      actual,
                    "runs_at_freeze":    int(row.runs_at_freeze),
                    "wickets_at_freeze": int(row.wickets_at_freeze),
                    "pred_p10":          round(pred_p10),
                    "pred_median":       round(pred_median),
                    "pred_p90":          round(pred_p90),
                    "pred_mean":         round(result.score_mean, 1),
                    "interval_width":    round(pred_p90 - pred_p10),
                    "in_band":           bool(pred_p10 <= actual <= pred_p90),
                    "abs_error":         abs(round(pred_median) - actual),
                    "signed_error":      round(pred_median) - actual,
                    "pred_p_win":        result.win_probability,
                    "actual_win":        float(row.batting_won) if not pd.isna(row.batting_won) else np.nan,
                    "brier_sq_error":    (result.win_probability - float(row.batting_won)) ** 2
                                         if result.win_probability is not None and not pd.isna(row.batting_won)
                                         else np.nan,
                }
                results.append(rec)

            except Exception as e:
                if verbose:
                    print(f"    SKIP: match {row.match_id} inn{row.innings} -- {e}")

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    backtest_df: pd.DataFrame,
    label: str = "Model",
) -> Dict[str, float]:
    """
    Compute aggregate backtesting metrics from the output of ``Backtester.run()``.

    Parameters
    ----------
    backtest_df : pd.DataFrame
    label : str
        Descriptive label for this model run (used for display).

    Returns
    -------
    dict with keys:
        n_predictions        : total predictions evaluated
        coverage_rate        : P(actual in [P10,P90]) -- should be ~0.80
        mae                  : mean absolute error of median prediction
        rmse                 : root mean square error of median prediction
        median_interval_width: median width of P10-P90 interval
        brier_score          : mean (P_win - actual_win)^2 (innings 2 only)
        n_innings2           : number of innings 2 predictions
    """
    n = len(backtest_df)
    if n == 0:
        return {"label": label, "n_predictions": 0}

    coverage = backtest_df["in_band"].mean()
    mae = backtest_df["abs_error"].mean()
    rmse = np.sqrt((backtest_df["signed_error"] ** 2).mean())
    median_width = backtest_df["interval_width"].median()

    inn2 = backtest_df[backtest_df["innings"] == 2].dropna(subset=["brier_sq_error"])
    brier = inn2["brier_sq_error"].mean() if len(inn2) > 0 else np.nan

    return {
        "label": label,
        "n_predictions": n,
        "coverage_rate": round(coverage, 3),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "median_interval_width": round(median_width, 1),
        "brier_score": round(brier, 4) if not np.isnan(brier) else np.nan,
        "n_innings2": len(inn2),
    }


def calibration_curve(
    backtest_df: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a win probability calibration curve for innings 2 predictions.

    Bins predicted P(win) into n_bins equal-width buckets and computes
    the actual win rate within each bin.

    A perfectly calibrated model produces points along the diagonal (y=x).
    Points above diagonal = model underestimates win probability.
    Points below diagonal = model overestimates win probability.

    Returns
    -------
    pd.DataFrame with columns:
        bin_center, pred_p_win_mean, actual_win_rate, n_predictions
    """
    inn2 = backtest_df[
        (backtest_df["innings"] == 2)
        & backtest_df["pred_p_win"].notna()
        & backtest_df["actual_win"].notna()
    ].copy()

    if len(inn2) == 0:
        return pd.DataFrame()

    bins = np.linspace(0, 1, n_bins + 1)
    inn2["bin"] = pd.cut(inn2["pred_p_win"], bins=bins, include_lowest=True)

    cal_df = inn2.groupby("bin", observed=True).agg(
        pred_p_win_mean=("pred_p_win", "mean"),
        actual_win_rate=("actual_win", "mean"),
        n_predictions=("actual_win", "count"),
    ).reset_index()

    cal_df["bin_center"] = bins[:-1] + 0.05
    return cal_df[cal_df["n_predictions"] >= 3].reset_index(drop=True)

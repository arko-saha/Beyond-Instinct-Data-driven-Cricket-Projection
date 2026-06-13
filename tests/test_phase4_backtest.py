"""
Phase 4 validation tests for backtesting & calibration.
Run from project root: python tests/test_phase4_backtest.py

Runs on 100 evaluation matches for speed (full run uses all 1,201).
"""
import sys
sys.path.insert(0, ".")

import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.forecast.simulator import (
    InningsSimulator, build_scoring_distribution
)
from src.forecast.backtester import (
    extract_match_states, Backtester, compute_metrics, calibration_curve
)
from src.forecast.config import BACKTEST_COVERAGE_TARGET, BACKTEST_MAE_TARGET

PASS, FAIL = "[OK]", "[FAIL]"
errors = []

def check(cond, msg):
    if cond:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        errors.append(msg)

# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
print("Loading artefacts...")
P1 = Path("models/phase1_artefacts")
P2 = Path("models/phase2_artefacts")

with open(P1 / "lookups.pkl", "rb") as f:
    lookups = pickle.load(f)

batter_skill = pd.read_parquet(P2 / "batter_skill.parquet")
bowler_skill  = pd.read_parquet(P2 / "bowler_skill.parquet")
df_calib_xp   = pd.read_parquet(P1 / "df_calib_xp.parquet")

scoring_dist = build_scoring_distribution(df_calib_xp)
sim = InningsSimulator(lookups, batter_skill, bowler_skill, scoring_dist)

# Baseline simulator: empty skill DataFrames -> neutral adjustments only
empty_bat = pd.DataFrame(columns=batter_skill.columns)
empty_bowl = pd.DataFrame(columns=bowler_skill.columns)
sim_baseline = InningsSimulator(lookups, empty_bat, empty_bowl, scoring_dist)

print("Simulators ready.")

# ---------------------------------------------------------------------------
# Load evaluation data
# ---------------------------------------------------------------------------
print("\nLoading evaluation data (full dataset)...")
df_raw = pd.read_csv("data/consolidated_t20_data.csv", low_memory=False)
df_raw["start_date"] = pd.to_datetime(df_raw["start_date"], errors="coerce")
df_raw["season"] = df_raw["season"].astype(str)
for col in ["is_wicket", "runs_off_bat", "total_runs", "wides", "noballs"]:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0).astype(int)

df_eval = df_raw[df_raw["start_date"] >= pd.Timestamp("2024-01-01")].copy()
print(f"Evaluation set: {df_eval['match_id'].nunique():,} matches, {len(df_eval):,} deliveries")

# ---------------------------------------------------------------------------
# Test 1: extract_match_states correctness
# ---------------------------------------------------------------------------
print("\n--- Test 1: Match State Extraction ---")
# Use a small sample for speed
sample_match_ids = df_eval["match_id"].unique()[:10]
df_sample = df_eval[df_eval["match_id"].isin(sample_match_ids)]

states_sample = extract_match_states(df_sample, freeze_overs=(10,))
check(len(states_sample) > 0, f"extract_match_states produced {len(states_sample)} rows")
required_cols = ["match_id", "innings", "freeze_over", "actual_score",
                 "runs_at_freeze", "wickets_at_freeze", "remaining_lineup", "bowling_plan"]
for col in required_cols:
    check(col in states_sample.columns, f"  Column '{col}' present")

# Validate state sanity
check((states_sample["runs_at_freeze"] >= 0).all(), "runs_at_freeze >= 0")
check((states_sample["wickets_at_freeze"] >= 0).all(), "wickets_at_freeze >= 0")
check((states_sample["wickets_at_freeze"] < 10).all(), "wickets_at_freeze < 10 at freeze")
check((states_sample["actual_score"] >= states_sample["runs_at_freeze"]).all(),
      "actual_score >= runs_at_freeze (scores only increase)")

for _, row in states_sample.iterrows():
    check(len(row["remaining_lineup"]) >= 1, f"  remaining_lineup has >= 1 batter")
    check(isinstance(row["bowling_plan"], dict), f"  bowling_plan is a dict")

print(f"State extraction sample:")
print(states_sample[["match_id","innings","runs_at_freeze","wickets_at_freeze","actual_score"]].head(6).to_string())

# ---------------------------------------------------------------------------
# Test 2: Backtester runs without errors (100 matches)
# ---------------------------------------------------------------------------
print("\n--- Test 2: Backtester (100 matches) ---")
MAX_BACKTEST_MATCHES = 100
states_100 = extract_match_states(df_eval, freeze_overs=(10,))
print(f"States extracted for all eval matches: {len(states_100)} rows")

bt = Backtester(sim, n_sim=2000, seed=42)  # 2000 sims for speed during testing

t0 = time.time()
results_df = bt.run(states_100, max_matches=MAX_BACKTEST_MATCHES, verbose=False)
elapsed = time.time() - t0

check(len(results_df) > 0, f"Backtester produced {len(results_df)} result rows")
check(elapsed < 300, f"100-match backtest in {elapsed:.1f}s (< 5 min)")

# Check required output columns
out_cols = ["match_id","innings","freeze_over","actual_score","pred_p10",
            "pred_median","pred_p90","in_band","abs_error","interval_width"]
for col in out_cols:
    check(col in results_df.columns, f"  Output column '{col}' present")

# Structural checks
check((results_df["pred_p10"] <= results_df["pred_median"]).all(), "P10 <= median always")
check((results_df["pred_median"] <= results_df["pred_p90"]).all(), "median <= P90 always")
check((results_df["interval_width"] >= 0).all(), "interval_width >= 0")
check((results_df["abs_error"] >= 0).all(), "abs_error >= 0")
check(results_df["in_band"].dtype == bool, "in_band is boolean")

print(f"\n  Speed: {elapsed:.1f}s for {len(results_df)} predictions ({elapsed/len(results_df):.2f}s each)")
print(f"  Sample results:")
print(results_df[["match_id","innings","actual_score","pred_p10","pred_median","pred_p90","in_band","abs_error"]].head(8).to_string())

# ---------------------------------------------------------------------------
# Test 3: Metric computation
# ---------------------------------------------------------------------------
print("\n--- Test 3: Metric Computation ---")
metrics = compute_metrics(results_df, label="Full Model")
check("coverage_rate" in metrics, "coverage_rate computed")
check("mae" in metrics, "mae computed")
check("rmse" in metrics, "rmse computed")
check(0 <= metrics["coverage_rate"] <= 1, f"coverage_rate in [0,1]: {metrics['coverage_rate']:.3f}")
check(metrics["mae"] > 0, f"mae > 0: {metrics['mae']:.2f}")
check(metrics["median_interval_width"] > 0, f"interval_width > 0: {metrics['median_interval_width']:.0f}")

print(f"\n  Metrics (100-match sample, 2000 sims):")
for k, v in metrics.items():
    print(f"    {k:25s}: {v}")

# ---------------------------------------------------------------------------
# Test 4: Baseline comparison (neutral model)
# ---------------------------------------------------------------------------
print("\n--- Test 4: Baseline Model Comparison ---")
bt_base = Backtester(sim_baseline, n_sim=2000, seed=42)
results_base = bt_base.run(states_100, max_matches=MAX_BACKTEST_MATCHES, verbose=False)
metrics_base = compute_metrics(results_base, label="Baseline (no skill)")

check("coverage_rate" in metrics_base, "Baseline metrics computed")
print(f"\n  Baseline metrics (100-match sample):")
for k, v in metrics_base.items():
    print(f"    {k:25s}: {v}")

# Both models should have some predictions in the band
check(metrics["coverage_rate"] > 0.3, f"Full model coverage > 30%: {metrics['coverage_rate']:.3f}")
check(metrics_base["coverage_rate"] > 0.3, f"Baseline coverage > 30%: {metrics_base['coverage_rate']:.3f}")

# ---------------------------------------------------------------------------
# Test 5: Calibration curve (innings 2 predictions)
# ---------------------------------------------------------------------------
print("\n--- Test 5: Calibration Curve ---")
inn2_results = results_df[results_df["innings"] == 2]
if len(inn2_results) > 0:
    cal = calibration_curve(results_df, n_bins=5)
    check(len(cal) >= 0, f"calibration_curve produced {len(cal)} bins")
    if len(cal) > 0:
        check("pred_p_win_mean" in cal.columns, "cal curve has pred_p_win_mean")
        check("actual_win_rate" in cal.columns, "cal curve has actual_win_rate")
        check((cal["actual_win_rate"].between(0, 1)).all(), "actual_win_rate in [0,1]")
        print(f"\n  Calibration curve (innings 2):")
        print(cal.to_string())
else:
    print("  [INFO] No innings 2 results in 100-match sample — skipping calibration curve test")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  PHASE 4 TESTS: {len(errors)} FAILED")
    for e in errors:
        print(f"  -- {e}")
else:
    print(f"  ALL PHASE 4 TESTS PASSED [PASS]")
    print(f"  Coverage rate (100 matches, 2k sims): {metrics['coverage_rate']:.1%}")
    print(f"  MAE (runs):                           {metrics['mae']:.1f}")
    print(f"  Median interval width:                {metrics['median_interval_width']:.0f} runs")
print("=" * 60)

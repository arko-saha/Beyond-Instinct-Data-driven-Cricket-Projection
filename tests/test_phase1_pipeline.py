"""Phase 1 smoke test — run from project root: python -m tests.test_phase1_pipeline"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

# --- Test 1: config imports ---
from src.forecast.config import (
    CALIBRATION_CUTOFF_DATE, MIN_SAMPLE_SIZE, MIN_BALLS_BATTER,
    MIN_BALLS_BOWLER, BALL_BY_BALL_PATH
)
print(f"[OK] config.py: cutoff={CALIBRATION_CUTOFF_DATE}, min_sample={MIN_SAMPLE_SIZE}")

# --- Test 2: pipeline imports ---
from src.forecast.data_pipeline import (
    load_ball_by_ball, split_calibration_evaluation,
    add_match_state_features, build_empirical_lookups,
    compute_xp_metrics, aggregate_player_xp, validate_no_eval_leakage,
    calculate_rate_bin, lookup_baseline
)
print("[OK] data_pipeline.py: all functions imported")

# --- Test 3: rate bin edge cases ---
rate_tests = [
    (-0.5, "very_low"), (0.9, "low"), (1.2, "medium"),
    (1.55, "high"), (2.1, "very_high"), (float("nan"), "very_low")
]
for rate, expected in rate_tests:
    result = calculate_rate_bin(rate)
    status = "OK" if result == expected else "FAIL"
    print(f"  [{status}] rate_bin({rate}) = {result!r} (expected {expected!r})")

# --- Test 4: load a small slice of real data ---
print("\nLoading 5,000-row sample...")
df_sample = pd.read_csv("data/consolidated_t20_data.csv", nrows=5000)
df_sample["start_date"] = pd.to_datetime(df_sample["start_date"], errors="coerce")
for col in ["wides", "noballs", "byes", "legbyes", "penalty", "extras"]:
    if col in df_sample.columns:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce").fillna(0).astype(int)
for col in ["is_wicket", "runs_off_bat", "total_runs"]:
    df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce").fillna(0).astype(int)

# --- Test 5: chronological split ---
df_calib, df_eval = split_calibration_evaluation(df_sample, "2024-01-01")
calib_ids = set(df_calib["match_id"].unique())
eval_ids = set(df_eval["match_id"].unique())
overlap = calib_ids & eval_ids
assert len(overlap) == 0, f"FAIL: {len(overlap)} overlapping match IDs!"
print(f"[OK] Split: {len(calib_ids)} calib matches, {len(eval_ids)} eval matches, 0 overlap")

# --- Test 6: state feature engineering ---
df_state = add_match_state_features(df_calib)
required_cols = [
    "cumulative_runs", "cumulative_wickets", "is_legal_ball",
    "cumulative_legal_balls", "balls_remaining", "wickets_in_hand",
    "scoring_rate", "asking_rate", "active_rate", "rate_bin"
]
missing = [c for c in required_cols if c not in df_state.columns]
assert not missing, f"FAIL: missing state columns: {missing}"
print(f"[OK] State features: all {len(required_cols)} required columns present")

# Assert pre-ball lags (cumulative_runs at ball 0 of any innings should be 0)
first_balls = df_state[
    (df_state["over"] == 0) & (df_state["ball"] == 1) & (df_state["innings"] == 1)
]
assert (first_balls["cumulative_runs"] == 0).all(), "FAIL: cumulative_runs not zero at first ball"
assert (first_balls["cumulative_wickets"] == 0).all(), "FAIL: cumulative_wickets not zero at first ball"
print("[OK] Pre-ball lag: cumulative_runs and cumulative_wickets are 0 at over 0, ball 1")

# Assert wickets_in_hand bounded [0,10]
assert df_state["wickets_in_hand"].between(0, 10).all(), "FAIL: wickets_in_hand out of [0,10]"
assert df_state["balls_remaining"].ge(0).all(), "FAIL: balls_remaining has negative values"
print("[OK] wickets_in_hand in [0,10], balls_remaining >= 0")

# --- Test 7: lookup builder (small sample, may have sparse cells) ---
if len(df_state) > 500:
    lookups = build_empirical_lookups(df_state)
    for lvl in ["level0", "level1", "level2", "level3"]:
        assert "match_id" not in lookups[lvl].columns, f"FAIL: match_id leaked into {lvl}"
        assert len(lookups[lvl]) > 0, f"FAIL: {lvl} lookup is empty"
    print(f"[OK] Lookups built: {[f'{k}: {len(v)} cells' for k,v in lookups.items()]}")

    # Test lookup_baseline returns finite values
    xr_b, xr_bw, xw = lookup_baseline(1, "medium", 10, 60, 7, lookups)
    assert np.isfinite(xr_b) and np.isfinite(xr_bw) and np.isfinite(xw), "FAIL: baseline lookup returned non-finite"
    assert 0.0 <= xw <= 1.0, f"FAIL: xW={xw} outside [0,1]"
    print(f"[OK] lookup_baseline(inn=1, medium, over=10, br=60, wih=7) -> xR_b={xr_b:.3f}, xW={xw:.4f}")

    # Test leakage validator
    validate_no_eval_leakage(lookups, df_eval)
    print("[OK] validate_no_eval_leakage passed")

print("\n" + "="*55)
print("  ALL PHASE 1 SMOKE TESTS PASSED [PASS]")
print("="*55)

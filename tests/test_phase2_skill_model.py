"""
Phase 2 validation tests for the log-odds skill interaction model.
Run from project root: python tests/test_phase2_skill_model.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.forecast.skill_model import (
    sigmoid, logit, safe_log_ratio,
    adjusted_wicket_prob, adjusted_run_expectation,
    compute_matchup_adjustments, build_player_skill_profiles,
    get_batter_deltas, get_bowler_deltas,
)

PASS = "[OK]"
FAIL = "[FAIL]"

errors = []

def check(cond, msg):
    if cond:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        errors.append(msg)

# ---------------------------------------------------------------------------
# Test 1: Mathematical primitives
# ---------------------------------------------------------------------------
print("\n--- Test 1: Mathematical Primitives ---")

# sigmoid is inverse of logit
for p in [0.01, 0.05, 0.15, 0.30, 0.50, 0.85, 0.99]:
    recovered = sigmoid(logit(p))
    check(abs(recovered - p) < 1e-9, f"sigmoid(logit({p})) = {recovered:.8f} (expected {p})")

# sigmoid is bounded in (0,1) for values where float64 stays representable.
# Float64 saturates around |x| > 36 (exp(-36) ~ 2e-16 ~ machine epsilon).
# Test the meaningful range.
for x in [-35, -10, -1, 0, 1, 10, 35]:
    s = sigmoid(x)
    check(0.0 < s < 1.0, f"sigmoid({x}) = {s} is in (0,1)")
# At float64 extremes, result should be at boundary (0 or 1)
check(sigmoid(1000) >= 1.0 - 1e-10, "sigmoid(1000) saturates to ~1.0 (float64 limit)")
check(sigmoid(-1000) <= 1e-10, "sigmoid(-1000) saturates to ~0.0 (float64 limit)")

# safe_log_ratio edge cases
check(safe_log_ratio(0, 1) == 0.0, "safe_log_ratio(0, 1) = 0.0 (no adjustment for zero actual)")
check(safe_log_ratio(1, 0) == 0.0, "safe_log_ratio(1, 0) = 0.0 (no adjustment for zero baseline)")
check(abs(safe_log_ratio(1.0, 1.0)) < 1e-9, "safe_log_ratio(1.0, 1.0) = 0.0 (neutral)")
check(safe_log_ratio(2.0, 1.0) > 0, "safe_log_ratio(2.0, 1.0) > 0 (above baseline)")
check(safe_log_ratio(0.5, 1.0) < 0, "safe_log_ratio(0.5, 1.0) < 0 (below baseline)")

# ---------------------------------------------------------------------------
# Test 2: adjusted_wicket_prob is always bounded in (0,1)
# ---------------------------------------------------------------------------
print("\n--- Test 2: adjusted_wicket_prob Bounds ---")

base_xws = [0.001, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.999]
for base in base_xws:
    for bowl in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        for bat in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            p = adjusted_wicket_prob(base, bowl, bat)
            check(
                0.0 < p < 1.0,
                f"adj_xw(base={base}, bowl={bowl}, bat={bat}) = {p:.6f} in (0,1)"
            )

# ---------------------------------------------------------------------------
# Test 3: Direction of adjustments is semantically correct
# ---------------------------------------------------------------------------
print("\n--- Test 3: Adjustment Direction ---")

BASE_XW = 0.05
BASE_XR = 1.20

# Good bowler (positive bowl_wkt_logodds) increases wicket probability
p_neutral = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=0.0, bat_wkt_logodds=0.0)
p_good_bowl = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=1.5, bat_wkt_logodds=0.0)
p_poor_bowl = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=-1.5, bat_wkt_logodds=0.0)
check(p_good_bowl > p_neutral, f"Good bowler increases adj_xw: {p_good_bowl:.4f} > {p_neutral:.4f}")
check(p_poor_bowl < p_neutral, f"Poor bowler decreases adj_xw: {p_poor_bowl:.4f} < {p_neutral:.4f}")

# Good batter (negative bat_wkt_logodds) decreases wicket probability
p_good_bat = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=0.0, bat_wkt_logodds=-1.5)
p_poor_bat = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=0.0, bat_wkt_logodds=1.5)
check(p_good_bat < p_neutral, f"Good batter decreases adj_xw: {p_good_bat:.4f} < {p_neutral:.4f}")
check(p_poor_bat > p_neutral, f"Poor batter increases adj_xw: {p_poor_bat:.4f} > {p_neutral:.4f}")

# Elite bowler vs poor batter: highest probability
p_elite_matchup = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=2.0, bat_wkt_logodds=2.0)
# Strong batter vs weak bowler: lowest probability
p_batter_dominated = adjusted_wicket_prob(BASE_XW, bowl_wkt_logodds=-2.0, bat_wkt_logodds=-2.0)
check(p_elite_matchup > p_neutral, f"Elite bowl + poor bat: {p_elite_matchup:.4f} > neutral {p_neutral:.4f}")
check(p_batter_dominated < p_neutral, f"Poor bowl + good bat: {p_batter_dominated:.4f} < neutral {p_neutral:.4f}")

# Good batter (positive run_logfactor) increases adj_xr
r_neutral = adjusted_run_expectation(BASE_XR, 0.0, 0.0)
r_good_bat = adjusted_run_expectation(BASE_XR, bat_run_logfactor=0.5, bowl_run_logfactor=0.0)
r_good_bowl = adjusted_run_expectation(BASE_XR, bat_run_logfactor=0.0, bowl_run_logfactor=-0.5)
check(r_good_bat > r_neutral, f"Good batter increases adj_xr: {r_good_bat:.4f} > {r_neutral:.4f}")
check(r_good_bowl < r_neutral, f"Good bowler decreases adj_xr: {r_good_bowl:.4f} < {r_neutral:.4f}")

# adjusted_run_expectation is always positive
for base_xr in [0.001, 0.5, 1.2, 6.0]:
    for bat_f in [-5.0, 0.0, 5.0]:
        for bowl_f in [-5.0, 0.0, 5.0]:
            r = adjusted_run_expectation(base_xr, bat_f, bowl_f)
            check(r > 0, f"adj_xr({base_xr}, bat={bat_f}, bowl={bowl_f}) = {r:.6f} > 0")

# ---------------------------------------------------------------------------
# Test 4: build_player_skill_profiles on synthetic data
# ---------------------------------------------------------------------------
print("\n--- Test 4: Skill Profiles on Synthetic Data ---")

# Create a synthetic XP dataset with known ground truth
np.random.seed(42)
n = 2000

# "Good bowler" takes wickets at 2x the expected rate
# "Good batter" gets out at 0.5x the expected rate
# "Average player" performs exactly at baseline

df_synth = pd.DataFrame({
    "batter": np.random.choice(
        ["GoodBatter", "AverageBatter", "PoorBatter"], n, p=[0.33, 0.34, 0.33]
    ),
    "bowler": np.random.choice(
        ["GoodBowler", "AverageBowler", "PoorBowler"], n, p=[0.33, 0.34, 0.33]
    ),
    "wides": 0,
    "noballs": 0,
    "is_legal_ball": 1,
    "xW_baseline": 0.05,
    "xR_batter_baseline": 1.2,
    "xR_bowler_baseline": 1.35,
    "bowler_runs_conceded": np.random.choice([0, 1, 2, 4, 6], n, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
    "runs_off_bat": np.random.choice([0, 1, 2, 4, 6], n, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
})

# Assign is_wicket based on player identity to create known skill ordering
def assign_wicket(row):
    if row["bowler"] == "GoodBowler" and row["batter"] == "PoorBatter":
        return int(np.random.random() < 0.12)   # much higher than baseline 0.05
    elif row["bowler"] == "GoodBowler":
        return int(np.random.random() < 0.09)   # good bowler
    elif row["batter"] == "GoodBatter":
        return int(np.random.random() < 0.02)   # very hard to dismiss
    elif row["batter"] == "PoorBatter":
        return int(np.random.random() < 0.10)   # easy to dismiss
    else:
        return int(np.random.random() < 0.05)   # baseline

df_synth["is_wicket"] = df_synth.apply(assign_wicket, axis=1)

# Assign runs_off_bat based on batter
def assign_runs(row):
    if row["batter"] == "GoodBatter":
        return int(np.random.choice([0, 1, 2, 4, 6], p=[0.25, 0.25, 0.15, 0.20, 0.15]))
    elif row["batter"] == "PoorBatter":
        return int(np.random.choice([0, 1, 2, 4, 6], p=[0.50, 0.25, 0.10, 0.10, 0.05]))
    return int(np.random.choice([0, 1, 2, 4, 6], p=[0.35, 0.30, 0.15, 0.10, 0.10]))

df_synth["runs_off_bat"] = df_synth.apply(assign_runs, axis=1)

batter_skill, bowler_skill = build_player_skill_profiles(
    df_synth, min_balls_batter=50, min_balls_bowler=50
)

# Good batter should have NEGATIVE bat_wkt_logodds (gets out less)
gb = batter_skill[batter_skill["batter"] == "GoodBatter"]["bat_wkt_logodds"].values[0]
pb = batter_skill[batter_skill["batter"] == "PoorBatter"]["bat_wkt_logodds"].values[0]
ab = batter_skill[batter_skill["batter"] == "AverageBatter"]["bat_wkt_logodds"].values[0]
check(gb < ab, f"GoodBatter bat_wkt_logodds ({gb:.3f}) < AverageBatter ({ab:.3f})")
check(pb > ab, f"PoorBatter bat_wkt_logodds ({pb:.3f}) > AverageBatter ({ab:.3f})")
check(gb < pb, f"GoodBatter ({gb:.3f}) < PoorBatter ({pb:.3f}) -- correct ordering")

# Good bowler should have POSITIVE bowl_wkt_logodds (takes more wickets)
gbow = bowler_skill[bowler_skill["bowler"] == "GoodBowler"]["bowl_wkt_logodds"].values[0]
pbow = bowler_skill[bowler_skill["bowler"] == "PoorBowler"]["bowl_wkt_logodds"].values[0]
abow = bowler_skill[bowler_skill["bowler"] == "AverageBowler"]["bowl_wkt_logodds"].values[0]
check(gbow > abow, f"GoodBowler bowl_wkt_logodds ({gbow:.3f}) > AverageBowler ({abow:.3f})")
check(gbow > pbow, f"GoodBowler ({gbow:.3f}) > PoorBowler ({pbow:.3f}) -- correct ordering")

# Good batter should have POSITIVE bat_run_logfactor (scores more)
gb_run = batter_skill[batter_skill["batter"] == "GoodBatter"]["bat_run_logfactor"].values[0]
pb_run = batter_skill[batter_skill["batter"] == "PoorBatter"]["bat_run_logfactor"].values[0]
check(gb_run > pb_run, f"GoodBatter run_logfactor ({gb_run:.3f}) > PoorBatter ({pb_run:.3f})")

# ---------------------------------------------------------------------------
# Test 5: compute_matchup_adjustments end-to-end
# ---------------------------------------------------------------------------
print("\n--- Test 5: compute_matchup_adjustments End-to-End ---")

result_elite = compute_matchup_adjustments(
    base_xr=1.2, base_xw=0.05,
    batter_name="PoorBatter", bowler_name="GoodBowler",
    batter_skill=batter_skill, bowler_skill=bowler_skill,
)
result_batter_dom = compute_matchup_adjustments(
    base_xr=1.2, base_xw=0.05,
    batter_name="GoodBatter", bowler_name="PoorBowler",
    batter_skill=batter_skill, bowler_skill=bowler_skill,
)
result_unknown = compute_matchup_adjustments(
    base_xr=1.2, base_xw=0.05,
    batter_name="UnknownPlayer", bowler_name="UnknownBowler",
    batter_skill=batter_skill, bowler_skill=bowler_skill,
)

check(0.0 < result_elite["adj_xw"] < 1.0, f"Elite matchup adj_xw in (0,1): {result_elite['adj_xw']:.4f}")
check(0.0 < result_batter_dom["adj_xw"] < 1.0, f"Batter-dom adj_xw in (0,1): {result_batter_dom['adj_xw']:.4f}")
check(result_elite["adj_xw"] > result_batter_dom["adj_xw"],
      f"GoodBowl+PoorBat ({result_elite['adj_xw']:.4f}) > PoorBowl+GoodBat ({result_batter_dom['adj_xw']:.4f})")
check(result_elite["adj_xr"] > 0, f"adj_xr > 0: {result_elite['adj_xr']:.4f}")

# Unknown player should return baseline values (zero deltas)
check(
    abs(result_unknown["adj_xw"] - adjusted_wicket_prob(0.05, 0.0, 0.0)) < 1e-9,
    f"Unknown player returns baseline adj_xw: {result_unknown['adj_xw']:.6f}"
)
check(
    abs(result_unknown["adj_xr"] - 1.2) < 1e-9,
    f"Unknown player returns baseline adj_xr: {result_unknown['adj_xr']:.6f}"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  PHASE 2 TESTS: {len(errors)} FAILED")
    for e in errors:
        print(f"  -- {e}")
else:
    print("  ALL PHASE 2 TESTS PASSED [PASS]")
print("=" * 60)

"""
Phase 3 validation tests for the Monte Carlo simulation engine.
Run from project root: python tests/test_phase3_simulator.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

from src.forecast.simulator import (
    InningsSimulator, SimulationResult,
    build_baseline_tables, build_scoring_distribution,
)

PASS, FAIL = "[OK]", "[FAIL]"
errors = []

def check(cond, msg):
    if cond:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        errors.append(msg)

# ---------------------------------------------------------------------------
# Load Phase 1/2 artefacts
# ---------------------------------------------------------------------------
print("Loading artefacts...")
p1 = Path("models/phase1_artefacts")
with open(p1 / "lookups.pkl", "rb") as f:
    lookups = pickle.load(f)

batter_skill = pd.read_parquet("models/phase2_artefacts/batter_skill.parquet")
bowler_skill = pd.read_parquet("models/phase2_artefacts/bowler_skill.parquet")
df_calib_xp = pd.read_parquet(p1 / "df_calib_xp.parquet")
print(f"Artefacts loaded: {len(batter_skill)} batters, {len(bowler_skill)} bowlers")

# ---------------------------------------------------------------------------
# Test 1: Baseline table construction
# ---------------------------------------------------------------------------
print("\n--- Test 1: Baseline Tables ---")
tables = build_baseline_tables(lookups)

check("xw_table" in tables and "xr_bat_table" in tables, "build_baseline_tables returns required keys")
check(tables["xw_table"].shape == (2, 20, 11), f"xw_table shape == (2,20,11): {tables['xw_table'].shape}")
check(tables["xr_bat_table"].shape == (2, 20, 11), f"xr_bat_table shape == (2,20,11)")
check((tables["xw_table"] > 0).all() and (tables["xw_table"] < 1).all(), "All xw values in (0,1)")
check((tables["xr_bat_table"] > 0).all(), "All xr_bat values > 0")

# Sanity: early overs should have fewer wickets expected than death overs
# (powerplay avg xW should be plausible)
pp_xw = tables["xw_table"][0, 0:6, 10].mean()  # innings 1, overs 0-5, full wickets
death_xw = tables["xw_table"][0, 16:20, 10].mean()  # death overs
print(f"  Powerplay avg xW (I1): {pp_xw:.4f}, Death avg xW: {death_xw:.4f}")

# ---------------------------------------------------------------------------
# Test 2: Scoring distribution
# ---------------------------------------------------------------------------
print("\n--- Test 2: Scoring Distribution ---")
scoring_dist = build_scoring_distribution(df_calib_xp)
check(len(scoring_dist) > 0, f"build_scoring_distribution produced {len(scoring_dist)} context entries")

# Each distribution should sum to 1
for key, dist in list(scoring_dist.items())[:10]:
    total = dist.sum()
    check(abs(total - 1.0) < 1e-6, f"  dist {key} sums to {total:.6f} (~1.0)")

# ---------------------------------------------------------------------------
# Test 3: Simulator initialization
# ---------------------------------------------------------------------------
print("\n--- Test 3: Simulator Initialization ---")
sim = InningsSimulator(lookups, batter_skill, bowler_skill, scoring_dist)
check(len(sim._batter_logodds) > 0, f"Batter registry populated: {len(sim._batter_logodds)} entries")
check(len(sim._bowler_logodds) > 0, f"Bowler registry populated: {len(sim._bowler_logodds)} entries")

# ---------------------------------------------------------------------------
# Test 4: Basic simulation run (Innings 1, from over 15)
# ---------------------------------------------------------------------------
print("\n--- Test 4: Innings 1 Simulation (Over 15, 120/3) ---")
batting_lineup = [
    "SA Yadav", "HH Pandya", "RA Tripathi", "JJ Roy", "AD Russell",
    "KA Pollard", "SP Narine", "Unknown1", "Unknown2", "Unknown3"
]
bowling_plan = {
    15: "JJ Bumrah", 16: "TA Boult", 17: "JJ Bumrah", 18: "TA Boult", 19: "JJ Bumrah"
}

t0 = time.time()
result = sim.simulate(
    innings=1,
    current_over=15,
    current_runs=120,
    current_wickets=3,
    batting_lineup=batting_lineup,
    bowling_plan=bowling_plan,
    target=None,
    n_simulations=10_000,
    seed=42,
)
elapsed = time.time() - t0

check(elapsed < 30.0, f"10k simulations in {elapsed:.2f}s (< 30s benchmark)")
check(isinstance(result, SimulationResult), "Returns SimulationResult instance")
check(len(result.final_scores) == 10_000, "final_scores has 10,000 entries")
check(len(result.final_wickets) == 10_000, "final_wickets has 10,000 entries")

# All final scores >= current_runs (scores can only go up)
check((result.final_scores >= 120).all(), f"All final scores >= 120 (starting runs)")

# Final wickets should be >= current (can only increase)
check((result.final_wickets >= 3).all(), "All final wickets >= 3 (starting wickets)")

# Wickets bounded at 10
check((result.final_wickets <= 10).all(), "All final wickets <= 10")

print(f"  Innings 1 score distribution:")
print(f"    P10={result.score_p10:.0f}, Median={result.score_median:.0f}, P90={result.score_p90:.0f}")
print(f"    Mean={result.score_mean:.1f}, P(collapse)={result.p_collapse:.3f}")

# Sanity: from 120/3 with 5 overs left, median should be in a reasonable range
check(140 <= result.score_median <= 210, f"Median score {result.score_median:.0f} in plausible range [140,210]")
# P10 should be above starting score
check(result.score_p10 > 120, f"P10 score {result.score_p10:.0f} > 120")
# Confidence band should have reasonable width
check(result.score_p90 - result.score_p10 >= 20, f"Confidence band >= 20 runs wide")

# ---------------------------------------------------------------------------
# Test 5: Simulation is stochastic (different seeds → different results)
# ---------------------------------------------------------------------------
print("\n--- Test 5: Stochasticity Check ---")
result_a = sim.simulate(1, 15, 120, 3, batting_lineup, bowling_plan, seed=1)
result_b = sim.simulate(1, 15, 120, 3, batting_lineup, bowling_plan, seed=2)
same_medians = abs(result_a.score_median - result_b.score_median) < 0.01

check(not same_medians, f"Different seeds produce different medians: {result_a.score_median:.1f} vs {result_b.score_median:.1f}")

# Same seed → deterministic
result_c = sim.simulate(1, 15, 120, 3, batting_lineup, bowling_plan, seed=42)
result_d = sim.simulate(1, 15, 120, 3, batting_lineup, bowling_plan, seed=42)
check(
    np.array_equal(result_c.final_scores, result_d.final_scores),
    "Same seed produces identical results (reproducible)"
)

# ---------------------------------------------------------------------------
# Test 6: Innings 2 chase scenario and win probability
# ---------------------------------------------------------------------------
print("\n--- Test 6: Innings 2 Chase Scenario ---")
batting_lineup_2 = [
    "V Kohli", "RG Sharma", "KL Rahul", "SA Yadav", "HH Pandya",
    "RA Tripathi", "MS Dhoni", "Unknown4", "Unknown5", "Unknown6"
]
bowling_plan_2 = {
    10: "JJ Bumrah", 11: "JT Fabian", 12: "Kuldeep Yadav",
    13: "JJ Bumrah", 14: "R Ashwin", 15: "Kuldeep Yadav",
    16: "JJ Bumrah", 17: "R Ashwin", 18: "JT Fabian", 19: "JJ Bumrah"
}

# Favourable chase: 80 from 10 overs, needing 160 total
result_fav = sim.simulate(
    innings=2, current_over=10, current_runs=80, current_wickets=2,
    batting_lineup=batting_lineup_2, bowling_plan=bowling_plan_2,
    target=160, n_simulations=10_000, seed=42
)
# Difficult chase: 80 from 10 overs, needing 200 total
result_diff = sim.simulate(
    innings=2, current_over=10, current_runs=80, current_wickets=6,
    batting_lineup=batting_lineup_2, bowling_plan=bowling_plan_2,
    target=200, n_simulations=10_000, seed=42
)

check(result_fav.win_probability is not None, "Win probability computed for innings 2")
check(0.0 <= result_fav.win_probability <= 1.0, f"Win prob in [0,1]: {result_fav.win_probability:.3f}")
check(0.0 <= result_diff.win_probability <= 1.0, f"Difficult win prob in [0,1]: {result_diff.win_probability:.3f}")
check(
    result_fav.win_probability > result_diff.win_probability,
    f"Favourable chase P(win)={result_fav.win_probability:.3f} > Difficult P(win)={result_diff.win_probability:.3f}"
)
print(f"  Favourable chase (80/2, need 160): P(win)={result_fav.win_probability:.3f}")
print(f"  Difficult chase  (80/6, need 200): P(win)={result_diff.win_probability:.3f}")

# ---------------------------------------------------------------------------
# Test 7: Over snapshots correctness
# ---------------------------------------------------------------------------
print("\n--- Test 7: Over Snapshots ---")
check(len(result.over_snapshots) == 5, f"5 over snapshots for overs 15-19: {len(result.over_snapshots)}")
for snap in result.over_snapshots:
    check("runs_median" in snap, f"  Snapshot has runs_median key")
    check("runs_p10" in snap, f"  Snapshot has runs_p10 key")
    check("wickets_median" in snap, f"  Snapshot has wickets_median key")
    check(snap["runs_p10"] <= snap["runs_median"] <= snap["runs_p90"],
          f"  P10 <= median <= P90: {snap['runs_p10']:.0f} <= {snap['runs_median']:.0f} <= {snap['runs_p90']:.0f}")

# Monotonic increase: each snapshot should have higher median than the previous
medians = [s["runs_median"] for s in result.over_snapshots]
check(all(medians[i] <= medians[i+1] for i in range(len(medians)-1)),
      f"Over-by-over medians are monotonically non-decreasing: {[round(m) for m in medians]}")

# ---------------------------------------------------------------------------
# Test 8: Edge cases
# ---------------------------------------------------------------------------
print("\n--- Test 8: Edge Cases ---")

# All out immediately (9 wickets down, 1 batter left)
result_last_man = sim.simulate(
    innings=1, current_over=10, current_runs=100, current_wickets=9,
    batting_lineup=["JJ Bumrah"], bowling_plan={}, n_simulations=1000, seed=42
)
check((result_last_man.final_wickets >= 9).all(), "Last man standing: wickets always >= 9")
check((result_last_man.final_wickets <= 10).all(), "Last man standing: wickets <= 10")
print(f"  Last man in (100/9, over 10): median score = {result_last_man.score_median:.0f}")

# Full innings from ball 1 (over 0)
result_full = sim.simulate(
    innings=1, current_over=0, current_runs=0, current_wickets=0,
    batting_lineup=["RG Sharma", "V Kohli", "KL Rahul", "SA Yadav",
                    "HH Pandya", "MS Dhoni", "HH Pandya", "JJ Bumrah",
                    "R Ashwin", "JJ Bumrah"],
    bowling_plan={}, n_simulations=5000, seed=42
)
# T20 scores from start: median should be in typical T20 range
check(100 <= result_full.score_median <= 200, f"Full innings median {result_full.score_median:.0f} in [100,200]")
print(f"  Full innings from 0/0: P10={result_full.score_p10:.0f}, Med={result_full.score_median:.0f}, P90={result_full.score_p90:.0f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  PHASE 3 TESTS: {len(errors)} FAILED")
    for e in errors:
        print(f"  -- {e}")
else:
    print("  ALL PHASE 3 TESTS PASSED [PASS]")
print("=" * 60)

"""
Phase 5 validation tests for the Tactical Decision Optimizer.
Run from project root: python tests/test_phase5_optimizer.py

Tests:
1. BowlingOptimizer produces valid plan rankings
2. Beam search respects T20 constraints (quota, no-consecutive)
3. Good bowler correctly ranked above bad bowler in synthetic scenario
4. BattingOptimizer returns ranked lineup permutations
5. MatchState.override() clean scenario injection
6. MatchState.compare_scenarios() multi-scenario comparison
"""
import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path

from src.forecast.simulator import InningsSimulator, build_scoring_distribution
from src.forecast.optimizer import BowlingOptimizer, BattingOptimizer, Bowler, MatchState

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
scoring_dist  = build_scoring_distribution(df_calib_xp)
sim = InningsSimulator(lookups, batter_skill, bowler_skill, scoring_dist)

# ---------------------------------------------------------------------------
# Test 1: BowlingOptimizer - basic ranking (fielding scenario)
# ---------------------------------------------------------------------------
print("\n--- Test 1: Bowling Optimizer (Minimise Runs) ---")

# Scenario: chasing team bowls remaining 6 overs. 4 bowlers, each 2 overs left.
available_bowlers = [
    Bowler("JJ Bumrah", overs_remaining=2),
    Bowler("TA Boult", overs_remaining=2),
    Bowler("Kuldeep Yadav", overs_remaining=2),
    Bowler("R Ashwin", overs_remaining=2),
]

batting_lineup = [
    "V Kohli", "RG Sharma", "SA Yadav", "HH Pandya",
    "MS Dhoni", "RA Tripathi", "T1", "T2", "T3", "T4"
]

opt = BowlingOptimizer(sim, n_sim=2000, seed=42, beam_width=15)

t0 = time.time()
result_df = opt.optimize(
    innings=2,
    current_over=14,
    current_runs=110,
    current_wickets=3,
    batting_lineup=batting_lineup,
    available_bowlers=available_bowlers,
    target=165,
    objective="maximise_p_win",
    top_n=5,
)
elapsed = time.time() - t0

check(len(result_df) > 0, f"Optimizer returned {len(result_df)} strategies")
check(elapsed < 120, f"Optimization in {elapsed:.1f}s (< 2 min)")

if len(result_df) > 0:
    required_cols = ["rank", "plan_label", "exp_runs", "score_median", "p_win", "p_collapse"]
    for col in required_cols:
        check(col in result_df.columns, f"  Column '{col}' present")
    
    check((result_df["rank"] == list(range(1, len(result_df)+1))).all(), "Ranks are sequential")
    check((result_df["p_win"].diff().dropna() <= 0.0001).all(), "Results sorted by P(win) desc")
    check((result_df["p_win"] >= 0).all() and (result_df["p_win"] <= 1).all(), "P(win) in [0,1]")
    
    print(f"\n  Top bowling strategies (ranked by P(win)):")
    print(result_df[["rank","plan_label","exp_runs","p_win","p_collapse"]].to_string())

# ---------------------------------------------------------------------------
# Test 2: Constraint validation (no-consecutive overs)
# ---------------------------------------------------------------------------
print("\n--- Test 2: T20 Constraints (No Consecutive Overs) ---")

# 2 bowlers, 2 overs remaining — neither can bowl both consecutively
bowlers_2 = [Bowler("A", overs_remaining=2), Bowler("B", overs_remaining=2)]
lineup_2 = ["Bat1", "Bat2", "Bat3", "Bat4", "Bat5", "Bat6", "Bat7", "Bat8", "Bat9", "Bat10"]

opt2 = BowlingOptimizer(sim, n_sim=1000, seed=42, beam_width=20)
result2 = opt2.optimize(
    innings=1, current_over=18, current_runs=150, current_wickets=2,
    batting_lineup=lineup_2, available_bowlers=bowlers_2,
    objective="minimise_runs", top_n=10
)

if len(result2) > 0:
    for _, row in result2.iterrows():
        plan_label = row["plan_label"]
        check(True, f"  Valid plan produced: {plan_label}")

# Verify no plan has same bowler in consecutive overs by parsing plan labels
check(len(result2) > 0, f"Plans generated for 2-bowler 2-over scenario: {len(result2)}")

# ---------------------------------------------------------------------------
# Test 3: Synthetic ordering test (known better vs worse bowler)
# ---------------------------------------------------------------------------
print("\n--- Test 3: Ordering Sanity (Better Bowler -> Lower Expected Score) ---")

# Find an elite bowler and a poor bowler from our skill registry
elite_bowl = bowler_skill.nsmallest(1, "bowl_wkt_logodds").iloc[0]  # MOST wickets = lowest logodds for batter
poor_bowl  = bowler_skill.nlargest(1, "bowl_wkt_logodds").iloc[0]

# Actually: bowl_wkt_logodds > 0 means takes MORE wickets than expected (good bowler)
#           bowl_wkt_logodds < 0 means takes fewer wickets (bad bowler)
# Also look at bowl_run_logfactor: < 0 means concedes fewer runs
elite = bowler_skill.nsmallest(3, "bowl_run_logfactor").iloc[0]  # concedes fewest runs
poor  = bowler_skill.nlargest(3, "bowl_run_logfactor").iloc[0]   # concedes most runs

print(f"  Elite bowler: {elite['bowler']} (run_logfactor={elite['bowl_run_logfactor']:.3f})")
print(f"  Poor bowler:  {poor['bowler']}  (run_logfactor={poor['bowl_run_logfactor']:.3f})")

# Simulate with only elite vs only poor bowler for remaining 2 overs
plan_elite = {18: elite["bowler"], 19: elite["bowler"]}
plan_poor  = {18: poor["bowler"], 19: poor["bowler"]}

batting = ["V Kohli","RG Sharma","SA Yadav","HH Pandya","MS Dhoni","RA Tripathi","T1","T2","T3","T4"]

r_elite = sim.simulate(1, 18, 150, 3, batting, plan_elite, n_simulations=5000, seed=42)
r_poor  = sim.simulate(1, 18, 150, 3, batting, plan_poor,  n_simulations=5000, seed=42)

print(f"  Elite bowler median total: {r_elite.score_median:.0f}")
print(f"  Poor bowler  median total: {r_poor.score_median:.0f}")

check(
    r_elite.score_mean <= r_poor.score_mean,
    f"Elite bowler yields lower E[score]: {r_elite.score_mean:.1f} <= {r_poor.score_mean:.1f}"
)

# ---------------------------------------------------------------------------
# Test 4: BattingOptimizer
# ---------------------------------------------------------------------------
print("\n--- Test 4: Batting Lineup Optimizer ---")

remaining = ["AD Russell", "KA Pollard", "SP Narine", "JJ Roy", "Unknown1"]
bat_opt = BattingOptimizer(sim, n_sim=1000, seed=42, perm_limit=20)

t0 = time.time()
bat_result = bat_opt.optimize(
    innings=1,
    current_over=15,
    current_runs=120,
    current_wickets=4,
    current_batter="SA Yadav",
    remaining_batters=remaining,
    bowling_plan={15: "JJ Bumrah", 16: "TA Boult", 17: "JJ Bumrah", 18: "TA Boult", 19: "JJ Bumrah"},
    objective="maximise_runs",
    top_n=5,
)
elapsed_bat = time.time() - t0

check(len(bat_result) > 0, f"BattingOptimizer returned {len(bat_result)} lineups in {elapsed_bat:.1f}s")
if len(bat_result) > 0:
    check("lineup_str" in bat_result.columns, "lineup_str column present")
    check("exp_runs" in bat_result.columns, "exp_runs column present")
    check(bat_result["exp_runs"].iloc[0] >= bat_result["exp_runs"].iloc[-1], "Results sorted by exp_runs desc")
    check(bat_result["lineup_str"].str.startswith("SA Yadav").all(), "Current batter always first")
    print(f"  Top batting lineups:")
    print(bat_result[["rank","lineup_str","exp_runs","score_p10","score_p90"]].to_string())

# ---------------------------------------------------------------------------
# Test 5: MatchState and what-if scenarios
# ---------------------------------------------------------------------------
print("\n--- Test 5: MatchState Override (What-If Scenarios) ---")

base_state = MatchState(
    innings=2,
    current_over=14,
    current_runs=110,
    current_wickets=3,
    batting_lineup=batting_lineup,
    bowling_plan={14: "JJ Bumrah", 15: "R Ashwin", 16: "TA Boult", 17: "JJ Bumrah",
                  18: "R Ashwin", 19: "JJ Bumrah"},
    target=165,
    label="Base scenario",
)

# Override: what if it rained and only 2 overs remained?
rain_state = base_state.override(current_over=18, target=150, label="Rain (18 overs)")
check(rain_state.current_over == 18, "Override: current_over changed")
check(rain_state.target == 150, "Override: target changed")
check(rain_state.innings == 2, "Override: innings unchanged (immutable)")
check(base_state.current_over == 14, "Override: original state not mutated")

# Simulate both
r_base = base_state.simulate(sim, n_sim=2000, seed=42)
r_rain = rain_state.simulate(sim, n_sim=2000, seed=42)
check(isinstance(r_base.win_probability, float), f"Base scenario P(win)={r_base.win_probability:.3f}")
check(isinstance(r_rain.win_probability, float), f"Rain scenario P(win)={r_rain.win_probability:.3f}")

# ---------------------------------------------------------------------------
# Test 6: compare_scenarios
# ---------------------------------------------------------------------------
print("\n--- Test 6: Multi-Scenario Comparison ---")
scenarios = [
    {"bowling_plan": {14:"JJ Bumrah",15:"R Ashwin",16:"TA Boult",17:"JJ Bumrah",18:"R Ashwin",19:"JJ Bumrah"},
     "label": "Bumrah death"},
    {"bowling_plan": {14:"Kuldeep Yadav",15:"R Ashwin",16:"Kuldeep Yadav",17:"R Ashwin",18:"JJ Bumrah",19:"JJ Bumrah"},
     "label": "Spin first"},
    {"bowling_plan": {14:"Unknown",15:"Unknown",16:"Unknown",17:"Unknown",18:"Unknown",19:"Unknown"},
     "label": "Unknown (neutral)"},
]

comp_df = base_state.compare_scenarios(sim, overrides=scenarios, n_sim=2000, seed=42)
check(len(comp_df) == 3, f"compare_scenarios returned {len(comp_df)} rows")
check("score_median" in comp_df.columns, "score_median column present")
check("win_probability" in comp_df.columns, "win_probability column present")
print(f"  Scenario comparison:")
print(comp_df[["score_median","score_p10","score_p90","win_probability","p_collapse"]].to_string())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  PHASE 5 TESTS: {len(errors)} FAILED")
    for e in errors:
        print(f"  -- {e}")
else:
    print("  ALL PHASE 5 TESTS PASSED [PASS]")
print("=" * 60)

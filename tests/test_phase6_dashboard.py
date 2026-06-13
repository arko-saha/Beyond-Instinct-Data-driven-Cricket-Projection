"""
Phase 6 validation tests for the Visualisation Dashboard.
Run from project root: python tests/test_phase6_dashboard.py
"""
import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for testing
import matplotlib.pyplot as plt
from pathlib import Path

from src.forecast.simulator import InningsSimulator, build_scoring_distribution
from src.forecast.backtester import calibration_curve
from src.forecast.dashboard import (
    plot_score_fan_chart,
    plot_win_probability_timeline,
    plot_strategy_comparison,
    plot_player_xp_leaderboard,
    plot_backtest_dashboard,
    plot_bias_heatmap,
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
# Load artefacts
# ---------------------------------------------------------------------------
print("Loading artefacts...")
P1 = Path("models/phase1_artefacts")
P2 = Path("models/phase2_artefacts")
P4 = Path("models/phase4_artefacts")

with open(P1 / "lookups.pkl", "rb") as f:
    lookups = pickle.load(f)
batter_skill = pd.read_parquet(P2 / "batter_skill.parquet")
bowler_skill  = pd.read_parquet(P2 / "bowler_skill.parquet")
df_xp = pd.read_parquet(P1 / "df_calib_xp.parquet")
scoring_dist = build_scoring_distribution(df_xp)
sim = InningsSimulator(lookups, batter_skill, bowler_skill, scoring_dist)

results_full = pd.read_parquet(P4 / "backtest_results_full.parquet")
results_base = pd.read_parquet(P4 / "backtest_results_baseline.parquet")
cal_full = pd.read_parquet(P4 / "calibration_full.parquet")

print(f"Artefacts loaded. Backtest: {len(results_full)} predictions.")

# ---------------------------------------------------------------------------
# Run simulation for chart inputs
# ---------------------------------------------------------------------------
batting_lineup = ["V Kohli","RG Sharma","SA Yadav","HH Pandya",
                  "MS Dhoni","RA Tripathi","JJ Roy","T1","T2","T3"]
bowling_plan = {10:"JJ Bumrah",11:"R Ashwin",12:"TA Boult",13:"JJ Bumrah",
                14:"R Ashwin",15:"Kuldeep Yadav",16:"JJ Bumrah",17:"TA Boult",
                18:"R Ashwin",19:"JJ Bumrah"}

r1 = sim.simulate(2, 10, 85, 3, batting_lineup, bowling_plan, target=165, n_simulations=3000, seed=42)
r2 = sim.simulate(2, 10, 85, 3, batting_lineup, {}, target=165, n_simulations=3000, seed=99)

# ---------------------------------------------------------------------------
# Test 1: Score fan chart
# ---------------------------------------------------------------------------
print("\n--- Test 1: Score Fan Chart ---")
fig1, ax1 = plot_score_fan_chart(r1, title="Test Fan Chart", target=165)
check(isinstance(fig1, plt.Figure), "plot_score_fan_chart returns Figure")
check(ax1.get_xlabel() != "", "Fan chart has x-axis label")
check(ax1.get_ylabel() != "", "Fan chart has y-axis label")
plt.close(fig1)
print("[OK] Score fan chart rendered successfully")

# ---------------------------------------------------------------------------
# Test 2: Win probability timeline
# ---------------------------------------------------------------------------
print("\n--- Test 2: Win Probability Timeline ---")
p_win_by_over = {}
for over in range(10, 20):
    r = sim.simulate(2, over, 85 + (over-10)*7, 3, batting_lineup, bowling_plan,
                     target=165, n_simulations=1000, seed=42)
    p_win_by_over[over] = r.win_probability or 0.0

fig2, ax2 = plot_win_probability_timeline(p_win_by_over, target=165)
check(isinstance(fig2, plt.Figure), "plot_win_probability_timeline returns Figure")
check(ax2.get_xlabel() != "", "Timeline has x-axis label")
plt.close(fig2)
print("[OK] Win probability timeline rendered")

# ---------------------------------------------------------------------------
# Test 3: Strategy comparison
# ---------------------------------------------------------------------------
print("\n--- Test 3: Strategy Comparison ---")
strategies = [
    ("Bumrah death", r1),
    ("Neutral",      r2),
]
fig3, (ax3a, ax3b) = plot_strategy_comparison(strategies, target=165)
check(isinstance(fig3, plt.Figure), "plot_strategy_comparison returns Figure")
plt.close(fig3)
print("[OK] Strategy comparison rendered")

# ---------------------------------------------------------------------------
# Test 4: Player XP leaderboard
# ---------------------------------------------------------------------------
print("\n--- Test 4: Player XP Leaderboard ---")
fig4, axes4 = plot_player_xp_leaderboard(batter_skill, bowler_skill, top_n=8)
check(isinstance(fig4, plt.Figure), "plot_player_xp_leaderboard returns Figure")
check(len(axes4) == 2, "Leaderboard has 2 axes (batter + bowler)")
plt.close(fig4)
print("[OK] XP leaderboard rendered")

# ---------------------------------------------------------------------------
# Test 5: Backtest dashboard
# ---------------------------------------------------------------------------
print("\n--- Test 5: Backtest Dashboard ---")
fig5 = plot_backtest_dashboard(results_full, cal_df=cal_full, baseline_df=results_base)
check(isinstance(fig5, plt.Figure), "plot_backtest_dashboard returns Figure")
fig5.savefig("models/phase6_backtest_dashboard.png", dpi=80, bbox_inches="tight")
plt.close(fig5)
print("[OK] Backtest dashboard rendered and saved to models/phase6_backtest_dashboard.png")

# ---------------------------------------------------------------------------
# Test 6: Bias heatmap
# ---------------------------------------------------------------------------
print("\n--- Test 6: Bias Heatmap ---")
fig6, ax6 = plot_bias_heatmap(results_full)
check(isinstance(fig6, plt.Figure), "plot_bias_heatmap returns Figure")
fig6.savefig("models/phase6_bias_heatmap.png", dpi=80, bbox_inches="tight")
plt.close(fig6)
print("[OK] Bias heatmap rendered and saved to models/phase6_bias_heatmap.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  PHASE 6 TESTS: {len(errors)} FAILED")
    for e in errors:
        print(f"  -- {e}")
else:
    print("  ALL PHASE 6 TESTS PASSED [PASS]")
print("=" * 60)

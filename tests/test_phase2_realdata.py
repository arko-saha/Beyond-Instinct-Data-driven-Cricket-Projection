"""Real-data Phase 2 validation against the 100k artefacts."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from pathlib import Path

from src.forecast.skill_model import (
    build_player_skill_profiles,
    adjusted_wicket_prob,
    adjusted_run_expectation,
)

ARTEFACT_DIR = Path("models/phase1_artefacts")
df_xp = pd.read_parquet(ARTEFACT_DIR / "df_calib_xp.parquet")
n_batters = df_xp["batter"].nunique()
print(f"Loaded real XP data: {len(df_xp):,} deliveries, {n_batters} unique batters")

batter_skill, bowler_skill = build_player_skill_profiles(
    df_xp, min_balls_batter=120, min_balls_bowler=120
)

grand_xw = df_xp["xW_baseline"].mean()
grand_xr = df_xp["xR_batter_baseline"].mean()

# Validate 50x50 sample of real matchup combinations
sample_bat = batter_skill.head(50)
sample_bowl = bowler_skill.head(50)

all_xw, all_xr = [], []
for _, b in sample_bat.iterrows():
    for _, bw in sample_bowl.iterrows():
        p = adjusted_wicket_prob(grand_xw, bw["bowl_wkt_logodds"], b["bat_wkt_logodds"])
        r = adjusted_run_expectation(grand_xr, b["bat_run_logfactor"], bw["bowl_run_logfactor"])
        all_xw.append(p)
        all_xr.append(r)

arr_xw = np.array(all_xw)
arr_xr = np.array(all_xr)
assert (arr_xw > 0).all() and (arr_xw < 1).all()
assert (arr_xr > 0).all()

print(f"[PASS] adj_xw in (0,1) for {len(arr_xw):,} real combos  range=[{arr_xw.min():.4f}, {arr_xw.max():.4f}]")
print(f"[PASS] adj_xr > 0     for {len(arr_xr):,} real combos  range=[{arr_xr.min():.4f}, {arr_xr.max():.4f}]")

print()
print("--- Top 10 Bowlers (most positive bowl_wkt_logodds = best wicket-takers) ---")
cols_b = ["bowler", "balls_bowled", "bowl_actual_wkt_rate", "bowl_context_wkt_rate",
          "bowl_wkt_logodds", "bowl_run_logfactor"]
print(bowler_skill.sort_values("bowl_wkt_logodds", ascending=False)[cols_b].head(10).round(4).to_string())

print()
print("--- Top 10 Batters by Survival (most negative bat_wkt_logodds = hardest to dismiss) ---")
cols_bat = ["batter", "balls_faced", "bat_actual_wkt_rate", "bat_context_wkt_rate",
            "bat_wkt_logodds", "bat_run_logfactor"]
print(batter_skill.sort_values("bat_wkt_logodds", ascending=True)[cols_bat].head(10).round(4).to_string())

# Save Phase 2 artefacts
p2_dir = Path("models/phase2_artefacts")
p2_dir.mkdir(parents=True, exist_ok=True)
batter_skill.to_parquet(p2_dir / "batter_skill.parquet", index=False)
bowler_skill.to_parquet(p2_dir / "bowler_skill.parquet", index=False)

print()
print("[PASS] Phase 2 artefacts saved to models/phase2_artefacts/")
print("=" * 55)
print("  PHASE 2 REAL-DATA VALIDATION COMPLETE [PASS]")
print("=" * 55)

"""
forecast/simulator.py
======================
Phase 3 -- Monte Carlo Ball-by-Ball Simulation Engine.

Replaces the original deterministic fractional EV accumulator
(``predict_innings_trajectory``) with a fully stochastic simulation that
produces a probability DISTRIBUTION of outcomes rather than a single point
estimate.

## Core Design

### Vectorized Across Simulations
All N=10,000 simulations are evolved in parallel using NumPy array operations.
State is maintained as four arrays of shape (N,):
    sim_runs[N]       -- cumulative runs in each path
    sim_wickets[N]    -- cumulative wickets in each path
    sim_batter_idx[N] -- index into batting_lineup for each path
    sim_alive[N]      -- True if innings still in progress

Each ball, ALL N outcomes are drawn simultaneously with a single
``rng.random(N)`` call, making 10,000 simulations take ~50-100ms.

### Per-Ball Pipeline (vectorized)
For each ball position in [current_ball, 120):
    1. Determine current over and bowler.
    2. Look up contextual baseline (xW, xR_bat) from pre-built 3D tables
       indexed by [innings, over, wickets_in_hand]. All N paths get their
       own lookup value since wickets_in_hand may differ across paths.
    3. Apply Phase 2 matchup adjustments per active batter:
           adj_xw[i] = sigmoid(logit(base_xw[i]) + bowl_logodds + bat_logodds[i])
           adj_xr[i] = base_xr[i] * exp(bat_run_logfactor[i] + bowl_run_logfactor)
    4. Draw outcomes:
           wicket_mask = rng.random(N) < adj_xw
           run_outcomes sampled from empirical T20 scoring distribution,
           scaled to match adj_xr mean, then clipped to valid cricket values.
    5. Update state arrays.

### Scoring Distribution
The empirical scoring distribution P(k runs | no wicket) is built from the
calibration data per (innings, over) bucket. When no calibration data is
available, the default T20 distribution is used:
    {0: 0.35, 1: 0.29, 2: 0.12, 3: 0.03, 4: 0.12, 5: 0.01, 6: 0.08}

To incorporate Phase 2 run skill adjustment (adj_xr vs base_xr):
    scale = adj_xr / base_xr_of_scoring_dist
    The empirical CDF is scaled: higher-value outcomes become more likely
    when scale > 1 (good batter/poor bowler) and less likely when scale < 1.
    Implementation uses a simple mix between the empirical distribution and
    a boundary-weighted (aggressive) or dot-weighted (defensive) distribution.

### Bowling State Machine
Each over maps to a bowler via ``bowling_plan: Dict[int, str]``.
For overs not in the plan, the model falls back to a neutral (unknown)
bowler with zero skill adjustment.

### Batting Order State Machine
``batting_lineup: List[str]`` lists batters in order of expected appearance.
When a wicket falls, ``sim_batter_idx[wicket_mask] += 1``.
Paths that have exhausted the lineup (sim_batter_idx >= len(lineup)) use
the last named batter (tail-ender).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast.config import (
    MAX_LEGAL_BALLS_PER_INNINGS,
    MAX_WICKETS,
    MC_N_SIMULATIONS,
    MC_RANDOM_SEED,
)
from src.forecast.skill_model import logit, sigmoid


# ---------------------------------------------------------------------------
# Default T20 scoring distributions
# ---------------------------------------------------------------------------

# Default empirical T20 scoring distribution when no calibration data is
# available. Source: approximate T20i and IPL aggregates.
# keys are runs scored off the bat (0-6), values are probabilities.
_T20_DEFAULT_SCORING_DIST = np.array([0.35, 0.29, 0.12, 0.03, 0.12, 0.01, 0.08])  # shape (7,)
_T20_SCORING_OUTCOMES = np.array([0, 1, 2, 3, 4, 5, 6])

# Aggressive distribution (more boundaries) used as mix target when adj_xr > base
_T20_AGGRESSIVE_DIST = np.array([0.22, 0.24, 0.10, 0.02, 0.22, 0.01, 0.19])
# Defensive distribution (more dots) used as mix target when adj_xr < base
_T20_DEFENSIVE_DIST = np.array([0.55, 0.25, 0.09, 0.02, 0.06, 0.01, 0.02])


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize a probability array to sum to 1."""
    s = arr.sum()
    return arr / s if s > 0 else np.ones(len(arr)) / len(arr)


# ---------------------------------------------------------------------------
# Pre-built baseline table
# ---------------------------------------------------------------------------

def build_baseline_tables(
    lookups: Dict[str, pd.DataFrame],
    innings_values: Tuple[int, ...] = (1, 2),
) -> Dict[str, np.ndarray]:
    """
    Convert the hierarchical lookup tables into pre-indexed 3D numpy arrays
    for O(1) per-simulation-per-ball lookup during vectorized simulation.

    The arrays are indexed as:
        table[innings_idx, over, wickets_in_hand]
    where innings_idx = 0 for innings 1, 1 for innings 2.

    Uses the Level 2 lookup (innings × over × wickets_in_hand) as the primary
    source. Falls back to Level 3 (innings-level grand mean) for missing cells.

    Returns
    -------
    dict with keys:
        xw_table    : np.ndarray, shape (2, 20, 11), wicket probabilities
        xr_bat_table: np.ndarray, shape (2, 20, 11), expected runs off bat
        xr_bowl_table: np.ndarray, shape (2, 20, 11), expected runs conceded
    """
    n_innings, n_overs, n_wih = 2, 20, 11
    xw_table = np.zeros((n_innings, n_overs, n_wih))
    xr_bat_table = np.zeros((n_innings, n_overs, n_wih))
    xr_bowl_table = np.zeros((n_innings, n_overs, n_wih))

    l2 = lookups["level2"]
    l3 = lookups["level3"]

    for inn_val, inn_idx in [(1, 0), (2, 1)]:
        # Grand mean fallback values for this innings
        gm = l3[l3["innings"] == inn_val]
        if len(gm) > 0:
            gm_xw = float(gm["xW"].iloc[0])
            gm_xr_bat = float(gm["xR_batter"].iloc[0])
            gm_xr_bowl = float(gm["xR_bowler"].iloc[0])
        else:
            gm_xw, gm_xr_bat, gm_xr_bowl = 0.054, 1.20, 1.35

        for over in range(n_overs):
            for wih in range(n_wih):
                row = l2[
                    (l2["innings"] == inn_val)
                    & (l2["over"] == over)
                    & (l2["wickets_in_hand"] == wih)
                ]
                if len(row) > 0:
                    xw_table[inn_idx, over, wih] = float(row["xW"].iloc[0])
                    xr_bat_table[inn_idx, over, wih] = float(row["xR_batter"].iloc[0])
                    xr_bowl_table[inn_idx, over, wih] = float(row["xR_bowler"].iloc[0])
                else:
                    xw_table[inn_idx, over, wih] = gm_xw
                    xr_bat_table[inn_idx, over, wih] = gm_xr_bat
                    xr_bowl_table[inn_idx, over, wih] = gm_xr_bowl

    # Clip probabilities to valid range
    xw_table = np.clip(xw_table, 1e-6, 0.999)
    xr_bat_table = np.clip(xr_bat_table, 1e-4, None)

    return {
        "xw_table": xw_table,
        "xr_bat_table": xr_bat_table,
        "xr_bowl_table": xr_bowl_table,
    }


# ---------------------------------------------------------------------------
# Scoring distribution builder
# ---------------------------------------------------------------------------

def build_scoring_distribution(
    df_calib_state: pd.DataFrame,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Build an empirical per-ball scoring distribution from calibration data.

    Computes P(runs=k | no wicket, innings, over) for k in {0,1,2,3,4,5,6}
    from the calibration ball-by-ball data.

    Parameters
    ----------
    df_calib_state : pd.DataFrame
        Calibration data with columns: innings, over, is_wicket, runs_off_bat,
        wides, noballs.

    Returns
    -------
    dict mapping (innings, over) -> np.ndarray of shape (7,)
        Probability of scoring k runs (index = k) given no wicket on that ball.
        Missing contexts use the global T20 default distribution.
    """
    dist_lookup: Dict[Tuple[int, int], np.ndarray] = {}

    # Only non-wicket, legal deliveries count for scoring distribution
    df_scoring = df_calib_state[
        (df_calib_state["is_wicket"] == 0) & (df_calib_state["wides"] == 0)
    ].copy()

    df_scoring["runs_capped"] = df_scoring["runs_off_bat"].clip(0, 6)

    for (inn, ov), grp in df_scoring.groupby(["innings", "over"]):
        if len(grp) < 30:
            continue
        counts = grp["runs_capped"].value_counts().sort_index()
        probs = np.zeros(7)
        for k in range(7):
            probs[k] = counts.get(k, 0)
        probs = _normalize(probs)
        dist_lookup[(int(inn), int(ov))] = probs

    return dist_lookup


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for Monte Carlo innings simulation output.

    All arrays have shape (n_simulations,), representing independent
    stochastic paths of the innings from the given match state.
    """

    innings: int
    n_simulations: int
    starting_over: int
    starting_runs: int
    starting_wickets: int
    target: Optional[int]

    # Final state distributions (shape: n_simulations)
    final_scores: np.ndarray
    final_wickets: np.ndarray

    # Per-over snapshots: list of dicts (one per completed over)
    over_snapshots: List[dict] = field(default_factory=list)

    # -------------------------------------------------------------------
    # Derived summary properties
    # -------------------------------------------------------------------

    @property
    def score_median(self) -> float:
        return float(np.median(self.final_scores))

    @property
    def score_p10(self) -> float:
        return float(np.percentile(self.final_scores, 10))

    @property
    def score_p90(self) -> float:
        return float(np.percentile(self.final_scores, 90))

    @property
    def score_mean(self) -> float:
        return float(np.mean(self.final_scores))

    @property
    def wickets_median(self) -> float:
        return float(np.median(self.final_wickets))

    @property
    def win_probability(self) -> Optional[float]:
        """P(batting team wins) = P(final_score >= target). Innings 2 only."""
        if self.target is None:
            return None
        return float(np.mean(self.final_scores >= self.target))

    @property
    def p_collapse(self) -> float:
        """
        P(3 or more additional wickets in the remaining innings).
        Captures tail-end collapse risk.
        """
        additional_wickets = self.final_wickets - self.starting_wickets
        return float(np.mean(additional_wickets >= 3))

    def summary(self) -> Dict:
        """Return a clean summary dict for display."""
        s = {
            "score_p10": round(self.score_p10),
            "score_median": round(self.score_median),
            "score_p90": round(self.score_p90),
            "score_mean": round(self.score_mean, 1),
            "wickets_median": round(self.wickets_median, 1),
            "p_collapse": round(self.p_collapse, 3),
        }
        if self.target is not None:
            s["target"] = self.target
            s["win_probability"] = round(self.win_probability, 3)
        return s

    def over_snapshot_df(self) -> pd.DataFrame:
        """Return over-by-over snapshot as a readable DataFrame."""
        return pd.DataFrame(self.over_snapshots).round(2)


# ---------------------------------------------------------------------------
# Main simulation engine
# ---------------------------------------------------------------------------

class InningsSimulator:
    """
    Monte Carlo ball-by-ball innings simulator.

    Vectorizes all N simulations across numpy arrays, evolving them in
    parallel ball-by-ball. Phase 2 skill adjustments are applied per ball
    using pre-indexed lookup dictionaries (O(1) player lookups).

    Parameters
    ----------
    lookups : dict
        Phase 1 empirical lookup tables (output of ``build_empirical_lookups``).
    batter_skill : pd.DataFrame
        Phase 2 batter skill profiles (output of ``build_player_skill_profiles``).
    bowler_skill : pd.DataFrame
        Phase 2 bowler skill profiles.
    scoring_dist : dict, optional
        Per-(innings, over) empirical scoring distributions. If None, the
        global T20 default distribution is used for all contexts.
    """

    def __init__(
        self,
        lookups: Dict[str, pd.DataFrame],
        batter_skill: pd.DataFrame,
        bowler_skill: pd.DataFrame,
        scoring_dist: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    ):
        # Pre-build 3D numpy baseline tables for O(1) vectorized lookup
        self._tables = build_baseline_tables(lookups)

        # Convert player skill DataFrames to O(1) dicts
        self._batter_logodds: Dict[str, float] = dict(
            zip(batter_skill["batter"], batter_skill["bat_wkt_logodds"])
        )
        self._batter_runfac: Dict[str, float] = dict(
            zip(batter_skill["batter"], batter_skill["bat_run_logfactor"])
        )
        self._bowler_logodds: Dict[str, float] = dict(
            zip(bowler_skill["bowler"], bowler_skill["bowl_wkt_logodds"])
        )
        self._bowler_runfac: Dict[str, float] = dict(
            zip(bowler_skill["bowler"], bowler_skill["bowl_run_logfactor"])
        )

        # Scoring distributions
        self._scoring_dist = scoring_dist or {}

        # Pre-compute base xR for each context's scoring distribution
        # (used to compute the adj_xr scaling factor)
        self._base_xr_of_scoring_dist: Dict[Tuple[int, int], float] = {
            k: float(np.dot(v, _T20_SCORING_OUTCOMES)) for k, v in self._scoring_dist.items()
        }

        print(
            f"[Simulator] Ready. "
            f"{len(self._batter_logodds)} batters, "
            f"{len(self._bowler_logodds)} bowlers in skill registry."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batter_deltas(self, name: str) -> Tuple[float, float]:
        """Return (bat_wkt_logodds, bat_run_logfactor) for a batter."""
        return (
            self._batter_logodds.get(name, 0.0),
            self._batter_runfac.get(name, 0.0),
        )

    def _bowler_deltas(self, name: str) -> Tuple[float, float]:
        """Return (bowl_wkt_logodds, bowl_run_logfactor) for a bowler."""
        return (
            self._bowler_logodds.get(name, 0.0),
            self._bowler_runfac.get(name, 0.0),
        )

    def _sample_run_outcomes(
        self,
        n: int,
        inn: int,
        over: int,
        adj_xr_arr: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample run outcomes for N paths on a single delivery (conditional on
        no wicket occurring).

        Fully vectorized: uses a pre-computed 101-step CDF interpolation table
        (100 mix ratios from 0→1) to avoid any per-path Python loops.

        The empirical scoring distribution for (inn, over) is mixed linearly
        toward an aggressive (more boundaries) or defensive (more dots)
        distribution based on each path's adj_xr vs. the context base xR.

        Returns
        -------
        np.ndarray of int, shape (N,), values in {0, 1, 2, 3, 4, 5, 6}
        """
        base_dist = self._scoring_dist.get((inn, over), _T20_DEFAULT_SCORING_DIST)
        base_xr_of_dist = self._base_xr_of_scoring_dist.get(
            (inn, over), float(np.dot(_T20_DEFAULT_SCORING_DIST, _T20_SCORING_OUTCOMES))
        )

        # Per-path scale factor: ratio of skill-adjusted to empirical mean xR
        scale = np.clip(adj_xr_arr / max(base_xr_of_dist, 1e-4), 0.1, 5.0)

        # Map scale to a mix ratio in [0, 1] for two regimes:
        #   scale >= 1: mix toward aggressive distribution (ratio = (scale-1)/2 capped at 1)
        #   scale <  1: mix toward defensive distribution (ratio = 1-scale capped at 0.9)
        aggressive_mask = scale >= 1.0
        mix_ratio = np.where(
            aggressive_mask,
            np.clip((scale - 1.0) / 2.0, 0.0, 1.0),   # [0,1]: 0=base, 1=aggressive
            np.clip(1.0 - scale, 0.0, 0.9),              # [0,0.9]: 0=base, 0.9=defensive
        )

        # Quantize mix_ratio to 101 discrete steps (0.00, 0.01, ..., 1.00)
        # so we can look up pre-computed CDFs without any per-path loop
        MIX_STEPS = 101
        mix_idx = np.round(mix_ratio * (MIX_STEPS - 1)).astype(int)  # shape (N,)
        mix_steps_arr = np.linspace(0.0, 1.0, MIX_STEPS)              # (101,)

        # Build CDF tables: shape (101, 7)
        # Aggressive table: CDF of (1-t)*base + t*aggressive for t in linspace
        agg_cdf_table = np.array([
            np.cumsum(_normalize((1.0 - t) * base_dist + t * _T20_AGGRESSIVE_DIST))
            for t in mix_steps_arr
        ])  # shape (101, 7)

        # Defensive table: CDF of (1-t)*base + t*defensive for t in linspace
        def_cdf_table = np.array([
            np.cumsum(_normalize((1.0 - t) * base_dist + t * _T20_DEFENSIVE_DIST))
            for t in mix_steps_arr
        ])  # shape (101, 7)

        # Draw uniform random values for each path
        uniform_draws = rng.random(n)  # shape (N,)

        # Look up each path's CDF row using mix_idx
        # agg_cdfs[i] = CDF for path i if aggressive, def_cdfs[i] if defensive
        agg_cdfs = agg_cdf_table[mix_idx]   # shape (N, 7)
        def_cdfs = def_cdf_table[mix_idx]   # shape (N, 7)

        # Select correct CDF per path based on regime
        cdfs = np.where(aggressive_mask[:, None], agg_cdfs, def_cdfs)  # (N, 7)

        # Vectorized searchsorted: for each path, find the outcome index
        # such that CDF[outcome_idx-1] < u <= CDF[outcome_idx]
        # np.searchsorted operates on a 2D array row-by-row via a broadcast trick
        u_expanded = uniform_draws[:, None]  # (N, 1)
        outcome_indices = (cdfs < u_expanded).sum(axis=1)  # (N,)
        outcome_indices = np.clip(outcome_indices, 0, 6)

        return _T20_SCORING_OUTCOMES[outcome_indices]

    # ------------------------------------------------------------------
    # Public simulation interface
    # ------------------------------------------------------------------

    def simulate(
        self,
        innings: int,
        current_over: int,
        current_runs: int,
        current_wickets: int,
        batting_lineup: List[str],
        bowling_plan: Dict[int, str],
        target: Optional[int] = None,
        n_simulations: int = MC_N_SIMULATIONS,
        seed: Optional[int] = MC_RANDOM_SEED,
    ) -> SimulationResult:
        """
        Run N Monte Carlo simulations from a given mid-innings match state.

        Parameters
        ----------
        innings : int
            1 or 2. Determines which baseline lookup table to use.
        current_over : int
            Over number at the start of projection (0-indexed; 0 = start of innings).
            E.g., if 14 overs have been bowled, pass 14.
        current_runs : int
            Runs scored so far in this innings.
        current_wickets : int
            Wickets fallen so far.
        batting_lineup : list of str
            Remaining batters in order. First entry is the current on-strike
            batter. When a wicket falls, the next batter in this list comes in.
            Typically 10 - current_wickets entries long.
        bowling_plan : dict of int -> str
            Maps each over number to the bowler's name.
            E.g., {14: 'JJ Bumrah', 15: 'TA Boult', ...}
            Overs not in the plan use a neutral (unknown) bowler.
        target : int, optional
            Required for innings 2. The score the batting team needs to win.
        n_simulations : int
            Number of Monte Carlo paths. Default 10,000.
        seed : int or None
            Random seed for reproducibility. Set to None for fresh randomness.

        Returns
        -------
        SimulationResult
        """
        if current_wickets >= MAX_WICKETS:
            raise ValueError("Innings is already over (10 wickets fallen).")
        if current_over >= 20:
            raise ValueError("Innings is already over (20 overs complete).")

        rng = np.random.default_rng(seed)
        N = n_simulations
        inn_idx = innings - 1  # 0 for innings 1, 1 for innings 2

        # ----------------------------------------------------------------
        # Initialise state arrays (shape: N)
        # ----------------------------------------------------------------
        sim_runs = np.full(N, float(current_runs))
        sim_wickets = np.full(N, float(current_wickets))
        sim_batter_idx = np.zeros(N, dtype=int)  # index into batting_lineup
        sim_alive = np.ones(N, dtype=bool)  # still active

        # Build lookup arrays for batter skills (one entry per lineup position)
        n_batters_in_lineup = len(batting_lineup)
        lineup_bat_logodds = np.array([
            self._batter_deltas(b)[0] for b in batting_lineup
        ])
        lineup_bat_runfac = np.array([
            self._batter_deltas(b)[1] for b in batting_lineup
        ])

        # ----------------------------------------------------------------
        # Per-ball simulation loop
        # ----------------------------------------------------------------
        balls_total = MAX_LEGAL_BALLS_PER_INNINGS - current_over * 6
        over_snapshots = []
        current_over_ball_count = 0

        for ball_idx in range(balls_total):
            if not sim_alive.any():
                break  # all paths concluded

            over = current_over + ball_idx // 6
            if over >= 20:
                break

            # Bowler for this over
            bowler_name = bowling_plan.get(over, "Unknown")
            bowl_logodds, bowl_runfac = self._bowler_deltas(bowler_name)

            # Clamp batter index to valid range (last tail-ender for exhausted paths)
            safe_batter_idx = np.minimum(sim_batter_idx, n_batters_in_lineup - 1)

            # Per-simulation batter deltas via pre-built lineup arrays
            bat_logodds = lineup_bat_logodds[safe_batter_idx]   # shape (N,)
            bat_runfac = lineup_bat_runfac[safe_batter_idx]      # shape (N,)

            # Wickets in hand per simulation, clamp to [0,10]
            sim_wih = np.clip(
                MAX_WICKETS - sim_wickets.astype(int), 0, MAX_WICKETS
            )  # shape (N,)

            # Base xW and xR from pre-built 3D tables
            # Index: [inn_idx, over, wih]
            base_xw = self._tables["xw_table"][inn_idx, over, sim_wih]   # (N,)
            base_xr = self._tables["xr_bat_table"][inn_idx, over, sim_wih]  # (N,)

            # Phase 2 adjustments (fully vectorized)
            adj_xw = sigmoid(
                logit(base_xw) + bowl_logodds + bat_logodds
            )  # shape (N,), bounded in (0,1)

            adj_xr = np.clip(
                base_xr * np.exp(bat_runfac + bowl_runfac), 1e-4, 12.0
            )  # shape (N,)

            # ---- Outcome draw ----
            # Wicket events (Bernoulli per path)
            wicket_mask = (rng.random(N) < adj_xw) & sim_alive

            # Run outcomes for non-wicket, alive paths
            run_outcomes = self._sample_run_outcomes(N, innings, over, adj_xr, rng)
            run_outcomes = np.where(wicket_mask | ~sim_alive, 0, run_outcomes)

            # ---- State update ----
            sim_runs += run_outcomes.astype(float)
            sim_wickets += wicket_mask.astype(float)

            # Advance batter on wicket (but don't exceed lineup length)
            sim_batter_idx = np.where(
                wicket_mask,
                np.minimum(sim_batter_idx + 1, n_batters_in_lineup - 1),
                sim_batter_idx,
            )

            # Kill paths where all wickets have fallen
            sim_alive &= sim_wickets < MAX_WICKETS

            # ---- End-of-over snapshot ----
            current_over_ball_count += 1
            if current_over_ball_count == 6:
                current_over_ball_count = 0
                alive_runs = sim_runs[sim_alive] if sim_alive.any() else sim_runs
                snapshot = {
                    "over_completed": over,
                    "runs_p10": float(np.percentile(sim_runs, 10)),
                    "runs_median": float(np.median(sim_runs)),
                    "runs_p90": float(np.percentile(sim_runs, 90)),
                    "wickets_median": float(np.median(sim_wickets)),
                    "paths_alive_pct": float(sim_alive.mean() * 100),
                }
                if target is not None:
                    snapshot["p_ahead"] = float(np.mean(sim_runs >= target))
                over_snapshots.append(snapshot)

        return SimulationResult(
            innings=innings,
            n_simulations=N,
            starting_over=current_over,
            starting_runs=current_runs,
            starting_wickets=current_wickets,
            target=target,
            final_scores=sim_runs.copy(),
            final_wickets=sim_wickets.copy(),
            over_snapshots=over_snapshots,
        )

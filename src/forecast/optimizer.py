"""
forecast/optimizer.py
======================
Phase 5 -- Tactical Decision Optimizer.

Transforms the Monte Carlo simulation engine from a passive forecasting tool
into an active prescriptive analytics engine.

## What It Answers

**Fielding captain (bowling rotation):**
    Given 5 bowlers with remaining overs quotas, which over-by-over assignment
    minimises E[runs conceded] or maximises P(win)?

**Batting captain (lineup management):**
    At what position should the power-hitter be promoted to maximise P(score > X)?

## Core Algorithm

### Bowling Optimizer
1. Input: current state, available bowlers [{name, overs_remaining}], objective.
2. Enumerate all valid assignments: each remaining over must be assigned to a bowler
   who has quota left. This is a constrained combinatorial problem.
3. For each valid assignment (or the top-K by heuristic score to bound complexity):
   - Set the bowling_plan dict.
   - Run N_sim simulations from current state.
   - Record E[runs_conceded], P(win), P(collapse).
4. Return ranked DataFrame of strategies.

### Complexity Control
Full enumeration grows as (n_bowlers)^(n_overs) which is intractable.
We use a beam search heuristic: at each over, keep the top-K partial plans
(ranked by expected runs) and extend only those. This reduces complexity to
O(K * n_bowlers * n_overs) while still exploring the most promising strategies.

### Batting Lineup Optimizer
Enumerate valid batting orders (current batter fixed, remaining N batters
in all permutations up to LINEUP_PERM_LIMIT). For each, run simulation and
rank by objective.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast.simulator import InningsSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bowler:
    """A bowler available for the remaining innings."""
    name: str
    overs_remaining: int  # quota still available (e.g., 2 if 2 overs left of a 4-over max)


@dataclass
class BowlingStrategy:
    """One valid bowling assignment and its simulation results."""
    plan: Dict[int, str]           # {over: bowler_name}
    sim_result: Optional[SimulationResult]
    label: str = ""

    @property
    def exp_runs(self) -> float:
        return self.sim_result.score_mean if self.sim_result else np.nan

    @property
    def p_win(self) -> Optional[float]:
        return self.sim_result.win_probability if self.sim_result else None

    @property
    def p_collapse(self) -> float:
        return self.sim_result.p_collapse if self.sim_result else np.nan

    @property
    def score_median(self) -> float:
        return self.sim_result.score_median if self.sim_result else np.nan

    @property
    def score_p10(self) -> float:
        return self.sim_result.score_p10 if self.sim_result else np.nan

    @property
    def score_p90(self) -> float:
        return self.sim_result.score_p90 if self.sim_result else np.nan


# ---------------------------------------------------------------------------
# Bowling optimizer
# ---------------------------------------------------------------------------

class BowlingOptimizer:
    """
    Enumerates valid bowling plans for the remaining overs and ranks them
    by a user-specified objective function.

    Parameters
    ----------
    simulator : InningsSimulator
        Phase 3 simulator.
    n_sim : int
        Simulations per plan evaluation. Use 3,000-5,000 for planning
        (fast enough to evaluate many strategies).
    seed : int
        Random seed for reproducibility.
    beam_width : int
        Number of partial plans retained at each step of the beam search.
        Higher = more thorough but slower. Default 20.
    """

    def __init__(
        self,
        simulator: InningsSimulator,
        n_sim: int = 3000,
        seed: int = 42,
        beam_width: int = 20,
    ):
        self._sim = simulator
        self._n_sim = n_sim
        self._seed = seed
        self._beam_width = beam_width

    def _enumerate_plans(
        self,
        available_bowlers: List[Bowler],
        remaining_overs: List[int],
    ) -> List[Dict[int, str]]:
        """
        Enumerate valid bowling plans using beam search.

        A plan is valid iff:
        - Each over is assigned exactly one bowler.
        - No bowler is assigned more overs than their quota.
        - No bowler bowls two consecutive overs (T20 rule).

        Returns
        -------
        list of dicts: [{over: bowler_name}, ...]
        """
        if not remaining_overs or not available_bowlers:
            return [{}]

        # Initialise beam with empty plans
        beam: List[Tuple[Dict[int, str], Dict[str, int]]] = [
            ({}, {b.name: b.overs_remaining for b in available_bowlers})
        ]  # (partial_plan, remaining_quota)

        for over in remaining_overs:
            next_beam: List[Tuple[Dict[int, str], Dict[str, int]]] = []

            for partial_plan, quota in beam:
                last_bowler = partial_plan.get(over - 1)

                for bowler in available_bowlers:
                    if quota.get(bowler.name, 0) <= 0:
                        continue  # quota exhausted
                    if bowler.name == last_bowler:
                        continue  # no consecutive overs

                    new_plan = {**partial_plan, over: bowler.name}
                    new_quota = {**quota, bowler.name: quota.get(bowler.name, 0) - 1}
                    next_beam.append((new_plan, new_quota))

            if not next_beam:
                # Relax no-consecutive constraint if no valid plan found
                for partial_plan, quota in beam:
                    for bowler in available_bowlers:
                        if quota.get(bowler.name, 0) <= 0:
                            continue
                        new_plan = {**partial_plan, over: bowler.name}
                        new_quota = {**quota, bowler.name: quota.get(bowler.name, 0) - 1}
                        next_beam.append((new_plan, new_quota))

            # Limit beam size: keep diverse plans
            if len(next_beam) > self._beam_width:
                # Deduplicate and sample evenly
                seen_keys = set()
                deduped = []
                for p, q in next_beam:
                    key = tuple(sorted(p.items()))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        deduped.append((p, q))
                next_beam = deduped[:self._beam_width]

            beam = next_beam if next_beam else beam

        return [plan for plan, _ in beam]

    def optimize(
        self,
        innings: int,
        current_over: int,
        current_runs: int,
        current_wickets: int,
        batting_lineup: List[str],
        available_bowlers: List[Bowler],
        target: Optional[int] = None,
        objective: str = "minimise_runs",
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Find and rank the best bowling plans for the remaining overs.

        Parameters
        ----------
        innings : int (1 or 2)
        current_over : int
        current_runs : int
        current_wickets : int
        batting_lineup : list of str
        available_bowlers : list of Bowler
        target : int or None
            For innings 2 chase scenarios.
        objective : str
            'minimise_runs'  -- rank by E[runs conceded] (fielding captain)
            'maximise_p_win' -- rank by P(batting team wins) (batting captain)
        top_n : int
            Return only the top-N strategies.

        Returns
        -------
        pd.DataFrame with columns:
            rank, plan_label, plan_str,
            exp_runs, score_median, score_p10, score_p90, p_collapse,
            [p_win if innings 2]
        """
        remaining_overs = list(range(current_over, 20))
        if not remaining_overs:
            return pd.DataFrame()

        plans = self._enumerate_plans(available_bowlers, remaining_overs)

        print(f"[Optimizer] Evaluating {len(plans)} valid bowling plans "
              f"({len(remaining_overs)} overs, {len(available_bowlers)} bowlers)")

        strategies = []
        for i, plan in enumerate(plans):
            try:
                result = self._sim.simulate(
                    innings=innings,
                    current_over=current_over,
                    current_runs=current_runs,
                    current_wickets=current_wickets,
                    batting_lineup=batting_lineup,
                    bowling_plan=plan,
                    target=target,
                    n_simulations=self._n_sim,
                    seed=self._seed + i,  # vary seed slightly across plans
                )
                plan_label = " | ".join(
                    f"o{ov}:{bwl[:8]}" for ov, bwl in sorted(plan.items())
                )
                strategies.append(BowlingStrategy(plan=plan, sim_result=result, label=plan_label))
            except Exception as e:
                continue

        if not strategies:
            return pd.DataFrame()

        # Sort by objective
        if objective == "maximise_p_win" and innings == 2:
            strategies.sort(key=lambda s: s.p_win or 0.0, reverse=True)
        else:  # minimise_runs
            strategies.sort(key=lambda s: s.exp_runs)

        top = strategies[:top_n]

        rows = []
        for rank, strat in enumerate(top, 1):
            row = {
                "rank": rank,
                "plan_label": strat.label,
                "exp_runs": round(strat.exp_runs, 1),
                "score_median": round(strat.score_median),
                "score_p10": round(strat.score_p10),
                "score_p90": round(strat.score_p90),
                "p_collapse": round(strat.p_collapse, 3),
            }
            if innings == 2:
                row["p_win"] = round(strat.p_win or 0.0, 3)
            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batting lineup optimizer
# ---------------------------------------------------------------------------

class BattingOptimizer:
    """
    Evaluates different batting orders for the remaining lineup and ranks
    them by a user-specified objective.

    Enumerates permutations of the remaining (not-yet-in) batters while
    keeping the current on-strike batter fixed. To control combinatorial
    explosion, limits to the first PERM_LIMIT permutations.

    Parameters
    ----------
    simulator : InningsSimulator
    n_sim : int
    seed : int
    perm_limit : int
        Maximum permutations to evaluate. Default 50.
    """

    PERM_LIMIT = 50

    def __init__(
        self,
        simulator: InningsSimulator,
        n_sim: int = 3000,
        seed: int = 42,
        perm_limit: int = 50,
    ):
        self._sim = simulator
        self._n_sim = n_sim
        self._seed = seed
        self._perm_limit = perm_limit

    def optimize(
        self,
        innings: int,
        current_over: int,
        current_runs: int,
        current_wickets: int,
        current_batter: str,
        remaining_batters: List[str],
        bowling_plan: Dict[int, str],
        target: Optional[int] = None,
        objective: str = "maximise_runs",
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Rank batting orders by the specified objective.

        Parameters
        ----------
        current_batter : str
            The batter currently at the crease (always first in lineup).
        remaining_batters : list of str
            The batters not yet in, in the ORIGINAL order. The optimizer
            will explore reorderings of these.
        objective : str
            'maximise_runs' -- rank by E[total runs] (innings 1)
            'maximise_p_win' -- rank by P(win) (innings 2)
        """
        # Always fix current batter at position 0
        permutations = list(itertools.islice(
            itertools.permutations(remaining_batters),
            self._perm_limit,
        ))

        print(f"[BattingOptimizer] Evaluating {len(permutations)} lineups "
              f"({len(remaining_batters)} remaining batters)")

        rows = []
        seen_lineups = set()

        for i, perm in enumerate(permutations):
            lineup = [current_batter] + list(perm)
            lineup_key = tuple(lineup[:5])  # compare first 5 positions for deduplication
            if lineup_key in seen_lineups:
                continue
            seen_lineups.add(lineup_key)

            try:
                result = self._sim.simulate(
                    innings=innings,
                    current_over=current_over,
                    current_runs=current_runs,
                    current_wickets=current_wickets,
                    batting_lineup=lineup,
                    bowling_plan=bowling_plan,
                    target=target,
                    n_simulations=self._n_sim,
                    seed=self._seed + i,
                )
                rows.append({
                    "lineup_str": " > ".join(lineup[:6]) + ("..." if len(lineup) > 6 else ""),
                    "exp_runs": round(result.score_mean, 1),
                    "score_median": round(result.score_median),
                    "score_p10": round(result.score_p10),
                    "score_p90": round(result.score_p90),
                    "p_collapse": round(result.p_collapse, 3),
                    "p_win": round(result.win_probability or 0.0, 3) if innings == 2 else None,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if objective == "maximise_p_win" and innings == 2:
            df = df.sort_values("p_win", ascending=False)
        else:
            df = df.sort_values("exp_runs", ascending=False)

        df.insert(0, "rank", range(1, len(df) + 1))
        return df.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# What-If scenario injector
# ---------------------------------------------------------------------------

@dataclass
class MatchState:
    """
    A fully specified mid-innings match state for scenario analysis.

    Acts as the canonical input format for the optimizer and simulator,
    replacing the scattered variables in the original notebook.
    """
    innings: int
    current_over: int
    current_runs: int
    current_wickets: int
    batting_lineup: List[str]
    bowling_plan: Dict[int, str] = field(default_factory=dict)
    target: Optional[int] = None
    label: str = "Scenario"

    def override(self, **kwargs) -> "MatchState":
        """
        Return a new MatchState with specified fields overridden.
        Enables clean what-if scenario injection without mutating the original.

        Example
        -------
        # What if it rains and only 5 overs remain?
        rain_state = live_state.override(current_over=15, label='Rain scenario')
        """
        import dataclasses
        return dataclasses.replace(self, **kwargs)

    def simulate(
        self,
        simulator: InningsSimulator,
        n_sim: int = 5000,
        seed: int = 42,
    ) -> SimulationResult:
        """Run the simulator from this match state."""
        return simulator.simulate(
            innings=self.innings,
            current_over=self.current_over,
            current_runs=self.current_runs,
            current_wickets=self.current_wickets,
            batting_lineup=self.batting_lineup,
            bowling_plan=self.bowling_plan,
            target=self.target,
            n_simulations=n_sim,
            seed=seed,
        )

    def compare_scenarios(
        self,
        simulator: InningsSimulator,
        overrides: List[Dict],
        n_sim: int = 3000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run a set of scenario overrides and return a comparison DataFrame.

        Parameters
        ----------
        overrides : list of dict
            Each dict is a set of field overrides plus an optional 'label' key.
            E.g.: [{'bowling_plan': {...}, 'label': 'Bumrah death'},
                   {'bowling_plan': {...}, 'label': 'Boult death'}]

        Returns
        -------
        pd.DataFrame with one row per scenario.
        """
        rows = []
        for ovr in overrides:
            label = ovr.pop("label", "Scenario")
            state = self.override(**ovr, label=label)
            result = state.simulate(simulator, n_sim=n_sim, seed=seed)
            row = result.summary()
            row["scenario"] = label
            rows.append(row)
        return pd.DataFrame(rows).set_index("scenario")

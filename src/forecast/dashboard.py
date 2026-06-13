"""
forecast/dashboard.py
======================
Phase 6 -- Decision-Ready Visualisation Dashboard.

All charts are self-contained functions that accept simulation/backtest results
and return matplotlib Figure objects. They are designed to be called from the
notebook OR from a standalone script, not hardcoded to specific cell data.

## Charts Available

1. `plot_score_fan_chart(result, title)` -- trajectory fan chart with P10/P50/P90 bands
2. `plot_win_probability_timeline(results_by_over, target)` -- rolling P(win) by over
3. `plot_strategy_comparison(sim_results, target)` -- grouped bar chart for Phase 5 strategies
4. `plot_player_xp_leaderboard(batter_skill, bowler_skill, top_n)` -- XP leaderboard
5. `plot_backtest_dashboard(backtest_df, cal_df)` -- 6-panel calibration summary
6. `plot_bias_heatmap(backtest_df)` -- signed error heatmap by over x wickets

All functions return (fig, axes) for further customisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":   "#1f77b4",   # Steel blue
    "secondary": "#ff7f0e",   # Orange
    "success":   "#2ca02c",   # Green
    "danger":    "#d62728",   # Red
    "neutral":   "#7f7f7f",   # Grey
    "accent":    "#9467bd",   # Purple
    "band":      "#aec7e8",   # Light blue (confidence band fill)
    "target":    "#d62728",   # Red dashed (target line)
}


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "",
              grid_axis: str = "y") -> None:
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, axis=grid_axis, linestyle="--")
    ax.tick_params(axis="both", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Score fan chart
# ---------------------------------------------------------------------------

def plot_score_fan_chart(
    sim_result,
    title: str = "Innings Score Projection",
    target: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot over-by-over trajectory with P10/P50/P90 bands.

    Parameters
    ----------
    sim_result : SimulationResult
        Output of ``InningsSimulator.simulate()``.
    title : str
    target : int or None
        Chase target for innings 2.
    ax : optional matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    snaps = sim_result.over_snapshots
    if not snaps:
        ax.text(0.5, 0.5, "No over snapshots available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    overs = [s["over_completed"] + 1 for s in snaps]
    p10s  = [s["runs_p10"]    for s in snaps]
    meds  = [s["runs_median"] for s in snaps]
    p90s  = [s["runs_p90"]    for s in snaps]

    ax.fill_between(overs, p10s, p90s, alpha=0.20, color=PALETTE["primary"], label="P10–P90 band")
    ax.plot(overs, meds, color=PALETTE["primary"], lw=2.5, marker="o", ms=4, label="Median projection")
    ax.plot(overs, p10s, color=PALETTE["primary"], lw=1, ls="--", alpha=0.5)
    ax.plot(overs, p90s, color=PALETTE["primary"], lw=1, ls="--", alpha=0.5)

    # Starting state dot
    ax.scatter([sim_result.starting_over], [sim_result.starting_runs],
               color=PALETTE["danger"], zorder=5, s=70, label="Freeze point")

    if target is not None:
        ax.axhline(target, color=PALETTE["target"], ls="--", lw=1.5,
                   label=f"Target: {target}", zorder=3)

    # Annotation: final summary
    summary = sim_result.summary()
    ann_text = (
        f"P10: {summary['score_p10']}  "
        f"Med: {summary['score_median']}  "
        f"P90: {summary['score_p90']}"
    )
    if "win_probability" in summary:
        ann_text += f"\nP(win): {summary['win_probability']:.1%}"
    ax.annotate(ann_text, xy=(overs[-1], meds[-1]),
                xytext=(overs[-1] - 1, meds[-1] + 8),
                fontsize=8, color=PALETTE["primary"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["primary"], lw=0.8))

    ax.set_xlim(sim_result.starting_over, 20)
    ax.set_ylim(max(0, min(p10s) - 10), max(p90s) + 15)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    _style_ax(ax, title=title, xlabel="Overs completed", ylabel="Projected runs", grid_axis="y")

    return fig, ax


# ---------------------------------------------------------------------------
# 2. Rolling win probability
# ---------------------------------------------------------------------------

def plot_win_probability_timeline(
    p_win_by_over: Dict[int, float],
    target: int,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple = (10, 4),
    title: str = "Rolling Win Probability",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot P(win) as a rolling line chart, updated over-by-over.

    Parameters
    ----------
    p_win_by_over : dict {over_completed: P(win)}
        Usually built by calling simulator.simulate() at each over and
        recording win_probability.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    overs = sorted(p_win_by_over.keys())
    probs = [p_win_by_over[o] for o in overs]

    # Colour gradient: green (high P win) to red (low P win)
    for i in range(len(overs) - 1):
        p = probs[i]
        color = plt.cm.RdYlGn(p)
        ax.plot([overs[i], overs[i+1]], [probs[i], probs[i+1]], color=color, lw=3)

    ax.scatter(overs, probs, c=probs, cmap="RdYlGn", s=40, zorder=5, vmin=0, vmax=1)
    ax.axhline(0.5, color=PALETTE["neutral"], ls=":", lw=1.5, label="Even odds")
    ax.fill_between(overs, probs, 0.5,
                    where=[p > 0.5 for p in probs], interpolate=True,
                    color=PALETTE["success"], alpha=0.10)
    ax.fill_between(overs, probs, 0.5,
                    where=[p < 0.5 for p in probs], interpolate=True,
                    color=PALETTE["danger"], alpha=0.10)

    ax.set_xlim(overs[0] - 0.5, 20.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=8, framealpha=0.85)
    _style_ax(ax, title=title, xlabel="Overs completed", ylabel="P(batting team wins)", grid_axis="y")

    return fig, ax


# ---------------------------------------------------------------------------
# 3. Strategy comparison chart
# ---------------------------------------------------------------------------

def plot_strategy_comparison(
    sim_results: List[Tuple[str, object]],
    target: Optional[int] = None,
    ax_metrics: Optional[plt.Axes] = None,
    ax_dist: Optional[plt.Axes] = None,
    figsize: Tuple = (14, 5),
    title: str = "Strategy Comparison",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Side-by-side strategy comparison: P(win) bars + score distribution.

    Parameters
    ----------
    sim_results : list of (label, SimulationResult)
    target : int, optional
    """
    if ax_metrics is None or ax_dist is None:
        fig, (ax_metrics, ax_dist) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=13, fontweight="bold")
    else:
        fig = ax_metrics.get_figure()

    n = len(sim_results)
    x_pos = np.arange(n)
    labels = [label for label, _ in sim_results]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
              PALETTE["accent"], PALETTE["neutral"]][:n]

    # ---- Metrics bars ----
    if any(r.win_probability is not None for _, r in sim_results):
        metric_vals = [r.win_probability or 0.0 for _, r in sim_results]
        metric_label = "P(win)"
    else:
        metric_vals = [r.score_mean for _, r in sim_results]
        metric_label = "E[runs]"

    bars = ax_metrics.bar(x_pos, metric_vals, color=colors, edgecolor="white",
                          width=0.55, linewidth=0.5)
    if target is not None and metric_label == "P(win)":
        ax_metrics.axhline(0.5, color="black", ls="--", lw=1, alpha=0.5, label="50%")
    ax_metrics.set_xticks(x_pos)
    ax_metrics.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    if metric_label == "P(win)":
        ax_metrics.set_ylim(0, 1)
        ax_metrics.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bar, v in zip(bars, metric_vals):
        label_str = f"{v:.1%}" if metric_label == "P(win)" else f"{v:.0f}"
        ax_metrics.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        label_str, ha="center", va="bottom", fontsize=8, fontweight="bold")
    _style_ax(ax_metrics, title=metric_label + " by Strategy",
              ylabel=metric_label, grid_axis="y")

    # ---- Distribution dots (P10/med/P90) ----
    for i, ((label, r), color) in enumerate(zip(sim_results, colors)):
        ax_dist.plot([i-0.15, i+0.15], [r.score_median]*2, "-", color=color, lw=3.5)
        ax_dist.plot([i, i], [r.score_p10, r.score_p90], "|", color=color, ms=16, mew=2)
        ax_dist.fill_between([i-0.12, i+0.12], [r.score_p10]*2, [r.score_p90]*2,
                             color=color, alpha=0.20)
        ax_dist.text(i, r.score_p90 + 1.5, f"P90={r.score_p90:.0f}",
                     ha="center", fontsize=7, color=color)
        ax_dist.text(i, r.score_p10 - 3.5, f"P10={r.score_p10:.0f}",
                     ha="center", fontsize=7, color=color)

    if target is not None:
        ax_dist.axhline(target, color=PALETTE["target"], ls="--", lw=2,
                        label=f"Target: {target}", zorder=3)
        ax_dist.legend(fontsize=9, framealpha=0.85)

    ax_dist.set_xticks(x_pos)
    ax_dist.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    _style_ax(ax_dist, title="Score Distribution (P10 | Median | P90)",
              ylabel="Projected score", grid_axis="y")

    plt.tight_layout()
    return fig, (ax_metrics, ax_dist)


# ---------------------------------------------------------------------------
# 4. Player XP Leaderboard
# ---------------------------------------------------------------------------

def plot_player_xp_leaderboard(
    batter_skill: pd.DataFrame,
    bowler_skill: pd.DataFrame,
    top_n: int = 10,
    figsize: Tuple = (14, 6),
    title: str = "Player Skill Leaderboard (Phase 2 Log-Odds XP)",
) -> Tuple[plt.Figure, Tuple]:
    """
    Horizontal bar chart leaderboard: top/bottom batters and bowlers by XP.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # ---- Batter leaderboard ----
    ax1 = axes[0]
    top_bat = batter_skill.nlargest(top_n, "bat_wkt_logodds")[
        ["batter", "bat_wkt_logodds", "bat_run_logfactor"]
    ].iloc[::-1]
    worst_bat = batter_skill.nsmallest(top_n, "bat_wkt_logodds")[
        ["batter", "bat_wkt_logodds", "bat_run_logfactor"]
    ]
    bat_display = pd.concat([top_bat, worst_bat]).drop_duplicates()

    colors_bat = [PALETTE["danger"] if v > 0 else PALETTE["success"]
                  for v in bat_display["bat_wkt_logodds"]]
    bars = ax1.barh(bat_display["batter"], bat_display["bat_wkt_logodds"],
                    color=colors_bat, edgecolor="white", linewidth=0.3)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("Wicket XP (log-odds delta)", fontsize=9)
    _style_ax(ax1, title=f"Batter Wicket Survival XP\n(+ve = lower wicket risk)",
              grid_axis="x")
    ax1.invert_xaxis()
    ax1.annotate("Better survivor -->", xy=(0.02, 0.98), xycoords="axes fraction",
                 ha="left", va="top", fontsize=7, color=PALETTE["success"])
    ax1.annotate("<-- Higher risk", xy=(0.98, 0.98), xycoords="axes fraction",
                 ha="right", va="top", fontsize=7, color=PALETTE["danger"])

    # ---- Bowler leaderboard ----
    ax2 = axes[1]
    top_bowl = bowler_skill.nlargest(top_n, "bowl_wkt_logodds")[
        ["bowler", "bowl_wkt_logodds", "bowl_run_logfactor"]
    ].iloc[::-1]
    worst_bowl = bowler_skill.nsmallest(top_n, "bowl_wkt_logodds")[
        ["bowler", "bowl_wkt_logodds", "bowl_run_logfactor"]
    ]
    bowl_display = pd.concat([top_bowl, worst_bowl]).drop_duplicates()

    colors_bowl = [PALETTE["success"] if v > 0 else PALETTE["danger"]
                   for v in bowl_display["bowl_wkt_logodds"]]
    ax2.barh(bowl_display["bowler"], bowl_display["bowl_wkt_logodds"],
             color=colors_bowl, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("Wicket XP (log-odds delta)", fontsize=9)
    _style_ax(ax2, title=f"Bowler Wicket Taking XP\n(+ve = takes more wickets)",
              grid_axis="x")
    ax2.annotate("<-- Less effective", xy=(0.02, 0.98), xycoords="axes fraction",
                 ha="left", va="top", fontsize=7, color=PALETTE["danger"])
    ax2.annotate("More wickets -->", xy=(0.98, 0.98), xycoords="axes fraction",
                 ha="right", va="top", fontsize=7, color=PALETTE["success"])

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Backtest dashboard
# ---------------------------------------------------------------------------

def plot_backtest_dashboard(
    backtest_df: pd.DataFrame,
    cal_df: Optional[pd.DataFrame] = None,
    baseline_df: Optional[pd.DataFrame] = None,
    figsize: Tuple = (16, 10),
    title: str = "Phase 4 -- Backtest Calibration Dashboard",
) -> plt.Figure:
    """
    6-panel backtest summary dashboard.
    """
    from src.forecast.calibration import (
        compute_conformalized_scale, apply_conformalized_intervals, bias_decomposition
    )
    from src.forecast.backtester import compute_metrics

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    m = compute_metrics(backtest_df)

    # 1. Predicted vs Actual scatter
    ax1 = fig.add_subplot(gs[0, 0])
    colors_pt = backtest_df["in_band"].map({True: PALETTE["primary"], False: PALETTE["danger"]})
    ax1.scatter(backtest_df["pred_median"], backtest_df["actual_score"],
                c=colors_pt, alpha=0.40, s=15, edgecolors="none")
    lims = [min(backtest_df["pred_median"].min(), backtest_df["actual_score"].min()) - 5,
            max(backtest_df["pred_median"].max(), backtest_df["actual_score"].max()) + 5]
    ax1.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.legend(fontsize=7, framealpha=0.85)
    _style_ax(ax1, title=f"Predicted vs Actual\n(Coverage: {m['coverage_rate']:.1%})",
              xlabel="Median pred", ylabel="Actual score", grid_axis="both")

    # 2. Signed error histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(backtest_df["signed_error"], bins=30, color=PALETTE["primary"],
             alpha=0.75, edgecolor="white", lw=0.5)
    ax2.axvline(0, color="black", lw=1.5, ls="--", label="No bias")
    mean_err = backtest_df["signed_error"].mean()
    ax2.axvline(mean_err, color=PALETTE["danger"], lw=1.5,
                label=f"Bias: {mean_err:+.1f}")
    ax2.legend(fontsize=7, framealpha=0.85)
    _style_ax(ax2, title="Error Distribution\n(+ve = over-predicted)",
              xlabel="pred - actual (runs)", ylabel="Count", grid_axis="y")

    # 3. Coverage by score quintile
    ax3 = fig.add_subplot(gs[0, 2])
    backtest_df["score_q"] = pd.qcut(backtest_df["runs_at_freeze"], q=5,
                                      labels=["VLow","Low","Med","High","VHigh"],
                                      duplicates="drop")
    cov_q = backtest_df.groupby("score_q", observed=True)["in_band"].mean()
    bar_colors = [PALETTE["danger"] if v < 0.6 else PALETTE["primary"] for v in cov_q.values]
    ax3.bar(range(len(cov_q)), cov_q.values, color=bar_colors, edgecolor="white")
    ax3.axhline(0.80, color=PALETTE["success"], ls="--", lw=1.5, label="80% target")
    ax3.set_xticks(range(len(cov_q)))
    ax3.set_xticklabels(cov_q.index.tolist(), rotation=20, ha="right", fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=7, framealpha=0.85)
    _style_ax(ax3, title="Coverage by Score at Freeze\n(Higher = better calibrated)",
              ylabel="Coverage rate", grid_axis="y")

    # 4. Win probability calibration
    ax4 = fig.add_subplot(gs[1, 0])
    if cal_df is not None and len(cal_df) > 0:
        ax4.plot(cal_df["pred_p_win_mean"], cal_df["actual_win_rate"],
                 "o-", color=PALETTE["primary"], lw=2, ms=6, label="Full model")
    if baseline_df is not None and len(baseline_df) > 0:
        from src.forecast.backtester import calibration_curve as _cc
        cal_base = _cc(baseline_df)
        if len(cal_base) > 0:
            ax4.plot(cal_base["pred_p_win_mean"], cal_base["actual_win_rate"],
                     "s--", color=PALETTE["danger"], lw=1.5, ms=5, alpha=0.7, label="Baseline")
    ax4.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.legend(fontsize=7, framealpha=0.85)
    _style_ax(ax4, title="Win Prob Calibration\n(Innings 2)",
              xlabel="Predicted P(win)", ylabel="Actual win rate", grid_axis="both")

    # 5. Conformalized coverage improvement
    ax5 = fig.add_subplot(gs[1, 1])
    n2 = len(backtest_df)
    calib_h = backtest_df.iloc[:n2//2].copy()
    hold_h  = backtest_df.iloc[n2//2:].copy()
    alpha = compute_conformalized_scale(calib_h, target_coverage=0.80)
    hold_recal = apply_conformalized_intervals(hold_h, alpha)
    orig_cov = backtest_df["in_band"].mean()
    cal_cov  = hold_recal["cal_in_band"].mean()

    model_labels = ["Full (raw)", f"Full (recal,\nalpha={alpha:.2f})", "Target"]
    if baseline_df is not None:
        m_base = compute_metrics(baseline_df)
        model_labels.insert(2, "Baseline")
        vals = [orig_cov, cal_cov, m_base["coverage_rate"], 0.80]
        bar_cols = [PALETTE["primary"], PALETTE["success"], PALETTE["secondary"], PALETTE["neutral"]]
    else:
        vals = [orig_cov, cal_cov, 0.80]
        bar_cols = [PALETTE["primary"], PALETTE["success"], PALETTE["neutral"]]

    bars5 = ax5.barh(model_labels, vals, color=bar_cols, edgecolor="white")
    ax5.axvline(0.80, color="black", ls="--", lw=1.5)
    ax5.set_xlim(0, 1)
    for bar, v in zip(bars5, vals):
        ax5.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.1%}", va="center", fontsize=8)
    _style_ax(ax5, title="Coverage Rate Comparison\n(Target: 80%)",
              xlabel="Coverage rate", grid_axis="x")

    # 6. Interval width distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(backtest_df["interval_width"], bins=25, color=PALETTE["primary"],
             alpha=0.75, edgecolor="white", lw=0.5, label="Full model")
    if baseline_df is not None:
        ax6.hist(baseline_df["interval_width"], bins=25, color=PALETTE["danger"],
                 alpha=0.45, edgecolor="white", lw=0.5, label="Baseline")
    med_w = backtest_df["interval_width"].median()
    ax6.axvline(med_w, color=PALETTE["primary"], lw=2, label=f"Median: {med_w:.0f}")
    ax6.legend(fontsize=7, framealpha=0.85)
    _style_ax(ax6, title=f"Interval Width\n(Sharpness, MAE={m['mae']:.1f} runs)",
              xlabel="P10-P90 width (runs)", ylabel="Count", grid_axis="y")

    return fig


# ---------------------------------------------------------------------------
# 6. Bias heatmap
# ---------------------------------------------------------------------------

def plot_bias_heatmap(
    backtest_df: pd.DataFrame,
    figsize: Tuple = (10, 5),
    title: str = "Signed Error Heatmap (Over x Wickets at Freeze)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Show mean signed error (bias) as a heatmap by freeze-over and wickets.
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = backtest_df.copy()
    df["over_bucket"] = pd.cut(df["freeze_over"], bins=[0,8,12,16,20],
                                labels=["0-7","8-11","12-15","16-19"])
    df["wkt_bucket"] = pd.cut(df["wickets_at_freeze"], bins=[-1,2,5,10],
                               labels=["0-2","3-5","6-9"])

    pivot = df.pivot_table(
        values="signed_error", index="wkt_bucket", columns="over_bucket",
        aggfunc="mean", observed=True
    )

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=-20, vmax=20, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xlabel("Freeze over bucket", fontsize=10)
    ax.set_ylabel("Wickets at freeze", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                        fontsize=9, color="black",
                        fontweight="bold" if abs(val) > 10 else "normal")

    plt.colorbar(im, ax=ax, label="Mean signed error (runs, +ve = over-predicted)")
    plt.tight_layout()
    return fig, ax

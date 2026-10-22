"""Evaluation figures (PNG into reports/figures). Functions return figures so the notebook can show them inline."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIGURES, PROCESSED, RESULTS

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
COL = {"naive prior": "#9a998f", "logistic regression": "#6250d6", "GBM": "#2a78d6",
       "GBM + Platt": "#eb6834", "GBM + isotonic": "#1baf7a", "GBM (monotone)": "#e87ba4",
       "logistic regression + Platt": "#008300"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titlelocation": "left"})


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.7, ls=(0, (2, 3)))
    ax.set_axisbelow(True)


def fig_reliability(models=("logistic regression", "GBM", "GBM + Platt"), n_bins=10):
    rel = pd.read_csv(RESULTS / "test_reliability.csv")
    info = pd.read_csv(RESULTS / "test_metrics_full.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150, sharex=True, sharey=True)
    for ax, (inn, title) in zip(axes, [(0, "All balls"), (1, "Innings 1 (batting first)"), (2, "Innings 2 (chasing)")]):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
        for m in models:
            t = rel[(rel.model == m) & (rel.innings == inn)]
            ax.plot(t.mean_pred, t.frac_pos, "-o", color=COL[m], lw=1.8, ms=4, label=m, zorder=3)
        grp = "overall" if inn == 0 else f"innings {inn}"
        ece = {m: info[(info.model == m) & (info.group == grp)].ece.iat[0] for m in models}
        ax.set_title(title)
        ax.text(0.02, 0.97, "\n".join(f"ECE {ece[m]:.3f}  {m}" for m in models), transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.set_xlabel("predicted win probability")
        _style(ax)
    axes[0].set_ylabel("observed win frequency")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Reliability diagrams on the held-out 2025-26 seasons (closer to the diagonal = better calibrated)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_by_ball(models=("naive prior", "logistic regression", "GBM", "GBM + Platt")):
    t = pd.read_csv(RESULTS / "test_by_ball_bucket.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150, sharey=True)
    for ax, inn in zip(axes, (1, 2)):
        for m in models:
            s = t[(t.innings == inn) & (t.model == m)]
            ax.plot(s.balls_bowled_from + 10, s.log_loss, "-o", ms=3, lw=1.8, color=COL[m], label=m)
        ax.set_title(f"Innings {inn}")
        ax.set_xlabel("legal balls bowled (20-ball buckets)")
        ax.set_xticks(range(10, 130, 20))
        _style(ax)
    axes[0].set_ylabel("log loss (lower is better)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Where does the model add information? Log loss by stage of the innings, test seasons", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_phase(models=("naive prior", "logistic regression", "GBM", "GBM + Platt")):
    t = pd.read_csv(RESULTS / "test_metrics_full.csv")
    groups = ["inn1 powerplay", "inn1 middle", "inn1 death", "inn2 powerplay", "inn2 middle", "inn2 death"]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [t[(t.model == m) & (t.group == g)].log_loss.iat[0] for g in groups]
        ax.bar(np.arange(len(groups)) + i * w - 0.4 + w / 2, v, width=w - 0.02, color=COL[m], label=m)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("inn1", "Inn 1").replace("inn2", "Inn 2") for g in groups])
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss by innings and phase, test seasons")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def fig_era():
    m = pd.read_parquet(PROCESSED / "matches.parquet")
    m = m[m.included]
    g = m.groupby("season").agg(first=("score1", "mean"), chase=("chase_won", "mean"), n=("match_id", "size"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=150)
    for ax in axes:
        ax.axvspan(2022.5, 2026.5, color="#eda100", alpha=0.10, lw=0)
        ax.axvline(2023 - 0.5, color=MUTED, lw=0.8, ls="--")
        _style(ax)
    axes[0].plot(g.index, g["first"], "-o", color="#2a78d6", ms=4, lw=2)
    axes[0].set_title("Average first-innings score")
    axes[0].text(2023, g["first"].min() + 1, "Impact Player era", color=MUTED, fontsize=9)
    axes[1].plot(g.index, g.chase * 100, "-o", color="#eb6834", ms=4, lw=2)
    axes[1].axhline(50, color=GRID, lw=1)
    axes[1].set_title("Chasing side win rate (%)")
    for s, x in ((2023, "train"), (2024, "val"), (2025, "test"), (2026, "test")):
        pass
    fig.tight_layout()
    return fig


def fig_backtest():
    f = pd.read_csv(RESULTS / "val_baselines.csv")
    f = f[f.fold != "all"]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    for mname, c in (("naive prior", COL["naive prior"]), ("logistic regression", COL["logistic regression"]),
                     ("GBM v0 (all features, 400 trees)", "#e34948"), ("GBM v1 (regularised, state feats)", COL["GBM"])):
        s = f[f.model == mname]
        ax.plot(s.fold.astype(int), s.log_loss, "-o", ms=4, lw=1.8, color=c, label=mname)
    ax.set_xticks([2021, 2022, 2023, 2024])
    ax.set_ylabel("log loss")
    ax.set_title("Rolling-origin backtests (each season scored by a model trained on earlier seasons)")
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    return fig


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fn in [("reliability", fig_reliability), ("logloss_by_ball", fig_by_ball), ("phase_logloss", fig_phase),
                     ("era_shift", fig_era), ("backtest_logloss", fig_backtest)]:
        fn().savefig(FIGURES / f"{name}.png")
        plt.close("all")
        print("saved", name)


if __name__ == "__main__":
    main()

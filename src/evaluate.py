"""Metrics for probabilistic win-prediction: log loss, Brier, AUC, ECE, reliability, bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def reliability_table(y, p, n_bins=10):
    """Equal-width bins on [0,1]. Returns DataFrame(bin, n, mean_pred, frac_pos)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append((b, int(mask.sum()), p[mask].mean(), y[mask].mean()))
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "frac_pos"])


def ece(y, p, n_bins=10):
    """Expected calibration error: weighted mean |observed frequency - mean predicted prob|."""
    t = reliability_table(y, p, n_bins)
    return float((t.n * (t.mean_pred - t.frac_pos).abs()).sum() / t.n.sum())


def all_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": len(y), "log_loss": log_loss(y, p), "brier": brier(y, p),
            "auc": auc(y, p), "ece": ece(y, p)}


def metrics_by_group(df, pcol, groups=("overall", "innings", "phase", "inn_phase")):
    """df needs columns: bat_team_wins, innings, phase and the prediction column pcol."""
    out = []
    def add(name, sub):
        r = all_metrics(sub.bat_team_wins.values, sub[pcol].values)
        r["group"] = name
        out.append(r)
    if "overall" in groups:
        add("overall", df)
    if "innings" in groups:
        for i, sub in df.groupby("innings"):
            add(f"innings {i}", sub)
    if "phase" in groups:
        for ph in ["powerplay", "middle", "death"]:
            add(ph, df[df.phase == ph])
    if "inn_phase" in groups:
        for i in (1, 2):
            for ph in ["powerplay", "middle", "death"]:
                add(f"inn{i} {ph}", df[(df.innings == i) & (df.phase == ph)])
    return pd.DataFrame(out)[["group", "n", "log_loss", "brier", "auc", "ece"]]


def bootstrap_ci(df, pcols, n_boot=500, seed=0, metrics=("log_loss", "brier", "auc", "ece"),
                 baseline_col=None):
    """Match-level (cluster) bootstrap: resample whole matches, since balls in a match are not independent.
    Returns {pcol: {metric: (lo, hi)}} and, if baseline_col is given, CI of (model - baseline) differences."""
    rng = np.random.default_rng(seed)
    mids = df.match_id.values
    uniq, inv = np.unique(mids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(inv))
    y = df.bat_team_wins.values.astype(float)
    P = {c: df[c].values.astype(float) for c in pcols}
    if baseline_col and baseline_col not in P:
        P[baseline_col] = df[baseline_col].values.astype(float)
    fns = {"log_loss": log_loss, "brier": brier, "auc": auc, "ece": ece}
    draws = {c: {m: [] for m in metrics} for c in P}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        yy = y[idx]
        for c, arr in P.items():
            for m in metrics:
                draws[c][m].append(fns[m](yy, arr[idx]))
    ci = {c: {m: tuple(np.nanpercentile(v, [2.5, 97.5])) for m, v in d.items()} for c, d in draws.items()}
    diff = {}
    if baseline_col:
        for c in pcols:
            diff[c] = {m: tuple(np.nanpercentile(np.array(draws[c][m]) - np.array(draws[baseline_col][m]), [2.5, 97.5]))
                       for m in metrics}
    return ci, diff

"""Backtest harness: rolling-origin (train on all earlier seasons, test on the next season)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED
from evaluate import all_metrics

FOLDS = [2021, 2022, 2023, 2024]   # each is evaluated by a model trained on strictly earlier seasons


def load():
    return pd.read_parquet(PROCESSED / "deliveries.parquet").reset_index(drop=True)


def decay_weights(df, test_season, half_life):
    """Weight = 0.5 ** (seasons_before_test / half_life); None half_life -> uniform."""
    if half_life is None:
        return None
    return pd.Series(0.5 ** ((test_season - 1 - df.season.values) / half_life), index=df.index)


def backtest(make_model, df, folds=FOLDS, half_life=None, train_min_season=None, name="m"):
    """make_model() -> object with fit(train_df[, weights=]) / predict(df). Returns preds DataFrame."""
    outs = []
    for y in folds:
        tr = df[df.season < y]
        if train_min_season:
            tr = tr[tr.season >= train_min_season]
        te = df[df.season == y]
        m = make_model()
        w = decay_weights(tr, y, half_life)
        try:
            m.fit(tr, weights=w) if w is not None else m.fit(tr)
        except TypeError:
            m.fit(tr)
        o = te[["match_id", "season", "innings", "phase", "balls_bowled", "bat_team_wins"]].copy()
        o["p"] = m.predict(te)
        o["fold"] = y
        outs.append(o)
    return pd.concat(outs)


def summarize(preds, label):
    """Per-fold + pooled metrics for one prediction frame."""
    rows = []
    for y, g in preds.groupby("fold"):
        rows.append({"model": label, "fold": y, **all_metrics(g.bat_team_wins.values, g.p.values)})
    pooled = {"model": label, "fold": "all", **all_metrics(preds.bat_team_wins.values, preds.p.values)}
    rows.append(pooled)
    return pd.DataFrame(rows)

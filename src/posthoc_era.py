"""POST-HOC exploration (run AFTER final_test.py).

The frozen test run showed a failure mode: models trained on <=2023 expect the absolute scoring level of
the training era, but 2025-26 scores are ~25 runs higher. This script asks whether features expressed
RELATIVE to the recent league scoring level (`*_rel`, built from the previous 40 matches only) fix it.
Because the idea was motivated by looking at the test seasons, treat these numbers as exploratory,
NOT as an unbiased test estimate. Nothing here changes the headline results table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, TEST_SEASONS, TRAIN_END, VAL_SEASONS
from evaluate import all_metrics, metrics_by_group
from final_config import FEAT_FINAL, build_gbm
from harness import FOLDS, backtest, load, summarize
from modeling import LR_FEATS_REL, LRBaseline, PlattCalibrator

REL = {1: ["proj_rel", "crr_rel"], 2: ["rrr_rel", "runs_req_rel", "crr_rel"]}


def main():
    df = load()
    tr = df[df.season <= TRAIN_END]
    va = df[df.season.isin(VAL_SEASONS)]
    te = df[df.season.isin(TEST_SEASONS)].reset_index(drop=True)
    feat_rel = {i: FEAT_FINAL[i] + REL[i] for i in (1, 2)}
    models = {
        "LR (relative feats)": LRBaseline(feats=LR_FEATS_REL).fit(tr),
        "GBM (relative feats, boost from rel-LR)": build_gbm(feat_set=feat_rel, lr_feats=LR_FEATS_REL).fit(tr),
    }
    rows, preds = [], te[["match_id", "season", "innings", "phase", "bat_team_wins"]].copy()
    for name, m in models.items():
        pt, pv = m.predict(te), m.predict(va)
        cal = PlattCalibrator().fit(pv, va.bat_team_wins.values, va.innings.values)
        for label, p in ((name, pt), (name + " + Platt", cal.predict(pt, te.innings.values))):
            preds[label] = p
            rows.append({"model": label, **all_metrics(te.bat_team_wins.values, p)})
    out = pd.DataFrame(rows)
    # same-table reference rows from the frozen run
    ref = pd.read_csv(RESULTS / "test_metrics_overall.csv")
    ref = ref[ref.model.isin(["logistic regression", "logistic regression + Platt", "GBM", "GBM + Platt"])]
    out = pd.concat([ref.assign(model=ref.model + "  [frozen run]"), out], ignore_index=True)
    out.to_csv(RESULTS / "posthoc_era_metrics.csv", index=False)
    print(out.round(4).to_string())
    by = pd.concat([metrics_by_group(preds.assign(p=preds[c]), "p", groups=("overall", "innings")).assign(model=c)
                    for c in preds.columns[5:]])
    by.to_csv(RESULTS / "posthoc_era_by_innings.csv", index=False)
    g = preds.groupby(["season", "innings"]).agg(actual=("bat_team_wins", "mean"),
                                                 **{c: (c, "mean") for c in preds.columns[5:]})
    g.to_csv(RESULTS / "posthoc_era_mean_prob.csv")
    print(g.round(3).T.to_string())

    # backtest evidence (validation folds only) for the same idea
    base = backtest(lambda: LRBaseline(), df[df.season <= 2024], folds=FOLDS)
    rel = backtest(lambda: LRBaseline(feats=LR_FEATS_REL), df[df.season <= 2024], folds=FOLDS)
    bt = pd.concat([summarize(base, "LR"), summarize(rel, "LR relative feats")])
    bt.to_csv(RESULTS / "posthoc_era_backtest_lr.csv", index=False)
    print(bt.pivot(index="model", columns="fold", values="log_loss").round(4))


if __name__ == "__main__":
    main()

"""FINAL EVALUATION. Run once, after every modelling decision is frozen.

  train        : seasons <= 2023
  validation   : 2024  (only used to fit the post-hoc calibrators; the GBM never sees it)
  test         : 2025, 2026

Writes results/test_*.csv, results/test_predictions.parquet and results/val2024_predictions.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, TEST_SEASONS, TRAIN_END, VAL_SEASONS
from evaluate import all_metrics, bootstrap_ci, metrics_by_group, reliability_table
from final_config import CALIBRATOR, build_gbm
from harness import load
from modeling import EloOnly, IsotonicCalibrator, LRBaseline, NaivePrior, PlattCalibrator


def main():
    df = load()
    tr = df[df.season <= TRAIN_END]
    va = df[df.season.isin(VAL_SEASONS)]
    te = df[df.season.isin(TEST_SEASONS)].reset_index(drop=True)
    meta = pd.read_parquet(RESULTS.parent / "data" / "processed" / "matches.parquet")
    split_sizes = {
        "train (<=2023)": (tr.match_id.nunique(), len(tr)),
        "validation (2024)": (va.match_id.nunique(), len(va)),
        "test (2025-26)": (te.match_id.nunique(), len(te)),
        **{f"test {s}": (te[te.season == s].match_id.nunique(), int((te.season == s).sum())) for s in TEST_SEASONS},
    }
    pd.DataFrame(split_sizes, index=["matches", "deliveries"]).T.to_csv(RESULTS / "split_sizes.csv")
    print(pd.DataFrame(split_sizes, index=["matches", "deliveries"]).T)

    models = {
        "naive prior": NaivePrior().fit(tr),
        "pre-match Elo only": EloOnly().fit(tr),
        "logistic regression": LRBaseline().fit(tr),
        "GBM": build_gbm().fit(tr),
        "GBM (monotone)": build_gbm(mono=True).fit(tr),
    }
    P_te = {k: m.predict(te) for k, m in models.items()}
    P_va = {k: m.predict(va) for k, m in models.items()}

    # --- post-hoc calibration, fitted on the 2024 validation season only
    for base in ("GBM", "GBM (monotone)", "logistic regression"):
        for name, Cal in (("Platt", PlattCalibrator), ("isotonic", IsotonicCalibrator)):
            cal = Cal().fit(P_va[base], va.bat_team_wins.values, va.innings.values)
            P_te[f"{base} + {name}"] = cal.predict(P_te[base], te.innings.values)
            P_va[f"{base} + {name}"] = cal.predict(P_va[base], va.innings.values)   # in-sample for the calibrator
    chosen = "GBM" if CALIBRATOR is None else f"GBM + {CALIBRATOR}"

    out = te[["match_id", "season", "innings", "k", "phase", "balls_bowled", "bat_team_wins"]].copy()
    for k, v in P_te.items():
        out[k] = v
    out.to_parquet(RESULTS / "test_predictions.parquet", index=False)
    vout = va[["match_id", "season", "innings", "k", "phase", "balls_bowled", "bat_team_wins"]].copy()
    for k, v in P_va.items():
        vout[k] = v
    vout.to_parquet(RESULTS / "val2024_predictions.parquet", index=False)

    # --- metrics
    rows = []
    for k in P_te:
        t = metrics_by_group(out.assign(p=out[k]), "p")
        t.insert(0, "model", k)
        rows.append(t)
    full = pd.concat(rows)
    full.to_csv(RESULTS / "test_metrics_full.csv", index=False)
    overall = full[full.group == "overall"].drop(columns="group").set_index("model")
    overall.to_csv(RESULTS / "test_metrics_overall.csv")

    # per test season
    rows = []
    for s in TEST_SEASONS:
        o = out[out.season == s]
        for k in P_te:
            rows.append({"season": s, "model": k, **all_metrics(o.bat_team_wins.values, o[k].values)})
    pd.DataFrame(rows).to_csv(RESULTS / "test_metrics_by_season.csv", index=False)

    # match-level bootstrap CIs and paired differences vs the baselines
    headline = list(dict.fromkeys(["naive prior", "logistic regression", "logistic regression + Platt", "GBM", "GBM (monotone)", "GBM + Platt", "GBM + isotonic", chosen]))
    ci, diff_lr = bootstrap_ci(out, headline, n_boot=500, baseline_col="logistic regression")
    _, diff_naive = bootstrap_ci(out, headline, n_boot=500, baseline_col="naive prior")
    rows = []
    for k in headline:
        r = {"model": k}
        for m in ("log_loss", "brier", "auc", "ece"):
            r[f"{m}_lo"], r[f"{m}_hi"] = ci[k][m]
            r[f"{m}_vs_LR_lo"], r[f"{m}_vs_LR_hi"] = diff_lr[k][m]
            r[f"{m}_vs_naive_lo"], r[f"{m}_vs_naive_hi"] = diff_naive[k][m]
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "test_bootstrap_ci.csv", index=False)

    # like-for-like: GBM + Platt vs LR + Platt, and the Platt gain on each test season separately
    rows = []
    _, d = bootstrap_ci(out, ["GBM + Platt"], n_boot=1000, baseline_col="logistic regression + Platt", metrics=("log_loss",))
    rng_cols = out[["match_id", "bat_team_wins", "GBM + Platt", "logistic regression + Platt"]]
    rows.append({"scope": "GBM+Platt minus LR+Platt, pooled", "log_loss_lo": d["GBM + Platt"]["log_loss"][0], "log_loss_hi": d["GBM + Platt"]["log_loss"][1]})
    for sname, sub in [("2025", out[out.season == 2025]), ("2026", out[out.season == 2026]), ("pooled", out)]:
        _, dd = bootstrap_ci(sub.assign(base=sub["GBM"]), ["GBM + Platt"], n_boot=1000, baseline_col="GBM", metrics=("log_loss",))
        rows.append({"scope": f"Platt gain on GBM, {sname}", "point": float(
            all_metrics(sub.bat_team_wins.values, sub["GBM + Platt"].values)["log_loss"] - all_metrics(sub.bat_team_wins.values, sub["GBM"].values)["log_loss"]),
            "log_loss_lo": dd["GBM + Platt"]["log_loss"][0], "log_loss_hi": dd["GBM + Platt"]["log_loss"][1]})
    pd.DataFrame(rows).to_csv(RESULTS / "test_platt_comparisons.csv", index=False)

    # reliability tables
    rel = []
    for k in P_te:
        for inn in (0, 1, 2):
            o = out if inn == 0 else out[out.innings == inn]
            t = reliability_table(o.bat_team_wins.values, o[k].values, 10)
            t.insert(0, "innings", inn)
            t.insert(0, "model", k)
            rel.append(t)
    pd.concat(rel).to_csv(RESULTS / "test_reliability.csv", index=False)

    # log-loss by balls bowled (20-ball buckets) -- where in the innings does the model add value?
    out["bucket"] = (out.balls_bowled // 20) * 20
    bl = []
    for (inn, b), g in out.groupby(["innings", "bucket"]):
        for k in headline:
            bl.append({"innings": inn, "balls_bowled_from": b, "model": k, **all_metrics(g.bat_team_wins.values, g[k].values)})
    pd.DataFrame(bl).to_csv(RESULTS / "test_by_ball_bucket.csv", index=False)

    json.dump({"chosen_model": chosen, "split_sizes": {k: list(map(int, v)) for k, v in split_sizes.items()}},
              open(RESULTS / "test_run_info.json", "w"), indent=2)
    print(overall.round(4).to_string())


if __name__ == "__main__":
    main()

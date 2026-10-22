"""All model-selection experiments. Uses ONLY seasons <= 2024 (rolling-origin backtests 2021-2024).
The test seasons (2025-26) are never loaded into any of these comparisons.

    python src/run_validation.py [baselines|features|structure|era|hparams|calibration|all]

Each section writes a CSV into results/ that DECISIONS.md refers to.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CRR_PRIOR_BALLS, RESULTS
from evaluate import all_metrics, ece, log_loss
from harness import FOLDS, backtest, load, summarize
from modeling import (CONTEXT, ERA, FORM, GBM, TOSS, EloOnly, IsotonicCalibrator, LRBaseline,
                      NaivePrior, PlattCalibrator)

RESULTS.mkdir(exist_ok=True)
# model-selection data: nothing from the test seasons
DF = load()
DF = DF[DF.season <= 2024].reset_index(drop=True)

from final_config import P_REG, STATE1, STATE2, USE_INIT_LR, build_gbm  # noqa: E402
GROUPS = {
    "form": (FORM, FORM),
    "relative_to_league": (["proj_rel", "crr_rel"], ["rrr_rel", "runs_req_rel", "crr_rel"]),
    "venue": (["venue_avg_score"], ["venue_avg_score", "venue_chase_win"]),
    "league_avg_l40": (["league_avg_l40"], ["league_avg_l40"]),
    "toss": (TOSS, TOSS),
    "elo_diff": (["elo_diff"], ["elo_diff"]),
    "impact_era_flag": (ERA, ERA),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_innings_fold_ll(make, df=DF, folds=FOLDS, half_life=None, train_min_season=None):
    """Log loss per (fold, innings) at the model's fixed number of iterations."""
    p = backtest(make, df, folds=folds, half_life=half_life, train_min_season=train_min_season)
    rows = []
    for (y, i), g in p.groupby(["fold", "innings"]):
        rows.append({"fold": y, "innings": i, "n": len(g), "log_loss": log_loss(g.bat_team_wins.values, g.p.values)})
    return pd.DataFrame(rows), p


def pooled(df_ll):
    """Row-weighted mean of per-fold log loss, by innings and overall."""
    out = {}
    for i, g in df_ll.groupby("innings"):
        out[f"inn{i}"] = np.average(g.log_loss, weights=g.n)
    out["overall"] = np.average(df_ll.log_loss, weights=df_ll.n)
    return out


# --------------------------------------------------------------------------------------
def baselines():
    log("baselines: naive / elo-only / LR / GBM (first attempt) / GBM (regularised)")
    models = {
        "naive prior": lambda: NaivePrior(),
        "pre-match Elo only": lambda: EloOnly(),
        "logistic regression": lambda: LRBaseline(),
        # first attempt: every feature incl. match-constant context + target, default-ish trees
        "GBM v0 (all features, 400 trees)": lambda: GBM(n_iter=400, params=dict(min_child_samples=300, num_leaves=15)),
        "GBM v1 (regularised, state feats)": lambda: GBM(params=P_REG, n_iter=400, feat_set={1: STATE1, 2: STATE2}),
    }
    rows = []
    for k, mk in models.items():
        p = backtest(mk, DF)
        rows.append(summarize(p, k))
    r = pd.concat(rows)
    r.to_csv(RESULTS / "val_baselines.csv", index=False)
    print(r.pivot(index="model", columns="fold", values="log_loss").round(4))
    print(r[r.fold == "all"].round(4).to_string())


def features():
    log("feature-group ablation (GBM, fixed 400 trees, +group vs base)")
    def run(g):
        f1 = STATE1 + (GROUPS[g][0] if g else [])
        f2 = STATE2 + (GROUPS[g][1] if g else [])
        ll, _ = per_innings_fold_ll(lambda: GBM(params=P_REG, n_iter=400, feat_set={1: f1, 2: f2}))
        ll["group"] = g or "base"
        return ll
    base = run(None)
    rows = [base]
    summ = []
    for g in GROUPS:
        r = run(g)
        rows.append(r)
        for i in (1, 2):
            a = r[r.innings == i].set_index("fold").log_loss
            b = base[base.innings == i].set_index("fold").log_loss
            summ.append({"group": g, "innings": i, "mean_delta": float((a - b).mean()),
                         "folds_better": int((a < b).sum()), "n_folds": len(a)})
        log(f"  {g}")
    pd.concat(rows).to_csv(RESULTS / "val_feature_ablation_folds.csv", index=False)
    s = pd.DataFrame(summ)
    s["accept"] = (s.mean_delta <= -0.001) & (s.folds_better >= 3)
    s.to_csv(RESULTS / "val_feature_ablation.csv", index=False)
    print(s.round(4).to_string())


from final_config import FEAT_FINAL  # noqa: E402


def structure():
    log("structure: separate vs joint, monotone, init-from-LR, bagging")
    variants = {
        "separate (state+venue)": dict(feat_set=FEAT_FINAL),
        "separate + monotone": dict(feat_set=FEAT_FINAL, mono=True),
        "separate, boost from LR": dict(feat_set=FEAT_FINAL, init_lr=True),
        "separate, boost from LR + monotone": dict(feat_set=FEAT_FINAL, init_lr=True, mono=True),
        "joint (innings flag)": dict(mode="joint"),
        "separate, 5 match-bagged": dict(feat_set=FEAT_FINAL, n_bags=5),
    }
    rows = []
    for k, kw in variants.items():
        mk = lambda kw=kw: GBM(params=P_REG, n_iter=400, **kw)
        ll, p = per_innings_fold_ll(mk)
        ll["variant"] = k
        rows.append(ll)
        log(f"  {k}: {pooled(ll)}")
    # joint model with the same venue+state features (fairer): union of both innings' features
    r = pd.concat(rows)
    r.to_csv(RESULTS / "val_structure_folds.csv", index=False)
    out = pd.DataFrame({k: pooled(g) for k, g in r.groupby("variant")}).T
    folds = r.groupby(["variant", "fold"]).apply(lambda g: np.average(g.log_loss, weights=g.n)).unstack()
    out = out.join(folds.add_prefix("fold_"))
    out.to_csv(RESULTS / "val_structure.csv")
    print(out.round(4).to_string())


def era():
    log("era: Impact Player (2023+) shift -- flag, recency weighting, shorter history")
    rows = []
    variants = {
        "baseline (all history, uniform)": dict(),
        "+ impact_era flag": dict(feat_set={1: FEAT_FINAL[1] + ERA, 2: FEAT_FINAL[2] + ERA}),
        "recency half-life 4 seasons": dict(half_life=4),
        "recency half-life 2 seasons": dict(half_life=2),
        "recency half-life 1 season": dict(half_life=1),
        "train only from 2016": dict(train_min_season=2016),
        "train only from 2018": dict(train_min_season=2018),
    }
    for k, kw in variants.items():
        hl, tm = kw.get("half_life"), kw.get("train_min_season")
        fs = kw.get("feat_set", FEAT_FINAL)
        ll, _ = per_innings_fold_ll(lambda fs=fs: GBM(params=P_REG, n_iter=400, feat_set=fs), half_life=hl, train_min_season=tm)
        ll["variant"] = k
        rows.append(ll)
        log(f"  {k}: {pooled(ll)}")
    r = pd.concat(rows)
    r.to_csv(RESULTS / "val_era_folds.csv", index=False)
    out = pd.DataFrame({k: pooled(g) for k, g in r.groupby("variant")}).T
    folds = r.groupby(["variant", "fold"]).apply(lambda g: np.average(g.log_loss, weights=g.n)).unstack()
    out = out.join(folds.add_prefix("fold_"))
    out.to_csv(RESULTS / "val_era.csv")
    print(out.round(4).to_string())
    # descriptive evidence of the shift
    m = pd.read_parquet(RESULTS.parent / "data" / "processed" / "matches.parquet")
    m = m[m.included]
    m["era"] = np.where(m.season >= 2023, "2023+ (Impact Player)", np.where(m.season >= 2018, "2018-22", "2008-17"))
    d = m.groupby("era").agg(matches=("match_id", "size"), avg_1st_inns=("score1", "mean"), avg_2nd_inns=("score2", "mean"),
                              chase_win_rate=("chase_won", "mean"), bat_first_choice=("toss_chose_bat", "mean"))
    d.to_csv(RESULTS / "era_scoring_shift.csv")
    print(d.round(3))


def hparams(n_trials=14, seed=11):
    log(f"hyper-parameter random search ({n_trials} trials, curves over 4 backtest folds)")
    rng = np.random.default_rng(seed)
    space = dict(num_leaves=[3, 4, 6, 8, 12], min_child_samples=[300, 600, 1000, 2000, 4000],
                 lambda_l2=[1.0, 10.0, 50.0, 200.0], feature_fraction=[0.6, 0.8, 1.0],
                 extra_trees=[True, False], learning_rate=[0.02, 0.03, 0.05])
    results = []
    cfgs = [dict(P_REG)] + [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n_trials)]
    for t, cfg in enumerate(cfgs):
        curves = {1: [], 2: []}
        for y in FOLDS:
            tr, te = DF[DF.season < y], DF[DF.season == y]
            m = GBM(params=cfg, n_iter=700, feat_set=FEAT_FINAL, init_lr=USE_INIT_LR).fit(tr, valid=te)
            for i in (1, 2):
                curves[i].append((m.curves[i], int((te.innings == i).sum())))
        row = {"trial": t, **{k: cfg.get(k) for k in space}}
        for i in (1, 2):
            c = sum(cv * n for cv, n in curves[i]) / sum(n for _, n in curves[i])
            row[f"inn{i}_best_iter"] = int(c.argmin() + 1)
            row[f"inn{i}_best_ll"] = float(c.min())
        results.append(row)
        log(f"  trial {t}: " + ", ".join(f"inn{i} {row[f'inn{i}_best_ll']:.4f}@{row[f'inn{i}_best_iter']}" for i in (1, 2)))
    r = pd.DataFrame(results)
    n1 = int((DF[DF.season.isin(FOLDS)].innings == 1).sum())
    n2 = int((DF[DF.season.isin(FOLDS)].innings == 2).sum())
    r["combined_ll"] = (n1 * r.inn1_best_ll + n2 * r.inn2_best_ll) / (n1 + n2)
    r.to_csv(RESULTS / "val_hparams.csv", index=False)
    b = r.loc[r.combined_ll.idxmin()]
    cast = lambda v: v.item() if hasattr(v, "item") else v
    best = {"params": {k: cast(b[k]) for k in space},
            "n_iter": {"1": int(b.inn1_best_iter), "2": int(b.inn2_best_iter)},
            "trial": int(b.trial), "backtest_ll": {"1": float(b.inn1_best_ll), "2": float(b.inn2_best_ll)}}
    json.dump(best, open(RESULTS / "best_hparams.json", "w"), indent=2)
    print(r.sort_values("combined_ll").round(4).to_string())
    print(json.dumps(best, indent=2))


def _perturb(te, kind):
    """Counterfactual copies of the states: +1 wicket, or +6 runs (derived features recomputed)."""
    x = te.copy()
    if kind == "extra_wicket":
        x["wickets"] = x.wickets + 1
    else:  # six more runs on the board
        x["runs"] = x.runs + 6
        x["runs_req"] = x.runs_req - 6
        x["rrr"] = x.runs_req * 6.0 / x.balls_left
        x["crr"] = 6.0 * (x.runs + x.league_avg_l40 / 120.0 * CRR_PRIOR_BALLS) / (x.balls_bowled + CRR_PRIOR_BALLS)
        x["rrr_minus_crr"] = x.rrr - x.crr
        x["proj_score"] = x.runs + x.crr * x.balls_left / 6.0
    return x


def mono():
    log("monotone constraints: do they help calibration / stability? (backtest folds 2021-24)")
    rows = []
    for name, mk in {"LR baseline": lambda: LRBaseline(),
                     "GBM (no constraints)": lambda: build_gbm(mono=False),
                     "GBM (monotone)": lambda: build_gbm(mono=True)}.items():
        for y in FOLDS:
            tr, te = DF[DF.season < y], DF[DF.season == y].reset_index(drop=True)
            m = mk().fit(tr)
            p = m.predict(te)
            rough = pd.Series(np.abs(np.diff(p))).where(
                (te.match_id.values[1:] == te.match_id.values[:-1]) & (te.innings.values[1:] == te.innings.values[:-1])).mean()
            r = {"model": name, "fold": y, **all_metrics(te.bat_team_wins.values, p), "roughness": rough}
            # wicket / runs counterfactual violation rates (innings 2 only for runs_req logic; both for wicket)
            v_w = (m.predict(_perturb(te, "extra_wicket")) > p + 1e-9).mean()
            t2 = te[te.innings == 2].reset_index(drop=True)
            v_r = (m.predict(_perturb(t2, "more_runs")) < m.predict(t2) - 1e-9).mean()
            r.update(viol_extra_wicket=v_w, viol_more_runs_inn2=v_r)
            rows.append(r)
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "val_monotone_folds.csv", index=False)
    out = r.groupby("model")[["log_loss", "brier", "ece", "roughness", "viol_extra_wicket", "viol_more_runs_inn2"]].mean()
    out.to_csv(RESULTS / "val_monotone.csv")
    print(out.round(5).to_string())


def calibration():
    log("calibration: nested backtest (model on <y-1, calibrator on y-1, scored on y)")
    rows, pred_rows = [], []
    for y in [2022, 2023, 2024]:
        tr, cal, te = DF[DF.season < y - 1], DF[DF.season == y - 1], DF[DF.season == y]
        m = build_gbm().fit(tr)
        pc, pt = m.predict(cal), m.predict(te)
        out = te[["match_id", "season", "innings", "phase", "bat_team_wins"]].copy()
        out["raw"] = pt
        out["platt"] = PlattCalibrator().fit(pc, cal.bat_team_wins.values, cal.innings.values).predict(pt, te.innings.values)
        out["isotonic"] = IsotonicCalibrator().fit(pc, cal.bat_team_wins.values, cal.innings.values).predict(pt, te.innings.values)
        out["fold"] = y
        pred_rows.append(out)
    P = pd.concat(pred_rows)
    for c in ("raw", "platt", "isotonic"):
        for y, g in list(P.groupby("fold")) + [("all", P)]:
            rows.append({"variant": c, "fold": y, **all_metrics(g.bat_team_wins.values, g[c].values)})
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "val_calibration.csv", index=False)
    print(r.pivot(index="variant", columns="fold", values="ece").round(4))
    print(r.pivot(index="variant", columns="fold", values="log_loss").round(4))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = dict(baselines=baselines, features=features, structure=structure, era=era,
                 hparams=hparams, mono=mono, calibration=calibration)
    for k, f in steps.items():
        if which in ("all", k):
            f()

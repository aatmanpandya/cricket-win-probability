"""The chosen model configuration (decisions documented in DECISIONS.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS
from modeling import GBM

# regularised starting point used throughout model selection
P_REG = dict(min_child_samples=1000, num_leaves=4, learning_rate=0.03, extra_trees=True)

STATE1 = ["runs", "wickets", "balls_left", "crr", "proj_score"]
# NB: no `target` / `runs` in innings 2 -- with runs_req they identify the match, so trees memorise (DECISIONS #7)
STATE2 = ["wickets", "balls_left", "runs_req", "rrr", "crr", "rrr_minus_crr"]
VENUE = {1: ["venue_avg_score"], 2: ["venue_avg_score", "venue_chase_win"]}
FEAT_FINAL = {1: STATE1 + VENUE[1], 2: STATE2 + VENUE[2]}

# Structural choices settled in run_validation.py (structure / era sections)
USE_MONOTONE = False
USE_INIT_LR = True        # boost from the logistic-regression logit (DECISIONS #10)
CALIBRATOR = None            # nested backtest: post-hoc calibration did NOT help -> headline model is uncalibrated GBM (DECISIONS #12)
RECENCY_HALF_LIFE = None     # None = uniform weights


def load_best():
    f = RESULTS / "best_hparams.json"
    if f.exists():
        return json.load(open(f))
    return None


def build_gbm(**override):
    best = load_best()
    if best:
        params = dict(best["params"])
        n_iter = {1: best["n_iter"]["1"], 2: best["n_iter"]["2"]}
    else:
        params, n_iter = dict(P_REG), {1: 400, 2: 400}
    kw = dict(params=params, n_iter=n_iter, feat_set=FEAT_FINAL, mono=USE_MONOTONE, init_lr=USE_INIT_LR)
    kw.update(override)
    return GBM(**kw)

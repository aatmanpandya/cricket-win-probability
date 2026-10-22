"""Feature sets, baselines and LightGBM wrappers."""
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ feature sets
LR_FEATS = {
    1: ["runs", "wickets", "balls_left", "crr"],
    2: ["runs_req", "wickets", "balls_left", "rrr", "crr"],
}
STATE = {
    1: ["runs", "wickets", "balls_left", "crr", "proj_score"],
    2: ["runs", "wickets", "balls_left", "crr", "target", "runs_req", "rrr", "rrr_minus_crr"],
}
FORM = ["runs_l12", "wkts_l12", "runs_l24", "wkts_l24"]
CONTEXT = ["venue_avg_score", "venue_chase_win", "league_avg_l40", "elo_diff", "bat_won_toss", "toss_chose_bat"]
TOSS = ["bat_won_toss", "toss_chose_bat"]
ERA = ["impact_era"]
# same idea as LR_FEATS but expressed relative to the recent league-wide scoring level
LR_FEATS_REL = {
    1: ["runs", "wickets", "balls_left", "crr_rel", "proj_rel"],
    2: ["runs_req_rel", "wickets", "balls_left", "rrr_rel", "crr_rel"],
}


def feats(innings, form=True, context=True, extra=()):
    f = list(STATE[innings]) + (FORM if form else []) + (CONTEXT if context else []) + list(extra)
    return list(dict.fromkeys(f))


# +1 = P(win) must not decrease as feature rises, -1 = must not increase
MONO = {
    "runs": +1, "wickets": -1, "proj_score": +1, "crr": +1, "runs_req": -1, "rrr": -1,
    "rrr_minus_crr": -1, "elo_diff": +1,
}

BASE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=300,
                   lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   verbosity=-1, num_threads=4, seed=7, deterministic=True, force_row_wise=True)


# ------------------------------------------------------------------ baselines
class NaivePrior:
    """P(win) = training win rate of the batting side in that innings (a constant per innings)."""
    def fit(self, df):
        self.rate = df.groupby("innings").bat_team_wins.mean().to_dict()
        return self

    def predict(self, df):
        return df.innings.map(self.rate).values.astype(float)


class EloOnly:
    """Pre-match strength only: logistic regression on elo_diff, per innings (ignores the live score)."""
    def fit(self, df):
        self.m = {i: LogisticRegression().fit(g[["elo_diff"]], g.bat_team_wins) for i, g in df.groupby("innings")}
        return self

    def predict(self, df):
        p = np.zeros(len(df))
        for i, g in df.groupby("innings"):
            p[df.innings.values == i] = self.m[i].predict_proba(g[["elo_diff"]])[:, 1]
        return p


class LRBaseline:
    """Logistic regression on a handful of raw state features, separate model per innings."""
    def __init__(self, C=1.0, feats=None):
        self.C, self.feats = C, feats or LR_FEATS

    def fit(self, df, weights=None):
        self.m = {}
        for i, g in df.groupby("innings"):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
            pipe.fit(g[self.feats[i]], g.bat_team_wins,
                     logisticregression__sample_weight=None if weights is None else weights[g.index])
            self.m[i] = pipe
        return self

    def predict(self, df):
        p = np.zeros(len(df))
        for i in (1, 2):
            mask = df.innings.values == i
            if not mask.any():
                continue
            p[mask] = self.m[i].predict_proba(df.loc[mask, self.feats[i]])[:, 1]
        return p


# ------------------------------------------------------------------ LightGBM
def _logit_arr(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GBM:
    """Gradient-boosted trees (LightGBM).

    mode        'separate' = one booster per innings, 'joint' = one booster with an `innings` flag
    mono        apply monotone constraints (see MONO)
    feat_set    explicit {1: [...], 2: [...]} feature lists (overrides form/context/extra/drop)
    init_lr     boost *from* the logistic-regression baseline's logit (residual boosting)
    n_bags      average this many boosters, each trained on a bootstrap sample of whole MATCHES
                (balls within a match are highly correlated, so row bagging does not decorrelate)
    """

    def __init__(self, params=None, n_iter=None, mode="separate", mono=False,
                 form=True, context=True, extra=(), drop=(), feat_set=None, init_lr=False, n_bags=1, lr_feats=None):
        self.params = {**BASE_PARAMS, **(params or {})}
        self.n_iter = n_iter if isinstance(n_iter, dict) else {1: n_iter or 400, 2: n_iter or 400, 0: n_iter or 400}
        self.mode, self.mono = mode, mono
        self.form, self.context, self.extra, self.drop = form, context, tuple(extra), tuple(drop)
        self.feat_set, self.init_lr, self.n_bags, self.lr_feats = feat_set, init_lr, n_bags, lr_feats

    def _feats(self, innings):
        if self.feat_set is not None and innings in self.feat_set:
            return list(self.feat_set[innings])
        if innings == 0:   # joint
            f = list(dict.fromkeys(feats(1, self.form, self.context, self.extra) + feats(2, self.form, self.context, self.extra)))
            f = ["innings"] + f
        else:
            f = feats(innings, self.form, self.context, self.extra)
        return [c for c in f if c not in self.drop]

    def _params(self, f):
        p = dict(self.params)
        if self.mono:
            p["monotone_constraints"] = [MONO.get(c, 0) for c in f]
            p["monotone_constraints_method"] = "advanced"
        return p

    def _groups(self):
        return [1, 2] if self.mode == "separate" else [0]

    def fit(self, df, weights=None, valid=None):
        self.boosters, self.curves = {}, {}
        if self.init_lr:
            self.lr = LRBaseline(feats=self.lr_feats).fit(df)
            df = df.assign(_init=_logit_arr(self.lr.predict(df)))
            if valid is not None:
                valid = valid.assign(_init=_logit_arr(self.lr.predict(valid)))
        rng = np.random.default_rng(self.params.get("seed", 7))
        for gi in self._groups():
            f = self._feats(gi)
            tr_all = df if gi == 0 else df[df.innings == gi]
            mids = tr_all.match_id.unique()
            boosters, curves = [], []
            for bag in range(self.n_bags):
                if self.n_bags > 1:
                    pick = rng.choice(mids, len(mids), replace=True)
                    cnt = pd.Series(pick).value_counts()
                    w_match = tr_all.match_id.map(cnt).fillna(0).values
                    tr = tr_all[w_match > 0]
                    w = w_match[w_match > 0].astype(float)
                else:
                    tr, w = tr_all, None
                if weights is not None:
                    ww = weights[tr.index].values
                    w = ww if w is None else w * ww
                ds = lgb.Dataset(tr[f], tr.bat_team_wins, weight=w,
                                 init_score=tr["_init"].values if self.init_lr else None)
                kw = {}
                if valid is not None:
                    va = valid if gi == 0 else valid[valid.innings == gi]
                    kw["valid_sets"] = [lgb.Dataset(va[f], va.bat_team_wins, reference=ds,
                                                    init_score=va["_init"].values if self.init_lr else None)]
                    rec = {}
                    kw["callbacks"] = [lgb.record_evaluation(rec)]
                p = self._params(f)
                p["seed"] = int(p.get("seed", 7)) + bag
                b = lgb.train(p, ds, num_boost_round=self.n_iter[gi], **kw)
                boosters.append(b)
                if valid is not None:
                    curves.append(np.array(rec["valid_0"]["binary_logloss"]))
            self.boosters[gi] = boosters
            if valid is not None:
                self.curves[gi] = np.mean(curves, axis=0)
        return self

    def predict(self, df):
        p = np.zeros(len(df))
        init = _logit_arr(self.lr.predict(df)) if self.init_lr else 0.0
        for gi in self._groups():
            mask = np.ones(len(df), bool) if gi == 0 else df.innings.values == gi
            if not mask.any():
                continue
            f = self._feats(gi)
            raw = np.mean([b.predict(df.loc[mask, f], raw_score=True) for b in self.boosters[gi]], axis=0)
            base = init[mask] if self.init_lr else 0.0
            p[mask] = 1 / (1 + np.exp(-(raw + base)))
        return p

    def importance(self):
        out = {}
        for gi, bs in self.boosters.items():
            out[gi] = pd.Series(np.mean([b.feature_importance("gain") for b in bs], axis=0),
                                index=bs[0].feature_name()).sort_values(ascending=False)
        return out


# ------------------------------------------------------------------ calibrators
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression on logit(p), fitted per innings: p' = sigmoid(a * logit(p) + b)."""
    def fit(self, p, y, innings):
        self.m = {i: LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p[innings == i]).reshape(-1, 1), y[innings == i])
                  for i in (1, 2)}
        return self

    def predict(self, p, innings):
        out = np.array(p, float)
        for i in (1, 2):
            mk = innings == i
            out[mk] = self.m[i].predict_proba(_logit(p[mk]).reshape(-1, 1))[:, 1]
        return out


class IsotonicCalibrator:
    def fit(self, p, y, innings):
        self.m = {i: IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p[innings == i], y[innings == i])
                  for i in (1, 2)}
        return self

    def predict(self, p, innings):
        out = np.array(p, float)
        for i in (1, 2):
            mk = innings == i
            out[mk] = self.m[i].predict(p[mk])
        return out

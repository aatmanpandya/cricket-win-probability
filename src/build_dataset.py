"""Build the modelling dataset from Cricsheet ball-by-ball CSVs.

Output (data/processed/):
  matches.parquet     one row per match (all 1,243), with inclusion flag + exclusion reason
  deliveries.parquet  one row per LEGAL delivery of every included match (innings 1 and 2),
                      holding the match state *before* that ball + label + event details
  terminals.parquet   one row per included innings: the state after the last ball, used only
                      for the replay (never for training / evaluation)

STATE CONVENTION  (documented in DECISIONS.md #3)
  A row describes the situation immediately BEFORE a legal ball is bowled. Any wide/no-ball
  bowled since the previous legal ball is already in the state (they change the score but not
  the ball count). Label = 1 if the team batting in that innings goes on to win the match.
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (BALLS_PER_INNINGS, CRR_PRIOR_BALLS, IMPACT_PLAYER_FIRST_SEASON,
                    PROCESSED, RAW_CSV, norm_team, norm_venue)

USECOLS = ["match_id", "start_date", "venue", "innings", "ball", "batting_team", "bowling_team",
           "striker", "non_striker", "bowler", "runs_off_bat", "extras", "wides", "noballs",
           "byes", "legbyes", "penalty", "wicket_type", "player_dismissed"]


# ----------------------------------------------------------------------------- loading
def load_balls() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW_CSV.glob("*.csv")):
        if re.fullmatch(r"\d+\.csv", f.name):
            frames.append(pd.read_csv(f, usecols=USECOLS, low_memory=False))
    d = pd.concat(frames, ignore_index=True)
    for c in ["wides", "noballs", "byes", "legbyes", "penalty"]:
        d[c] = d[c].fillna(0).astype(float)
    d["batting_team"] = d.batting_team.map(norm_team)
    d["bowling_team"] = d.bowling_team.map(norm_team)
    return d


def load_info() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW_CSV.glob("*_info.csv")):
        rec = {"match_id": int(f.name.split("_")[0])}
        for line in f.read_text().splitlines():
            parts = line.split(",", 2)
            if len(parts) < 3 or parts[0] != "info":
                continue
            key, val = parts[1], parts[2].strip().strip('"')
            if key in ("team", "player", "registry", "umpire", "reserve_umpire", "tv_umpire",
                       "match_referee", "player_of_match"):
                continue
            if key in ("target_overs", "target_runs"):   # stored as "<innings>,<value>"
                val = val.split(",")[-1]
            rec.setdefault(key, val)
        rows.append(rec)
    m = pd.DataFrame(rows)
    m["date"] = pd.to_datetime(m["date"].str.replace("/", "-"))
    # season label in Cricsheet is "2007/08", "2009/10", "2020/21": use the calendar year of
    # the first match day, which gives the real IPL edition (2008, 2010, 2020).
    m["season"] = m["date"].dt.year
    m["venue_raw"] = m["venue"]
    m["venue"] = m["venue"].map(norm_venue)
    for c in ("toss_winner", "winner", "eliminator"):
        if c in m:
            m[c] = m[c].map(lambda x: norm_team(x) if isinstance(x, str) else x)
    m["match_number"] = pd.to_numeric(m.get("match_number"), errors="coerce")
    return m


# ----------------------------------------------------------------------------- innings states
def innings_states(d: pd.DataFrame):
    """Return (states, terminals) for innings 1 and 2 of every match in d."""
    d = d[d.innings.isin([1, 2])].copy()
    d["wide"] = d.wides > 0
    d["noball"] = d.noballs > 0
    d["legal"] = ~(d.wide | d.noball)
    d["runs"] = d.runs_off_bat + d.extras                  # extras already include penalties
    d["wkt"] = d.wicket_type.notna() & (d.wicket_type != "retired hurt")
    keys = ["match_id", "innings"]
    d["w"] = d.groupby(keys).legal.cumsum().astype(int)    # legal balls bowled incl. this one

    win = (d.groupby(keys + ["w"])
             .agg(runs_w=("runs", "sum"), wk_w=("wkt", "sum"),
                  n_wide=("wide", "sum"), n_noball=("noball", "sum"))
             .reset_index())
    lead = win[win.w == 0].set_index(keys)[["runs_w", "wk_w"]].rename(
        columns={"runs_w": "R0", "wk_w": "W0"})
    legal = (d[d.legal][keys + ["w", "striker", "non_striker", "bowler", "runs_off_bat",
                                "wicket_type", "player_dismissed", "batting_team",
                                "bowling_team", "extras", "wides", "noballs", "byes",
                                "legbyes"]]
             .merge(win[win.w > 0], on=keys + ["w"], how="left"))
    legal = legal.sort_values(keys + ["w"]).reset_index(drop=True)
    legal = legal.join(lead, on=keys).fillna({"R0": 0, "W0": 0})

    g = legal.groupby(keys)
    legal["k"] = legal.w
    legal["balls_bowled"] = legal.k - 1
    cr = g.runs_w.cumsum()
    cw = g.wk_w.cumsum()
    legal["runs"] = legal.R0 + cr - legal.runs_w            # score BEFORE this ball
    legal["wickets"] = legal.W0 + cw - legal.wk_w
    # form over the previous 12 / 24 legal balls (sum over windows k-n .. k-1)
    pre_r = cr - legal.runs_w
    pre_w = cw - legal.wk_w
    for n in (12, 24):
        legal[f"runs_l{n}"] = pre_r - pre_r.groupby([legal.match_id, legal.innings]).shift(n).fillna(0)
        legal[f"wkts_l{n}"] = pre_w - pre_w.groupby([legal.match_id, legal.innings]).shift(n).fillna(0)

    term = (g.agg(n_legal=("k", "max"), runs_w_sum=("runs_w", "sum"), wk_sum=("wk_w", "sum"),
                  R0=("R0", "first"), W0=("W0", "first"),
                  batting_team=("batting_team", "first"), bowling_team=("bowling_team", "first"))
              .reset_index())
    term["runs_final"] = term.R0 + term.runs_w_sum
    term["wickets_final"] = term.W0 + term.wk_sum
    return legal, term[["match_id", "innings", "n_legal", "runs_final", "wickets_final",
                        "batting_team", "bowling_team"]]


# ----------------------------------------------------------------------------- match priors
def elo_priors(matches: pd.DataFrame, k=16.0, regress=0.3, base=1500.0) -> pd.DataFrame:
    """Pre-match Elo for team1 (bats first) and team2, from strictly earlier matches only.

    Every match with a recorded winner updates ratings (including D/L matches and super-over
    deciders, because the *result* is real evidence of strength even if we do not model the
    ball-by-ball). Ratings are pulled 30% toward 1500 at each new season (squads change).
    """
    order = matches.sort_values(["date", "match_number", "match_id"])
    r, last_season, out = {}, None, {}
    for row in order.itertuples():
        if last_season is not None and row.season != last_season:
            r = {t: base + (1 - regress) * (v - base) for t, v in r.items()}
        last_season = row.season
        a, b = row.team1, row.team2
        ra, rb = r.get(a, base), r.get(b, base)
        out[row.match_id] = (ra, rb)
        win = row.winner if isinstance(row.winner, str) else (
            row.eliminator if isinstance(row.eliminator, str) else None)
        if win in (a, b):
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            sa = 1.0 if win == a else 0.0
            r[a] = ra + k * (sa - ea)
            r[b] = rb - k * (sa - ea)
    return pd.DataFrame([(i, *v) for i, v in out.items()], columns=["match_id", "elo_team1", "elo_team2"])


def venue_league_priors(matches: pd.DataFrame, shrink=3.0, window=40) -> pd.DataFrame:
    """Leak-free (strictly earlier matches only) scoring-environment features per match.

    venue_avg_score : shrunk mean first-innings score at the (canonical) ground
    venue_chase_win : shrunk share of matches won by the chasing side at that ground
    league_avg_l40  : mean first-innings score over the previous 40 included matches
                      (tracks drift such as the 2023 Impact Player rule)
    """
    inc = matches[matches.included].sort_values(["date", "match_number", "match_id"])
    hist, venue_hist, league = [], {}, []
    rows = {}
    for r in inc.itertuples():
        g_avg = np.mean([x[0] for x in hist]) if hist else 160.0
        g_chase = np.mean([x[1] for x in hist]) if hist else 0.5
        vh = venue_hist.get(r.venue, [])
        n = len(vh)
        v_avg = (sum(x[0] for x in vh) + shrink * g_avg) / (n + shrink)
        v_chase = (sum(x[1] for x in vh) + shrink * g_chase) / (n + shrink)
        l40 = np.mean([x[0] for x in hist[-window:]]) if hist else 160.0
        rows[r.match_id] = (v_avg, v_chase, n, l40)
        rec = (r.score1, float(r.chase_won))
        hist.append(rec)
        venue_hist.setdefault(r.venue, []).append(rec)
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["venue_avg_score", "venue_chase_win", "venue_n", "league_avg_l40"])
    out.index.name = "match_id"
    return out.reset_index()


# ----------------------------------------------------------------------------- main
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    balls = load_balls()
    info = load_info()
    print(f"loaded {len(balls):,} delivery rows, {info.match_id.nunique()} matches")

    states, term = innings_states(balls)

    # ---- match table: teams, totals, flags
    t1 = term[term.innings == 1].set_index("match_id")
    t2 = term[term.innings == 2].set_index("match_id")
    m = info.copy()
    m = m.join(t1[["batting_team", "runs_final", "wickets_final", "n_legal"]]
               .add_suffix("_1").rename(columns={"batting_team_1": "team1"}), on="match_id")
    m = m.join(t2[["batting_team", "runs_final", "wickets_final", "n_legal"]]
               .add_suffix("_2").rename(columns={"batting_team_2": "team2"}), on="match_id")
    m = m.rename(columns={"runs_final_1": "score1", "runs_final_2": "score2",
                          "wickets_final_1": "wkts1", "wickets_final_2": "wkts2",
                          "n_legal_1": "legal1", "n_legal_2": "legal2"})

    reason = pd.Series("", index=m.index)
    has_winner = m.winner.notna()
    reason[(reason == "") & (m.get("method") == "D/L")] = "duckworth_lewis"
    reason[(reason == "") & (m.target_overs.astype(str) != "20") & m.target_overs.notna()] = "reduced_overs"
    reason[(reason == "") & (m.outcome == "tie")] = "tie_super_over"
    reason[(reason == "") & (m.outcome == "no result")] = "no_result"
    reason[(reason == "") & ~has_winner] = "no_winner_other"
    reason[(reason == "") & (m.legal1.isna() | m.legal2.isna())] = "missing_innings"
    # complete-innings sanity: innings 1 must be 20 overs or all out; chase must end legitimately
    inn1_ok = (m.legal1 == BALLS_PER_INNINGS) | (m.wkts1 >= 10)
    reason[(reason == "") & ~inn1_ok] = "truncated_innings1"
    m["exclusion_reason"] = reason
    m["included"] = reason == ""
    m["chase_won"] = (m.winner == m.team2)
    m["bat_first_won"] = (m.winner == m.team1)

    # sanity: computed result (score2 >= score1+1) must agree with recorded winner
    inc = m[m.included]
    computed_chase_won = inc.score2 > inc.score1
    mism = inc[computed_chase_won != inc.chase_won]
    print(f"result cross-check mismatches (included matches): {len(mism)}")
    if len(mism):
        print(mism[["match_id", "date", "team1", "team2", "score1", "score2", "winner"]])
        m.loc[mism.index, "included"] = False
        m.loc[mism.index, "exclusion_reason"] = "score_vs_winner_mismatch"

    m = m.merge(elo_priors(m), on="match_id", how="left")
    m = m.merge(venue_league_priors(m), on="match_id", how="left")
    m["toss_chose_bat"] = (m.toss_decision == "bat")
    m["impact_era"] = (m.season >= IMPACT_PLAYER_FIRST_SEASON).astype(int)

    # ---- delivery table
    keep = m[m.included].set_index("match_id")
    s = states[states.match_id.isin(keep.index)].copy()
    s = s.join(keep[["season", "date", "venue", "team1", "team2", "winner", "score1", "toss_winner",
                     "toss_chose_bat", "elo_team1", "elo_team2", "venue_avg_score", "venue_chase_win",
                     "venue_n", "league_avg_l40", "impact_era", "match_number"]], on="match_id")
    s["bat_team_wins"] = (s.batting_team == s.winner).astype(int)
    s["balls_left"] = BALLS_PER_INNINGS - s.balls_bowled
    s["wickets_left"] = 10 - s.wickets
    s["crr_raw"] = np.where(s.balls_bowled > 0, s.runs * 6.0 / s.balls_bowled.clip(lower=1), 0.0)
    s["crr"] = 6.0 * (s.runs + s.league_avg_l40 / 120.0 * CRR_PRIOR_BALLS) / (s.balls_bowled + CRR_PRIOR_BALLS)
    s["target"] = np.where(s.innings == 2, s.score1 + 1, np.nan)
    s["runs_req"] = s.target - s.runs
    s["rrr"] = s.runs_req * 6.0 / s.balls_left
    s["rrr_minus_crr"] = s.rrr - s.crr
    s["proj_score"] = s.runs + s.crr * s.balls_left / 6.0
    # environment-relative versions (scoring drifts upward over the years, e.g. Impact Player rule)
    s["rrr_rel"] = s.rrr * 20.0 / s.league_avg_l40
    s["crr_rel"] = s.crr * 20.0 / s.league_avg_l40
    s["proj_rel"] = s.proj_score / s.league_avg_l40
    s["runs_req_rel"] = s.runs_req / s.league_avg_l40
    first = s.innings == 1
    s["elo_diff"] = np.where(first, s.elo_team1 - s.elo_team2, s.elo_team2 - s.elo_team1)
    s["bat_won_toss"] = (s.batting_team == s.toss_winner).astype(int)
    s["toss_chose_bat"] = s.toss_chose_bat.astype(int)
    s["over"] = s.balls_bowled // 6 + 1
    s["ball_in_over"] = s.balls_bowled % 6 + 1
    s["phase"] = pd.cut(s.balls_bowled, [-1, 35, 89, 119], labels=["powerplay", "middle", "death"]).astype(str)
    s = s.rename(columns={"venue": "venue_name"})
    # runs / wickets that occurred in this ball's window (the legal ball + any wides/no-balls after it)
    s = s.rename(columns={"runs_w": "ball_runs", "wk_w": "ball_wkts"})
    s = s.drop(columns=["R0", "W0", "w"], errors="ignore")
    s = s.sort_values(["date", "match_number", "match_id", "innings", "k"]).reset_index(drop=True)

    t = term[term.match_id.isin(keep.index)].join(keep[["season", "winner", "score1"]], on="match_id")
    t["bat_team_wins"] = (t.batting_team == t.winner).astype(int)

    m.to_parquet(PROCESSED / "matches.parquet", index=False)
    s.to_parquet(PROCESSED / "deliveries.parquet", index=False)
    t.to_parquet(PROCESSED / "terminals.parquet", index=False)

    # ---- report
    print("\nExclusions (all seasons):")
    print(m.exclusion_reason.replace("", "INCLUDED").value_counts().to_string())
    print(f"\nincluded matches: {int(m.included.sum())}, legal deliveries (innings 1-2): {len(s):,}")
    summary = (s.groupby("season").agg(matches=("match_id", "nunique"), rows=("k", "size")).T)
    print(summary.to_string())
    excl = m[~m.included].groupby(["season", "exclusion_reason"]).size().unstack(fill_value=0)
    excl.to_csv(PROCESSED.parent.parent / "results" / "exclusions_by_season.csv")
    m.groupby("exclusion_reason").size().rename("matches").to_csv(
        PROCESSED.parent.parent / "results" / "exclusions_summary.csv")


if __name__ == "__main__":
    main()

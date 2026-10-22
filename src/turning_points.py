"""Ball-by-ball win-probability traces and turning-point detection.

For every match we build ONE continuous series from the point of view of the team that batted first
("team 1"):   P1[j]  = P(team 1 wins) in the state before ball j     (j = 0 .. N1+N2-1)
              P1[N1+N2] = the real result (1 or 0)
Innings-1 states use the innings-1 model's P(batting side wins); innings-2 states use 1 - the
innings-2 model's P(batting side wins). The ball j "moves" the series from P1[j] to P1[j+1].

Known wrinkle: at the innings break the two models hand over to each other, so the step between
the last ball of innings 1 and the first ball of innings 2 is a model hand-over, not a cricket
event. We measure it (`handover_gap`) and exclude that step from turning points.
"""
import numpy as np
import pandas as pd

from config import ABBR


def _ball_label(row):
    return f"{int(row.balls_bowled // 6)}.{int(row.balls_bowled % 6) + 1}"


def _event(row):
    """Classify what happened on the ball and write a short human-readable description."""
    who = row.striker
    if row.ball_wkts > 0:
        how = row.wicket_type if isinstance(row.wicket_type, str) else "wicket"
        out = row.player_dismissed if isinstance(row.player_dismissed, str) else who
        return "wicket", f"WICKET: {out} {how}, b {row.bowler}"
    if row.runs_off_bat == 6:
        return "six", f"SIX: {who} off {row.bowler}"
    if row.runs_off_bat == 4:
        return "four", f"FOUR: {who} off {row.bowler}"
    if row.noballs > 0:
        return "extra", f"No-ball ({int(row.ball_runs)} runs) by {row.bowler}"
    if row.wides > 0:
        return "extra", f"Wide ({int(row.ball_runs)} runs) by {row.bowler}"
    if row.ball_runs == 0:
        return "dot", f"Dot ball, {row.bowler} to {who}"
    return "runs", f"{int(row.ball_runs)} run(s), {who} off {row.bowler}"


def build_match_series(d, term, match_row):
    """d: delivery rows of ONE match (both innings) with column `p` (batting-side win prob)."""
    d = d.sort_values(["innings", "k"]).reset_index(drop=True)
    inn = d.innings.values
    p_bat = d.p.values
    p1 = np.where(inn == 1, p_bat, 1 - p_bat)
    t1_won = float(match_row.winner == match_row.team1)
    P = np.append(p1, t1_won)                                  # length N1 + N2 + 1
    n1 = int((inn == 1).sum())
    swing = np.diff(P)                                         # swing[j] = effect of ball j
    swing[n1 - 1] = np.nan                                     # model hand-over, not a cricket event
    ev = [_event(r) for r in d.itertuples()]
    out = d[["innings", "k", "balls_bowled", "runs", "wickets", "ball_runs", "ball_wkts",
             "striker", "bowler"]].copy()
    out["ball"] = [_ball_label(r) for r in d.itertuples()]
    out["p_before"] = P[:-1]
    out["p_after"] = P[1:]
    out["swing"] = swing
    out["event_type"] = [e[0] for e in ev]
    out["event_text"] = [e[1] for e in ev]
    out["match_id"] = int(d.match_id.iloc[0])
    return out, P, n1


MIN_BALLS = 12   # the first two overs of each innings are too unstable to call a "turning point" (DECISIONS #14)


def pick_turning_points(ser, top=3, min_swing=0.04, gap=4):
    """Top-|swing| balls, greedily, skipping balls within `gap` of one already chosen (same innings)."""
    cand = (ser[ser.balls_bowled >= MIN_BALLS].dropna(subset=["swing"])
            .assign(a=lambda x: x.swing.abs()).sort_values("a", ascending=False))
    chosen = []
    for r in cand.itertuples():
        if r.a < min_swing or len(chosen) >= top:
            break
        if all(not (r.innings == c.innings and abs(r.k - c.k) <= gap) for c in chosen):
            chosen.append(r)
    return [c.Index for c in chosen]


def over_swings(ser):
    """Win-probability change per over (sum of ball swings), with runs/wickets in the over."""
    s = ser.copy()
    s["over"] = (s.balls_bowled // 6) + 1
    g = s.groupby(["innings", "over"])
    o = g.agg(swing=("swing", "sum"), runs=("ball_runs", "sum"), wkts=("ball_wkts", "sum"),
              bowler=("bowler", lambda x: x.mode().iat[0]), n=("k", "size"),
              first_ball=("ball", "first"), match_id=("match_id", "first"))
    return o.reset_index()


def swing_label(swing, t1, t2):
    """'DC +18.2 pts' -- which team gained, and by how many win-probability points."""
    gainer = t1 if swing > 0 else t2
    return f"{ABBR.get(gainer, gainer)} +{abs(swing) * 100:.1f} pts"

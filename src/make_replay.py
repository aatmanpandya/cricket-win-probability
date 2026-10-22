"""Turning points + match replays for the TEST seasons.

Reads results/test_predictions.parquet (model chosen in REPLAY_MODEL), writes
  results/test_turning_points.csv   top turning point(s) of every test match
  results/test_biggest_swings.csv   biggest single-ball swings overall
  results/test_biggest_overs.csv    biggest single-over swings overall
  results/handover_gap.csv          innings-break consistency diagnostic
  demo/replay_data.json             data embedded into demo/index.html
  reports/figures/replay_*.png      replay charts for 3 selected matches
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ABBR, DEMO, FIGURES, PROCESSED, RESULTS
from turning_points import MIN_BALLS, build_match_series, over_swings, pick_turning_points, swing_label

REPLAY_MODEL = "GBM + Platt"     # GBM fitted on <=2023, Platt recalibration fitted on 2024
BLUE, ORANGE, GRID, INK, MUTED = "#2a78d6", "#eb6834", "#d9d8d3", "#0b0b0b", "#52514e"


def main():
    pred = pd.read_parquet(RESULTS / "test_predictions.parquet")
    deliv = pd.read_parquet(PROCESSED / "deliveries.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet").set_index("match_id")
    d = deliv.merge(pred[["match_id", "innings", "k", REPLAY_MODEL]], on=["match_id", "innings", "k"]).rename(columns={REPLAY_MODEL: "p"})
    assert d.match_id.nunique() == pred.match_id.nunique()

    all_ser, tp_rows, over_rows, gap_rows, payload, store = [], [], [], [], [], {}
    for mid, g in d.groupby("match_id", sort=False):
        mr = matches.loc[mid]
        ser, P, n1 = build_match_series(g, None, mr)
        ser["match_id"] = mid
        t1, t2 = mr.team1, mr.team2
        picks = pick_turning_points(ser, top=3)
        for rank, ix in enumerate(picks, 1):
            r = ser.loc[ix]
            tp_rows.append({"match_id": mid, "season": int(mr.season), "date": str(mr.date.date()), "match": f"{ABBR[t1]} v {ABBR[t2]}",
                            "rank": rank, "innings": int(r.innings), "ball": r.ball, "event_type": r.event_type, "event": r.event_text,
                            "p_before_t1": r.p_before, "p_after_t1": r.p_after, "swing_t1": r.swing,
                            "gain": swing_label(r.swing, t1, t2), "state": f"{int(r.runs)}/{int(r.wickets)}"})
        o = over_swings(ser)
        o["match"] = f"{ABBR[t1]} v {ABBR[t2]}"; o["season"] = int(mr.season); o["date"] = str(mr.date.date())
        over_rows.append(o)
        gap_rows.append({"match_id": mid, "season": int(mr.season), "handover_gap": abs(P[n1] - P[n1 - 1])})
        all_ser.append(ser.assign(season=int(mr.season), match=f"{ABBR[t1]} v {ABBR[t2]}", date=str(mr.date.date())))
        store[mid] = (ser, P, n1, picks)

        # ---- JSON payload for the HTML replay
        evs = {}
        for j, r in enumerate(ser.itertuples()):
            if r.event_type in ("wicket", "six", "four"):
                evs[j] = r.event_text
        sc = lambda inn: (lambda x: f"{int(x.runs + x.ball_runs)}/{int(x.wickets + x.ball_wkts)}")(ser[ser.innings == inn].iloc[-1])
        res = (f"{ABBR[mr.winner]} won by {mr.winner_runs} runs" if mr.winner == t1 else f"{ABBR[mr.winner]} won by {mr.winner_wickets} wickets")
        payload.append({
            "id": int(mid), "date": str(mr.date.date()), "season": int(mr.season), "venue": mr.venue,
            "t1": ABBR[t1], "t2": ABBR[t2], "t1f": t1, "t2f": t2, "s1": sc(1), "s2": sc(2), "result": res,
            "toss": f"{ABBR[mr.toss_winner]} won the toss and chose to {'bat' if mr.toss_chose_bat else 'field'}",
            "n1": n1, "p": [int(round(x * 1000)) for x in P],
            "r": [int(x) for x in ser.ball_runs], "w": [int(x) for x in ser.ball_wkts],
            "ball": list(ser.ball), "ev": evs,
            "tp": [{"i": int(ser.index.get_loc(ix)), "text": ser.loc[ix, "event_text"], "swing": round(float(ser.loc[ix, "swing"]), 3),
                    "gain": swing_label(ser.loc[ix, "swing"], t1, t2), "ball": f'{int(ser.loc[ix, "innings"])}st/nd {ser.loc[ix, "ball"]}'.replace("1st/nd", "Inn 1,").replace("2st/nd", "Inn 2,")}
                   for ix in picks],
        })

    tp = pd.DataFrame(tp_rows)
    tp.to_csv(RESULTS / "test_turning_points.csv", index=False)
    S = pd.concat(all_ser).dropna(subset=["swing"])
    S = S[S.balls_bowled >= MIN_BALLS]
    S["abs_swing"] = S.swing.abs()
    top = S.sort_values("abs_swing", ascending=False).head(25)
    cols = ["season", "date", "match", "innings", "ball", "runs", "wickets", "event_type", "event_text", "p_before", "p_after", "swing"]
    top[cols].to_csv(RESULTS / "test_biggest_swings.csv", index=False)
    O = pd.concat(over_rows)
    O = O[O.over >= 3]
    O["abs_swing"] = O.swing.abs()
    O.sort_values("abs_swing", ascending=False).head(25).to_csv(RESULTS / "test_biggest_overs.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(RESULTS / "handover_gap.csv", index=False)

    # what kind of ball causes the largest swings?
    big = S.sort_values("abs_swing", ascending=False).head(200)
    kinds = big.event_type.value_counts(normalize=True).rename("share_of_top200_swings")
    base = S.event_type.value_counts(normalize=True).rename("share_of_all_balls")
    pd.concat([kinds, base], axis=1).fillna(0).to_csv(RESULTS / "swing_event_mix.csv")
    S.groupby("event_type").abs_swing.agg(["mean", "median", "size"]).to_csv(RESULTS / "swing_by_event_type.csv")

    # --------------------------------------------------------------- embed data in the demo
    payload.sort(key=lambda m: (m["date"], m["id"]))
    DEMO.mkdir(exist_ok=True)
    json.dump(payload, open(DEMO / "replay_data.json", "w"), separators=(",", ":"))
    print(f"replay data: {len(payload)} matches")

    # --------------------------------------------------------------- pick 3 matches for PNGs
    stats = []
    for mid, (ser, P, n1, picks) in store.items():
        mr = matches.loc[mid]
        w1 = mr.winner == mr.team1
        pw = P[:-1] if w1 else 1 - P[:-1]            # winner's win-prob path
        cross = int(((P[1:n1 + 1 + 1][:-1] - 0.5) * (P[2:n1 + 2] - 0.5) < 0).sum()) if False else int((np.diff(np.sign(P[12:-1] - 0.5)) != 0).sum())
        stats.append({"match_id": mid, "min_winner_p": float(pw[12:].min()), "max_abs_swing": float(np.nanmax(np.abs(np.diff(P)[np.arange(len(P) - 1) != n1 - 1]))),
                      "crossings": cross, "margin_close": (mr.score1 - mr.score2 if w1 else 0)})
    st = pd.DataFrame(stats)
    comeback = int(st.sort_values("min_winner_p").match_id.iloc[0])
    swingy = int(st[st.match_id != comeback].sort_values("max_abs_swing", ascending=False).match_id.iloc[0])
    seesaw = int(st[~st.match_id.isin([comeback, swingy])].sort_values("crossings", ascending=False).match_id.iloc[0])
    chosen = {"comeback": comeback, "biggest_swing": swingy, "seesaw": seesaw}
    json.dump(chosen, open(RESULTS / "replay_png_matches.json", "w"))
    for tag, mid in chosen.items():
        ser, P, n1, picks = store[mid]
        plot_match(matches.loc[mid], ser, P, n1, picks, FIGURES / f"replay_{tag}.png")
        print(tag, mid, matches.loc[mid, ["date", "team1", "team2"]].tolist())


def plot_match(mr, ser, P, n1, picks, path):
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
    x = np.arange(len(P))
    t1, t2 = ABBR[mr.team1], ABBR[mr.team2]
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.fill_between(x, 0.5, P, where=P >= 0.5, color=BLUE, alpha=0.16, lw=0, interpolate=True)
    ax.fill_between(x, 0.5, P, where=P <= 0.5, color=ORANGE, alpha=0.16, lw=0, interpolate=True)
    ax.plot(x[:n1], P[:n1], color=INK, lw=1.8, zorder=3)
    ax.plot(x[n1:], P[n1:], color=INK, lw=1.8, zorder=3)
    ax.plot(x[n1 - 1:n1 + 1], P[n1 - 1:n1 + 1], color=INK, lw=1, ls=":", zorder=3)
    ax.axvline(n1, color=MUTED, lw=1, ls="--")
    ax.text(n1 / 2, 1.02, f"Innings 1: {t1} batting", ha="center", color=MUTED, fontsize=10)
    ax.text(n1 + (len(P) - n1) / 2, 1.02, f"Innings 2: {t2} chasing", ha="center", color=MUTED, fontsize=10)
    w = np.where(ser.ball_wkts.values > 0)[0] + 1
    ax.scatter(w, P[w], s=34, color="#e34948", edgecolor="white", linewidth=1, zorder=5, label="Wicket")
    six = np.where(ser.event_type.values == "six")[0] + 1
    ax.scatter(six, P[six], s=26, color="#1baf7a", edgecolor="white", linewidth=1, zorder=5, marker="D", label="Six")
    for n, ix in enumerate(picks, 1):
        j = ser.index.get_loc(ix) + 1
        ax.scatter([j], [P[j]], s=200, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=6)
        ax.text(j, P[j], str(n), ha="center", va="center", fontsize=9, fontweight="bold", zorder=7)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(P) - 1)
    tk = [(0, "0"), (36, "6"), (90, "15"), (n1, "20 ov | 0"), (n1 + 36, "6"), (n1 + 90, "15"), (len(P) - 1, "end")]
    tk = [(x_, l_) for x_, l_ in tk if x_ < len(P)]
    ax.set_xticks([a for a, _ in tk])
    ax.set_xticklabels([b for _, b in tk], color=MUTED)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels([f"{t2} 100%", "75", "50", "75", f"{t1} 100%"] if False else ["0%", "25%", "50%", "75%", "100%"], color=MUTED)
    ax.set_ylabel(f"P({t1} wins)", color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    res = f"{ABBR[mr.winner]} won by {mr.winner_runs} runs" if mr.winner == mr.team1 else f"{ABBR[mr.winner]} won by {mr.winner_wickets} wickets"
    ax.set_title(f"{t1} v {t2}, {mr.date.date()} - {res}\n{mr.venue}", loc="left", fontsize=12, color=INK, pad=24)
    lines = []
    for n, ix in enumerate(picks, 1):
        r = ser.loc[ix]
        lines.append(f"{n}. Inn {int(r.innings)}, over {r.ball} ({int(r.runs)}/{int(r.wickets)}): {r.event_text}  "
                     f"[{swing_label(r.swing, mr.team1, mr.team2)}, win prob {r.p_before * 100:.0f}% -> {r.p_after * 100:.0f}% for {t1}]")
    fig.text(0.07, 0.01, "\n".join(lines), fontsize=8.5, color=MUTED, va="bottom", ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08 + 0.035 * len(lines))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

# IPL ball-by-ball win probability

A live win-probability model for the IPL, in the spirit of the win predictor shown during broadcasts, built to be **honestly evaluated**: it is trained on past seasons, tuned on 2024, and scored once on the two seasons it has never seen (2025 and 2026). The question is not "what is the accuracy" but "when the model says 70%, do teams win 70% of the time, and does that hold up on future seasons?"

**[Open the match replay demo](demo/index.html)** (a single self-contained HTML file; double-click it, no server needed). Pick any of the 138 held-out matches, see the win-probability line ball by ball, wickets and sixes marked, and the three biggest turning points annotated.

![Replay of PBKS v KKR, 2025-04-15: PBKS defend 111](reports/figures/replay_comeback.png)

*PBKS v KKR, 15 April 2025. PBKS were bowled out for 111 and the model gave them almost no chance; KKR were bowled out for 95 and PBKS won by 16 runs. The model never saw this match.*

## What was built

| Piece | Where |
|---|---|
| Data pipeline (Cricsheet ball-by-ball to one row per legal delivery) | `src/build_dataset.py` |
| Model selection on past seasons only (backtests, ablations, tuning) | `src/run_validation.py` |
| The one-off test on 2025-26, with bootstrap confidence intervals | `src/final_test.py` |
| Turning-point detection and replay data | `src/turning_points.py`, `src/make_replay.py` |
| Static HTML replay | `demo/index.html` |
| Evaluation notebook (executed, with charts) | `notebooks/evaluation.ipynb` |
| Every decision, with the numbers behind it | `DECISIONS.md` |

## Data

Ball-by-ball and match data for all IPL matches from [Cricsheet](https://cricsheet.org/) (Open Database License; please credit Cricsheet if you reuse anything here). 1,243 matches from 2008 to 2026.

Matches that cannot be fairly labelled by "who won the chase" were excluded and counted:

| Reason | Matches |
|---|---|
| Duckworth-Lewis / rain-revised target | 23 |
| Tie decided by a super over | 16 |
| Other reduced-overs matches | 11 |
| No result | 8 |
| First innings cut short | 3 |
| **Kept** | **1,182** (273,644 legal deliveries) |

Splits are by season, never by ball:

| Split | Seasons | Matches | Deliveries |
|---|---|---|---|
| Train | 2008-2023 | 974 | 225,926 |
| Validation (tuning, calibration) | 2024 | 70 | 16,107 |
| **Test** | 2025-2026 | 138 | 31,611 |

2026 is a complete season in the data (70 of its 74 matches kept; 4 excluded as no-result / tie / reduced-overs / D/L).

## Method in one page

* **State before the ball.** Each row is the situation just before a legal delivery: runs, wickets, balls left, smoothed run rate and, in the chase, the target, runs required and required run rate. The label is whether the team batting in that innings goes on to win.
* **One model per innings.** The questions are different (an open-ended projection versus a chase with a target) and separate models beat a joint model with an innings flag on every backtest fold (log loss 0.534 vs 0.544).
* **Baseline:** logistic regression on a handful of raw features. **Main model:** LightGBM with shallow trees, boosted *from* the logistic-regression logit, plus the ground's historical scoring level. Hyper-parameters were chosen on rolling-origin backtests over 2021-2024.
* **Metrics:** log loss and Brier score (both reward honest probabilities), ROC-AUC, expected calibration error (ECE) and reliability diagrams, overall and by innings and phase. Confidence intervals come from resampling whole matches, since the balls of one match are not independent.

## Results on the held-out 2025-26 seasons

| Model | Log loss | Brier | ROC-AUC | ECE |
|---|---|---|---|---|
| Naive prior (win rate by innings) | 0.686 | 0.247 | 0.569 | 0.039 |
| Pre-match team-strength (Elo) only | 0.687 | 0.247 | 0.572 | 0.053 |
| Logistic regression (baseline) | 0.541 | 0.185 | 0.797 | 0.043 |
| LightGBM | 0.546 | 0.187 | 0.798 | 0.056 |
| LightGBM + Platt (fitted on 2024) | 0.530 | 0.179 | 0.815 | 0.057 |
| LightGBM + isotonic (fitted on 2024) | 0.560 | 0.185 | 0.802 | 0.071 |
| Logistic regression + Platt (fitted on 2024) | 0.519 | 0.175 | 0.820 | 0.046 |

Lower is better for log loss, Brier and ECE. A coin flip scores 0.693 log loss.

Match-level bootstrap 95% intervals (138 matches): LightGBM + Platt minus naive prior is -0.216 to -0.088 in log loss, so both real models are clearly better than the prior. LightGBM minus logistic regression is -0.006 to +0.018, and LightGBM + Platt minus logistic regression is -0.030 to +0.011: **a tie, not a win**.

By stage (log loss, test seasons):

| | Naive | Logistic regression | LightGBM | LightGBM + Platt |
|---|---|---|---|---|
| Innings 1 | 0.684 | 0.673 | 0.690 | 0.673 |
| Innings 2 (chase) | 0.689 | 0.397 | 0.392 | 0.377 |
| Powerplay | 0.685 | 0.615 | 0.608 | 0.586 |
| Middle overs | 0.686 | 0.518 | 0.532 | 0.519 |
| Death overs | 0.689 | 0.482 | 0.488 | 0.476 |

![Reliability diagrams](reports/figures/reliability.png)

### What the results say

1. **The chase is predictable; the first innings barely is.** In the chase the model cuts log loss from 0.689 to about 0.38 (AUC 0.91). In the first innings it is no better than a coin flip with a prior (0.673 vs 0.684, AUC 0.70): nobody knows what a total of 190 is worth until they see the chase.
2. **A gradient-boosted model did not beat logistic regression.** After heavy tuning the GBM matched the baseline within noise (and was slightly worse before recalibration). With roughly 1,100 matches and strongly correlated rows, a well-specified parametric model is hard to beat. This is the honest headline.
3. **Calibration is where the time-shift shows.** Models trained up to 2023 were over-optimistic about teams batting first and under-confident about chasers on 2025-26. Platt scaling fitted on 2024 fixed a large part of it (log loss 0.546 to 0.530, chase ECE 0.067 to 0.033). Isotonic regression made things worse because one season is too little data for a free-form calibrator.
4. **Neither toss nor pre-match team strength helped.** Both were tested properly (see `DECISIONS.md`); an Elo rating built from earlier results scored *worse* than the base rate out of time, which is a fair summary of how hard it is to predict IPL matches before they start.
5. **The Impact Player era is a real shift** (average first-innings score rose from 161 in 2008-17 to 190 in 2023-26) but an era flag and recency weighting both *hurt* the backtests. Expressing everything relative to the target and balls left already transfers across eras. A later, exploratory check found features relative to the recent league scoring rate help on 2025-26 (log loss 0.524 for logistic regression versus 0.541), but that idea came from looking at the test seasons, so it is reported as a lead and not as a result.
6. **Monotonic constraints made no measurable difference** (log loss 0.5279 vs 0.5271 on backtests; the unconstrained model almost never violated "more wickets, lower win probability" anyway).

## Biggest turning points in the test seasons

Ball-level swings in the win probability of the team batting first (balls in the first two overs of an innings are excluded because the model is not stable there):

| Match | Ball | What happened | Win prob. (team batting first) |
|---|---|---|---|
| PBKS v KKR, 2025-04-15 | 2nd inns 15.1 | Russell bowled by Jansen, last wicket: PBKS win by 16 | 10% to 100% |
| KKR v LSG, 2026-04-09 | 2nd inns 19.5 | MD Choudhary six off Arora, LSG chase it down | 94% to 5% |
| GT v DC, 2026-04-08 | 2nd inns 19.4 | Miller six off Prasidh Krishna | 75% to 4% |
| MI v RCB, 2026-05-10 | 2nd inns 19.6 | Last ball, 2 runs: RCB win | 70% to 0% |
| RCB v CSK, 2025-05-03 | 2nd inns 19.5 | Jadeja single off Yash Dayal | 39% to 94% |

Wickets are 5% of balls but 57% of the 200 biggest swings; sixes are 9% of balls and 29% of the biggest swings. The largest single over was Avesh Khan's last over in LSG v RR (2025-04-19), a +94-point swing in one over. Full lists: `results/test_biggest_swings.csv`, `results/test_biggest_overs.csv`, `results/test_turning_points.csv`.

## How to run

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
# put the Cricsheet IPL CSV zip contents in data/raw/ipl_csv/  (https://cricsheet.org/downloads/ipl_male_csv2.zip)
./run_all.sh                 # ~10 minutes: dataset, model selection, test, replay, figures, demo, notebook
open demo/index.html         # the match replay
```

Individual steps: `python src/build_dataset.py`, `python src/run_validation.py all`, `python src/final_test.py`, `python src/make_replay.py`, `python src/make_demo.py`. On macOS, LightGBM needs `brew install libomp`.

## Limitations

* **138 test matches is small.** Confidence intervals are wide, and a model that looks 0.01 better is not reliably better. ECE is particularly noisy on this much data.
* **The test seasons are unusual.** 2026 saw the chasing side win 63% of matches, which no ball-state model can anticipate; the "calibrated on 2024" step only partly absorbs that.
* **No player information.** The model does not know who is batting or bowling, which is where most of the missing signal in the first innings probably is.
* **Early-innings estimates are jittery.** The first two overs of each innings move by up to about 8 points between consecutive balls at times; these are excluded from turning points.
* **Innings hand-over.** The two innings use different models, so the line has a small step at the break (mean 8.9 points on the test matches). It is a model artefact, drawn dotted in the demo and excluded from the swings.
* **The test was looked at more than once.** The headline run was executed on a frozen configuration, but I re-ran the full pipeline after fixing a run-rate instability found while inspecting replays, and I added post-hoc diagnostics after seeing the results. Everything is logged in `DECISIONS.md`.
* **Not a betting model**, and not affiliated with the IPL, any broadcaster or Cricsheet.

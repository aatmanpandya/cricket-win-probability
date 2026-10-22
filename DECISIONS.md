# DECISIONS.md: how this project was thought through

A running log of every meaningful choice: what I decided, what else I considered, why, the numbers that settled it, and the dead ends. Written to be read by someone learning how to do this kind of project, so the reasoning is spelled out plainly.

Conventions: "log loss", "Brier" and "ECE" are explained in decision 5. "Backtest folds" means 2021, 2022, 2023 and 2024, each scored by a model trained only on strictly earlier seasons. Numbers in this log come from the files in `results/`; the table name is given in brackets.

---

## 0. The overall plan, and the order of work

**Decision.** Data prep, then splits, then baseline, then gradient boosting, then evaluation and calibration, then turning points, then the HTML replay. Finish each before the next.

**Why.** Every later step depends on the earlier ones being right, and a bug in the labels (for example, wrongly saying who won) silently ruins everything downstream. So I built cross-checks into step 1 (decision 3) and only then modelled.

**The framing that shaped everything.** A win-probability product is a *probability* product. It is shown to viewers as "RCB 63%". The question is therefore not "did you pick the winner?" but "are the percentages honest, and do they stay honest on future seasons?". That is why the evaluation (decisions 5, 6, 12) matters more than the model.

---

## 1. Which data, which files

**Decision.** Use Cricsheet's CSV ball-by-ball files plus the `_info.csv` per match. Ignore the JSON copy.

**Alternatives.** The JSON has the same matches with slightly richer fields.

**Why.** The CSV contains every field I need (striker, bowler, runs off the bat, extras, wides, no-balls, wicket type) and the info file gives toss, venue, winner and the method flag (D/L). Two parsers for the same data would only add bug surface. One wrinkle: `all_matches.csv` sits in the same folder, so I select match files with a strict regex (`^\d+\.csv$`); a naive glob would double count the whole dataset.

**Treated as data.** Downloaded files were only ever read, never executed.

---

## 2. Normalising names and seasons

**Teams.** Delhi Daredevils to Delhi Capitals, Kings XI Punjab to Punjab Kings, Royal Challengers Bangalore to Royal Challengers Bengaluru, Rising Pune Supergiants to Rising Pune Supergiant. 19 spellings collapse to 15 franchises. Deccan Chargers stay separate from Sunrisers Hyderabad (different franchise legally; no model feature depends on it).

**Venues.** The same ground appears under many names ("Wankhede Stadium" and "Wankhede Stadium, Mumbai"; Feroz Shah Kotla became Arun Jaitley Stadium; Sardar Patel Stadium became Narendra Modi Stadium). Rule: drop everything after the first comma, then a small alias table for real renames. 60 raw spellings become 37 grounds. This matters because one of the features (decision 8) is "how much does this ground normally score", which is only meaningful if a ground is one entity.

**Seasons.** Cricsheet labels some seasons "2007/08", "2009/10", "2020/21". I use the calendar year of the first match, which gives the real IPL editions 2008, 2010, 2020 and gives exactly 19 seasons totalling 1,243 matches.

---

## 3. What is one row, and what is excluded?

**Decision.** One row per *legal* delivery (wides and no-balls are not legal deliveries). The row holds the match state **before** the ball. Label: 1 if the team batting in that innings wins the match.

**State before vs after.** Before is the natural definition for a predictor ("here is the situation, what will happen?") and it means a row never contains the outcome of its own ball. Wides and no-balls bowled since the previous legal ball are already in the state: they add runs (and can carry a run-out) but do not use up a ball. For the replay I additionally keep a terminal state after the final ball, set to the real result; it is never used in training or scoring.

**Exclusions** [`exclusions_summary.csv`]. Of 1,243 matches:

| Reason | Matches |
|---|---|
| Duckworth-Lewis method | 23 |
| Tie decided by super over | 16 |
| Reduced overs without the D/L tag (target_overs != 20) | 11 |
| No result | 8 |
| First innings cut short | 3 |
| Kept | 1,182 |

Why drop them: in a rain-reduced match the target and overs are rewritten mid-game, so "runs required over balls left" means something else; a super-over tie has no winner in the 20-over game. Kept matches give 273,644 legal deliveries in innings 1 and 2.

**Cross-check built in.** For every kept match I recompute the winner from the scores (second-innings total greater than first) and compare it with the recorded winner. Mismatches: 0. That catches off-by-one errors in the target, in extras (penalty runs are included in `extras`), and in team mapping.

**Window bookkeeping.** Wides after a legal ball belong to the window of that ball when attributing swings (decision 14), so the swing "caused by" ball k includes any wide/no-ball that followed it.

---

## 4. Splits: by season, never by ball

**Decision.** Train on seasons up to 2023, validate/tune/calibrate on 2024, test once on 2025 and 2026 [`split_sizes.csv`]:

| Split | Matches | Deliveries |
|---|---|---|
| Train (to 2023) | 974 | 225,926 |
| Validation (2024) | 70 | 16,107 |
| Test (2025: 68, 2026: 70) | 138 | 31,611 |

2026 is a complete season in the data (74 scheduled, 70 kept).

**Why split by season.** The product's job is to predict *future* matches. Cricket changes over time (the Impact Player rule, bigger scores, new grounds), so a random split would be optimistic: a model that has seen other balls of the same match, or the same season, is cheating. **Why never by ball:** the 240 balls of one match share a label and share match-level context. Put some in train and some in test and the model just memorises the match. Everything here is split by match, and in fact by season.

**Rolling-origin backtests.** 2024 alone is only 70 matches, far too noisy to choose between models. So during selection I also score 2021, 2022 and 2023, each with a model trained on strictly earlier seasons. All selection used these four folds. The 2025-26 seasons are never loaded by `run_validation.py` (it filters to `season <= 2024` at load time).

---

## 5. Which metrics, and why accuracy is not one of them

* **Log loss** (cross-entropy). Punishes confident wrong answers heavily. Saying 99% and being wrong costs far more than saying 60% and being wrong. This is the natural score for a probability.
* **Brier score.** Mean squared error of the probability; gentler on extremes, easy to interpret.
* **ROC-AUC.** Does the model rank winning states above losing ones? Useful but blind to calibration: multiplying every probability by a constant does not change it.
* **ECE and reliability diagrams.** Group predictions into 10 bins; in each bin compare the average predicted probability with how often teams actually won. A calibrated model's bins sit on the diagonal.
* **Accuracy is reported nowhere.** If a team needs 6 off 6 balls and the model says 52%, accuracy cannot tell a good answer (52%) from a bad one (90%). A viewer-facing product lives or dies on the numbers themselves.

**Reference points.** A coin flip scores log loss 0.693. The naive prior (training win rate by innings) scores 0.686 on test. Anything claiming skill must beat that clearly.

**Honest caveats on the metrics.** (a) Rows within a match are strongly correlated, so the effective sample size is matches (138 in test), not 31,611 rows. I therefore bootstrap by *match* for every confidence interval. (b) ECE is biased upwards on small samples and swings a lot (per-season backtest ECEs of 0.03 to 0.10 for the same model). I report it, but trust log loss and Brier first.

**By phase.** Powerplay = overs 1-6 (balls 0-35 bowled before the ball), middle = 7-15, death = 16-20.

---

## 6. One model per innings, not a joint model

**Question.** One model with an `innings` flag, or two?

**Numbers** [`val_structure.csv`, backtest log loss]: separate 0.5338 vs joint 0.5442; better in all four folds.

**Why separate wins.** The two situations are structurally different. In the first innings there is no target (the feature is missing); in the second there is a target and everything is measured against it. A joint tree has to spend splits on separating them, and shared features take different meanings. Two small models are simpler, easier to monitor and more accurate.

**Cost.** The two models hand over at the innings break, so the probability has a small step there (mean 8.9 points on test matches, max 28; `handover_gap.csv`). I draw that segment dotted in the demo and exclude it from turning points.

---

## 7. The first GBM was terrible, and why [dead end with a lesson]

**What happened.** My first LightGBM (all features, 400 trees) scored log loss **0.743** on the backtests, worse than the 0.695 naive prior, versus 0.537 for plain logistic regression [`val_baselines.csv`]. On 2024 alone it was 0.924.

**Diagnosis.** Several features are constant within a match (venue average score, Elo, toss) and in the chase the *target* is too. Every one of a match's ~120 rows carries the same values, so a tree can isolate one match by those values and learn "this match was won by the chaser". That is memorisation, and it does not transfer. The tell-tale sign was validation loss turning up after only ~70 trees.

**Fixes that worked.**
* Drop `target` and `runs` from the innings-2 features. `runs_required`, `balls_left`, `wickets` and run rates already express the situation, and `target`, `runs` and `runs_required` together reveal the match identity.
* Very shallow trees (3 leaves) and large leaves (300+ rows), heavy L2 regularisation.
* Test each extra feature group against a pre-agreed rule (decision 8).

**Result.** Regularised state-only GBM: 0.5363 vs logistic regression 0.5364. A tie, which is the first hint of a recurring theme (decision 10).

**Lesson for the guide.** With grouped data, rows are not independent. Anything constant within a group is a group identifier in disguise.

---

## 8. Which features earn their place

**Rule, decided before running the ablation.** A feature group is kept only if adding it lowers backtest log loss by at least 0.001 on average *and* in at least 3 of the 4 folds [`val_feature_ablation.csv`]. This stops me from keeping features because of one lucky fold.

| Group | Innings 1 mean change | Innings 2 mean change | Kept? |
|---|---|---|---|
| Recent form (runs/wickets in last 12 and 24 balls) | -0.0006 | -0.0003 | No |
| Features relative to league scoring | -0.0003 | -0.0008 | No |
| **Ground's historical average score and chase win rate** | **-0.0027** | **-0.0026** | **Yes (3 of 4 folds)** |
| League average score, last 40 matches | +0.0059 | +0.0024 | No (hurts) |
| Toss winner / decision | +0.0002 | +0.0009 | No |
| Pre-match Elo difference | -0.0016 | +0.0050 | No (hurts, 0 of 4 folds in innings 2) |
| `impact_era` flag | +0.0023 | +0.0011 | No (hurts) |

(Negative = better.)

**Surprising and worth saying plainly.**
* **Toss does not matter** once you know the score and the ground. This fits the usual finding that the toss is worth very little in T20.
* **Pre-match team strength did not help, even on its own.** I built an Elo rating from earlier results (strictly earlier matches only, regressing 30% to the mean each season). As a pure pre-match predictor, scored out of time, it was *worse* than the base rate (0.699 vs 0.695 log loss). I swept K from 4 to 32 and season regression from 0 to 0.6; nothing beat the base rate. IPL squads change every year and matches are close to coin flips; a crude rating cannot find the signal.
* **Recent form (last two to four overs)** adds nothing beyond the current state. Momentum may exist but the state already contains it.
* **The ground helps a bit:** at a high-scoring ground a given score means less.

**Ground features are leak-free by construction.** For each match they use only matches played strictly before it (a shrunk mean, with 3 pseudo-matches of the league average, so grounds with 2 matches do not get extreme values).

---

## 9. Handling the Impact Player era (2023 onwards)

**The shift is real** [`era_scoring_shift.csv`]:

| Era | Matches | Avg 1st-innings score | Avg 2nd-innings | Chasing side wins | Bat first after winning toss |
|---|---|---|---|---|---|
| 2008-17 | 600 | 161 | 148 | 54% | 43% |
| 2018-22 | 303 | 168 | 156 | 56% | 25% |
| 2023-26 | 279 | 190 | 176 | 53% | 23% |

By season the average first innings is 183 (2023), 190 (2024), 192 (2025), 196 (2026).

**What I tested** (backtest log loss, [`val_era.csv`]; baseline = all history, uniform weights = 0.5338):

| Variant | Log loss |
|---|---|
| `impact_era` flag | 0.5356 |
| Recency weights, half-life 4 seasons | 0.5386 |
| Half-life 2 seasons | 0.5439 |
| Half-life 1 season | 0.5526 |
| Train only from 2016 | 0.5465 |
| Train only from 2018 | 0.5499 |

**Decision.** None of them. Using all history with uniform weights was best in the backtests; the more I down-weighted the past, the worse it got.

**Why.** There is little data: a season is ~70 matches. Throwing away old matches costs more in variance than it saves in bias. And the second-innings features are all relative to the target and balls left, which transfers across eras; the cricket of "need 40 off 18 with 3 wickets in hand" has not changed.

**But the test seasons showed the shift anyway** (decision 12 and 13). Records the honest conclusion: the backtest could not see this problem clearly (the 2023 fold was the worst, 0.589, but nothing I tried improved it), and the test seasons exposed it for the first-innings model.

---

## 10. Baseline vs gradient boosting: a tie

**Baseline.** Logistic regression per innings. Innings 1: runs, wickets, balls left, run rate. Innings 2: runs required, wickets, balls left, required run rate, run rate. Standardised, C = 1.

**Structure options tested on the backtests** [`val_structure.csv`]:

| Variant | Log loss |
|---|---|
| GBM, state + ground features | 0.5338 |
| + monotone constraints | 0.5340 |
| 5 GBMs, each bagged over bootstrap samples of whole *matches* | 0.5334 |
| **GBM boosted from the logistic-regression logit** | **0.5297** |
| same + monotone | 0.5309 |

**Decision.** Boost from the LR logit (also called "init score"): LightGBM starts from the logistic-regression prediction and learns only a correction. It was better in all four folds. Why it helps: the LR supplies a smooth, monotone, extrapolating backbone and the trees add non-linearities, and because they only model the residual they have less to memorise.

**Hyper-parameters.** A 15-trial random search over leaves (3-12), min leaf size (300-4000), L2 (1-200), feature fraction, extra-trees on/off, learning rate; number of trees chosen from the averaged backtest validation curve [`val_hparams.csv`]. Winner: 3 leaves, min leaf 300, L2 = 200, lr 0.03, 358 trees (innings 1) and 477 trees (innings 2). The spread across configs was small (combined log loss 0.527 to 0.532): the model is not sensitive to these knobs.

**The honest result.** On the backtests the final GBM scores 0.5297 against 0.5364 for the plain logistic regression, a gain of about 0.007 (part of it just from giving the model the ground feature that the plain baseline does not have). On the **test seasons the two are tied** (decision 12: 0.5463 vs 0.5405, bootstrap interval for the difference -0.006 to +0.018). Cricket's win probability is close to a smooth function of a few numbers, and a logistic regression captures most of it. I am reporting this rather than hiding it: a useful finding of the project is that careful evaluation shows complexity buying almost nothing here.

---

## 11. Monotone constraints: do they help?

**Idea.** Win probability should not fall when the batting side scores more runs, and should not rise when they lose a wicket. LightGBM can enforce this.

**Test** [`val_monotone.csv`, backtest folds]:

| Model | Log loss | Brier | ECE | Roughness (avg ball-to-ball change) | Wicket-violations | Runs-violations |
|---|---|---|---|---|---|---|
| Logistic regression | 0.5362 | 0.1821 | 0.0446 | 0.0142 | 0 | 0 |
| GBM unconstrained | 0.5271 | 0.1785 | 0.0425 | 0.0152 | 0 | 0.07% |
| GBM monotone | 0.5279 | 0.1788 | 0.0407 | 0.0153 | 0 | 0 |

"Violations" = fraction of states where adding one wicket raises the probability, or adding six runs in a chase lowers it. **Conclusion:** constraints cost 0.0008 log loss, improve nothing measurable on stability or calibration, and the unconstrained model already behaves sensibly because it starts from the monotone LR. I kept the unconstrained GBM as the main model and report the constrained variant in the test table too (0.5485 vs 0.5463: still no help).

---

## 12. Calibration: what the validation said, and what the test said

**Pre-test evidence** [`val_calibration.csv`]. Nested backtest: train before season y-1, fit Platt or isotonic on y-1, score on y, for y = 2022, 2023, 2024. Log loss pooled: raw 0.5307, Platt 0.5532, isotonic 0.5850. **Calibration hurt in every fold.** With one year of calibration data, the calibrator learns that year's quirks (which do not repeat). Isotonic is worse than Platt as expected: it has far more freedom and 70 matches is little to fit it on.

**So the pre-declared headline model was the uncalibrated GBM.** Platt and isotonic variants (fitted on 2024, per innings) were still computed and are reported, because the project brief asks for before-and-after.

**What the test seasons showed** [`test_metrics_overall.csv`]:

| Model | Log loss | Brier | AUC | ECE |
|---|---|---|---|---|
| Naive prior | 0.6863 | 0.2466 | 0.569 | 0.039 |
| Elo only | 0.6874 | 0.2471 | 0.572 | 0.053 |
| Logistic regression | 0.5405 | 0.1850 | 0.797 | 0.043 |
| GBM | 0.5463 | 0.1867 | 0.798 | 0.056 |
| GBM + Platt | 0.5303 | 0.1794 | 0.815 | 0.057 |
| GBM + isotonic | 0.5595 | 0.1851 | 0.802 | 0.071 |
| GBM monotone | 0.5485 | 0.1874 | 0.798 | 0.063 |
| Logistic regression + Platt | 0.5193 | 0.1753 | 0.820 | 0.046 |

* Both real models crush the naive prior; the match-level bootstrap interval for GBM + Platt minus naive is -0.216 to -0.088.
* GBM vs LR is a tie (interval -0.006 to +0.018). After Platt, LR is actually 0.011 better than GBM.
* **Platt scaling on 2024 *did* help on the test seasons** (0.5463 to 0.5303), the opposite of what the nested backtest suggested. Why: the test seasons are the era the older models were *not* trained for. Mean predictions for the team batting first were about 0.57 vs an observed 0.49 (2025) and 0.36 (2026, a season where chasing sides won 63%); in the chase, models under-predicted the chaser. A calibrator fitted on 2024 (the first season of the new, high-scoring regime) captures part of the shift. The nested backtest used older, calmer seasons where there was no systematic shift to correct, so calibration only added noise there.
* **ECE barely moved overall** (0.056 to 0.057) but it moved a lot where it should: chase ECE 0.067 to 0.033, powerplay unchanged. ECE on 138 matches has intervals of roughly [0.04, 0.09], so differences of 0.01 are noise.
* **First-innings probabilities are not much better than the prior** (log loss 0.673-0.690 vs 0.684; GBM before calibration is *worse* than the prior). Honest reading: a first-innings score only means something relative to the era, and the 2025-26 era is far from what the training data saw.

**How I treat this in writing.** The pre-registered headline was raw GBM, and the calibrated model beat it on test, against my pre-test evidence. I report both side by side and call the calibration result a *finding I did not predict*, not a tuned success. The demo uses GBM + Platt (it is the spec's recipe: GBM fitted on the training seasons, calibrated on the validation season).

---

## 13. The run-rate instability bug [found by looking at replays]

**What I saw.** In the first replay charts, win probability jumped around wildly in the first over and a six on ball one swung the probability by ~47 points in one match.

**Cause.** Current run rate = runs / balls bowled. After one ball and a six that is 36 runs per over, and `proj_score = runs + run rate × balls left` multiplied that into a projected score of ~700. The models were trained on that feature, so they learned a muted reaction, but the early-innings outputs were still erratic.

**Fix 1.** Shrink the run rate toward a prior as if 18 balls had already been bowled: `crr = 6 × (runs + prior_rate × 18) / (balls + 18)`. On the backtests this also *improved* the first six balls' log loss (0.650 to 0.641 for LR) and cut ball-to-ball roughness there from 0.025 to 0.016; overall log loss was unchanged (0.5375 to 0.5370). The prior weight was insensitive (6 to 48 balls gave the same overall score), so I took 18.

**Fix 2 (second iteration).** With a fixed prior of 7.8 runs per over, every innings *started* below par once scoring drifted up to 9.6 per over, so e.g. a team batting first started the match at 20% to win. The prior is now the **recent league scoring rate** (average first-innings score of the previous 40 matches over 120 balls), so at ball 0 projected score equals the league average and the start-of-match probability is sensible (mean 44.5%, range 33% to 59% across the 138 test matches).

**Disclosure.** Both fixes were made after I had seen test-season outputs, and I re-ran the entire pipeline (validation selection included) each time. The first complete run gave test log losses of LR 0.5518, GBM 0.5521, GBM + Platt 0.5305, LR + Platt 0.5193 (before any run-rate smoothing); the final run is in the table above. The conclusions were the same in every version: tie between LR and GBM, Platt helps on test, first innings is hard.

---

## 14. Turning points

**Definition.** For each match, build one series from the point of view of the team that batted first: innings-1 states use the innings-1 model, innings-2 states use 1 minus the innings-2 model, and a final point at the real result. The *swing* of a ball is the change in that series caused by the ball.

**Choices.**
* **Innings break excluded.** The step between the last ball of innings 1 and the first ball of innings 2 is a model hand-over, not an event.
* **First 12 balls of each innings excluded.** Even after the fix in decision 13 the model moves by up to ~8 points between consecutive balls in the opening overs at times (mean maximum step 7.9 points in the first 12 balls of a test match). Calling those "turning points" would be calling noise a story. A coarse rule, openly stated.
* **Top 3 per match**, greedily, skipping balls within 4 of one already chosen so that "six, six, six" is one story, and with a floor of 4 points.
* **Labels** come from the ball itself: wicket (who, how, bowler), six, four, no-ball/wide, or runs. Over-level swings (sum of ball swings in an over) are reported separately.
* **The last ball counts.** The final ball's swing runs to the true result (0 or 100%). A model saying 70% before a last-ball finish and then 0% is the "biggest swing" mathematically; it is a real turning point but partly mechanical. I keep them and say so.

**Findings on test** [`test_biggest_swings.csv`, `swing_event_mix.csv`]. Wickets are 5.3% of balls but 56.5% of the 200 biggest swings; sixes are 8.5% of balls and 29%; fours 13% of balls but 6%. Mean absolute swing by event: wicket 6.0 points, six 4.7, four 2.8, dot 1.3, other runs 0.4. The very biggest are in the final overs of tight chases (PBKS v KKR 2025-04-15, where the last KKR wicket moved PBKS from 10% to 100%).

**Replay PNGs.** Three matches picked by rules, not by hand: the biggest comeback (lowest win probability of the eventual winner after over 2: PBKS v KKR, 2025), the biggest single-ball swing after that (KKR v LSG, 2026), and the most lead changes (PBKS v MI, 2026).

---

## 15. The HTML replay

**Choice.** A single static HTML file with inline SVG and no libraries: data embedded as JSON (0.9 MB for 138 matches), no server, no CDN, works offline and from a double-click.

**Why not a chart library.** A pinned CDN script is allowed but is a way for a demo to break in two years; inline SVG also let me do exactly the marks I wanted (wickets as red dots, sixes as green diamonds, numbered turning-point circles, an innings divider, a dotted hand-over).

**Features.** Dropdown grouped by season, previous/next buttons, hover crosshair with tooltip (over, score, event, win probability for both teams, swing), replay button and scrubber, toggles for markers, turning-point list below, an over-by-over table (the data view of the chart), light and dark themes. Colours: blue and orange for the two sides, validated for colour-blind separation in the project's palette.

**Bugs found by actually opening it.** (1) A `const` used before initialisation made the whole page blank; fixed by using function declarations. (2) Axis labels were clipped, so the left margin was widened. (3) Legend icons stretched because a global `svg { width:100% }` rule hit them; scoped to the chart.

---

## 16. Post-hoc: features relative to the league scoring level

**Why.** The test showed the failure mode was the scoring-era shift. A natural fix: express runs, run rate and projected score *relative to the recent league level*.

**Result** [`posthoc_era_metrics.csv`], test seasons: LR with relative features 0.5241 (vs 0.5405), + Platt 0.5137; GBM with relative features 0.5323 (vs 0.5463), + Platt 0.5219. The gain is concentrated in the chase (innings-2 log loss 0.368 vs 0.397 for LR).

**Why this is NOT in the headline.** The backtests did not support it before the test (decision 8: kept in 3 of 4 folds in the chase but average gain 0.0008 < 0.001), and I proposed it *after* seeing test results, so it has been tuned on the data it is scored on. On the pre-test backtests it is a wash for LR (0.5374 vs 0.5364). I report it as a lead for the next version, not a result.

---

## 17. Things I did and did not do

* Did not tune anything on the test seasons. `final_test.py` is the only script that loads them for scoring, and its configuration (features, hyper-parameters, calibrator candidates) was fixed beforehand. The things that changed after seeing test numbers are listed in decisions 13 and 16, each with the reason.
* Did not use player identity, weather, dew, pitch reports or live odds. They are the obvious next step and likely matter most in the first innings.
* Did not refit on 2024 before testing. It would give the model a season closer to the test years but would remove the clean validation season used for calibration.
* XGBoost was installed during setup but not used; LightGBM was enough.
* `requirements.txt` lists exact versions; the project used Python 3.12.

## 18. Where the time went (dead ends, for honesty)

1. First GBM: 0.743 log loss (decision 7).
2. Elo and toss features: cost an afternoon, gave nothing (decision 8).
3. Recency weighting and era flags: all worse (decision 9).
4. Match-level bagging: tiny gain (0.5334 vs 0.5338), not kept.
5. Monotone constraints: no gain (decision 11).
6. Isotonic calibration: always worse than Platt.
7. Run-rate shrinkage with a fixed prior: fixed one problem and created another (decision 13).

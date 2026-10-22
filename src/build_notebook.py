"""Generate notebooks/evaluation.ipynb (then execute it with nbconvert; see run_all.sh)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# IPL ball-by-ball win probability: evaluation notebook

Everything in this notebook is read from files written by the pipeline (`results/*.csv`, `data/processed/*.parquet`), so it can be re-executed in seconds.
The structure mirrors how the project was judged:

1. **The data**: what was kept, what was excluded, and the scoring-era shift
2. **Model selection** (seasons up to 2024 only; the test seasons are not loaded anywhere in this part)
3. **The test** (train up to 2023, calibrate on 2024, score 2025-26)
4. **Calibration** before and after
5. **Turning points**
6. **Post-hoc, exploratory follow-up** (clearly separated; informed by looking at the test results)
""")
code("""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('..').resolve() / 'src'))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
import make_figures as F
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 80)
R = Path('..') / 'results'; FIG = Path('..') / 'reports' / 'figures'
matches = pd.read_parquet(Path('..') / 'data' / 'processed' / 'matches.parquet')
deliv = pd.read_parquet(Path('..') / 'data' / 'processed' / 'deliveries.parquet')
%matplotlib inline""")

md("## 1. The data")
code("""print('Matches in the Cricsheet IPL files:', len(matches))
display(matches.exclusion_reason.replace('', 'INCLUDED').value_counts().rename('matches').to_frame())
split = pd.read_csv(R / 'split_sizes.csv', index_col=0); display(split)
display(pd.read_csv(R / 'exclusions_by_season.csv', index_col=0).T)""")
md("""### The scoring-era shift (Impact Player rule, 2023+)
Average first-innings scores jumped from roughly 160-170 (2008-22) to 190+ (2024-26). A model trained on older seasons has never seen a typical score this high, which is the single biggest threat to out-of-time performance.""")
code("""display(pd.read_csv(R / 'era_scoring_shift.csv', index_col=0).round(3))
F.fig_era(); plt.show()""")

md("## 2. Model selection (seasons <= 2024 only)\nRolling-origin backtests: each of 2021, 2022, 2023, 2024 is scored by a model trained only on strictly earlier seasons.")
code("""b = pd.read_csv(R / 'val_baselines.csv')
display(b.pivot(index='model', columns='fold', values='log_loss').round(4))
F.fig_backtest(); plt.show()""")
md("""**Lesson 1: the first GBM was terrible (log loss 0.74, worse than coin-flipping on 2024).** It had match-constant features (venue, Elo, toss) and the chase target, so the trees memorised individual matches: every ball of a match shares those values, so a tree can carve out a single match and learn its outcome. The fix was to remove `target`/`runs` from the innings-2 features, use very shallow trees and large leaves, and to test every feature group with a pre-declared rule.""")
code("""a = pd.read_csv(R / 'val_feature_ablation.csv'); display(a.round(4))""")
md("Rule: a feature group is kept only if it lowers log loss by at least 0.001 on average **and** in at least 3 of the 4 backtest folds.")
code("""display(pd.read_csv(R / 'val_structure.csv', index_col=0).round(4))""")
code("""display(pd.read_csv(R / 'val_era.csv', index_col=0).round(4))""")
md("Recency weighting, restricting to recent seasons and an `impact_era` flag all made the backtests **worse**: more history beat more relevant history. (The state-based features already express a ball in terms of target and balls left, which transfers across eras.)")
code("""h = pd.read_csv(R / 'val_hparams.csv').sort_values('combined_ll'); display(h.head(6).round(4))
print(json.dumps(json.load(open(R / 'best_hparams.json')), indent=1))""")
code("""display(pd.read_csv(R / 'val_monotone.csv', index_col=0).round(5))""")
md("Monotone constraints did not change anything measurable: because the booster starts from the logistic regression (which is monotone by construction), the unconstrained GBM already never gave a higher win probability for an extra wicket and almost never for fewer runs.")
code("""c = pd.read_csv(R / 'val_calibration.csv')
display(c.pivot(index='variant', columns='fold', values='log_loss').round(4))
display(c.pivot(index='variant', columns='fold', values='ece').round(4))""")
md("Nested backtest of post-hoc calibration (model trained before season y-1, calibrator fitted on y-1, scored on y): **it did not help** in any fold. That was the evidence available before the test, so the pre-declared headline model was the *uncalibrated* GBM; both calibrated variants are still reported below.")

md("## 3. The test: train <= 2023, calibrate on 2024, score 2025-26")
code("""o = pd.read_csv(R / 'test_metrics_overall.csv', index_col=0)
order = ['naive prior','pre-match Elo only','logistic regression','logistic regression + Platt','logistic regression + isotonic','GBM','GBM + Platt','GBM + isotonic','GBM (monotone)','GBM (monotone) + Platt','GBM (monotone) + isotonic']
display(o.loc[order].round(4))""")
code("""ci = pd.read_csv(R / 'test_bootstrap_ci.csv').set_index('model')
t = pd.DataFrame({'log_loss 95% CI': ci.apply(lambda r: f"[{r.log_loss_lo:.3f}, {r.log_loss_hi:.3f}]", axis=1),
                  'log_loss diff vs LR 95% CI': ci.apply(lambda r: f"[{r.log_loss_vs_LR_lo:+.3f}, {r.log_loss_vs_LR_hi:+.3f}]", axis=1),
                  'brier diff vs LR 95% CI': ci.apply(lambda r: f"[{r.brier_vs_LR_lo:+.4f}, {r.brier_vs_LR_hi:+.4f}]", axis=1),
                  'ECE 95% CI': ci.apply(lambda r: f"[{r.ece_lo:.3f}, {r.ece_hi:.3f}]", axis=1)})
display(t)""")
md("Intervals come from resampling whole **matches** (500 draws), because balls within a match are not independent. 138 test matches is not many, so the intervals are wide: the honest reading is that GBM vs logistic regression is a tie, while both are clearly better than the naive prior.")
code("""s = pd.read_csv(R / 'test_metrics_by_season.csv')
display(s.pivot(index='model', columns='season', values='log_loss').loc[order].round(4))""")
code("""full = pd.read_csv(R / 'test_metrics_full.csv')
for m in ['naive prior', 'logistic regression', 'GBM', 'GBM + Platt']:
    display(Markdown(f'**{m}**')); display(full[full.model == m].drop(columns='model').set_index('group').round(4))""")
code("""F.fig_phase(); plt.show(); F.fig_by_ball(); plt.show()""")

md("## 4. Calibration, before and after")
code("""F.fig_reliability(); plt.show()
act = pd.read_parquet(R / 'test_predictions.parquet')
g = act.groupby(['season', 'innings']).agg(actual=('bat_team_wins', 'mean'), LR=('logistic regression', 'mean'), GBM=('GBM', 'mean'), GBM_Platt=('GBM + Platt', 'mean'))
display(g.round(3))""")
md("""The raw models are **optimistic about the side that bats first** in both test seasons (mean predicted ~0.57 vs observed 0.49 in 2025, and 0.36 observed in 2026, a season where chasing teams won 63% of matches). Platt scaling fitted on 2024 removes part of this, which is why it improves log loss and Brier. It does not improve ECE in every slice: ECE on ~138 matches is noisy, so look at the confidence intervals above before reading anything into a 0.00x gap.""")

md("## 5. Turning points")
code("""sw = pd.read_csv(R / 'test_biggest_swings.csv'); sw['swing_pts'] = (sw.swing * 100).round(1)
display(sw[['season','date','match','innings','ball','event_text','p_before','p_after','swing_pts']].head(15).round(3))""")
code("""display(pd.read_csv(R / 'swing_event_mix.csv', index_col=0).round(3))
display(pd.read_csv(R / 'swing_by_event_type.csv', index_col=0).round(4))
ov = pd.read_csv(R / 'test_biggest_overs.csv'); ov['swing_pts'] = (ov.swing * 100).round(1)
display(ov[['season','date','match','innings','over','bowler','runs','wkts','swing_pts']].head(10))
gap = pd.read_csv(R / 'handover_gap.csv'); print('innings-break hand-over gap (|P2_start - P1_end|), mean/median/max:', gap.handover_gap.mean().round(3), gap.handover_gap.median().round(3), gap.handover_gap.max().round(3))""")
code("""sel = json.load(open(R / 'replay_png_matches.json'))
for tag in sel:
    display(Markdown(f'**{tag}**')); display(Image(filename=str(FIG / f'replay_{tag}.png')))""")

md("## 6. Post-hoc follow-up (exploratory)\nAfter seeing that the test-season failure mode was era drift, I tried features expressed relative to the recent league scoring level. Because the idea came from looking at the test seasons, **these numbers are not an unbiased estimate**; they are a lead for future work.")
code("""display(pd.read_csv(R / 'posthoc_era_metrics.csv').round(4))
display(pd.read_csv(R / 'posthoc_era_by_innings.csv').pivot(index='model', columns='group', values='log_loss').round(4))
display(pd.read_csv(R / 'posthoc_era_backtest_lr.csv').pivot(index='model', columns='fold', values='log_loss').round(4))""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "evaluation.ipynb"
nbf.write(nb, out)
print("wrote", out)

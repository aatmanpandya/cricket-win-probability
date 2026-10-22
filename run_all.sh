#!/usr/bin/env bash
# Rebuild everything from the raw Cricsheet files. ~10 minutes on an M-series laptop.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/build_dataset.py            # raw CSV -> data/processed/*.parquet
$PY src/run_validation.py all       # model selection on seasons <= 2024 only
$PY src/final_test.py               # train <=2023, calibrate on 2024, score 2025-26 (the test)
$PY src/posthoc_era.py              # exploratory follow-up (clearly labelled post-hoc)
$PY src/make_replay.py              # turning points, demo data, replay PNGs
$PY src/make_figures.py             # evaluation figures
$PY src/make_demo.py                # self-contained demo/index.html
$PY src/build_notebook.py           # evaluation notebook, executed with outputs
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/evaluation.ipynb

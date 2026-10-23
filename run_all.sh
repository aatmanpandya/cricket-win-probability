#!/usr/bin/env bash
# Rebuild everything from the raw Cricsheet files. ~10 minutes on an M-series laptop.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p results/logs
L=results/logs
$PY src/build_dataset.py       2>&1 | tee $L/build.txt            # raw CSV -> data/processed/*.parquet
$PY src/run_validation.py all  2>&1 | tee $L/validation.txt       # model selection on seasons <= 2024 only
$PY src/final_test.py          2>&1 | tee $L/final_test.txt       # train <=2023, calibrate on 2024, score 2025-26
$PY src/posthoc_era.py         2>&1 | tee $L/posthoc.txt          # exploratory follow-up (post-hoc)
$PY src/make_replay.py         2>&1 | tee $L/replay.txt           # turning points, demo data, replay PNGs
$PY src/make_figures.py        2>&1 | tee $L/figures.txt
$PY src/make_demo.py           2>&1 | tee $L/demo.txt             # self-contained demo/index.html
$PY src/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/evaluation.ipynb

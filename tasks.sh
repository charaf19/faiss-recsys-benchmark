#!/usr/bin/env bash
set -e
CMD=${1:-help}
if [ "$CMD" = "smoke" ]; then
  python src/smoke_tests.py
elif [ "$CMD" = "grid50" ]; then
  python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 50
elif [ "$CMD" = "grid100" ]; then
  python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 100
elif [ "$CMD" = "grid150" ]; then
  python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 150
elif [ "$CMD" = "figs" ]; then
  python src/figures.py --summary results/summary_all.csv --out_dir results/figures
  python src/report.py  --summary results/summary_all.csv --out results/report.md --fig_dir results/figures
else
  echo "Usage: ./tasks.sh [smoke|grid50|grid100|grid150|figs]"
fi

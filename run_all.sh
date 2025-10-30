#!/usr/bin/env bash
set -euo pipefail
echo "[run_all] creating results directories..."
mkdir -p results results/figures results/sweeps
DATASETS=("ml-1m" "ml-20m" "goodbooks" "amazon-books")
BUDGETS=(50 100 150)
echo "[run_all] running full grids for budgets: ${BUDGETS[*]}"
for B in "${BUDGETS[@]}"; do
  echo "[run_all] === budget ${B}MB ==="
  python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb "$B" --datasets "${DATASETS[@]}"
done
echo "[run_all] generating main figures from combined CSV..."
python src/figures.py --summary results/summary_all.csv --out_dir results/figures
echo "[run_all] example sweeps (ml-1m, 100MB)..."
python src/sweep.py --dataset ml-1m --method hnsw --budget_mb 100 --ef_list 32 64 128
python src/sweep.py --dataset ml-1m --method ivfpq --budget_mb 100 --nprobe_list 4 8 16 32 --m_list 16 32 --with_opq --without_opq
echo "[run_all] rendering sensitivity figures..."
python - <<'PY'
from src.figures import fig_hnsw_sensitivity, fig_ivfpq_sensitivity, fig_recall_to_ndcg
fig_hnsw_sensitivity('results/sweeps/ml-1m_hnsw_budget100.csv','results/figures/hnsw_sensitivity.png')
fig_ivfpq_sensitivity('results/sweeps/ml-1m_ivfpq_budget100.csv','results/figures/ivfpq_sensitivity.png')
fig_recall_to_ndcg('results/sweeps/ml-1m_ivfpq_budget100.csv','results/figures/recall_to_ndcg.png')
PY
echo "[run_all] compiling markdown report..."
python src/report.py --summary results/summary_all.csv --out results/report.md --fig_dir results/figures
echo "[run_all] done. See results/report.md and results/figures/"

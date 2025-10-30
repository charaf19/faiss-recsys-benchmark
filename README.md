# Tiny-Index, Big Impact (Local-Only Benchmark)

Everything runs **locally**: dataset prep, training, index build, latency/memory measurement, end-to-end quality, figures, and a Markdown report.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download datasets (auto, normalized to `user_id,item_id,timestamp`)
```bash
python src/prepare_dataset.py --dataset ml-1m          --out data/ml1m.csv
python src/prepare_dataset.py --dataset ml-20m         --out data/ml20m.csv
python src/prepare_dataset.py --dataset goodbooks      --out data/goodbooks.csv
python src/prepare_dataset.py --dataset oulad          --out data/oulad.csv
python src/prepare_dataset.py --dataset bookcrossing   --out data/bookcrossing.csv
python src/prepare_dataset.py --dataset movietweetings --out data/movietweetings.csv
```

*(Alternatively)* run all at once:
```bash
./prepare_all_datasets.sh     # or:  ./prepare_all_datasets.ps1  (PowerShell)
```

## Train embeddings (TruncatedSVD, float32)
```bash
python src/train_embeddings.py --interactions data/ml1m.csv --dim 128 --out_dir data/emb_ml1m
```

## Build indices (verbose)
- HNSW
```bash
python src/build_index.py --method hnsw --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_hnsw --budget_mb 100 --M 24 --efc 200
```
- IVF-PQ (+OPQ) **auto-safeguards for small N** (OPQ auto-disabled <10k items unless `--force_opq`)
```bash
python src/build_index.py --method ivfpq --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_ivfpq --budget_mb 100 --nlist auto --m 32 --bits 8 --opq
```
- IVF-Flat
```bash
python src/build_index.py --method ivfflat --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_ivfflat --budget_mb 100 --nlist auto
```
- Flat-PQ
```bash
python src/build_index.py --method flatpq --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_flatpq --budget_mb 100 --m 32 --bits 8
```
- Flat (exact)
```bash
python src/build_index.py --method flat --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_flat --budget_mb 100
```

## Measure latency & memory
```bash
python src/run_device.py --index data/index_ml1m_hnsw --item_vecs data/emb_ml1m/item_vecs.npy --queries 10000 --topk 100 --ef 64
```

## End-to-end metrics (Recall, Precision, MAP, MRR, NDCG, HR; + Coverage, Gini, Long-tail uplift)
```bash
python src/eval_end2end.py --item_vecs data/emb_ml1m/item_vecs.npy --index data/index_ml1m_hnsw   --ann_method hnsw --queries 10000 --topk 100 --metric_topk 10 --ef 64
```

## Full grid (multi-dataset × methods) and report
```bash
python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 100
python src/figures.py --summary results/summary_all.csv --out_dir results/figures
python src/report.py  --summary results/summary_all.csv --out results/report.md --fig_dir results/figures
```

## One-shot (everything: 50/100/150 MB, all datasets)
```bash
./run_all.sh         # or:  ./run_all.ps1    (PowerShell)
```
Outputs to `results/`: per-dataset CSVs, `summary_all.csv`, figures, and `report.md`.

## Smoke test
```bash
./tasks.sh smoke
```

## Ablation sweeps
```bash
python src/sweep.py --dataset ml-1m --method hnsw --budget_mb 100 --ef_list 32 64 128
python src/sweep.py --dataset ml-1m --method ivfpq --budget_mb 100 --nprobe_list 4 8 16 32 --m_list 16 32 --with_opq --without_opq
```

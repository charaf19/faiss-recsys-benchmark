# Adding a new method (plugin-style)

1) Create a new method key, e.g., `mynew`.
2) Extend **build_index.py** with `elif args.method == "mynew": ...` to:
   - build your index,
   - write it to `data/index_<dataset>_mynew`,
   - save any needed metadata.
3) Extend **run_device.py** to detect your index files and measure p50/p95 latency.
4) Extend **eval_end2end.py**:
   - add `elif ann_method == "mynew":` to perform ANN search and return candidate IDs,
   - keep the rest (rerank, metrics) unchanged.
5) Add your method label to `METHODS` in **run_grid.py** to include it in all tables.
6) Run `python src/smoke_tests.py` to ensure metrics & outputs are intact.

This "plugin by convention" approach avoids a framework layer and keeps reviewer-driven changes fast.

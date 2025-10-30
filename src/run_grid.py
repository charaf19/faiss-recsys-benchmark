# src/run_grid.py
import argparse, os, subprocess, sys, json, numpy as np, pandas as pd
from pathlib import Path

DATASETS = {
  "ml-1m":         {"prepare": ["python","src/prepare_dataset.py","--dataset","ml-1m","--out","data/ml1m.csv"],           "csv":"data/ml1m.csv"},
  "ml-20m":        {"prepare": ["python","src/prepare_dataset.py","--dataset","ml-20m","--out","data/ml20m.csv"],         "csv":"data/ml20m.csv"},
  "goodbooks":     {"prepare": ["python","src/prepare_dataset.py","--dataset","goodbooks","--out","data/goodbooks.csv"],  "csv":"data/goodbooks.csv"},
  "amazon-books":  {"prepare": ["python","src/prepare_dataset.py","--dataset","amazon-books","--out","data/amazon_books.csv"], "csv":"data/amazon_books.csv"},
}

METHODS = ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]

def run(cmd):
    py = sys.executable
    full = [py] + cmd[1:] if isinstance(cmd, list) and len(cmd)>0 and cmd[0]=="python" else cmd
    print(">>", " ".join(str(c) for c in full))
    subprocess.run(full, check=True)

def run_capture_json(cmd) -> dict:
    """Run a python script and parse a single JSON object from stdout."""
    py = sys.executable
    full = [py] + cmd[1:] if isinstance(cmd, list) and len(cmd)>0 and cmd[0]=="python" else cmd
    print(">>", " ".join(str(c) for c in full))
    out = subprocess.check_output(full, text=True)
    out = out.strip()
    # If the script prints extra lines, try to find the last JSON block
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # naive fallback: scan for first '{' and last '}'
        s = out.find("{")
        e = out.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(out[s:e+1])
        print(out)
        raise

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    print("[run_grid] starting full grid...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--metric_topk", type=int, default=10)
    ap.add_argument("--budget_mb", type=int, default=100)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), help="subset of datasets to run")
    args = ap.parse_args()

    for dname, info in DATASETS.items():
        if dname not in args.datasets:
            continue

        print(f"[run_grid] dataset={dname}")
        # 1) prepare dataset
        run(info["prepare"])

        # 2) train embeddings
        emb_dir = f"data/emb_{dname}"
        run(["python","src/train_embeddings.py","--interactions",info["csv"],"--dim",str(args.emb_dim),"--out_dir",emb_dir])
        item_vecs = f"{emb_dir}/item_vecs.npy"
        item_ids  = f"{emb_dir}/item_ids.npy"

        # 3) per-method build/run/eval
        for m in METHODS:
            print(f"[run_grid] building method={m}")
            out_dir = f"data/index_{dname}_{m}"
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            # Build index
            if m == "hnsw":
                run(["python","src/build_index.py","--method","hnsw","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",out_dir,"--budget_mb",str(args.budget_mb),"--M","24","--efc","200"])
            elif m == "ivfflat":
                run(["python","src/build_index.py","--method","ivfflat","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",out_dir,"--budget_mb",str(args.budget_mb),"--nlist","auto"])
            elif m == "ivfpq":
                run(["python","src/build_index.py","--method","ivfpq","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",out_dir,"--budget_mb",str(args.budget_mb),"--nlist","auto","--m","32","--bits","8","--opq"])
            elif m == "flatpq":
                run(["python","src/build_index.py","--method","flatpq","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",out_dir,"--budget_mb",str(args.budget_mb),"--m","32","--bits","8"])
            elif m == "flat":
                run(["python","src/build_index.py","--method","flat","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",out_dir,"--budget_mb",str(args.budget_mb)])
            else:
                raise ValueError(f"unknown method {m}")

            # Run latency test (capture + save)
            latency_cmd = ["python","src/run_device.py","--index",out_dir,"--item_vecs",item_vecs,"--queries",str(args.queries),"--topk",str(args.topk)]
            if m == "hnsw":
                latency_cmd += ["--ef","128"]
            if m in ("ivfflat","ivfpq"):
                latency_cmd += ["--nprobe","16"]
            latency_json = run_capture_json(latency_cmd)
            write_json(Path(out_dir) / f"latency_{m}.json", latency_json)

            # Evaluate end-to-end (IMPORTANT: pass --interactions) (capture + save)
            eval_cmd = [
                "python","src/eval_end2end.py",
                "--interactions", info["csv"],
                "--item_vecs", item_vecs,
                "--index", out_dir,
                "--ann_method", m,
                "--queries", str(args.queries),
                "--topk", str(args.topk),
                "--metric_topk", str(args.metric_topk),
            ]
            if m == "hnsw":
                eval_cmd += ["--ef","128"]
            if m in ("ivfflat","ivfpq"):
                eval_cmd += ["--nprobe","16"]

            end2end_json = run_capture_json(eval_cmd)
            write_json(Path(out_dir) / f"end2end_{m}.json", end2end_json)

        # 4) collect results for this dataset
        rows = []
        for m in METHODS:
            idx_dir = Path(f"data/index_{dname}_{m}")
            efile = idx_dir / f"end2end_{m}.json"
            lfile = idx_dir / f"latency_{m}.json"
            estats = json.load(open(efile)) if efile.exists() else {}
            lstats = json.load(open(lfile)) if lfile.exists() else {}
            row = {
                "dataset": dname, "method": m,
                "recall_at_topk_mean": estats.get("recall_at_topk_mean"),
                "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                "hr_at_k_mean": estats.get("hr_at_k_mean"),
                "precision_at_k_mean": estats.get("precision_at_k_mean"),
                "map_at_k_mean": estats.get("map_at_k_mean"),
                "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                "coverage_at_k": estats.get("coverage_at_k"),
                "gini_exposure": estats.get("gini_exposure"),
                "long_tail_uplift": estats.get("long_tail_uplift"),
                "latency_p50_ms": (lstats.get("latency_ms") or {}).get("p50"),
                "latency_p95_ms": (lstats.get("latency_ms") or {}).get("p95"),
                "rss_mb_after": lstats.get("rss_mb_after"),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        Path("results").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"results/summary_{dname}.csv", index=False)
        print(f"[run_grid] wrote results/summary_{dname}.csv")

    # 5) aggregate summaries
    frames = []
    for dname in args.datasets:
        p = Path(f"results/summary_{dname}.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv("results/summary_all.csv", index=False)
        print("[run_grid] wrote results/summary_all.csv")

if __name__ == "__main__":
    main()

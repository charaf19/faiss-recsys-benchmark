\
import argparse, subprocess, json, pandas as pd, sys
from pathlib import Path
def run(cmd):
    py = sys.executable
    full = [py] + cmd[1:] if isinstance(cmd, list) and len(cmd)>0 and cmd[0]=="python" else cmd
    print(">>", " ".join(str(c) for c in full)); subprocess.run(full, check=True)
def prepare(dataset):
    run(["python","src/prepare_dataset.py","--dataset",dataset,"--out",f"data/{dataset}.csv"])
    run(["python","src/train_embeddings.py","--interactions",f"data/{dataset}.csv","--dim","128","--out_dir",f"data/emb_{dataset}"])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ml-1m","ml-20m","goodbooks","amazon-books"])
    ap.add_argument("--method", required=True, choices=["hnsw","ivfpq"])
    ap.add_argument("--budget_mb", type=int, default=100)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--metric_topk", type=int, default=10)
    ap.add_argument("--ef_list", nargs="*", type=int, default=[32,64,128])
    ap.add_argument("--M", type=int, default=24); ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--nprobe_list", nargs="*", type=int, default=[4,8,16,32])
    ap.add_argument("--m_list", nargs="*", type=int, default=[16,32]); ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--with_opq", action="store_true"); ap.add_argument("--without_opq", action="store_true")
    args = ap.parse_args()
    Path("results/sweeps").mkdir(parents=True, exist_ok=True); prepare(args.dataset)
    item_vecs = f"data/emb_{args.dataset}/item_vecs.npy"; item_ids = f"data/emb_{args.dataset}/item_ids.npy"
    if args.method == "hnsw":
        idx_dir = f"data/index_{args.dataset}_hnsw_sweep"
        run(["python","src/build_index.py","--method","hnsw","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",idx_dir,"--budget_mb",str(args.budget_mb),"--M",str(args.M),"--efc",str(args.efc)])
        rows = []
        for ef in args.ef_list:
            run(["python","src/run_device.py","--index",idx_dir,"--item_vecs",item_vecs,"--queries",str(args.queries),"--topk",str(args.topk),"--ef",str(ef)])
            run(["python","src/eval_end2end.py","--item_vecs",item_vecs,"--index",idx_dir,"--ann_method","hnsw","--queries",str(args.queries),"--topk",str(args.topk),"--metric_topk",str(args.metric_topk),"--ef",str(ef)])
            estats = json.load(open(Path(idx_dir)/"end2end_hnsw.json")); lstats = json.load(open(Path(idx_dir)/"latency_hnsw.json"))
            rows.append({"dataset": args.dataset, "method":"hnsw", "budget_mb": args.budget_mb, "ef": ef,
                         "recall_at_topk_mean": estats.get("recall_at_topk_mean"), "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                         "hr_at_k_mean": estats.get("hr_at_k_mean"), "precision_at_k_mean": estats.get("precision_at_k_mean"),
                         "map_at_k_mean": estats.get("map_at_k_mean"), "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                         "latency_p50_ms": (lstats.get("latency_ms") or {}).get("p50"), "latency_p95_ms": (lstats.get("latency_ms") or {}).get("p95")})
        pd.DataFrame(rows).to_csv(f"results/sweeps/{args.dataset}_hnsw_budget{args.budget_mb}.csv", index=False)
    elif args.method == "ivfpq":
        all_rows = []; flags = []
        if args.with_opq: flags.append(True)
        if args.without_opq: flags.append(False)
        if not flags: flags = [True]
        for m in args.m_list:
            for opq in flags:
                idx_dir = f"data/index_{args.dataset}_ivfpq_m{m}_{'opq' if opq else 'nopq'}"
                cmd = ["python","src/build_index.py","--method","ivfpq","--item_vecs",item_vecs,"--item_ids",item_ids,"--out_dir",idx_dir,"--budget_mb",str(args.budget_mb),"--nlist","auto","--m",str(m),"--bits",str(args.bits)]
                if opq: cmd.append("--opq")
                run(cmd)
                for nprobe in args.nprobe_list:
                    run(["python","src/run_device.py","--index",idx_dir,"--item_vecs",item_vecs,"--queries",str(args.queries),"--topk",str(args.topk),"--nprobe",str(nprobe)])
                    run(["python","src/eval_end2end.py","--item_vecs",item_vecs,"--index",idx_dir,"--ann_method","ivfpq","--queries",str(args.queries),"--topk",str(args.topk),"--metric_topk",str(args.metric_topk),"--nprobe",str(nprobe)])
                    estats = json.load(open(Path(idx_dir)/"end2end_ivfpq.json")); lstats = json.load(open(Path(idx_dir)/"latency_ivfpq.json"))
                    all_rows.append({"dataset": args.dataset, "method":"ivfpq", "budget_mb": args.budget_mb,
                                     "m": m, "opq": opq, "nprobe": nprobe,
                                     "recall_at_topk_mean": estats.get("recall_at_topk_mean"), "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                                     "hr_at_k_mean": estats.get("hr_at_k_mean"), "precision_at_k_mean": estats.get("precision_at_k_mean"),
                                     "map_at_k_mean": estats.get("map_at_k_mean"), "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                                     "latency_p50_ms": (lstats.get("latency_ms") or {}).get("p50"), "latency_p95_ms": (lstats.get("latency_ms") or {}).get("p95")})
        pd.DataFrame(all_rows).to_csv(f"results/sweeps/{args.dataset}_ivfpq_budget{args.budget_mb}.csv", index=False)
if __name__ == "__main__":
    main()

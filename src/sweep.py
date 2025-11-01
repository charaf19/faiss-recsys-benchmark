"""Parameter sweeps for ANN indices.

This script prepares a dataset, trains embeddings, builds an index, and then
collects latency and quality metrics for a sweep over key hyper-parameters. The
outputs are written to ``results/sweeps`` as CSV tables so they can be plotted
or inspected manually.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DATASETS = {
    "ml-1m": {
        "csv": Path("data/ml1m.csv"),
        "prepare": [
            "python",
            "src/prepare_dataset.py",
            "--dataset",
            "ml-1m",
            "--out",
            "data/ml1m.csv",
        ],
        "emb_dir": Path("data/emb_ml1m"),
    },
    "ml-20m": {
        "csv": Path("data/ml20m.csv"),
        "prepare": [
            "python",
            "src/prepare_dataset.py",
            "--dataset",
            "ml-20m",
            "--out",
            "data/ml20m.csv",
        ],
        "emb_dir": Path("data/emb_ml20m"),
    },
    "goodbooks": {
        "csv": Path("data/goodbooks.csv"),
        "prepare": [
            "python",
            "src/prepare_dataset.py",
            "--dataset",
            "goodbooks",
            "--out",
            "data/goodbooks.csv",
        ],
        "emb_dir": Path("data/emb_goodbooks"),
    },
    "amazon-books": {
        "csv": Path("data/amazon_books.csv"),
        "prepare": [
            "python",
            "src/prepare_dataset.py",
            "--dataset",
            "amazon-books",
            "--out",
            "data/amazon_books.csv",
        ],
        "emb_dir": Path("data/emb_amazon_books"),
    },
}


def _python_cmd(cmd: Iterable[str]) -> List[str]:
    cmd = list(cmd)
    if cmd and cmd[0] == "python":
        return [sys.executable, *cmd[1:]]
    return cmd


def run(cmd: Iterable[str]) -> None:
    full = _python_cmd(cmd)
    print(">>", " ".join(str(c) for c in full))
    subprocess.run(full, check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_prepared(dataset: str, emb_dim: int) -> tuple[Path, Path, Path]:
    cfg = DATASETS[dataset]
    run(cfg["prepare"])

    csv_path = cfg["csv"]
    emb_dir = cfg["emb_dir"]
    run(
        [
            "python",
            "src/train_embeddings.py",
            "--interactions",
            str(csv_path),
            "--dim",
            str(emb_dim),
            "--out_dir",
            str(emb_dir),
        ]
    )

    item_vecs = emb_dir / "item_vecs.npy"
    item_ids = emb_dir / "item_ids.npy"
    return csv_path, item_vecs, item_ids


def sweep_hnsw(args: argparse.Namespace, interactions: Path, item_vecs: Path, item_ids: Path) -> None:
    idx_dir = Path(f"data/index_{args.dataset}_hnsw_sweep")
    run(
        [
            "python",
            "src/build_index.py",
            "--method",
            "hnsw",
            "--item_vecs",
            str(item_vecs),
            "--item_ids",
            str(item_ids),
            "--out_dir",
            str(idx_dir),
            "--budget_mb",
            str(args.budget_mb),
            "--M",
            str(args.M),
            "--efc",
            str(args.efc),
        ]
    )

    rows = []
    for ef in args.ef_list:
        run(
            [
                "python",
                "src/run_device.py",
                "--index",
                str(idx_dir),
                "--item_vecs",
                str(item_vecs),
                "--queries",
                str(args.queries),
                "--topk",
                str(args.topk),
                "--ef",
                str(ef),
            ]
        )

        run(
            [
                "python",
                "src/eval_end2end.py",
                "--interactions",
                str(interactions),
                "--item_vecs",
                str(item_vecs),
                "--index",
                str(idx_dir),
                "--ann_method",
                "hnsw",
                "--queries",
                str(args.queries),
                "--topk",
                str(args.topk),
                "--metric_topk",
                str(args.metric_topk),
                "--ef",
                str(ef),
            ]
        )

        estats = load_json(idx_dir / "end2end_hnsw.json")
        lstats = load_json(idx_dir / "latency_hnsw.json")
        latencies = lstats.get("latency_ms") or {}

        rows.append(
            {
                "dataset": args.dataset,
                "method": "hnsw",
                "budget_mb": args.budget_mb,
                "ef": ef,
                "recall_at_topk_mean": estats.get("recall_at_topk_mean"),
                "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                "hr_at_k_mean": estats.get("hr_at_k_mean"),
                "precision_at_k_mean": estats.get("precision_at_k_mean"),
                "map_at_k_mean": estats.get("map_at_k_mean"),
                "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                "latency_p50_ms": latencies.get("p50"),
                "latency_p95_ms": latencies.get("p95"),
            }
        )

    Path("results/sweeps").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        Path("results/sweeps") / f"{args.dataset}_hnsw_budget{args.budget_mb}.csv",
        index=False,
    )


def sweep_ivfpq(args: argparse.Namespace, interactions: Path, item_vecs: Path, item_ids: Path) -> None:
    flags = []
    if args.with_opq:
        flags.append(True)
    if args.without_opq:
        flags.append(False)
    if not flags:
        flags = [True]

    rows = []
    for m in args.m_list:
        for use_opq in flags:
            idx_dir = Path(f"data/index_{args.dataset}_ivfpq_m{m}_{'opq' if use_opq else 'nopq'}")
            build_cmd = [
                "python",
                "src/build_index.py",
                "--method",
                "ivfpq",
                "--item_vecs",
                str(item_vecs),
                "--item_ids",
                str(item_ids),
                "--out_dir",
                str(idx_dir),
                "--budget_mb",
                str(args.budget_mb),
                "--nlist",
                "auto",
                "--m",
                str(m),
                "--bits",
                str(args.bits),
            ]
            if use_opq:
                build_cmd.append("--opq")
            run(build_cmd)

            for nprobe in args.nprobe_list:
                run(
                    [
                        "python",
                        "src/run_device.py",
                        "--index",
                        str(idx_dir),
                        "--item_vecs",
                        str(item_vecs),
                        "--queries",
                        str(args.queries),
                        "--topk",
                        str(args.topk),
                        "--nprobe",
                        str(nprobe),
                    ]
                )

                run(
                    [
                        "python",
                        "src/eval_end2end.py",
                        "--interactions",
                        str(interactions),
                        "--item_vecs",
                        str(item_vecs),
                        "--index",
                        str(idx_dir),
                        "--ann_method",
                        "ivfpq",
                        "--queries",
                        str(args.queries),
                        "--topk",
                        str(args.topk),
                        "--metric_topk",
                        str(args.metric_topk),
                        "--nprobe",
                        str(nprobe),
                    ]
                )

                estats = load_json(idx_dir / "end2end_ivfpq.json")
                lstats = load_json(idx_dir / "latency_ivfpq.json")
                latencies = lstats.get("latency_ms") or {}

                rows.append(
                    {
                        "dataset": args.dataset,
                        "method": "ivfpq",
                        "budget_mb": args.budget_mb,
                        "m": m,
                        "opq": use_opq,
                        "nprobe": nprobe,
                        "recall_at_topk_mean": estats.get("recall_at_topk_mean"),
                        "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                        "hr_at_k_mean": estats.get("hr_at_k_mean"),
                        "precision_at_k_mean": estats.get("precision_at_k_mean"),
                        "map_at_k_mean": estats.get("map_at_k_mean"),
                        "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                        "latency_p50_ms": latencies.get("p50"),
                        "latency_p95_ms": latencies.get("p95"),
                    }
                )

    Path("results/sweeps").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        Path("results/sweeps") / f"{args.dataset}_ivfpq_budget{args.budget_mb}.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=["hnsw", "ivfpq"])
    parser.add_argument("--budget_mb", type=int, default=100)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--metric_topk", type=int, default=10)
    parser.add_argument("--ef_list", nargs="*", type=int, default=[32, 64, 128])
    parser.add_argument("--M", type=int, default=24)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--nprobe_list", nargs="*", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--m_list", nargs="*", type=int, default=[16, 32])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--with_opq", action="store_true")
    parser.add_argument("--without_opq", action="store_true")
    parser.add_argument("--emb_dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions, item_vecs, item_ids = ensure_prepared(args.dataset, args.emb_dim)

    if args.method == "hnsw":
        sweep_hnsw(args, interactions, item_vecs, item_ids)
    else:
        sweep_ivfpq(args, interactions, item_vecs, item_ids)


if __name__ == "__main__":
    main()


\
import argparse, pandas as pd
from pathlib import Path

def to_md_table(df: pd.DataFrame) -> str:
    try: return df.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"]*len(df.columns)) + " |"
        rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.values]
        return "\n".join([header, sep] + rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary_all.csv")
    ap.add_argument("--out", default="results/report.md")
    ap.add_argument("--fig_dir", default="results/figures")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    if not Path(args.summary).exists():
        print(f"[report] missing {args.summary}. Run run_grid.py first.")
        return
    df = pd.read_csv(args.summary)

    cols = ["dataset","method","recall_at_topk_mean","precision_at_k_mean","map_at_k_mean","mrr_at_k_mean",
            "ndcg_at_k_mean","hr_at_k_mean","coverage_at_k","gini_exposure","long_tail_uplift",
            "latency_p50_ms","latency_p95_ms","rss_mb_after"]
    present_cols = [c for c in cols if c in df.columns]

    lines = []
    lines.append("# Tiny-Index, Big Impact — Results\n")
    lines.append("## Combined table\n")
    lines.append(to_md_table(df[present_cols].round(4)))
    lines.append("")
    for d in sorted(df["dataset"].unique()):
        sub = df[df["dataset"]==d][present_cols].round(4)
        lines.append(f"## {d}\n"); lines.append(to_md_table(sub)); lines.append("")
    lines.append("## Figures\n")
    for f in ["latency_vs_ndcg.png","recall_vs_latency.png","coverage_vs_ndcg.png"]:
        p = Path(args.fig_dir)/f
        if p.exists(): lines.append(f"![{f}]({p.as_posix()})\n")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {args.out}")

if __name__ == "__main__":
    main()

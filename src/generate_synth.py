\
import argparse, numpy as np, pandas as pd, random, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--items", type=int, default=5000)
    ap.add_argument("--interactions", type=int, default=100000)
    ap.add_argument("--out", type=str, default="data/synth.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    users = [f"u{i}" for i in range(args.users)]
    items = [f"i{j}" for j in range(args.items)]

    rows = []; ts = 1_700_000_000
    for _ in range(args.interactions):
        u = users[rng.integers(0, len(users))]
        j = int(rng.pareto(1.5) * 50) % len(items)
        i = items[j]
        rows.append((u, i, ts)); ts += rng.integers(0,5)

    df = pd.DataFrame(rows, columns=["user_id","item_id","timestamp"])
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[generate_synth] wrote {args.out} rows={len(df)}")

if __name__ == "__main__":
    main()

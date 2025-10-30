# src/datasets/goodbooks.py

import os
from pathlib import Path
import pandas as pd
import numpy as np

from .common import _download

RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"


def prepare_goodbooks(out_csv: str = "data/goodbooks.csv", min_rating: int = 4) -> str:
    """
    Download Goodbooks-10k ratings and convert to the unified interactions CSV:
      columns: user_id, item_id, timestamp

    - Downloads ratings.csv if not present locally.
    - Optionally filters out interactions with rating < min_rating.
    - Synthesizes a per-user 'timestamp' (1..n) because Goodbooks-10k has no timestamps.
    """
    # Where we'll cache the raw CSV
    csv_path = Path("data/goodbooks-10k/ratings.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _download(RATINGS_URL, str(csv_path))

    # Read and normalize columns
    df = pd.read_csv(csv_path)

    # Basic schema handling (Goodbooks-10k columns: user_id, book_id, rating)
    # If any variants appear, rename them.
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "user_id" not in lower_map and "user" in lower_map:
        rename_map[lower_map["user"]] = "user_id"
    if "book_id" not in lower_map and "book" in lower_map:
        rename_map[lower_map["book"]] = "book_id"
    if "rating" not in lower_map and "ratings" in lower_map:
        rename_map[lower_map["ratings"]] = "rating"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Validate required columns
    required = {"user_id", "book_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"[goodbooks] Expected columns {sorted(list(required))}, got {sorted(df.columns.tolist())}"
        )

    # Optional rating filter
    if "rating" in df.columns and min_rating is not None:
        df = df[pd.to_numeric(df["rating"], errors="coerce") >= min_rating]

    # Coerce ids (keep as strings for consistency across datasets)
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)

    # If timestamp is missing, synthesize an interaction order per user
    if "timestamp" not in df.columns:
        # preserve original file order within each user
        df["_row"] = np.arange(len(df), dtype=np.int64)
        df = df.sort_values(["user_id", "_row"], kind="mergesort")
        df["timestamp"] = df.groupby("user_id").cumcount() + 1
        df = df.drop(columns=["_row"])
    else:
        # ensure integer-like timestamps with no NaNs
        ts = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        df["timestamp"] = ts

    # Build output schema
    out_df = df[["user_id", "book_id", "timestamp"]].copy()
    out_df = out_df.rename(columns={"book_id": "item_id"})

    # Write
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(
        f"[goodbooks] src={csv_path} -> wrote {out_path} "
        f"rows={len(out_df)} users={out_df['user_id'].nunique()} items={out_df['item_id'].nunique()}"
    )
    return str(out_path)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/goodbooks.csv")
    ap.add_argument("--min_rating", type=int, default=4)
    args = ap.parse_args()
    prepare_goodbooks(args.out, args.min_rating)


if __name__ == "__main__":
    main()

# src/datasets/amazon_books.py

import os
import gzip
import json
from pathlib import Path
from typing import Iterable, Optional, Union, List

import pandas as pd
import numpy as np

from .common import _download


# Known public mirrors for the Amazon “Books” 5-core reviews.
# We try them in order unless the user provides AMAZON_BOOKS_URL.
CANDIDATE_URLS = [
    # Classic 5-core (2014-era) location
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
    # UCSD mirror (paths have changed over time; some mirrors keep this layout)
    "https://jmcauley.ucsd.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
    # Fallback: often used in forks/mirrors (kept for convenience; may be unavailable)
    "https://raw.githubusercontent.com/entitize/amazon-reviews-archive/master/data/reviews_Books_5.json.gz",
]


def _resolve_source(dst_dir: Path) -> Path:
    """
    Resolve a local path to the compressed JSONL reviews for Amazon Books 5-core.
    Priority:
      1) If AMAZON_BOOKS_URL is set, download from it.
      2) If a local file already exists in dst_dir (amazon_books_5.json.gz), use it.
      3) Try known public URLs in CANDIDATE_URLS until one succeeds.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    local_path = dst_dir / "amazon_books_5.json.gz"

    env_url = os.environ.get("AMAZON_BOOKS_URL", "").strip()
    if env_url:
        _download(env_url, str(local_path))
        return local_path

    if local_path.is_file():
        return local_path

    last_err: Optional[Exception] = None
    for url in CANDIDATE_URLS:
        try:
            _download(url, str(local_path))
            return local_path
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        "Could not obtain Amazon Books 5-core reviews. "
        "Set AMAZON_BOOKS_URL to a valid .json.gz (one JSON object per line) or place a file at "
        f"{local_path}. Last error: {last_err}"
    )


def _read_jsonl_gz_in_chunks(path: Union[str, Path], chunksize: int = 1_000_000) -> Iterable[pd.DataFrame]:
    """
    Stream a large .json.gz (one JSON object per line) into DataFrame chunks.
    Keeps only the columns we need to minimize memory: reviewerID, asin, unixReviewTime, overall.
    """
    wanted = {"reviewerID", "asin", "unixReviewTime", "overall"}
    buf: List[dict] = []
    n = 0

    with gzip.open(path, "rb") as f:
        for line in f:
            if not line:
                continue
            rec = json.loads(line)
            # Keep only the wanted keys
            rec_small = {k: rec.get(k, None) for k in wanted}
            # Discard rows missing core ids
            if rec_small.get("reviewerID") is None or rec_small.get("asin") is None:
                continue
            buf.append(rec_small)
            n += 1
            if n % chunksize == 0:
                yield pd.DataFrame.from_records(buf)
                buf.clear()

    if buf:
        yield pd.DataFrame.from_records(buf)


def prepare_amazon_books(
    out_csv: str = "data/amazon_books.csv",
    min_rating: Optional[float] = None,
) -> str:
    """
    Convert Amazon “Books” 5-core reviews into the unified interactions CSV expected by the pipeline:
      columns: user_id, item_id, timestamp

    - Downloads the source if not found locally. You can override the URL via env AMAZON_BOOKS_URL.
    - Streams and filters the JSONL to avoid high memory usage.
    - If `min_rating` is provided, keeps only rows with overall >= min_rating (e.g., 4.0).
    - If `unixReviewTime` is missing, synthesizes a per-user order timestamp.

    Returns the path written (out_csv).
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_path = _resolve_source(Path("data/raw"))

    # Accumulate per chunk, then concatenate
    parts: List[pd.DataFrame] = []
    for df in _read_jsonl_gz_in_chunks(src_path, chunksize=1_000_000):
        # Rename & filter columns
        df = df.rename(
            columns={
                "reviewerID": "user_id",
                "asin": "item_id",
                "unixReviewTime": "timestamp",
                "overall": "rating",
            }
        )

        # Optional rating filter
        if min_rating is not None and "rating" in df.columns:
            df = df[df["rating"].astype(float) >= float(min_rating)]

        # Keep only required columns (we may need rating to derive later logic; drop it now)
        keep = ["user_id", "item_id", "timestamp"]
        # If timestamp is entirely missing in this chunk, synthesize later globally
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NA

        df = df[keep]
        parts.append(df)

    if not parts:
        raise RuntimeError("No data parsed from Amazon Books JSONL; file empty or unreadable.")

    full = pd.concat(parts, ignore_index=True)

    # Ensure types
    # Keep IDs as strings (stable identifiers), timestamp as integer when available
    full["user_id"] = full["user_id"].astype(str)
    full["item_id"] = full["item_id"].astype(str)

    # If timestamps are missing (NA) or all-null, synthesize per-user order
    if full["timestamp"].isna().all():
        # Use per-user interaction order as pseudo-time
        full["timestamp"] = (
            full.groupby("user_id", sort=False).cumcount() + 1
        ).astype(np.int64)
    else:
        # Where timestamp missing, fill with per-user order; otherwise cast to int
        missing_mask = full["timestamp"].isna()
        if missing_mask.any():
            full.loc[missing_mask, "timestamp"] = (
                full[missing_mask]
                .groupby("user_id", sort=False)
                .cumcount()
                .add(1)
                .astype(np.int64)
            )
        full["timestamp"] = full["timestamp"].astype(np.int64, errors="ignore")

    # Write final CSV
    full.to_csv(out_path, index=False)

    print(
        "[amazon_books] wrote",
        out_path,
        f"rows={len(full)} users={full['user_id'].nunique()} items={full['item_id'].nunique()}",
    )
    return str(out_path)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Prepare Amazon Books 5-core interactions CSV.")
    ap.add_argument("--out", default="data/amazon_books.csv", help="Output CSV path.")
    ap.add_argument(
        "--min_rating",
        type=float,
        default=None,
        help="Optional rating threshold (e.g., 4.0). If set, keep only reviews with rating >= min_rating.",
    )
    args = ap.parse_args()
    prepare_amazon_books(out_csv=args.out, min_rating=args.min_rating)


if __name__ == "__main__":
    main()

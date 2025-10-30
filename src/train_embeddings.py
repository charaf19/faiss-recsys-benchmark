\
import argparse, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=str, required=True, help="CSV user_id,item_id,timestamp")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="data/emb")
    args = ap.parse_args()

    print(f"[train_embeddings] loading {args.interactions}")
    df = pd.read_csv(args.interactions)
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    user_codes = users.cat.codes.values
    item_codes = items.cat.codes.values
    n_users = users.cat.categories.size
    n_items = items.cat.categories.size
    print(f"[train_embeddings] users={n_users} items={n_items}")

    R = sp.coo_matrix((np.ones(len(df)), (user_codes, item_codes)), shape=(n_users, n_items)).tocsr()

    svd = TruncatedSVD(n_components=args.dim, random_state=42)
    item_vecs = svd.fit_transform(R.transpose()).astype("float32")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "item_vecs.npy", item_vecs)
    np.save(out / "item_ids.npy", items.cat.categories.values)
    print(f"[train_embeddings] saved vectors to {out} shape={item_vecs.shape}")

if __name__ == "__main__":
    main()

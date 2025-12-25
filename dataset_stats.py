import pandas as pd

def stats(train_csv, test_csv):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)

    users = pd.concat([tr["user_id"], te["user_id"]]).nunique()
    items = pd.concat([tr["item_id"], te["item_id"]]).nunique()

    r_tr = len(tr)
    r_te = len(te)
    r_total = r_tr + r_te
    return users, items, r_tr, r_te, r_total

datasets = {
    "ML-1M": ("data/ml-1m_train.csv", "data/ml-1m_test.csv"),
    "ML-20M": ("data/ml-20m_train.csv", "data/ml-20m_test.csv"),
    "Goodbooks-10k": ("data/goodbooks_train.csv", "data/goodbooks_test.csv"),
}

for name, (tr, te) in datasets.items():
    u,i,rtr,rte,rt = stats(tr, te)
    print(f"{name}: Users={u}, Items={i}, Train={rtr}, Test={rte}, Total={rt}")

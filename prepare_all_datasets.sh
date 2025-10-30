#!/usr/bin/env bash
set -e
python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv
python src/prepare_dataset.py --dataset ml-20m --out data/ml20m.csv
python src/prepare_dataset.py --dataset goodbooks --out data/goodbooks.csv
python src/prepare_dataset.py --dataset amazon-books --out data/amazon_books.csv

echo "[prepare_all] done."

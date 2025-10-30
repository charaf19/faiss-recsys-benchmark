# Public datasets (download locally)

This project includes simple download/prepare scripts for widely available datasets:

- **MovieLens 1M** and **MovieLens 20M** (GroupLens)
- **Goodbooks-10k** (GitHub-hosted CSVs)
- **OULAD** (Open University Learning Analytics Dataset)

> Note: These scripts fetch files from public URLs using `requests`. If a mirror changes, update the URLs in the scripts.
> All datasets are converted to `user_id,item_id,timestamp` implicit CSVs under `data/`.

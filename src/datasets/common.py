\
import os, pandas as pd, numpy as np, zipfile, io, time, requests
from pathlib import Path

def _download(url: str, dest: str):
    Path(os.path.dirname(dest) or ".").mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f: f.write(r.content)
    return dest

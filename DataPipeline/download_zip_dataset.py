#!/usr/bin/env python3
import os
from typing import Optional
import requests
from tqdm import tqdm

DATA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, "../DR-VCTK.zip")
CHUNK_BYTES = 1024 * 512

def download_zip(url: str = DATA_URL, dst_path: str = ZIP_PATH, overwrite: bool = False, timeout_s: Optional[float] = 60.0) -> str:
    dst_path = os.path.abspath(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and not overwrite:
        print(f"[download] ZIP already exists: {dst_path}")
        return dst_path
    with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        tmp_path = dst_path + ".part"
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading DR-VCTK") as bar:
            for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
                    if total: bar.update(len(chunk))
    os.replace(tmp_path, dst_path)
    print(f"[download] Saved: {dst_path}")
    return dst_path

if __name__ == "__main__":
    download_zip()

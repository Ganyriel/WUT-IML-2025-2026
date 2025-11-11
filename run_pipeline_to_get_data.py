#!/usr/bin/env python3
from download_zip_dataset import download_zip, ZIP_PATH as DEFAULT_ZIP
from build_data_subsets import build_subset
from build_manifest import build_manifest

def main():
    zip_path = download_zip()
    out_dir = build_subset(zip_path=zip_path)
    build_manifest(out_dir)

if __name__ == "__main__":
    main()
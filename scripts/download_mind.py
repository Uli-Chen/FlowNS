#!/usr/bin/env python
"""Download the MINDsmall dataset from Azure Blob Storage."""

import argparse
import os
import sys
import urllib.request
import zipfile

TRAIN_URL = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
DEV_URL = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)


def _reporthook(block_num, block_size, total_size):
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r  {pct:5.1f}%  ({mb_down:.1f} / {mb_total:.1f} MB)"
        )
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  {mb_down:.1f} MB downloaded")
    sys.stdout.flush()


def download_and_extract(url, dest_dir):
    """Download a zip from *url* and extract into *dest_dir*.

    If *dest_dir* already exists and is non-empty the download is skipped.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Skip if already extracted
    if os.listdir(dest_dir):
        print(f"[skip] {dest_dir} already exists and is non-empty.")
        return

    zip_name = url.rsplit("/", 1)[-1]
    zip_path = os.path.join(os.path.dirname(dest_dir), zip_name)

    print(f"Downloading {url}")
    print(f"  -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)
    print()  # newline after progress bar

    print(f"Extracting to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    os.remove(zip_path)
    print(f"  Removed {zip_path}")
    print(f"  Done: {dest_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download the MINDsmall dataset (train + dev)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_raw",
        help="Root directory for raw data (default: data_raw).",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "MINDsmall_train")
    dev_dir = os.path.join(args.data_dir, "MINDsmall_dev")

    download_and_extract(TRAIN_URL, train_dir)
    download_and_extract(DEV_URL, dev_dir)

    print("All downloads complete.")


if __name__ == "__main__":
    main()

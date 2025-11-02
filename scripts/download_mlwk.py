import os
import argparse
import requests
import zipfile
from io import BytesIO
from pathlib import Path

def download_movielens_100k(target_dir):
    """Download MovieLens 100K dataset (small, older movies from 1990s)."""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading MovieLens 100K from {url}...")
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('content-length', 0))
    z = zipfile.ZipFile(BytesIO(resp.content))
    z.extractall(target_dir)
    print(f"Extracted to {target_dir}")

def download_movielens_25m(target_dir):
    """Download MovieLens 25M dataset (large, recent movies up to 2019)."""
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading MovieLens 25M from {url}...")
    print("Note: This is a ~250MB download. Please be patient...")
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192
    content = BytesIO()
    
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            content.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='', flush=True)
    
    print("\nExtracting...")
    content.seek(0)
    z = zipfile.ZipFile(content)
    z.extractall(target_dir)
    print(f"Extracted to {target_dir}")

def detect_dataset_format(data_dir):
    """Detect which MovieLens format is in the directory."""
    # Check for 100K format
    if os.path.exists(os.path.join(data_dir, 'ml-100k', 'u.data')):
        return '100k'
    # Check for 25M format
    if os.path.exists(os.path.join(data_dir, 'ml-25m', 'ratings.csv')):
        return '25m'
    # Check for 25M in root
    if os.path.exists(os.path.join(data_dir, 'ratings.csv')):
        return '25m'
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract MovieLens dataset.")
    parser.add_argument("--target", type=str, default="./data/ml-25m",
                        help="Directory to extract dataset (default: ./data/ml-25m)")
    parser.add_argument("--dataset", type=str, choices=['100k', '25m'], default='25m',
                        help="Dataset to download: '100k' (small, old) or '25m' (large, recent). Default: 25m")
    args = parser.parse_args()
    
    if args.dataset == '100k':
        download_movielens_100k(args.target)
    else:
        download_movielens_25m(args.target)
    
    print("\nDownload complete!")

import os
import argparse
import requests
import zipfile
from io import BytesIO

def download_movielens_100k(target_dir):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {url}...")
    resp = requests.get(url)
    z = zipfile.ZipFile(BytesIO(resp.content))
    z.extractall(target_dir)
    print(f"Extracted to {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract MovieLens 100K dataset.")
    parser.add_argument("--target", type=str, default="./data/ml-100k",
                        help="Directory to extract MovieLens 100K (default: ./data/ml-100k)")
    args = parser.parse_args()
    download_movielens_100k(args.target)

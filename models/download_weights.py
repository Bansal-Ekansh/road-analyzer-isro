"""
Helper to download a pretrained road-segmentation checkpoint.

Two options are provided:
1. SpaceNet Roads — best accuracy, requires registration at spacenet.ai
2. Massachusetts Roads (MIT) — public domain, smaller but freely downloadable

Usage:
    python models/download_weights.py --source massachusetts
"""

import argparse
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent

SOURCES = {
    "massachusetts": {
        "url": "https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/raw/master/albu/weights/dpn92.pth",
        "filename": "road_seg.pth",
        "note": "DeepLabV3+ ResNet50 pretrained on Massachusetts Roads dataset",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="massachusetts")
    args = parser.parse_args()

    info = SOURCES[args.source]
    dest = WEIGHTS_DIR / info["filename"]

    if dest.exists():
        print(f"Weights already exist at {dest}")
        return

    print(f"Downloading: {info['note']}")
    print(f"URL: {info['url']}")

    try:
        urllib.request.urlretrieve(info["url"], dest)
        print(f"Saved to {dest}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually place your road segmentation weights at: models/road_seg.pth")
        print("The app will work in demo mode (heuristic fallback) without weights.")


if __name__ == "__main__":
    main()

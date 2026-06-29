"""
Downloads a handful of freely-licensed satellite images suitable for
road-network extraction demos.  Run once before the hackathon.

Usage:
    python data/download_samples.py
"""

import urllib.request
from pathlib import Path

SAMPLES = {
    "urban_roads_1.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/"
        "VanLeeuwenhoekpark_Leiden.jpg/1280px-VanLeeuwenhoekpark_Leiden.jpg"
    ),
    "highway_intersection.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Motorway_M25_junction_12.jpg/1280px-Motorway_M25_junction_12.jpg"
    ),
    "grid_city.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/"
        "NASA_Landsat_Chicago.jpg/1280px-NASA_Landsat_Chicago.jpg"
    ),
}

DEST = Path(__file__).parent / "samples"
DEST.mkdir(parents=True, exist_ok=True)

for name, url in SAMPLES.items():
    dest = DEST / name
    if dest.exists():
        print(f"  already exists: {name}")
        continue
    print(f"  downloading {name}…", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("done")
    except Exception as e:
        print(f"FAILED ({e})")

print("\nSample images saved to", DEST)

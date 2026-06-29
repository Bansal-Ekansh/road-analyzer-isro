"""
Creates a road segmentation training dataset for INDIAN cities by:
  1. Downloading road networks from OpenStreetMap via osmnx
  2. Fetching matching satellite tiles from a public tile server
  3. Rasterizing OSM roads onto the tiles to create binary masks

Output structure:
  data/indian_roads/
    images/   *.png   (RGB satellite patches, 512x512)
    masks/    *.png   (binary road masks, 512x512, 0 or 255)

Usage:
    pip install osmnx mercantile pillow requests tqdm
    python train/prepare_indian_data.py
"""

import math
import os
import io
import time
from pathlib import Path

import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import osmnx as ox
    import mercantile
except ImportError:
    raise SystemExit(
        "Missing packages — run: pip install osmnx mercantile"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Indian cities to include — add or remove as needed
INDIAN_CITIES = [
    "Bengaluru, Karnataka, India",
    "Chennai, Tamil Nadu, India",
    "Mumbai, Maharashtra, India",
    "Delhi, India",
    "Hyderabad, Telangana, India",
    "Kolkata, West Bengal, India",
    "Pune, Maharashtra, India",
    "Ahmedabad, Gujarat, India",
    "Jaipur, Rajasthan, India",
    "Kochi, Kerala, India",          # laterite / red roads
    "Bhopal, Madhya Pradesh, India",
    "Nagpur, Maharashtra, India",
]

ZOOM_LEVEL    = 17        # 17 = ~1.2 m/px (good for road widths)
TILE_SIZE     = 256       # standard OSM tile size
PATCH_SIZE    = 512       # output patch = 2×2 tiles merged
TILES_PER_CITY = 80       # how many tile-patches to extract per city
OUTPUT_DIR    = Path("data/indian_roads")
TILE_SERVER   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # free OSM tiles
USER_AGENT    = "ISROHackathon2026/1.0 (kdsp9206@gmail.com)"

ROAD_TYPES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "service", "track",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
]

# Road width in pixels at zoom 17 by road type (approximate)
ROAD_WIDTHS = {
    "motorway": 8,  "trunk": 7,   "primary": 6,  "secondary": 5,
    "tertiary": 4,  "residential": 3, "unclassified": 3, "service": 2,
    "track": 2,     "default": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_pixel(lat, lon, tile_x, tile_y, zoom):
    """Convert (lat, lon) → pixel (x, y) within a 256×256 tile grid."""
    n = 2 ** zoom
    # Tile origin in pixel space
    origin_px = tile_x * TILE_SIZE
    origin_py = tile_y * TILE_SIZE
    # Global pixel coords
    gx = (lon + 180.0) / 360.0 * n * TILE_SIZE
    sin_lat = math.sin(math.radians(lat))
    gy = (1.0 - math.log((1 + sin_lat) / (1 - sin_lat)) / (2 * math.pi)) / 2 * n * TILE_SIZE
    return gx - origin_px, gy - origin_py


def fetch_tile(z, x, y, session: requests.Session) -> Image.Image | None:
    """Download one OSM tile as a PIL Image."""
    url = TILE_SERVER.format(z=z, x=x, y=y)
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Road rasterizer
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_roads(
    road_network,
    tile_x: int, tile_y: int, zoom: int,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Draw OSM road edges onto a blank canvas of size×size pixels.
    Returns a uint8 binary mask (0 / 255).
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    edges = road_network.edges(data=True)
    for u, v, data in edges:
        highway = data.get("highway", "default")
        if isinstance(highway, list):
            highway = highway[0]
        width = ROAD_WIDTHS.get(highway, ROAD_WIDTHS["default"])

        # Get geometry
        geom = data.get("geometry")
        if geom is not None:
            coords = list(geom.coords)
        else:
            coords = [
                (road_network.nodes[u]["x"], road_network.nodes[u]["y"]),
                (road_network.nodes[v]["x"], road_network.nodes[v]["y"]),
            ]

        pixels = [
            latlon_to_pixel(lat, lon, tile_x, tile_y, zoom)
            for lon, lat in coords
        ]

        if len(pixels) >= 2:
            flat = [coord for px in pixels for coord in px]
            if len(flat) >= 4:
                draw.line(flat, fill=255, width=width)

    return np.array(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def prepare_city(city_name: str, session: requests.Session, img_dir: Path, msk_dir: Path):
    print(f"\n  Processing: {city_name}")

    try:
        G = ox.graph_from_place(
            city_name,
            network_type="drive",
            retain_all=False,
            simplify=True,
        )
    except Exception as e:
        print(f"    Could not fetch OSM data: {e}")
        return 0

    # Get bounding box of the city road network
    nodes = ox.graph_to_gdfs(G, edges=False)
    lat_min, lat_max = nodes.geometry.y.min(), nodes.geometry.y.max()
    lon_min, lon_max = nodes.geometry.x.min(), nodes.geometry.x.max()

    # Get all tiles covering this bbox at our zoom level
    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=ZOOM_LEVEL))
    if not tiles:
        return 0

    # Sample TILES_PER_CITY 2×2 tile patches
    import random
    random.shuffle(tiles)
    saved = 0

    for tile in tiles:
        if saved >= TILES_PER_CITY:
            break

        tx, ty, tz = tile.x, tile.y, tile.z

        # Fetch 2×2 tiles and stitch into a 512×512 image
        imgs = []
        valid = True
        for dy in range(2):
            row = []
            for dx in range(2):
                img = fetch_tile(tz, tx + dx, ty + dy, session)
                time.sleep(0.05)   # be polite to tile server
                if img is None:
                    valid = False
                    break
                row.append(img)
            if not valid:
                break
            imgs.append(row)

        if not valid:
            continue

        # Stitch tiles
        patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
        for dy in range(2):
            for dx in range(2):
                patch.paste(imgs[dy][dx], (dx * TILE_SIZE, dy * TILE_SIZE))

        patch_np = np.array(patch)

        # Rasterize roads onto the same 512×512 canvas
        mask = rasterize_roads(G, tx, ty, tz, size=PATCH_SIZE)

        # Skip if less than 0.5% of pixels are roads (empty patch)
        road_ratio = (mask > 0).sum() / (PATCH_SIZE * PATCH_SIZE)
        if road_ratio < 0.005:
            continue

        # Save
        stem = f"{city_name.split(',')[0].replace(' ', '_')}_{tx}_{ty}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_dir / f"{stem}.png"), mask)
        saved += 1

    print(f"    Saved {saved} patches")
    return saved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / "images"
    msk_dir = OUTPUT_DIR / "masks"
    img_dir.mkdir(exist_ok=True)
    msk_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    total = 0
    for city in tqdm(INDIAN_CITIES, desc="Cities"):
        total += prepare_city(city, session, img_dir, msk_dir)

    print(f"\n✅ Done — {total} training patches saved to {OUTPUT_DIR}/")
    print("   Next step: python train/train.py")


if __name__ == "__main__":
    main()

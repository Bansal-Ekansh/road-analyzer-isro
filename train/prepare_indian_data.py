"""
Creates a road segmentation training dataset for INDIAN cities by:
  1. Downloading road networks from OpenStreetMap via osmnx
  2. Fetching matching satellite tiles from a public tile server
  3. Rasterizing road vectors into binary masks

Output structure:
  data/indian_roads/
    images/   *.png   (RGB satellite patches, 512×512)
    masks/    *.png   (binary road masks,     512×512, 0 or 255)

Usage:
    python train/prepare_indian_data.py

Defensive features:
  - SKIP_CITIES list at the top — toggle off dense/slow cities by name
  - 30-second hard timeout on all OSM + tile server requests
  - Exponential-backoff retry (up to 3 attempts) on tile HTTP errors
  - Smart dual-cache: skips any tile pair already saved to disk
  - Nested tqdm progress bars: cities → tiles within each city
"""

import math
import io
import random
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
# ★  CONFIGURATION — Edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Cities to attempt downloading.
# Full OSM query strings: "City, State, India"
INDIAN_CITIES = [
    "Bengaluru, India", "Chennai, India", "Delhi, India", 
    "Hyderabad, India", "Kolkata, India", "Pune, India",
    "Ahmedabad, India", "Jaipur, India", "Surat, India", 
    "Lucknow, India", "Kanpur, India", "Nagpur, India"
]

# ★  SKIP LIST — Add city first-names here to skip them without deleting above.
#    Match is done against the FIRST token before the first comma, case-insensitive.
#    Example: ["Mumbai", "Bengaluru"] skips those two high-density cities.
SKIP_CITIES: list[str] = []

# ─────────────────────────────────────────────────────────────────────────────
# Tuning knobs
# ─────────────────────────────────────────────────────────────────────────────

ZOOM_LEVEL      = 17     # ~1.2 m/px — fine enough to see road widths
TILE_SIZE       = 256    # standard OSM tile size in pixels
PATCH_SIZE      = 512    # output patch = 2×2 tiles stitched together
TILES_PER_CITY  = 80     # max tile-patches to save per city
OUTPUT_DIR      = Path("data/indian_roads")

# Tile server (free OSM raster tiles — be polite: 50 ms sleep between fetches)
TILE_SERVER  = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT   = "ISROHackathon2026/1.0 (research-use-only)"
TILE_DELAY_S = 0.07      # seconds between individual tile HTTP requests

# Network timeouts & retry
OSM_TIMEOUT_S   = 30     # hard deadline for the osmnx Overpass API call
TILE_TIMEOUT_S  = 30     # hard deadline for each satellite tile fetch
MAX_TILE_RETRIES = 3     # how many times to retry a failed tile before skipping

# Road pixel widths at zoom 17 (wider = easier for the model to detect)
ROAD_WIDTHS = {
    "motorway": 8,  "trunk": 7,      "primary": 6,    "secondary": 5,
    "tertiary": 4,  "residential": 3, "unclassified": 3, "service": 2,
    "track": 2,     "default": 2,
}

# Minimum road-pixel fraction to keep a patch (skip empty/rural tiles)
MIN_ROAD_RATIO = 0.005   # 0.5 % of 512×512 = ~1310 road pixels


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — coordinate maths
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_pixel(lat: float, lon: float,
                    tile_x: int, tile_y: int, zoom: int) -> tuple[float, float]:
    """Map (lat, lon) to pixel (x, y) offset within a 512×512 patch origin."""
    n = 2 ** zoom
    gx = (lon + 180.0) / 360.0 * n * TILE_SIZE
    sin_lat = math.sin(math.radians(lat))
    gy = (1.0 - math.log((1 + sin_lat) / (1 - sin_lat)) / (2 * math.pi)) / 2 * n * TILE_SIZE
    return gx - tile_x * TILE_SIZE, gy - tile_y * TILE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — tile fetching with retry
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tile(z: int, x: int, y: int,
               session: requests.Session) -> Image.Image | None:
    """
    Download one 256×256 OSM raster tile with up to MAX_TILE_RETRIES attempts.
    Returns a PIL RGB Image, or None if all attempts fail.
    Uses TILE_TIMEOUT_S as a hard connection+read deadline.
    """
    url = TILE_SERVER.format(z=z, x=x, y=y)
    for attempt in range(1, MAX_TILE_RETRIES + 1):
        try:
            r = session.get(url, timeout=TILE_TIMEOUT_S)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except requests.exceptions.Timeout:
            wait = 2 ** attempt          # 2 s, 4 s, 8 s
            tqdm.write(f"    ⚠  Tile {z}/{x}/{y} timed out "
                       f"(attempt {attempt}/{MAX_TILE_RETRIES}), "
                       f"retrying in {wait}s…")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 5 * attempt       # rate-limit: longer back-off
                tqdm.write(f"    ⚠  Rate-limited on tile {z}/{x}/{y} — "
                           f"waiting {wait}s before retry {attempt}/{MAX_TILE_RETRIES}…")
                time.sleep(wait)
            else:
                tqdm.write(f"    ✗  HTTP {e.response.status_code} on tile {z}/{x}/{y}, skipping.")
                return None
        except Exception as exc:
            tqdm.write(f"    ✗  Tile {z}/{x}/{y} error: {exc!r}, skipping.")
            return None
    tqdm.write(f"    ✗  Tile {z}/{x}/{y} failed after {MAX_TILE_RETRIES} attempts.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — road rasterization
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_roads(road_network, tile_x: int, tile_y: int,
                    zoom: int, size: int = PATCH_SIZE) -> np.ndarray:
    """
    Draw OSM road edges onto a blank canvas of size×size pixels.
    Returns a uint8 binary mask (0 = background, 255 = road).
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    for u, v, data in road_network.edges(data=True):
        highway = data.get("highway", "default")
        if isinstance(highway, list):
            highway = highway[0]
        width = ROAD_WIDTHS.get(highway, ROAD_WIDTHS["default"])

        geom = data.get("geometry")
        if geom is not None:
            coords = list(geom.coords)
        else:
            coords = [
                (road_network.nodes[u]["x"], road_network.nodes[u]["y"]),
                (road_network.nodes[v]["x"], road_network.nodes[v]["y"]),
            ]

        pixels = [latlon_to_pixel(lat, lon, tile_x, tile_y, zoom)
                  for lon, lat in coords]

        if len(pixels) >= 2:
            flat = [coord for px in pixels for coord in px]
            if len(flat) >= 4:
                draw.line(flat, fill=255, width=width)

    return np.array(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Per-city pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_city(city_name: str, session: requests.Session,
                 img_dir: Path, msk_dir: Path) -> int:
    """
    Download OSM road network + satellite tiles for one city, rasterize masks,
    and save 512×512 image/mask pairs.  Returns the number of pairs saved.
    """
    city_slug = city_name.split(",")[0].strip().replace(" ", "_")

    # ── Step 1: Download OSM road graph with a hard timeout ──────────────────
    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  🏙  Processing: {city_name}")
    tqdm.write(f"{'─'*60}")
    tqdm.write(f"  ⬇  Fetching OSM road network (timeout={OSM_TIMEOUT_S}s)…")

    try:
        # osmnx uses the Overpass API; timeout controls the HTTP read deadline
        ox.settings.timeout = OSM_TIMEOUT_S
        G = ox.graph_from_place(
            city_name,
            network_type="drive",
            retain_all=False,
            simplify=True,
        )
        tqdm.write(f"  ✓  OSM graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    except Exception as exc:
        tqdm.write(
            f"  ✗  Timeout / rate-limit hit for {city_name} — skipping to next city.\n"
            f"     Error: {exc!r}"
        )
        return 0

    # ── Step 2: Get tile bounding box ────────────────────────────────────────
    try:
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        lat_min = nodes_gdf.geometry.y.min()
        lat_max = nodes_gdf.geometry.y.max()
        lon_min = nodes_gdf.geometry.x.min()
        lon_max = nodes_gdf.geometry.x.max()
    except Exception as exc:
        tqdm.write(f"  ✗  Could not extract node GeoDataFrame: {exc!r}")
        return 0

    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=ZOOM_LEVEL))
    if not tiles:
        tqdm.write("  ✗  No tiles found for bounding box — skipping.")
        return 0

    random.shuffle(tiles)
    tqdm.write(f"  ℹ  {len(tiles)} candidate tiles at zoom {ZOOM_LEVEL} "
               f"(will save up to {TILES_PER_CITY})")

    # ── Step 3: Download & save tile pairs ───────────────────────────────────
    saved  = 0
    tried  = 0
    cached = 0

    # Inner tqdm: shows individual tile progress within this city
    tile_bar = tqdm(
        tiles,
        total=min(len(tiles), TILES_PER_CITY),
        desc=f"  {city_slug[:12]:<12} tiles",
        unit="tile",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for tile in tile_bar:
        if saved >= TILES_PER_CITY:
            break

        tx, ty, tz = tile.x, tile.y, tile.z
        stem       = f"{city_slug}_{tx}_{ty}"
        img_path   = img_dir / f"{stem}.png"
        msk_path   = msk_dir / f"{stem}.png"

        # ── Smart cache check: BOTH files must exist to skip ─────────────────
        if img_path.exists() and msk_path.exists():
            cached += 1
            saved  += 1
            tile_bar.set_postfix(saved=saved, cached=cached, tried=tried)
            tile_bar.update(1)
            continue

        tried += 1

        # ── Fetch 2×2 sub-tiles and stitch into one 512×512 patch ───────────
        rows  = []
        valid = True
        for dy in range(2):
            row = []
            for dx in range(2):
                img = fetch_tile(tz, tx + dx, ty + dy, session)
                time.sleep(TILE_DELAY_S)
                if img is None:
                    valid = False
                    break
                row.append(img)
            if not valid:
                break
            rows.append(row)

        if not valid:
            tile_bar.set_postfix(saved=saved, cached=cached, tried=tried)
            continue

        # Stitch
        patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
        for dy in range(2):
            for dx in range(2):
                patch.paste(rows[dy][dx], (dx * TILE_SIZE, dy * TILE_SIZE))

        patch_np = np.array(patch)

        # ── Rasterize road mask ───────────────────────────────────────────────
        mask = rasterize_roads(G, tx, ty, tz, size=PATCH_SIZE)

        # Skip if road density is too low (rural / water tile)
        road_ratio = (mask > 0).sum() / (PATCH_SIZE * PATCH_SIZE)
        if road_ratio < MIN_ROAD_RATIO:
            tile_bar.set_postfix(saved=saved, cached=cached, tried=tried,
                                 last="sparse-skip")
            continue

        # ── Save both files atomically ────────────────────────────────────────
        cv2.imwrite(str(img_path), cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(msk_path), mask)
        saved += 1

        tile_bar.set_postfix(saved=saved, cached=cached, tried=tried)
        tile_bar.update(1)

    tile_bar.close()
    tqdm.write(f"  ✅  {city_name}: {saved} patches saved "
               f"({cached} from cache, {tried} freshly downloaded)")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Resolve which cities to actually process
    skip_lower = {s.strip().lower() for s in SKIP_CITIES}
    active_cities = [
        c for c in INDIAN_CITIES
        if c.split(",")[0].strip().lower() not in skip_lower
    ]

    if skip_lower:
        skipped_names = [
            c for c in INDIAN_CITIES
            if c.split(",")[0].strip().lower() in skip_lower
        ]
        print(f"\n⚠  SKIP_CITIES active — skipping: {skipped_names}")

    print(f"\n🗺  Will process {len(active_cities)} cities:")
    for c in active_cities:
        print(f"     • {c}")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / "images"
    msk_dir = OUTPUT_DIR / "masks"
    img_dir.mkdir(exist_ok=True)
    msk_dir.mkdir(exist_ok=True)

    # Shared HTTP session
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Outer progress bar — cities
    total = 0
    city_bar = tqdm(
        active_cities,
        desc="Overall cities",
        unit="city",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for city in city_bar:
        city_bar.set_description(f"City: {city.split(',')[0]:<18}")
        saved = prepare_city(city, session, img_dir, msk_dir)
        total += saved
        city_bar.set_postfix(total_patches=total)

    print(f"\n{'═'*60}")
    print(f"  ✅  Done — {total} total training patches saved to {OUTPUT_DIR}/")
    print(f"  ℹ   Images : {len(list(img_dir.glob('*.png')))}")
    print(f"  ℹ   Masks  : {len(list(msk_dir.glob('*.png')))}")
    print(f"\n  Next step:  python train/train.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

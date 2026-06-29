"""
Generates a synthetic demo dataset so the app runs instantly
without uploading a real satellite image.

Creates a realistic-looking Indian city grid satellite image,
a matching road mask, then runs the full graph + analytics pipeline.
"""

import numpy as np
import cv2
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.graph_builder import GraphBuilder
from pipeline.healing import GraphHealer
from pipeline.analytics import GraphAnalyzer


def _make_synthetic_satellite(size: int = 512) -> np.ndarray:
    """Produce a satellite-style city image: buildings, vegetation, roads."""
    rng = np.random.default_rng(42)
    img = np.full((size, size, 3), [38, 35, 28], dtype=np.uint8)

    # Scatter rectangular city blocks (buildings, rooftops)
    block_colors = [
        [62, 58, 52],  [75, 70, 60],  [85, 80, 65],
        [45, 55, 38],  [90, 85, 72],  [55, 50, 45],
        [70, 65, 55],  [50, 60, 42],
    ]
    for _ in range(180):
        x  = int(rng.integers(0, size - 40))
        y  = int(rng.integers(0, size - 40))
        w  = int(rng.integers(12, 42))
        h  = int(rng.integers(12, 42))
        c  = block_colors[int(rng.integers(0, len(block_colors)))]
        var = int(rng.integers(-8, 8))
        img[y:y+h, x:x+w] = np.clip(np.array(c) + var, 0, 255)

    # Vegetation patches
    for _ in range(60):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r,
                   (int(rng.integers(28, 55)), int(rng.integers(55, 85)), int(rng.integers(22, 45))), -1)

    # Road grid — main arterials
    road_color = [108, 105, 100]
    for pos in [96, 192, 288, 384, 448]:
        img[pos-7:pos+7, 10:size-10] = road_color           # horizontal
        img[10:size-10, pos-7:pos+7] = road_color           # vertical

    # Secondary roads (thinner)
    sec_color = [88, 85, 80]
    for pos in [48, 144, 240, 336, 416]:
        img[pos-3:pos+3, 10:size-10] = sec_color
        img[10:size-10, pos-3:pos+3] = sec_color

    # Diagonal connector road (makes graph more interesting)
    pts = np.array([[10, 10], [160, 96], [256, 192], [352, 288], [448, 384]], np.int32)
    cv2.polylines(img, [pts], False, [98, 95, 90], 5)

    # Add realistic noise
    noise = rng.integers(-12, 12, img.shape, dtype=int)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    # Slight Gaussian blur — simulates atmospheric blur in satellite imagery
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _make_road_mask(size: int = 512) -> np.ndarray:
    """Create a binary road mask matching the synthetic city image above."""
    mask = np.zeros((size, size), dtype=np.uint8)

    # Main arterials
    for pos in [96, 192, 288, 384, 448]:
        mask[pos-7:pos+7, 10:size-10] = 255
        mask[10:size-10, pos-7:pos+7] = 255

    # Secondary roads
    for pos in [48, 144, 240, 336, 416]:
        mask[pos-3:pos+3, 10:size-10] = 255
        mask[10:size-10, pos-3:pos+3] = 255

    # Diagonal connector
    pts = np.array([[10, 10], [160, 96], [256, 192], [352, 288], [448, 384]], np.int32)
    cv2.polylines(mask, [pts], False, 255, 5)

    # Simulate occlusion gaps (shadows / tree cover)
    rng = np.random.default_rng(7)
    for _ in range(12):
        gx = int(rng.integers(20, size - 20))
        gy = int(rng.integers(20, size - 20))
        gw = int(rng.integers(8, 20))
        gh = int(rng.integers(8, 20))
        mask[gy:gy+gh, gx:gx+gw] = 0

    return mask


@st.cache_data(show_spinner=False)
def load_demo() -> dict:
    """Build and cache the full demo pipeline result."""
    image   = _make_synthetic_satellite()
    mask    = _make_road_mask()

    builder  = GraphBuilder()
    skeleton, G = builder.build(mask)

    healer = GraphHealer(max_gap_px=20)
    G      = healer.heal(G)

    analyzer = GraphAnalyzer(G)
    analyzer.compute_centrality()
    cfi = analyzer.cascading_failure_index(top_k=5)

    return {
        "image":    image,
        "mask":     mask,
        "skeleton": skeleton,
        "graph":    G,
        "analyzer": analyzer,
        "cfi":      cfi,
        "report":   None,
    }

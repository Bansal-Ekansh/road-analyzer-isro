"""
Converts a binary road mask into a weighted NetworkX graph.

Pipeline:
  mask (H×W uint8)
    → skeletonize          (1-pixel-wide centrelines)
    → classify pixels      (endpoint / road / junction)
    → trace road segments  (walk skeleton between junctions/endpoints)
    → build NetworkX graph (nodes = junctions+endpoints, edges = segments)
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import cv2
from typing import Optional


# 8-connectivity offsets
_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class GraphBuilder:
    def __init__(self, min_edge_length: int = 5):
        self.min_edge_length = min_edge_length   # discard very short spurious edges

    # ------------------------------------------------------------------
    def build(self, road_mask: np.ndarray) -> tuple[np.ndarray, nx.Graph]:
        """
        Parameters
        ----------
        road_mask : np.ndarray  uint8 (0 / 255)

        Returns
        -------
        skeleton : np.ndarray   uint8 (0 / 255)
        G        : nx.Graph     weighted graph
        """
        skeleton = self._skeletonize(road_mask)
        G        = self._skeleton_to_graph(skeleton)
        G        = self._attach_pixel_coords(G)
        return skeleton, G

    # ------------------------------------------------------------------
    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        binary   = (mask > 127).astype(bool)
        skeleton = skeletonize(binary)
        return skeleton.astype(np.uint8) * 255

    # ------------------------------------------------------------------
    def _skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        ys, xs = np.where(skeleton > 0)
        if len(ys) == 0:
            return nx.Graph()

        pixel_set = set(zip(ys.tolist(), xs.tolist()))

        def neighbors(y, x):
            return [(y+dy, x+dx) for dy, dx in _OFFSETS if (y+dy, x+dx) in pixel_set]

        # Classify every skeleton pixel
        endpoints  = set()
        junctions  = set()
        for y, x in pixel_set:
            n = len(neighbors(y, x))
            if n == 1:
                endpoints.add((y, x))
            elif n >= 3:
                junctions.add((y, x))

        special = endpoints | junctions

        # Add isolated pixels as single nodes
        if not special:
            # Entire skeleton is a single loop or short segment
            p = next(iter(pixel_set))
            G = nx.Graph()
            G.add_node(p, y=p[0], x=p[1])
            return G

        G = nx.Graph()
        for p in special:
            G.add_node(p, y=p[0], x=p[1])

        visited_edges: set[tuple] = set()

        for start in special:
            for nb in neighbors(*start):
                edge_key = (min(start, nb), max(start, nb))
                if edge_key in visited_edges:
                    continue

                # Trace path from start → nb → ... until another special pixel
                path  = [start, nb]
                prev  = start
                curr  = nb

                while curr not in special:
                    nxt_list = [n for n in neighbors(*curr) if n != prev]
                    if not nxt_list:
                        break          # dead end (artifact)
                    prev = curr
                    curr = nxt_list[0]
                    path.append(curr)

                if curr in special and curr != start:
                    length = self._path_length(path)
                    if length >= self.min_edge_length:
                        G.add_node(curr, y=curr[0], x=curr[1])
                        G.add_edge(start, curr, weight=length, path=path)
                        visited_edges.add((min(start, curr), max(start, curr)))

        return G

    # ------------------------------------------------------------------
    @staticmethod
    def _path_length(path: list) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            dy = path[i+1][0] - path[i][0]
            dx = path[i+1][1] - path[i][1]
            total += (dy*dy + dx*dx) ** 0.5
        return total

    # ------------------------------------------------------------------
    @staticmethod
    def _attach_pixel_coords(G: nx.Graph) -> nx.Graph:
        """Ensure every node has explicit 'y' and 'x' attributes."""
        for node in G.nodes():
            if "y" not in G.nodes[node]:
                G.nodes[node]["y"] = node[0]
                G.nodes[node]["x"] = node[1]
        return G


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def pixel_to_latlon(
    y: float, x: float,
    image_shape: tuple,
    bbox: Optional[tuple] = None,
) -> tuple[float, float]:
    """
    Map pixel (y, x) to (lat, lon) given a bounding-box (lat_min, lat_max, lon_min, lon_max).
    If no bbox is provided, returns normalised [0,1] floats as placeholder coords.
    """
    h, w = image_shape[:2]
    if bbox is None:
        return (1 - y / h, x / w)
    lat_min, lat_max, lon_min, lon_max = bbox
    lat = lat_max - (y / h) * (lat_max - lat_min)
    lon = lon_min + (x / w) * (lon_max - lon_min)
    return lat, lon

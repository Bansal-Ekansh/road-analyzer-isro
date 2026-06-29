"""
Topological healing: find disconnected terminal nodes and bridge small gaps
caused by shadows, trees, or segmentation errors.
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


class GraphHealer:
    def __init__(
        self,
        max_gap_px:   int   = 25,   # max pixel distance to bridge
        angle_tol_deg: float = 35.0, # directional alignment tolerance
    ):
        self.max_gap_px    = max_gap_px
        self.angle_tol_deg = angle_tol_deg

    # ------------------------------------------------------------------
    def heal(self, G: nx.Graph) -> nx.Graph:
        """Return a new graph with bridged gaps.  Original graph is untouched."""
        G2        = G.copy()
        endpoints = self._find_endpoints(G2)
        if len(endpoints) < 2:
            return G2

        coords = np.array([[G2.nodes[n]["y"], G2.nodes[n]["x"]] for n in endpoints])
        tree   = cKDTree(coords)

        pairs  = tree.query_pairs(r=self.max_gap_px)
        healed = 0

        for i, j in pairs:
            n1, n2 = endpoints[i], endpoints[j]
            if n1 == n2 or G2.has_edge(n1, n2):
                continue
            if self._same_component(G2, n1, n2):
                continue
            if self._aligned(G2, n1, n2):
                dist = self._dist(G2, n1, n2)
                G2.add_edge(n1, n2, weight=dist, bridged=True)
                healed += 1

        if healed:
            print(f"[Healer] Bridged {healed} gap(s).")
        return G2

    # ------------------------------------------------------------------
    @staticmethod
    def _find_endpoints(G: nx.Graph) -> list:
        return [n for n in G.nodes() if G.degree(n) == 1]

    @staticmethod
    def _same_component(G: nx.Graph, u, v) -> bool:
        return nx.node_connectivity(G, u, v) > 0

    @staticmethod
    def _dist(G: nx.Graph, u, v) -> float:
        dy = G.nodes[u]["y"] - G.nodes[v]["y"]
        dx = G.nodes[u]["x"] - G.nodes[v]["x"]
        return (dy*dy + dx*dx) ** 0.5

    def _aligned(self, G: nx.Graph, u, v) -> bool:
        """
        Check whether the 'outgoing' direction of endpoint u roughly points
        toward v (and vice versa), to avoid bridging across perpendicular roads.
        """
        vec_uv  = self._endpoint_direction(G, u, v)
        vec_vu  = self._endpoint_direction(G, v, u)
        if vec_uv is None or vec_vu is None:
            return True  # no context — allow bridge

        dot = np.dot(vec_uv, -vec_vu)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        return angle <= self.angle_tol_deg

    @staticmethod
    def _endpoint_direction(G: nx.Graph, endpoint, target) -> np.ndarray | None:
        """Unit vector from the endpoint's neighbour toward the endpoint (road direction)."""
        nbrs = list(G.neighbors(endpoint))
        if not nbrs:
            return None
        nb = nbrs[0]
        dy = G.nodes[endpoint]["y"] - G.nodes[nb]["y"]
        dx = G.nodes[endpoint]["x"] - G.nodes[nb]["x"]
        length = (dy*dy + dx*dx) ** 0.5
        if length < 1e-6:
            return None
        return np.array([dy / length, dx / length])

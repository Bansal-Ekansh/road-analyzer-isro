"""
Graph analytics: centrality metrics, resilience scoring, and
emergency alternate-route finding.
"""

import math
import networkx as nx
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ResilienceReport:
    baseline_efficiency: float
    post_failure_efficiency: float
    resilience_score: float          # 0–100
    failed_nodes: list
    n_components_before: int
    n_components_after: int
    n_isolated_nodes: int
    alternate_routes: dict = field(default_factory=dict)   # (src, tgt) → path


class GraphAnalyzer:
    def __init__(self, G: nx.Graph):
        self.G = G
        self._centrality_computed = False

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def compute_centrality(self) -> dict[str, dict]:
        """
        Adds betweenness, closeness, and degree centrality to every node.
        Returns a dict-of-dicts: {metric_name: {node: value}}.
        """
        G = self.G

        if len(G) == 0:
            return {}

        bet = nx.betweenness_centrality(G, normalized=True, weight="weight")
        clo = nx.closeness_centrality(G, distance="weight")
        deg = nx.degree_centrality(G)

        # Composite bottleneck score (weighted blend)
        nodes  = list(G.nodes())
        bmax   = max(bet.values()) if bet else 1.0
        cmax   = max(clo.values()) if clo else 1.0
        dmax   = max(deg.values()) if deg else 1.0

        for n in nodes:
            G.nodes[n]["betweenness"] = bet[n]
            G.nodes[n]["closeness"]   = clo[n]
            G.nodes[n]["degree_c"]    = deg[n]
            # Bottleneck index: 60% betweenness + 25% closeness + 15% degree
            G.nodes[n]["bottleneck"] = (
                0.60 * (bet[n] / max(bmax, 1e-9)) +
                0.25 * (clo[n] / max(cmax, 1e-9)) +
                0.15 * (deg[n] / max(dmax, 1e-9))
            )

        self._centrality_computed = True
        return {"betweenness": bet, "closeness": clo, "degree": deg}

    # ------------------------------------------------------------------
    # Top bottlenecks
    # ------------------------------------------------------------------

    def top_bottlenecks(self, k: int = 10) -> list[tuple]:
        """Return [(node, bottleneck_score), ...] sorted descending."""
        if not self._centrality_computed:
            self.compute_centrality()
        return sorted(
            [(n, self.G.nodes[n]["bottleneck"]) for n in self.G.nodes()],
            key=lambda t: t[1],
            reverse=True,
        )[:k]

    # ------------------------------------------------------------------
    # Global efficiency  (Latora & Marchiori 2001)
    # ------------------------------------------------------------------

    @staticmethod
    def global_efficiency(G: nx.Graph) -> float:
        n = len(G)
        if n < 2:
            return 0.0
        total = 0.0
        nodes = list(G.nodes())
        for u in nodes:
            lengths = nx.single_source_dijkstra_path_length(G, u, weight="weight")
            for v, d in lengths.items():
                if v != u and d > 0:
                    total += 1.0 / d
        denom = n * (n - 1)
        return total / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # Resilience simulation
    # ------------------------------------------------------------------

    def simulate_failure(
        self,
        failed_nodes: list,
        find_alternates: bool = True,
    ) -> ResilienceReport:
        """
        Remove `failed_nodes` from a copy of the graph, measure efficiency
        drop, and optionally find alternate routes between previously-connected
        pairs that are now disconnected.
        """
        G_orig = self.G

        baseline_eff = self.global_efficiency(G_orig)
        n_comp_before = nx.number_connected_components(G_orig)

        G_fail = G_orig.copy()
        valid_fails = [n for n in failed_nodes if n in G_fail]
        G_fail.remove_nodes_from(valid_fails)

        post_eff      = self.global_efficiency(G_fail)
        n_comp_after  = nx.number_connected_components(G_fail)
        n_isolated    = sum(1 for n in G_fail if G_fail.degree(n) == 0)

        if baseline_eff > 0:
            resilience = max(0.0, (post_eff / baseline_eff) * 100)
        else:
            resilience = 0.0

        alternates: dict = {}
        if find_alternates and valid_fails:
            alternates = self._find_alternate_routes(
                G_orig, G_fail, valid_fails
            )

        return ResilienceReport(
            baseline_efficiency   = baseline_eff,
            post_failure_efficiency = post_eff,
            resilience_score      = resilience,
            failed_nodes          = valid_fails,
            n_components_before   = n_comp_before,
            n_components_after    = n_comp_after,
            n_isolated_nodes      = n_isolated,
            alternate_routes      = alternates,
        )

    # ------------------------------------------------------------------
    # Alternate route finder  (WINNING FEATURE)
    # ------------------------------------------------------------------

    def _find_alternate_routes(
        self,
        G_orig: nx.Graph,
        G_fail: nx.Graph,
        failed_nodes: list,
        n_pairs: int = 3,
    ) -> dict:
        """
        Identify important node pairs that were connected before failure
        but are still connected after (i.e., there IS an alternate route).
        Returns the best alternate path for each such pair.
        """
        if not self._centrality_computed:
            self.compute_centrality()

        # Pick pairs: neighbours of failed nodes (most practically relevant)
        affected = set()
        for fn in failed_nodes:
            affected.update(G_orig.neighbors(fn))
        affected -= set(failed_nodes)
        affected = list(affected)[:8]   # cap for performance

        routes = {}
        checked = 0
        for i, u in enumerate(affected):
            for v in affected[i+1:]:
                if checked >= n_pairs:
                    break
                if u not in G_fail or v not in G_fail:
                    continue
                if not nx.has_path(G_fail, u, v):
                    continue
                try:
                    path = nx.shortest_path(G_fail, u, v, weight="weight")
                    routes[(u, v)] = path
                    checked += 1
                except nx.NetworkXNoPath:
                    pass

        return routes

    # ------------------------------------------------------------------
    # Cascading Failure Index  (unique differentiator)
    # ------------------------------------------------------------------

    def cascading_failure_index(self, top_k: int = 5) -> dict:
        """
        Remove the top-k bottleneck nodes and measure network degradation.
        Returns a dict with CFI score (0–100) and fragmentation count.
        """
        if not self._centrality_computed:
            self.compute_centrality()

        top_nodes = [n for n, _ in self.top_bottlenecks(top_k)]
        report    = self.simulate_failure(top_nodes, find_alternates=False)

        cfi = 100 - report.resilience_score   # higher = more fragile
        return {
            "cfi":                round(cfi, 1),
            "resilience_score":   round(report.resilience_score, 1),
            "fragmentation":      report.n_components_after - report.n_components_before,
            "isolated_nodes":     report.n_isolated_nodes,
            "top_nodes_removed":  top_nodes,
        }

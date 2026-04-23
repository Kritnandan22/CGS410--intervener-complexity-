"""
Encode dependency trees as networkx graphs for representation learning.
Computes graph-level structural features for each sentence.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.info("networkx not installed; graph encoding disabled")

from src.data.loader import Sentence
from src.parsing.tree import DependencyTree


class DependencyGraphEncoder:
    """Convert dependency trees to networkx directed graphs with node/edge attributes."""

    def __init__(self):
        if not HAS_NX:
            raise ImportError("networkx is required for graph encoding")

    def sentence_to_graph(self, sentence: Sentence, tree: DependencyTree) -> Optional[nx.DiGraph]:
        """Build a directed graph from a parsed sentence."""
        G = nx.DiGraph()

        # add nodes with attributes
        for tok in sentence.tokens:
            node_info = tree.nodes.get(tok.id)
            G.add_node(tok.id, **{
                "form": tok.form,
                "upos": tok.upos,
                "deprel": tok.deprel,
                "arity": tree.arity(tok.id),
                "depth": tree.depth(tok.id),
                "subtree_size": tree.subtree_size(tok.id),
            })

        # add edges (head -> dependent)
        for tok in sentence.tokens:
            if tok.head != 0 and tok.head in tree.nodes:
                G.add_edge(tok.head, tok.id, deprel=tok.deprel)

        return G

    def graph_features(self, G: nx.DiGraph) -> Dict[str, float]:
        """Compute graph-level structural features from a dependency graph."""
        n = G.number_of_nodes()
        if n == 0:
            return {"n_nodes": 0, "n_edges": 0, "density": 0, "avg_degree": 0,
                    "max_degree": 0, "avg_clustering": 0, "diameter": 0}

        # basic graph stats
        degrees = [d for _, d in G.degree()]
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        # clustering on undirected version
        G_und = G.to_undirected()
        try:
            avg_clustering = nx.average_clustering(G_und)
        except Exception:
            avg_clustering = 0.0

        # diameter (on largest connected component of undirected)
        try:
            if nx.is_connected(G_und):
                diameter = nx.diameter(G_und)
            else:
                largest_cc = max(nx.connected_components(G_und), key=len)
                diameter = nx.diameter(G_und.subgraph(largest_cc))
        except Exception:
            diameter = 0

        # degree centrality stats
        try:
            centrality = nx.degree_centrality(G)
            avg_centrality = float(np.mean(list(centrality.values())))
            max_centrality = float(max(centrality.values()))
        except Exception:
            avg_centrality = 0.0
            max_centrality = 0.0

        # betweenness
        try:
            betweenness = nx.betweenness_centrality(G)
            avg_betweenness = float(np.mean(list(betweenness.values())))
        except Exception:
            avg_betweenness = 0.0

        return {
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": float(np.mean(degrees)),
            "max_degree": max(degrees) if degrees else 0,
            "avg_in_degree": float(np.mean(in_degrees)),
            "avg_out_degree": float(np.mean(out_degrees)),
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "avg_clustering": avg_clustering,
            "diameter": diameter,
            "avg_centrality": avg_centrality,
            "max_centrality": max_centrality,
            "avg_betweenness": avg_betweenness,
        }

    def batch_graph_features(
        self,
        sentences: List[Sentence],
        max_sentences: int = 5000,
    ) -> List[Dict]:
        """Compute graph features for a batch of sentences."""
        rows = []
        for i, sent in enumerate(sentences[:max_sentences]):
            tree = DependencyTree(sent)
            G = self.sentence_to_graph(sent, tree)
            if G is None:
                continue
            feats = self.graph_features(G)
            feats["sentence_id"] = sent.sent_id
            rows.append(feats)
        return rows

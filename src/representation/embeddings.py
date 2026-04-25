"""
Generate fixed-length tree embeddings from dependency graph features.
Uses graph-level statistics as a feature vector for each sentence,
then aggregates per-language for cross-lingual comparison.
"""

from __future__ import annotations

import logging

from typing import Dict, List, Optional

import numpy as np

import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDING_FEATURES = [

    "n_nodes", "n_edges", "density", "avg_degree", "max_degree",

    "avg_in_degree", "avg_out_degree", "max_out_degree",

    "avg_clustering", "diameter", "avg_centrality", "max_centrality",

    "avg_betweenness",

]

class TreeEmbedder:

    """Aggregate graph-level features into per-language embedding vectors."""

    def __init__(self):

        self.scaler = None

    def sentence_embeddings(self, graph_features: List[Dict]) -> np.ndarray:

        """Convert list of graph feature dicts to a matrix (n_sentences x n_features)."""

        df = pd.DataFrame(graph_features)

        avail = [c for c in EMBEDDING_FEATURES if c in df.columns]

        if not avail:

            return np.zeros((len(df), 1))

        return df[avail].fillna(0).values.astype(float)

    def language_embedding(self, graph_features: List[Dict]) -> np.ndarray:

        """Compute a single embedding vector for a language (mean of sentence embeddings)."""

        mat = self.sentence_embeddings(graph_features)

        if len(mat) == 0:

            return np.zeros(len(EMBEDDING_FEATURES))

        return mat.mean(axis=0)

    def language_embedding_stats(self, graph_features: List[Dict]) -> Dict[str, float]:

        """Return mean and std for each embedding dimension — richer representation."""

        mat = self.sentence_embeddings(graph_features)

        if len(mat) == 0:

            return {}

        avail = [c for c in EMBEDDING_FEATURES if c in pd.DataFrame(graph_features).columns]

        result = {}

        for i, feat in enumerate(avail):

            if i < mat.shape[1]:

                result[f"{feat}_mean"] = float(mat[:, i].mean())

                result[f"{feat}_std"] = float(mat[:, i].std())

        return result

    def multi_language_matrix(

        self,

        lang_features: Dict[str, List[Dict]],

    ) -> tuple:

        """
        Build a (n_languages x n_features) matrix for cross-lingual comparison.
        Returns (language_names, embedding_matrix).
        """

        names = []

        vectors = []

        for lang, feats in sorted(lang_features.items()):

            vec = self.language_embedding(feats)

            names.append(lang)

            vectors.append(vec)

        if not vectors:

            return [], np.array([])

        mat = np.vstack(vectors)

        return names, mat

    def pca_reduce(self, matrix: np.ndarray, n_components: int = 2) -> np.ndarray:

        """Reduce embedding matrix to 2D via PCA for visualization."""

        from sklearn.decomposition import PCA

        from sklearn.preprocessing import StandardScaler

        if matrix.shape[0] < 3:

            return matrix[:, :n_components] if matrix.shape[1] >= n_components else matrix

        X = StandardScaler().fit_transform(matrix)

        pca = PCA(n_components=min(n_components, X.shape[1]), random_state=42)

        return pca.fit_transform(X)


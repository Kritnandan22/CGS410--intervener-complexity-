"""
Cluster intervener patterns across languages using KMeans and optional DBSCAN.
Operates on per-language embedding vectors produced by TreeEmbedder.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IntervenorClusterer:
    """Cluster languages by their intervener complexity patterns."""

    def __init__(self, n_clusters: int = 4, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed

    def cluster_languages(
        self,
        names: List[str],
        matrix: np.ndarray,
    ) -> pd.DataFrame:
        """
        Cluster languages using KMeans on their embedding vectors.
        Returns a DataFrame with language, cluster_id, and embedding coords.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        if len(names) < self.n_clusters:
            logger.warning("fewer languages (%d) than clusters (%d), adjusting",
                           len(names), self.n_clusters)
            k = max(2, len(names) // 2)
        else:
            k = self.n_clusters

        X = StandardScaler().fit_transform(matrix)
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        labels = km.fit_predict(X)

        rows = []
        for i, name in enumerate(names):
            row = {"language": name, "cluster_id": int(labels[i])}
            for j in range(matrix.shape[1]):
                row[f"dim_{j}"] = float(matrix[i, j])
            rows.append(row)

        return pd.DataFrame(rows)

    def dbscan_cluster(
        self,
        names: List[str],
        matrix: np.ndarray,
        eps: float = 1.5,
        min_samples: int = 2,
    ) -> pd.DataFrame:
        """Alternative clustering with DBSCAN (density-based, finds outliers)."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(matrix)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        rows = []
        for i, name in enumerate(names):
            rows.append({
                "language": name,
                "cluster_id": int(labels[i]),  # -1 = noise/outlier
            })
        return pd.DataFrame(rows)

    def silhouette_score(self, matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score to evaluate clustering quality."""
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        if len(set(labels)) < 2:
            return 0.0
        X = StandardScaler().fit_transform(matrix)
        return float(silhouette_score(X, labels))

    def cluster_summary(self, cluster_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Merge cluster assignments with language summary stats for interpretation."""
        if "language" not in cluster_df.columns or "language" not in summary_df.columns:
            return cluster_df
        merged = cluster_df.merge(summary_df, on="language", how="left")
        return merged

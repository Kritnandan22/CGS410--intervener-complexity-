"""
Distribution analysis: arity, POS, dependency length, KL divergence.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


class DistributionAnalyzer:

    @staticmethod
    def arity_distribution(arities: Sequence[int]) -> Dict[int, int]:
        return dict(Counter(arities))

    @staticmethod
    def pos_distribution(pos_list: Sequence[str]) -> Dict[str, int]:
        return dict(Counter(pos_list))

    @staticmethod
    def kl_divergence(p_counts: Dict, q_counts: Dict) -> float:
        """
        KL(P || Q) — symmetric (Jensen-Shannon) divergence.
        Smoothed with epsilon to avoid log(0).
        """
        vocab = set(p_counts) | set(q_counts)
        eps = 1e-9
        p_total = sum(p_counts.values()) or 1
        q_total = sum(q_counts.values()) or 1

        kl = 0.0
        for k in vocab:
            p = (p_counts.get(k, 0) + eps) / (p_total + eps * len(vocab))
            q = (q_counts.get(k, 0) + eps) / (q_total + eps * len(vocab))
            kl += p * math.log(p / q)
        return kl

    @staticmethod
    def js_divergence(p_counts: Dict, q_counts: Dict) -> float:
        """Jensen-Shannon divergence (symmetric, bounded [0,1])."""
        vocab = set(p_counts) | set(q_counts)
        eps = 1e-9
        p_total = sum(p_counts.values()) or 1
        q_total = sum(q_counts.values()) or 1

        p_vals = [(p_counts.get(k, 0) + eps) / (p_total + eps * len(vocab)) for k in vocab]
        q_vals = [(q_counts.get(k, 0) + eps) / (q_total + eps * len(vocab)) for k in vocab]
        m_vals = [(p + q) / 2 for p, q in zip(p_vals, q_vals)]

        def _kl(a, b):
            return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0)

        return (_kl(p_vals, m_vals) + _kl(q_vals, m_vals)) / 2

    @staticmethod
    def entropy(counts: Dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    @staticmethod
    def build_distribution_rows(
        language: str,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create distribution_data.csv rows for multiple metrics.
        Each row: (language, metric_type, value)
        """
        rows = []
        for col in ["arity", "subtree_size", "depth",
                    "dependency_distance", "complexity_score"]:
            if col in features_df.columns:
                for val in features_df[col].dropna():
                    rows.append({"language": language, "metric_type": col, "value": float(val)})
        if "intervener_upos" in features_df.columns:
            for pos in features_df["intervener_upos"].dropna():
                rows.append({"language": language, "metric_type": "intervener_upos", "value": pos})
        return pd.DataFrame(rows)

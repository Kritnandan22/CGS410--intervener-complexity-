"""
Composite complexity score and Intervener Efficiency Ratio (IER).

Complexity Score = w1*arity + w2*subtree_size + w3*depth + w4*pos_weight
IER = dependency_length / complexity_score
"""
from __future__ import annotations

import math
from typing import Dict


class ComplexityScorer:
    def __init__(self, weights: Dict, pos_weights: Dict):
        self.w_arity = weights.get("w_arity", 0.35)
        self.w_subtree = weights.get("w_subtree_size", 0.25)
        self.w_depth = weights.get("w_depth", 0.20)
        self.w_pos = weights.get("w_pos_weight", 0.20)
        self.pos_weights = pos_weights

    def score(
        self,
        arity: int,
        subtree_size: int,
        depth: int,
        upos: str,
    ) -> float:
        pos_w = self.pos_weights.get(upos, self.pos_weights.get("_", 0.3))
        return (
            self.w_arity * arity
            + self.w_subtree * subtree_size
            + self.w_depth * depth
            + self.w_pos * pos_w
        )

    def efficiency_ratio(self, dependency_length: int, complexity: float) -> float:
        """IER = dependency_length / complexity_score (avoid division by zero)."""
        if complexity <= 0:
            return float("nan")
        return dependency_length / complexity

    def label(self, complexity: float, threshold: float = 1.5) -> str:
        """Binary label for ML: 'high' or 'low' complexity."""
        return "high" if complexity >= threshold else "low"

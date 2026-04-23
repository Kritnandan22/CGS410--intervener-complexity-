"""Config loading and miscellaneous helpers."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def load_config(path: str = "config/config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_languages(path: str = "config/languages.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("languages", {})


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_contributor_languages(languages_cfg: Dict, contributor: str) -> List[str]:
    return [
        code for code, meta in languages_cfg.items()
        if meta.get("contributor", "").strip() == contributor.strip()
    ]


def build_language_summary_row(
    language: str,
    features_df,
    entropy: float,
) -> Dict:
    """Build a single language_summary.csv row from the features dataframe."""
    import pandas as pd
    from collections import Counter

    df = features_df
    n = len(df)
    if n == 0:
        return {"language": language}

    left = (df["direction"] == "left").sum() if "direction" in df.columns else 0
    right = (df["direction"] == "right").sum() if "direction" in df.columns else 0
    total_dir = left + right

    pos_counts = Counter(df["intervener_upos"].dropna().tolist()) if "intervener_upos" in df.columns else {}
    most_common_pos = pos_counts.most_common(1)[0][0] if pos_counts else ""

    return {
        "language": language,
        "avg_dependency_length": float(df["dependency_distance"].mean()) if "dependency_distance" in df.columns else None,
        "avg_complexity": float(df["complexity_score"].mean()) if "complexity_score" in df.columns else None,
        "avg_arity": float(df["arity"].mean()) if "arity" in df.columns else None,
        "avg_subtree_size": float(df["subtree_size"].mean()) if "subtree_size" in df.columns else None,
        "avg_depth": float(df["depth"].mean()) if "depth" in df.columns else None,
        "percent_left_dependencies": float(100 * left / total_dir) if total_dir > 0 else None,
        "percent_right_dependencies": float(100 * right / total_dir) if total_dir > 0 else None,
        "most_common_pos": most_common_pos,
        "entropy_pos_distribution": float(entropy),
        "avg_efficiency_ratio": float(
            df["efficiency_ratio"].replace([np.inf, -np.inf], np.nan).mean()
        ) if "efficiency_ratio" in df.columns else None,
    }

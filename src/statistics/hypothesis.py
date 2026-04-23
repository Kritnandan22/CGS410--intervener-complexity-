"""
Hypothesis testing: ANOVA, Mann-Whitney U, Cohen's d, chi-square, z-scores.
Mixed-effects models via statsmodels.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def z_score(
    real_value: float,
    random_mean: float,
    random_std: float,
) -> float:
    if random_std == 0:
        return float("nan")
    return (real_value - random_mean) / random_std


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    mean_a, mean_b = np.mean(a), np.mean(b)
    pooled_std = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return (mean_a - mean_b) / pooled_std


class HypothesisTester:

    @staticmethod
    def anova(*groups: Sequence[float]) -> Dict:
        from scipy import stats
        groups_clean = [np.array(g, dtype=float) for g in groups if len(g) >= 2]
        if len(groups_clean) < 2:
            return {"f_stat": float("nan"), "p_value": float("nan")}
        try:
            f_stat, p_value = stats.f_oneway(*groups_clean)
            return {"f_stat": float(f_stat), "p_value": float(p_value)}
        except Exception as e:
            logger.warning("ANOVA failed: %s", e)
            return {"f_stat": float("nan"), "p_value": float("nan")}

    @staticmethod
    def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict:
        from scipy import stats
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        if len(a_arr) < 2 or len(b_arr) < 2:
            return {"u_stat": float("nan"), "p_value": float("nan")}
        try:
            u_stat, p_value = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
            return {"u_stat": float(u_stat), "p_value": float(p_value)}
        except Exception as e:
            logger.warning("Mann-Whitney failed: %s", e)
            return {"u_stat": float("nan"), "p_value": float("nan")}

    @staticmethod
    def chi_square(observed: Dict, expected: Dict = None) -> Dict:
        """Chi-square test for POS distribution."""
        from scipy import stats
        vocab = sorted(observed.keys())
        obs = np.array([observed.get(k, 0) for k in vocab], dtype=float)
        if expected is None:
            # Uniform expected
            exp = np.full_like(obs, obs.sum() / len(obs))
        else:
            exp = np.array([expected.get(k, 0) for k in vocab], dtype=float)

        # Remove zeros in expected
        mask = exp > 0
        obs, exp = obs[mask], exp[mask]
        if len(obs) < 2:
            return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}

        try:
            chi2, p = stats.chisquare(obs, f_exp=exp)
            return {"chi2": float(chi2), "p_value": float(p), "dof": len(obs) - 1}
        except Exception as e:
            logger.warning("Chi-square failed: %s", e)
            return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}

    @staticmethod
    def mixed_effects_model(df: pd.DataFrame, outcome: str, fixed: List[str]) -> Optional[object]:
        """
        Fit a mixed-effects model with 'language' as random effect.
        Returns the fitted model result or None on failure.
        """
        try:
            import statsmodels.formula.api as smf
            warnings.filterwarnings("ignore")
            clean = df[[outcome] + fixed + ["language"]].dropna()
            if len(clean) < 30:
                return None
            formula = f"{outcome} ~ " + " + ".join(fixed)
            model = smf.mixedlm(formula, clean, groups=clean["language"])
            result = model.fit(reml=False)
            return result
        except Exception as e:
            logger.warning("Mixed-effects model failed: %s", e)
            return None

    @staticmethod
    def compute_zscores(
        language: str,
        real_df: pd.DataFrame,
        random_stats: Dict,
    ) -> pd.DataFrame:
        """Build zscore_results.csv rows for a language."""
        rows = []
        metric_map = {
            "complexity_score": ("random_mean_complexity", "random_std_complexity"),
            "dependency_distance": ("random_mean_distance", "random_std_distance"),
        }
        for metric, (mean_key, std_key) in metric_map.items():
            if metric not in real_df.columns:
                continue
            real_val = real_df[metric].mean()
            r_mean = random_stats.get(mean_key, float("nan"))
            r_std = random_stats.get(std_key, float("nan"))
            z = z_score(real_val, r_mean, r_std)
            rows.append({
                "language": language,
                "metric": metric,
                "real_value": real_val,
                "random_mean": r_mean,
                "random_std": r_std,
                "z_score": z,
            })
        return pd.DataFrame(rows)

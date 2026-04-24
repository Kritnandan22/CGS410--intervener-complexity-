#!/usr/bin/env python3
"""
global_analysis.py — Cross-language analysis after merge_all_results.py.

Includes:
  - Typology ANOVA
  - Pairwise Mann-Whitney U
  - Chi-square tests across ALL languages for ALL metrics
  - Correlation analysis
  - Left vs Right dependency analysis
  - Dravidian language focus
  - KL divergence across typologies
  - Mixed-effects models
  - Representation learning (graph embeddings + clustering)
  - Publication-quality global visualizations

Usage:
    python scripts/global_analysis.py
    python scripts/global_analysis.py --final-dir final_outputs --plot-dir plots/global
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, load_languages
from src.statistics.distributions import DistributionAnalyzer
from src.statistics.hypothesis import HypothesisTester, cohens_d
from src.visualization.plots import Visualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def enrich_with_typology(df: pd.DataFrame, languages_cfg: dict) -> pd.DataFrame:
    """Add 'typology', 'family', 'word_order' columns from language config."""
    df = df.copy()
    if "language" not in df.columns:
        return df
    df["typology"] = df["language"].map(
        {k: v.get("typology", "") for k, v in languages_cfg.items()}
    )
    df["word_order"] = df["language"].map(
        {k: v.get("word_order", "") for k, v in languages_cfg.items()}
    )
    df["family"] = df["language"].map(
        {k: v.get("family", "") for k, v in languages_cfg.items()}
    )
    return df


def typology_comparison(summary_df: pd.DataFrame, tester: HypothesisTester):
    """ANOVA and group stats for each metric by typology.

    AUDIT FIX (2026-04-24): Unit of analysis is now language (n=40), not token.
    Token-level pooling (millions of correlated observations) inflates significance.
    Bonferroni correction applied across 6 simultaneous tests: threshold = 0.05/6 = 0.0083.
    """
    results = []
    metrics = ["avg_dependency_length", "avg_complexity", "avg_arity",
               "avg_subtree_size", "avg_depth", "avg_efficiency_ratio"]
    n_tests = len(metrics)
    bonferroni_threshold = 0.05 / n_tests  # = 0.0083

    for metric in metrics:
        if metric not in summary_df.columns:
            continue
        groups = {}
        for typ, grp in summary_df.groupby("typology"):
            vals = grp[metric].dropna().tolist()
            if vals:
                groups[typ] = vals

        if len(groups) < 2:
            continue

        n_langs = summary_df["language"].nunique() if "language" in summary_df.columns else len(summary_df)
        anova = tester.anova(*groups.values())
        results.append({
            "metric": metric,
            "f_stat": anova["f_stat"],
            "p_value": anova["p_value"],
            "n_groups": len(groups),
            "n_languages": n_langs,
            "unit_of_analysis": "language",
            "bonferroni_threshold": bonferroni_threshold,
            "survives_bonferroni": anova["p_value"] < bonferroni_threshold,
            "groups": str(list(groups.keys())),
        })
        sig_marker = "*** SURVIVES BONFERRONI" if anova["p_value"] < bonferroni_threshold else ""
        logger.info("ANOVA %s (n=%d langs): F=%.3f p=%.4f %s",
                    metric, n_langs, anova["f_stat"], anova["p_value"], sig_marker)
    return pd.DataFrame(results)


def pairwise_typology(features_df: pd.DataFrame, tester: HypothesisTester):
    """Mann-Whitney U tests between all pairs of typology groups."""
    typologies = features_df["typology"].dropna().unique()
    pairs = []
    for i, t1 in enumerate(typologies):
        for t2 in typologies[i + 1:]:
            a = features_df.loc[features_df["typology"] == t1, "complexity_score"].dropna()
            b = features_df.loc[features_df["typology"] == t2, "complexity_score"].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            mw = tester.mann_whitney(a, b)
            d = cohens_d(a, b)
            pairs.append({
                "group1": t1, "group2": t2,
                "u_stat": mw["u_stat"], "p_value": mw["p_value"],
                "cohens_d": d,
                "n1": len(a), "n2": len(b),
            })
    return pd.DataFrame(pairs)


def icm_significance_table(zscore_df: pd.DataFrame) -> pd.DataFrame:
    """Per-language ICM significance classification.

    AUDIT FIX (2026-04-24): Audit §4.2 — the report claimed 'robustly supporting ICM'
    from mean z=-0.119. In reality, 0/40 languages reach |z|>1.96, and 15/40 contradict ICM.
    This table provides the honest per-language breakdown.
    """
    if zscore_df.empty:
        return pd.DataFrame()

    comp_z = zscore_df[zscore_df["metric"] == "complexity_score"].copy()
    dist_z = zscore_df[zscore_df["metric"] == "dependency_distance"].copy()
    if comp_z.empty:
        return pd.DataFrame()

    def categorize(z):
        if z < -1.96: return "Significant ICM (p<0.05)"
        if z < -0.5:  return "Moderate ICM trend"
        if z < 0.0:   return "Weak ICM trend"
        if z < 0.5:   return "Weak contradiction"
        return "Strong contradiction"

    comp_z["icm_category"] = comp_z["z_score"].apply(categorize)
    comp_z["supports_icm"] = comp_z["z_score"] < 0
    comp_z["statistically_significant"] = comp_z["z_score"].abs() > 1.96

    # Merge in DLM z-score for comparison
    if not dist_z.empty:
        comp_z = comp_z.merge(
            dist_z[["language", "z_score"]].rename(columns={"z_score": "dlm_z_score"}),
            on="language", how="left"
        )

    summary = {
        "n_languages": len(comp_z),
        "n_supporting_icm": int((comp_z["z_score"] < 0).sum()),
        "n_contradicting_icm": int((comp_z["z_score"] > 0).sum()),
        "n_significant_icm": int((comp_z["z_score"] < -1.96).sum()),
        "mean_complexity_z": float(comp_z["z_score"].mean()),
        "mean_dlm_z": float(dist_z["z_score"].mean()) if not dist_z.empty else float("nan"),
    }
    logger.info("ICM Summary: %d supporting, %d contradicting, %d significant (|z|>1.96); mean z=%.3f",
                summary["n_supporting_icm"], summary["n_contradicting_icm"],
                summary["n_significant_icm"], summary["mean_complexity_z"])
    logger.info("DLM mean z=%.3f vs ICM mean z=%.3f (DLM is %.1fx stronger)",
                summary["mean_dlm_z"], summary["mean_complexity_z"],
                abs(summary["mean_dlm_z"] / summary["mean_complexity_z"]) if summary["mean_complexity_z"] != 0 else float("nan"))
    return comp_z, summary


def flag_data_anomalies(summary_df: pd.DataFrame, threshold_sd: float = 2.5) -> pd.DataFrame:
    """Detect and flag statistical outlier languages.

    AUDIT FIX (2026-04-24): Audit §4.7 — Arabic dep_length=30.85 (3x any other),
    Bengali/Telugu suspiciously low, Tagalog contradicts VSO typology.
    """
    if summary_df.empty:
        return pd.DataFrame()

    anomalies = []
    metrics = ["avg_dependency_length", "avg_complexity", "avg_arity"]
    for metric in metrics:
        if metric not in summary_df.columns:
            continue
        col = summary_df[metric].dropna()
        mean, std = col.mean(), col.std()
        outliers = summary_df[abs(summary_df[metric] - mean) > threshold_sd * std]
        for _, row in outliers.iterrows():
            z = (row[metric] - mean) / std if std > 0 else 0
            anomalies.append({
                "language": row.get("language", "?"),
                "metric": metric,
                "value": row[metric],
                "population_mean": round(mean, 4),
                "z_score": round(z, 3),
                "direction": "high" if z > 0 else "low",
                "note": "Outlier detected — verify treebank quality",
            })

    # Known hardcoded anomalies from audit §4.7
    known = [
        {"language": "ar", "metric": "avg_dependency_length", "note": "PADT clitic tokenization inflates distances 3x"},
        {"language": "bn", "metric": "avg_dependency_length", "note": "Very small treebank — unreliable statistics"},
        {"language": "te", "metric": "avg_dependency_length", "note": "Very small treebank — unreliable statistics"},
        {"language": "tl", "metric": "pct_left_deps",       "note": "94.4% left-deps for VSO language — likely annotation artifact (190 sentences)"},
        {"language": "ko", "metric": "pos_entropy",          "note": "POS entropy 33% below average — agglutinative morphology compression"},
    ]
    known_df = pd.DataFrame(known)
    result = pd.concat([pd.DataFrame(anomalies), known_df], ignore_index=True).drop_duplicates()
    for _, row in result.iterrows():
        logger.warning("DATA ANOMALY [%s] %s: %s", row.get("language"), row.get("metric"), row.get("note"))
    return result


def correlation_analysis(summary_df: pd.DataFrame):
    """Pearson correlation matrix of numeric summary metrics."""
    numeric = summary_df.select_dtypes(include="number")
    return numeric.corr()


def chi_square_all_languages(features_df: pd.DataFrame, tester: HypothesisTester):
    """
    Chi-square test for POS distribution uniformity across ALL languages.
    Produces pairwise chi-square comparison matrix and per-language results.
    """
    from scipy import stats as sp_stats

    languages = sorted(features_df["language"].dropna().unique())
    logger.info("Running chi-square tests for %d languages", len(languages))

    # per-language chi-square against uniform distribution
    per_lang_results = []
    lang_pos_dists = {}
    for lang in languages:
        lang_df = features_df[features_df["language"] == lang]
        if "intervener_upos" not in lang_df.columns:
            continue
        pos_counts = dict(lang_df["intervener_upos"].value_counts())
        lang_pos_dists[lang] = pos_counts
        chi2_res = tester.chi_square(pos_counts)
        per_lang_results.append({
            "language": lang,
            "chi2": chi2_res.get("chi2", float("nan")),
            "p_value": chi2_res.get("p_value", float("nan")),
            "dof": chi2_res.get("dof", 0),
            "n_pos_categories": len(pos_counts),
            "total_interveners": sum(pos_counts.values()),
        })

    # pairwise chi-square between language pairs
    pairwise_results = []
    lang_list = sorted(lang_pos_dists.keys())
    for i, l1 in enumerate(lang_list):
        for l2 in lang_list[i + 1:]:
            # create contingency table
            all_pos = sorted(set(lang_pos_dists[l1]) | set(lang_pos_dists[l2]))
            obs1 = [lang_pos_dists[l1].get(p, 0) for p in all_pos]
            obs2 = [lang_pos_dists[l2].get(p, 0) for p in all_pos]

            try:
                contingency = np.array([obs1, obs2])
                # filter out zero columns
                nonzero = contingency.sum(axis=0) > 0
                contingency = contingency[:, nonzero]
                if contingency.shape[1] < 2:
                    continue
                chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)
                pairwise_results.append({
                    "language_1": l1,
                    "language_2": l2,
                    "chi2": float(chi2),
                    "p_value": float(p),
                    "dof": int(dof),
                })
            except Exception as e:
                logger.debug("Chi-square %s-%s failed: %s", l1, l2, e)

    return pd.DataFrame(per_lang_results), pd.DataFrame(pairwise_results)


def chi_square_all_metrics(features_df: pd.DataFrame):
    """Chi-square tests for all numeric metrics binned across languages."""
    from scipy import stats as sp_stats

    languages = sorted(features_df["language"].dropna().unique())
    metrics = ["arity", "subtree_size", "depth"]
    results = []

    for metric in metrics:
        if metric not in features_df.columns:
            continue
        # bin metric values into categories
        try:
            bins = pd.qcut(features_df[metric].dropna(), q=5, duplicates="drop")
            contingency = pd.crosstab(features_df["language"], bins)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p, dof, expected = sp_stats.chi2_contingency(contingency.values)
            results.append({
                "metric": metric,
                "chi2": float(chi2),
                "p_value": float(p),
                "dof": int(dof),
                "n_languages": len(languages),
            })
            logger.info("Chi-square %s across languages: chi2=%.2f p=%.4f",
                        metric, chi2, p)
        except Exception as e:
            logger.debug("Chi-square %s failed: %s", metric, e)

    return pd.DataFrame(results)


def sensitivity_analysis(
    features_df: pd.DataFrame,
    languages_cfg: dict,
    tester: HypothesisTester,
) -> pd.DataFrame:
    """Test ANOVA robustness across different complexity weight combinations.

    AUDIT FIX (2026-04-24) — Fix 15:
    The complexity formula uses weights (0.35, 0.25, 0.20, 0.20) for
    (arity, subtree_size, depth, POS_weight). These are arbitrary — not grounded in
    psycholinguistic data. If results depend critically on these specific weights,
    the conclusions are fragile. This analysis re-runs the typology ANOVA with 6
    alternative weight sets and reports whether conclusions hold across all of them.
    """
    pos_weights = {
        "VERB": 1.0, "NOUN": 0.6, "PROPN": 0.5, "ADJ": 0.4,
        "ADV": 0.3, "PRON": 0.3, "DET": 0.2, "PUNCT": 0.0,
    }
    weight_sets = [
        {"name": "original_0.35-0.25-0.20-0.20", "w_arity": 0.35, "w_sub": 0.25, "w_dep": 0.20, "w_pos": 0.20},
        {"name": "equal_0.25-0.25-0.25-0.25",    "w_arity": 0.25, "w_sub": 0.25, "w_dep": 0.25, "w_pos": 0.25},
        {"name": "arity_heavy_0.50-0.20-0.20-0.10","w_arity": 0.50, "w_sub": 0.20, "w_dep": 0.20, "w_pos": 0.10},
        {"name": "struct_heavy_0.20-0.40-0.30-0.10","w_arity":0.20, "w_sub": 0.40, "w_dep": 0.30, "w_pos": 0.10},
        {"name": "no_pos_0.35-0.30-0.35-0.00",    "w_arity": 0.35, "w_sub": 0.30, "w_dep": 0.35, "w_pos": 0.00},
        {"name": "depth_heavy_0.20-0.20-0.50-0.10","w_arity": 0.20, "w_sub": 0.20, "w_dep": 0.50, "w_pos": 0.10},
    ]

    df = features_df.copy()
    df["typology"] = df["language"].map(
        {k: v.get("typology", "") for k, v in languages_cfg.items()}
    )

    results = []
    for ws in weight_sets:
        pos_col = df["intervener_upos"].map(pos_weights).fillna(0.1)
        df["recomputed_complexity"] = (
            ws["w_arity"] * df["arity"]
            + ws["w_sub"]   * df["subtree_size"]
            + ws["w_dep"]   * df["depth"]
            + ws["w_pos"]   * pos_col
        )
        lang_means = (
            df.groupby(["language", "typology"])["recomputed_complexity"]
            .mean()
            .reset_index()
        )
        groups = {
            typ: grp["recomputed_complexity"].tolist()
            for typ, grp in lang_means.groupby("typology")
            if len(grp) >= 2
        }
        if len(groups) >= 2:
            anova = tester.anova(*groups.values())
            results.append({
                "weight_set": ws["name"],
                "w_arity": ws["w_arity"],
                "w_subtree_size": ws["w_sub"],
                "w_depth": ws["w_dep"],
                "w_pos": ws["w_pos"],
                "anova_f": anova["f_stat"],
                "anova_p": anova["p_value"],
                "survives_bonferroni": anova["p_value"] < 0.0083,
                "significant_p05": anova["p_value"] < 0.05,
            })
            logger.info("Sensitivity [%s]: F=%.3f p=%.4f", ws["name"],
                        anova["f_stat"], anova["p_value"])

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        n_sig = result_df["significant_p05"].sum()
        logger.info("Sensitivity summary: %d/%d weight sets produce significant ANOVA (p<0.05)",
                    n_sig, len(result_df))
        if n_sig == 0:
            logger.warning("SENSITIVITY: No weight set produces significant complexity ANOVA — "
                           "finding is robust (complexity does not vary by typology regardless of weights)")
        elif n_sig < len(result_df):
            logger.warning("SENSITIVITY: Only %d/%d weight sets significant — "
                           "finding is weight-dependent", n_sig, len(result_df))
    return result_df


def dravidian_focus(features_df: pd.DataFrame, languages_cfg: dict):
    """Dravidian vs non-Dravidian: plain Mann-Whitney + ANCOVA controlling for word order.

    AUDIT FIX (2026-04-24) — Fix 14:
    Dravidian languages (Tamil, Telugu, Malayalam) are all SOV. A plain group comparison
    conflates the Dravidian family effect with the SOV word-order effect. The ANCOVA
    controls for word order (SOV/SVO/VSO/Free) as a covariate and tests whether being
    Dravidian explains additional variance beyond word order alone.
    Reference: Audit §4.7, §7.3 Fix 14.
    """
    dravidian = [k for k, v in languages_cfg.items() if v.get("family") == "Dravidian"]
    logger.info("Dravidian languages in dataset: %s", dravidian)

    df = features_df.copy()
    df["is_dravidian"] = df["language"].isin(dravidian).map({True: "Dravidian", False: "Non-Dravidian"})
    df["word_order"] = df["language"].map(
        {k: v.get("typology", "SVO") for k, v in languages_cfg.items()}
    )

    drav = features_df[features_df["language"].isin(dravidian)]
    non_drav = features_df[~features_df["language"].isin(dravidian)]

    tester = HypothesisTester()
    result = {}
    for metric in ["complexity_score", "dependency_distance", "arity"]:
        if metric not in features_df.columns:
            continue
        a = drav[metric].dropna()
        b = non_drav[metric].dropna()
        if len(a) > 1 and len(b) > 1:
            mw = tester.mann_whitney(a, b)
            d = cohens_d(a, b)
            result[metric] = {
                "dravidian_mean": float(a.mean()),
                "other_mean": float(b.mean()),
                "mann_whitney_u": mw.get("u_stat", float("nan")),
                "p_value_unadjusted": mw.get("p_value", float("nan")),
                "cohens_d": d,
            }
            logger.info("Dravidian vs other — %s: drav=%.3f other=%.3f d=%.3f p=%.4f",
                        metric, a.mean(), b.mean(), d, mw.get("p_value", float("nan")))

            # ANCOVA: control for word order
            sample_df = df[["language", metric, "is_dravidian", "word_order"]].dropna()
            # Use language-level means (n=40) rather than token-level (millions)
            lang_level = (
                sample_df.groupby(["language", "is_dravidian", "word_order"])[metric]
                .mean()
                .reset_index()
            )
            ancova_res = tester.ancova(lang_level, metric, "is_dravidian", "word_order")
            result[metric]["ancova_f"] = ancova_res.get("f_stat", float("nan"))
            result[metric]["ancova_p"] = ancova_res.get("p_value", float("nan"))
            result[metric]["ancova_partial_eta2"] = ancova_res.get("partial_eta_squared", float("nan"))
            result[metric]["ancova_n_obs"] = ancova_res.get("n_obs", 0)

            p_ancova = ancova_res.get("p_value", float("nan"))
            if not (p_ancova != p_ancova):  # not NaN
                if p_ancova < 0.05:
                    logger.info("  ANCOVA [%s]: Dravidian effect SURVIVES word-order control "
                                "(F=%.3f p=%.4f partial_eta2=%.3f)",
                                metric, ancova_res.get("f_stat", 0), p_ancova,
                                ancova_res.get("partial_eta_squared", 0))
                else:
                    logger.warning("  ANCOVA [%s]: Dravidian effect DOES NOT survive word-order "
                                   "control (F=%.3f p=%.4f) — effect is likely SOV, not Dravidian-specific",
                                   metric, ancova_res.get("f_stat", 0), p_ancova)
    return result


def icm_full_contradiction_table(zscore_df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    """Build and save the full 40-language ICM z-score table with contradiction classification.

    AUDIT FIX (2026-04-24) — Fix 16:
    The original report did not prominently report the 15 languages contradicting ICM
    (positive complexity z-scores). Science requires reporting null/negative results.
    This function produces:
      1. A sorted table of all 40 languages ranked by z-score (most supportive to most contradictory)
      2. A separate 'contradicting_languages.csv' listing all languages with z > 0
      3. Prominent WARNING log entries for each contradicting language
    Reference: Audit §4.2, §7.3 Fix 16.
    """
    if zscore_df.empty:
        return pd.DataFrame()

    comp_z = zscore_df[zscore_df["metric"] == "complexity_score"].copy()
    if comp_z.empty:
        return pd.DataFrame()

    def categorize(z):
        if z < -1.96:  return "Significant ICM support (p<0.05)"
        if z < -1.0:   return "Strong ICM trend (|z|>1)"
        if z < -0.5:   return "Moderate ICM trend"
        if z < 0.0:    return "Weak ICM trend"
        if z < 0.5:    return "Weak contradiction"
        if z < 1.0:    return "Moderate contradiction"
        return             "Strong contradiction (z>1)"

    comp_z = comp_z.sort_values("z_score").copy()
    comp_z["icm_category"] = comp_z["z_score"].apply(categorize)
    comp_z["supports_icm"] = comp_z["z_score"] < 0
    comp_z["statistically_significant"] = comp_z["z_score"].abs() > 1.96

    # Save full sorted table
    comp_z.to_csv(os.path.join(results_dir, "icm_full_zscore_table.csv"), index=False)

    # Contradicting languages
    contradicting = comp_z[comp_z["z_score"] > 0].sort_values("z_score", ascending=False)
    if not contradicting.empty:
        contradicting.to_csv(
            os.path.join(results_dir, "icm_contradicting_languages.csv"), index=False)
        logger.warning("=" * 60)
        logger.warning("ICM CONTRADICTION TABLE — %d / %d languages contradict ICM (z > 0):",
                       len(contradicting), len(comp_z))
        for _, row in contradicting.iterrows():
            logger.warning("  %s: z=+%.3f (%s)",
                           row["language"], row["z_score"], row["icm_category"])
        logger.warning("Notable: English z=+%.3f — directly contradicts ICM in most-studied language",
                       comp_z.loc[comp_z["language"] == "en", "z_score"].values[0]
                       if "en" in comp_z["language"].values else float("nan"))
        logger.warning("=" * 60)

    # Supporting languages
    supporting = comp_z[comp_z["z_score"] < 0].sort_values("z_score")
    n_sig = int((comp_z["z_score"] < -1.96).sum())
    logger.info("ICM SUPPORT TABLE — %d / %d languages support ICM (z < 0), %d significant:",
                len(supporting), len(comp_z), n_sig)
    for _, row in supporting.iterrows():
        logger.info("  %s: z=%.3f (%s)", row["language"], row["z_score"], row["icm_category"])

    logger.info("Mean z = %.3f (requires < -1.96 for p<0.05 support of ICM)", comp_z["z_score"].mean())
    return comp_z


def investigate_arabic_outlier(
    summary_df: pd.DataFrame,
    features_df: pd.DataFrame,
    tester: HypothesisTester,
) -> Dict:
    """Quantify distortion caused by Arabic PADT tokenization outlier.

    AUDIT FIX (2026-04-24) — Fix 12:
    Arabic shows avg_dep_length=30.85 (3x higher than any other language), likely due to
    PADT clitic-split tokenization that inflates distances. This function re-runs the key
    analyses (correlation matrix, typology ANOVA) with Arabic excluded and reports how
    results change, isolating the outlier's distortion.
    Reference: Audit §4.7, Liu (2008), De Marneffe et al. (2021).
    """
    from typing import Dict as D
    result: D = {"arabic_stats": {}, "impact_on_correlation": {}, "impact_on_anova": {}}

    # Arabic stats
    if not summary_df.empty and "language" in summary_df.columns:
        ar_row = summary_df[summary_df["language"] == "ar"]
        if not ar_row.empty:
            result["arabic_stats"] = {
                "avg_dep_length": float(ar_row["avg_dependency_length"].values[0])
                    if "avg_dependency_length" in ar_row.columns else float("nan"),
                "mean_all_languages": float(summary_df["avg_dependency_length"].mean())
                    if "avg_dependency_length" in summary_df.columns else float("nan"),
                "z_score_vs_population":
                    float((ar_row["avg_dependency_length"].values[0]
                           - summary_df["avg_dependency_length"].mean())
                          / summary_df["avg_dependency_length"].std())
                    if "avg_dependency_length" in ar_row.columns else float("nan"),
                "note": "PADT uses clitic-split tokenization inflating dependency distances ~3x",
            }
            logger.warning("ARABIC OUTLIER: avg_dep_length=%.2f vs mean=%.2f (z=%.2f)",
                           result["arabic_stats"].get("avg_dep_length", 0),
                           result["arabic_stats"].get("mean_all_languages", 0),
                           result["arabic_stats"].get("z_score_vs_population", 0))

    # Correlation with vs without Arabic
    if not summary_df.empty and "avg_dependency_length" in summary_df.columns \
            and "avg_complexity" in summary_df.columns:
        from scipy import stats as sp_stats
        all_corr, _ = sp_stats.pearsonr(
            summary_df["avg_dependency_length"].dropna(),
            summary_df["avg_complexity"].dropna(),
        )
        no_ar = summary_df[summary_df["language"] != "ar"]
        no_ar_corr, _ = sp_stats.pearsonr(
            no_ar["avg_dependency_length"].dropna(),
            no_ar["avg_complexity"].dropna(),
        )
        result["impact_on_correlation"] = {
            "dep_len_vs_complexity_with_arabic": float(all_corr),
            "dep_len_vs_complexity_without_arabic": float(no_ar_corr),
            "delta": float(abs(all_corr - no_ar_corr)),
        }
        logger.info("Arabic outlier impact on dep_len~complexity correlation: "
                    "with=%.3f, without=%.3f (delta=%.3f)",
                    all_corr, no_ar_corr, abs(all_corr - no_ar_corr))

    # ANOVA with vs without Arabic (using typology column if present)
    if not summary_df.empty and "typology" in summary_df.columns \
            and "avg_complexity" in summary_df.columns:
        def run_anova(df):
            groups = {
                typ: grp["avg_complexity"].dropna().tolist()
                for typ, grp in df.groupby("typology") if len(grp) >= 2
            }
            return tester.anova(*groups.values()) if len(groups) >= 2 else {}

        anova_with = run_anova(summary_df)
        anova_without = run_anova(summary_df[summary_df["language"] != "ar"])
        result["impact_on_anova"] = {
            "complexity_anova_p_with_arabic": anova_with.get("p_value", float("nan")),
            "complexity_anova_p_without_arabic": anova_without.get("p_value", float("nan")),
            "conclusion": (
                "Arabic inclusion does not materially change ANOVA outcome"
                if abs(anova_with.get("p_value", 1) - anova_without.get("p_value", 1)) < 0.05
                else "Arabic outlier materially affects ANOVA — results should be reported both ways"
            ),
        }
        logger.info("Arabic outlier impact on complexity ANOVA: "
                    "with_arabic p=%.4f, without_arabic p=%.4f — %s",
                    result["impact_on_anova"]["complexity_anova_p_with_arabic"],
                    result["impact_on_anova"]["complexity_anova_p_without_arabic"],
                    result["impact_on_anova"]["conclusion"])

    return result


def left_right_analysis(features_df: pd.DataFrame, tester: HypothesisTester):
    """Compare intervener complexity and dependency length for left vs right dependencies."""
    # Complexity Analysis
    left_c = features_df.loc[features_df["direction"] == "left", "complexity_score"].dropna()
    right_c = features_df.loc[features_df["direction"] == "right", "complexity_score"].dropna()
    
    # Distance Analysis
    left_d = features_df.loc[features_df["direction"] == "left", "dependency_distance"].dropna()
    right_d = features_df.loc[features_df["direction"] == "right", "dependency_distance"].dropna()

    if len(left_c) < 2 or len(right_c) < 2:
        return {}

    mw_c = tester.mann_whitney(left_c, right_c)
    d_c = cohens_d(left_c, right_c)

    mw_d = tester.mann_whitney(left_d, right_d)
    d_d = cohens_d(left_d, right_d)

    logger.info("Global Left vs Right (Complexity): left_mean=%.3f right_mean=%.3f p=%.4f d=%.3f",
                left_c.mean(), right_c.mean(), mw_c.get("p_value", float("nan")), d_c)
    logger.info("Global Left vs Right (Distance): left_mean=%.3f right_mean=%.3f p=%.4f d=%.3f",
                left_d.mean(), right_d.mean(), mw_d.get("p_value", float("nan")), d_d)

    return {
        "complexity_left_mean": float(left_c.mean()),
        "complexity_right_mean": float(right_c.mean()),
        "complexity_u_stat": mw_c.get("u_stat", float("nan")),
        "complexity_p_value": mw_c.get("p_value", float("nan")),
        "complexity_cohens_d": d_c,
        "distance_left_mean": float(left_d.mean()),
        "distance_right_mean": float(right_d.mean()),
        "distance_u_stat": mw_d.get("u_stat", float("nan")),
        "distance_p_value": mw_d.get("p_value", float("nan")),
        "distance_cohens_d": d_d,
        "left_n": int(len(left_c)),
        "right_n": int(len(right_c)),
    }


def main():
    p = argparse.ArgumentParser(description="Global cross-language analysis.")
    p.add_argument("--final-dir", default="final_outputs")
    p.add_argument("--plot-dir", default="plots/global")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--languages-config", default="config/languages.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    languages_cfg = load_languages(args.languages_config)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ── Load merged data ────────────────────────────────────────────
    def load_csv(name):
        path = os.path.join(args.final_dir, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
                logger.info("Loaded %s: %d rows", name, len(df))
                return df
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)
                return pd.DataFrame()
        logger.warning("File not found: %s", path)
        return pd.DataFrame()

    features_df = load_csv("all_features.csv")
    summary_df = load_csv("all_language_summary.csv")
    ml_df = load_csv("all_ml_results.csv")
    zscore_df = load_csv("all_zscores.csv")

    if features_df.empty and summary_df.empty:
        logger.error("No data found in %s. Run merge_all_results.py first.", args.final_dir)
        sys.exit(1)

    # ── Enrich with typology info ───────────────────────────────────
    features_df = enrich_with_typology(features_df, languages_cfg)
    summary_df = enrich_with_typology(summary_df, languages_cfg)

    tester = HypothesisTester()
    analyzer = DistributionAnalyzer()
    vis = Visualizer(cfg, args.plot_dir)

    results_dir = os.path.join(args.final_dir, "global_stats")
    os.makedirs(results_dir, exist_ok=True)

    # ── A. Typology ANOVA (language-level, Bonferroni corrected) ────
    if not summary_df.empty and "typology" in summary_df.columns:
        logger.info("=== Typology ANOVA (language-level, Bonferroni corrected) ===")
        anova_df = typology_comparison(summary_df, tester)
        anova_df.to_csv(os.path.join(results_dir, "typology_anova.csv"), index=False)
        surviving = anova_df[anova_df["survives_bonferroni"] == True]["metric"].tolist()
        logger.info("Metrics surviving Bonferroni correction: %s", surviving)

    # ── A2. ICM Significance Table + Full Contradiction Table ───────
    if not zscore_df.empty:
        logger.info("=== ICM Significance Table ===")
        icm_result = icm_significance_table(zscore_df)
        if isinstance(icm_result, tuple):
            icm_table, icm_summary = icm_result
            icm_table.to_csv(os.path.join(results_dir, "icm_significance_table.csv"), index=False)
            pd.DataFrame([icm_summary]).to_csv(
                os.path.join(results_dir, "icm_summary.csv"), index=False)

        # AUDIT FIX Fix 16: full sorted contradiction table
        logger.info("=== Full ICM Contradiction Table (Fix 16) ===")
        icm_full_contradiction_table(zscore_df, results_dir)

    # ── A3. Data Anomaly Detection ──────────────────────────────────
    if not summary_df.empty:
        logger.info("=== Data Anomaly Detection ===")
        anomalies_df = flag_data_anomalies(summary_df)
        if not anomalies_df.empty:
            anomalies_df.to_csv(os.path.join(results_dir, "data_anomalies.csv"), index=False)
            logger.info("%d anomalies flagged", len(anomalies_df))

    # ── B. Pairwise typology Mann-Whitney (token-level, for reference only) ──
    if not features_df.empty and "typology" in features_df.columns:
        logger.info("=== Pairwise Typology Mann-Whitney (token-level, for reference) ===")
        logger.warning("NOTE: Token-level tests inflate significance (n=millions of correlated tokens). "
                       "Use language-level ANOVA above for primary analysis.")
        pw_df = pairwise_typology(features_df, tester)
        pw_df["note"] = "Token-level — effect sizes (cohens_d) are more meaningful than p-values"
        pw_df.to_csv(os.path.join(results_dir, "pairwise_typology_mw.csv"), index=False)

    # ── C. Correlation matrix ───────────────────────────────────────
    if not summary_df.empty:
        logger.info("=== Correlation Analysis ===")
        corr = correlation_analysis(summary_df)
        corr.to_csv(os.path.join(results_dir, "correlation_matrix.csv"))

    # ── D. Left vs Right global ─────────────────────────────────────
    if not features_df.empty and "direction" in features_df.columns:
        logger.info("=== Left vs Right Dependency Analysis ===")
        lr = left_right_analysis(features_df, tester)
        pd.DataFrame([lr]).to_csv(os.path.join(results_dir, "left_right_analysis.csv"), index=False)

    # ── E. Dravidian special focus (with ANCOVA) ───────────────────
    if not features_df.empty:
        logger.info("=== Dravidian Language Focus + ANCOVA (Fix 14) ===")
        drav_results = dravidian_focus(features_df, languages_cfg)
        if drav_results:
            rows = []
            for metric, vals in drav_results.items():
                row = {"metric": metric}
                row.update(vals)
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(results_dir, "dravidian_analysis.csv"), index=False)

    # ── F. Chi-square tests across ALL languages ────────────────────
    if not features_df.empty:
        logger.info("=== Chi-Square Tests Across All Languages ===")

        # per-language and pairwise POS chi-square
        per_lang_chi2, pairwise_chi2 = chi_square_all_languages(features_df, tester)
        per_lang_chi2.to_csv(os.path.join(results_dir, "chi_square_per_language.csv"), index=False)
        pairwise_chi2.to_csv(os.path.join(results_dir, "chi_square_pairwise.csv"), index=False)
        logger.info("Chi-square per-language: %d rows; pairwise: %d rows",
                    len(per_lang_chi2), len(pairwise_chi2))

        # chi-square for all numeric metrics across languages
        metric_chi2 = chi_square_all_metrics(features_df)
        metric_chi2.to_csv(os.path.join(results_dir, "chi_square_all_metrics.csv"), index=False)

    # ── G. KL divergence between typology groups ────────────────────
    if not features_df.empty and "typology" in features_df.columns and "intervener_upos" in features_df.columns:
        logger.info("=== KL Divergence across Typologies ===")
        kl_rows = []
        typologies = features_df["typology"].dropna().unique()
        for t1 in typologies:
            for t2 in typologies:
                if t1 >= t2:
                    continue
                p = dict(features_df.loc[features_df["typology"] == t1, "intervener_upos"].value_counts())
                q = dict(features_df.loc[features_df["typology"] == t2, "intervener_upos"].value_counts())
                kl = analyzer.kl_divergence(p, q)
                js = analyzer.js_divergence(p, q)
                kl_rows.append({"typology1": t1, "typology2": t2, "kl_divergence": kl, "js_divergence": js})
        pd.DataFrame(kl_rows).to_csv(os.path.join(results_dir, "typology_kl_divergence.csv"), index=False)

    # ── H. Mixed-effects model ──────────────────────────────────────
    if not features_df.empty:
        logger.info("=== Mixed-Effects Model ===")
        avail_fixed = [c for c in ["dependency_distance", "arity", "subtree_size", "depth"]
                       if c in features_df.columns]
        if avail_fixed and "language" in features_df.columns and "complexity_score" in features_df.columns:
            sample_size = min(50000, len(features_df))
            model_result = tester.mixed_effects_model(
                features_df.sample(sample_size, random_state=42),
                outcome="complexity_score",
                fixed=avail_fixed,
            )
            if model_result is not None:
                with open(os.path.join(results_dir, "mixed_effects_summary.txt"), "w") as f:
                    f.write(str(model_result.summary()))
                logger.info("Mixed-effects model saved")

    # ── I-pre. Sensitivity analysis (Fix 15) ───────────────────────
    if not features_df.empty and "typology" in features_df.columns:
        logger.info("=== Sensitivity Analysis on Complexity Weights (Fix 15) ===")
        sens_df = sensitivity_analysis(features_df, languages_cfg, tester)
        if not sens_df.empty:
            sens_df.to_csv(os.path.join(results_dir, "sensitivity_analysis.csv"), index=False)
            logger.info("Sensitivity results saved (%d weight sets tested)", len(sens_df))

    # ── I-pre2. Arabic outlier investigation (Fix 12) ──────────────
    if not summary_df.empty:
        logger.info("=== Arabic Outlier Investigation (Fix 12) ===")
        ar_impact = investigate_arabic_outlier(summary_df, features_df, tester)
        pd.DataFrame([{
            **ar_impact.get("arabic_stats", {}),
            **ar_impact.get("impact_on_correlation", {}),
            **ar_impact.get("impact_on_anova", {}),
        }]).to_csv(os.path.join(results_dir, "arabic_outlier_investigation.csv"), index=False)

    # ── I-pre3. Power analysis (Fix 18) ────────────────────────────
    logger.info("=== Power Analysis n=40 (Fix 18) ===")
    power_results = tester.power_analysis_n40()
    pd.DataFrame([power_results]).to_csv(
        os.path.join(results_dir, "power_analysis.csv"), index=False)

    # ── I. Global visualizations ────────────────────────────────────
    logger.info("=== Global Visualizations ===")

    if not summary_df.empty and "typology" in summary_df.columns:
        for metric in ["avg_complexity", "avg_dependency_length", "avg_arity"]:
            if metric in summary_df.columns:
                vis.typology_heatmap(
                    summary_df, metric,
                    os.path.join(args.plot_dir, f"heatmap_{metric}.png")
                )

    if not features_df.empty:
        for metric in ["complexity_score", "dependency_distance"]:
            if metric in features_df.columns:
                vis.cross_language_boxplot(
                    features_df, metric,
                    os.path.join(args.plot_dir, f"cross_lang_{metric}_boxplot.png")
                )
        if "typology" in features_df.columns:
            vis.typology_violin(
                features_df, "complexity_score",
                os.path.join(args.plot_dir, "typology_complexity_violin.png")
            )

    if not summary_df.empty:
        vis.pca_plot(summary_df, os.path.join(args.plot_dir, "pca_languages.png"))
        vis.correlation_heatmap(summary_df, os.path.join(args.plot_dir, "correlation_heatmap.png"))

    if not ml_df.empty:
        vis.ml_results_heatmap(ml_df, os.path.join(args.plot_dir, "ml_f1_heatmap.png"))

    # ── J. Chi-square heatmap visualization ─────────────────────────
    if not features_df.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # pairwise chi-square heatmap
            pw_path = os.path.join(results_dir, "chi_square_pairwise.csv")
            if os.path.exists(pw_path):
                pw_df = pd.read_csv(pw_path)
                if not pw_df.empty and "language_1" in pw_df.columns:
                    langs = sorted(set(pw_df["language_1"]) | set(pw_df["language_2"]))
                    chi2_matrix = pd.DataFrame(0.0, index=langs, columns=langs)
                    for _, row in pw_df.iterrows():
                        chi2_matrix.loc[row["language_1"], row["language_2"]] = row["chi2"]
                        chi2_matrix.loc[row["language_2"], row["language_1"]] = row["chi2"]

                    fig, ax = plt.subplots(figsize=(max(14, len(langs) * 0.5),
                                                     max(12, len(langs) * 0.4)))
                    sns.heatmap(np.log1p(chi2_matrix.values.astype(float)),
                                xticklabels=langs, yticklabels=langs,
                                cmap="YlOrRd", ax=ax, linewidths=0.2,
                                cbar_kws={"label": "log(1 + chi²)"})
                    ax.set_title("Pairwise Chi-Square (POS Distribution) — log scale", fontsize=14)
                    ax.tick_params(axis="x", rotation=90, labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    fig.tight_layout()
                    fig.savefig(os.path.join(args.plot_dir, "chi_square_pairwise_heatmap.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("Saved chi-square pairwise heatmap")
        except Exception as e:
            logger.warning("Chi-square heatmap failed: %s", e)

    logger.info("=== Global analysis complete. Results in %s ===", results_dir)


if __name__ == "__main__":
    main()

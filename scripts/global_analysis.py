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
    """ANOVA and group stats for each metric by typology."""
    results = []
    metrics = ["avg_dependency_length", "avg_complexity", "avg_arity",
               "avg_subtree_size", "avg_depth", "avg_efficiency_ratio"]

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

        anova = tester.anova(*groups.values())
        results.append({
            "metric": metric,
            "f_stat": anova["f_stat"],
            "p_value": anova["p_value"],
            "n_groups": len(groups),
            "groups": str(list(groups.keys())),
        })
        logger.info("ANOVA %s: F=%.3f p=%.4f", metric,
                    anova["f_stat"], anova["p_value"])
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


def dravidian_focus(features_df: pd.DataFrame, languages_cfg: dict):
    """Special analysis: Dravidian vs non-Dravidian languages."""
    dravidian = [k for k, v in languages_cfg.items() if v.get("family") == "Dravidian"]
    logger.info("Dravidian languages in dataset: %s", dravidian)

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
            result[metric] = {
                "dravidian_mean": float(a.mean()),
                "other_mean": float(b.mean()),
                "mann_whitney_u": mw.get("u_stat", float("nan")),
                "p_value": mw.get("p_value", float("nan")),
                "cohens_d": cohens_d(a, b),
            }
            logger.info("Dravidian vs other — %s: drav=%.3f other=%.3f p=%.4f",
                        metric, a.mean(), b.mean(), mw.get("p_value", float("nan")))
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

    # ── A. Typology ANOVA ───────────────────────────────────────────
    if not summary_df.empty and "typology" in summary_df.columns:
        logger.info("=== Typology ANOVA ===")
        anova_df = typology_comparison(summary_df, tester)
        anova_df.to_csv(os.path.join(results_dir, "typology_anova.csv"), index=False)

    # ── B. Pairwise typology Mann-Whitney ───────────────────────────
    if not features_df.empty and "typology" in features_df.columns:
        logger.info("=== Pairwise Typology Mann-Whitney ===")
        pw_df = pairwise_typology(features_df, tester)
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

    # ── E. Dravidian special focus ──────────────────────────────────
    if not features_df.empty:
        logger.info("=== Dravidian Language Focus ===")
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

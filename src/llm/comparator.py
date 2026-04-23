"""
Compare LLM-generated sentence intervener distributions
against real corpus distributions. Supports GPT-2 and BERT comparison.
Generates high-quality comparison plots.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.statistics.distributions import DistributionAnalyzer

logger = logging.getLogger(__name__)


class LLMComparator:
    def __init__(self):
        self.analyzer = DistributionAnalyzer()

    def compare(
        self,
        real_df: pd.DataFrame,
        llm_df: pd.DataFrame,
        language: str = "en",
    ) -> Dict:
        """
        Compare real vs LLM distributions on:
         - arity, POS, dependency_distance, complexity_score
        Returns a dict of KL divergence / JS divergence scores.
        """
        results = {}

        numeric_metrics = ["arity", "dependency_distance", "subtree_size", "complexity_score"]
        for metric in numeric_metrics:
            if metric not in real_df.columns or metric not in llm_df.columns:
                continue
            # bin into integer buckets for distribution comparison
            real_vals = real_df[metric].dropna().round(0).astype(int)
            llm_vals = llm_df[metric].dropna().round(0).astype(int)
            real_counts = dict(real_vals.value_counts())
            llm_counts = dict(llm_vals.value_counts())
            kl = self.analyzer.kl_divergence(real_counts, llm_counts)
            js = self.analyzer.js_divergence(real_counts, llm_counts)
            results[f"kl_{metric}"] = kl
            results[f"js_{metric}"] = js

        # POS distribution
        if "intervener_upos" in real_df.columns and "intervener_upos" in llm_df.columns:
            real_pos = dict(real_df["intervener_upos"].value_counts())
            llm_pos = dict(llm_df["intervener_upos"].value_counts())
            results["kl_upos"] = self.analyzer.kl_divergence(real_pos, llm_pos)
            results["js_upos"] = self.analyzer.js_divergence(real_pos, llm_pos)

        results["language"] = language
        results["real_n_interveners"] = len(real_df)
        results["llm_n_interveners"] = len(llm_df)
        return results

    def summary_table(self, results: Dict) -> pd.DataFrame:
        """Convert comparison results to a tidy DataFrame."""
        rows = []
        for key, val in results.items():
            if key.startswith("kl_") or key.startswith("js_"):
                metric_type = "KL" if key.startswith("kl_") else "JS"
                metric_name = key[3:]
                rows.append({
                    "language": results.get("language", ""),
                    "divergence_type": metric_type,
                    "metric": metric_name,
                    "value": val,
                })
        return pd.DataFrame(rows)

    def multi_llm_compare(
        self,
        real_df: pd.DataFrame,
        gpt2_df: Optional[pd.DataFrame],
        bert_df: Optional[pd.DataFrame],
        language: str = "en",
    ) -> pd.DataFrame:
        """Compare real corpus against both GPT-2 and BERT generated sentences."""
        rows = []
        for llm_name, llm_df in [("GPT-2", gpt2_df), ("BERT", bert_df)]:
            if llm_df is None or llm_df.empty:
                continue
            comp = self.compare(real_df, llm_df, language)
            for key, val in comp.items():
                if key.startswith("kl_") or key.startswith("js_"):
                    div_type = "KL" if key.startswith("kl_") else "JS"
                    metric = key[3:]
                    rows.append({
                        "language": language,
                        "llm": llm_name,
                        "divergence_type": div_type,
                        "metric": metric,
                        "value": val,
                    })
        return pd.DataFrame(rows)

    # ── Visualization methods ──────────────────────────────────────────

    def plot_distribution_comparison(
        self,
        real_df: pd.DataFrame,
        llm_df: pd.DataFrame,
        metric: str,
        llm_name: str,
        out_path: str,
        dpi: int = 150,
    ):
        """Side-by-side histogram comparison of a metric between real and LLM."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        if metric in real_df.columns:
            real_vals = real_df[metric].dropna()
            axes[0].hist(real_vals, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
            axes[0].set_title(f"Real Corpus — {metric}", fontsize=12, fontweight="bold")
            axes[0].set_xlabel(metric, fontsize=10)
            axes[0].set_ylabel("Count", fontsize=10)

        if metric in llm_df.columns:
            llm_vals = llm_df[metric].dropna()
            axes[1].hist(llm_vals, bins=30, color="#DD8452", edgecolor="white", alpha=0.85)
            axes[1].set_title(f"{llm_name} Generated — {metric}", fontsize=12, fontweight="bold")
            axes[1].set_xlabel(metric, fontsize=10)

        fig.suptitle(f"Real vs {llm_name}: {metric} Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved LLM comparison plot: %s", out_path)

    def plot_pos_comparison(
        self,
        real_df: pd.DataFrame,
        llm_df: pd.DataFrame,
        llm_name: str,
        out_path: str,
        dpi: int = 150,
    ):
        """Grouped bar chart comparing POS distributions between real and LLM."""
        if "intervener_upos" not in real_df.columns or "intervener_upos" not in llm_df.columns:
            return

        real_pos = real_df["intervener_upos"].value_counts(normalize=True)
        llm_pos = llm_df["intervener_upos"].value_counts(normalize=True)

        # merge on all POS tags
        all_pos = sorted(set(real_pos.index) | set(llm_pos.index))
        real_vals = [real_pos.get(p, 0) for p in all_pos]
        llm_vals = [llm_pos.get(p, 0) for p in all_pos]

        x = np.arange(len(all_pos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width / 2, real_vals, width, label="Real Corpus", color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, llm_vals, width, label=f"{llm_name}", color="#DD8452", alpha=0.85)
        ax.set_xlabel("UPOS Tag", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=11)
        ax.set_title(f"POS Distribution: Real vs {llm_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_pos, rotation=45, fontsize=9)
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved POS comparison plot: %s", out_path)

    def plot_divergence_heatmap(
        self,
        comparison_df: pd.DataFrame,
        out_path: str,
        dpi: int = 150,
    ):
        """Heatmap of KL/JS divergence values across metrics and LLMs."""
        if comparison_df.empty:
            return

        # filter to JS only for cleaner visualization
        js_df = comparison_df[comparison_df["divergence_type"] == "JS"]
        if js_df.empty:
            js_df = comparison_df

        pivot = js_df.pivot_table(index="metric", columns="llm", values="value")
        if pivot.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "JS Divergence"})
        ax.set_title("JS Divergence: Real Corpus vs LLM Outputs", fontsize=14, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved divergence heatmap: %s", out_path)

    def plot_all_comparisons(
        self,
        real_df: pd.DataFrame,
        gpt2_df: Optional[pd.DataFrame],
        bert_df: Optional[pd.DataFrame],
        plot_dir: str,
        dpi: int = 150,
    ):
        """Generate all LLM comparison plots."""
        metrics = ["arity", "dependency_distance", "complexity_score", "subtree_size"]

        for llm_name, llm_df in [("GPT2", gpt2_df), ("BERT", bert_df)]:
            if llm_df is None or llm_df.empty:
                continue
            for metric in metrics:
                self.plot_distribution_comparison(
                    real_df, llm_df, metric, llm_name,
                    os.path.join(plot_dir, f"llm_{llm_name.lower()}_{metric}_comparison.png"),
                    dpi=dpi,
                )
            self.plot_pos_comparison(
                real_df, llm_df, llm_name,
                os.path.join(plot_dir, f"llm_{llm_name.lower()}_pos_comparison.png"),
                dpi=dpi,
            )

        # multi-LLM divergence heatmap
        comp_df = self.multi_llm_compare(real_df, gpt2_df, bert_df)
        if not comp_df.empty:
            self.plot_divergence_heatmap(
                comp_df,
                os.path.join(plot_dir, "llm_divergence_heatmap.png"),
                dpi=dpi,
            )

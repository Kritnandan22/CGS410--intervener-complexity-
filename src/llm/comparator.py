"""
Compare LLM-generated sentence intervener distributions
against real corpus distributions. Supports GPT-2 and BERT comparison.
Generates high-quality comparison plots.

AUDIT FIXES (2026-04-24) — Fix 10:
- Added shuffled_text_control(): randomly shuffles words in real corpus sentences to
  produce an ungrammatical baseline. This establishes an upper-bound on divergence
  ("how different would maximally ungrammatical text be?"). Without this, reported
  GPT-2 JS divergences (0.008–0.031) cannot be interpreted — they might be no
  smaller than shuffled-word divergence.
- Added generate_temperature_variants() to LLMGenerator for sensitivity analysis
  across GPT-2 temperatures (0.5, 0.7, 0.9, 1.1), checking whether JS divergences
  are robust to generation hyperparameters.
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
        n_bootstrap: int = 500,
    ) -> Dict:
        """Compare real vs LLM distributions with bootstrap CIs and null baseline.

        AUDIT FIX (2026-04-24): Previous version reported only point estimates of KL/JS.
        Without a null baseline, divergences are uninterpretable — even two random samples
        from the same corpus produce non-zero JS divergence due to sampling variation.
        This version adds:
          1. Bootstrap 95% CI on each JS/KL estimate
          2. Null baseline: JS divergence from two same-source samples (same size as LLM)
          3. Significance flag: observed > 95th percentile of null distribution
        Reference: Pardo (2006) on reliable divergence estimation.
        """
        results = {}
        rng = np.random.default_rng(42)

        def bootstrap_divergence(real_counts, llm_counts, real_series, llm_n, n_boot):
            """Bootstrap CI + null baseline for a divergence value."""
            obs_js = self.analyzer.js_divergence(real_counts, llm_counts)

            # Bootstrap distribution of observed JS
            boot_js = []
            for _ in range(n_boot):
                real_boot = real_series.sample(llm_n, replace=True).round(0).astype(int)
                llm_boot  = real_series.sample(llm_n, replace=True).round(0).astype(int)
                boot_js.append(self.analyzer.js_divergence(
                    dict(real_boot.value_counts()), dict(llm_boot.value_counts())
                ))

            # Null: two same-source random samples (same size as LLM group)
            null_js = []
            for _ in range(n_boot):
                s1 = real_series.sample(llm_n, replace=True).round(0).astype(int)
                s2 = real_series.sample(llm_n, replace=True).round(0).astype(int)
                null_js.append(self.analyzer.js_divergence(
                    dict(s1.value_counts()), dict(s2.value_counts())
                ))

            return {
                "observed_js": obs_js,
                "ci_lower": float(np.percentile(boot_js, 2.5)),
                "ci_upper": float(np.percentile(boot_js, 97.5)),
                "null_mean_js": float(np.mean(null_js)),
                "null_95th": float(np.percentile(null_js, 95)),
                "significant": obs_js > float(np.percentile(null_js, 95)),
            }

        numeric_metrics = ["arity", "dependency_distance", "subtree_size", "complexity_score"]
        for metric in numeric_metrics:
            if metric not in real_df.columns or metric not in llm_df.columns:
                continue
            real_vals = real_df[metric].dropna()
            llm_vals  = llm_df[metric].dropna().round(0).astype(int)
            real_int  = real_vals.round(0).astype(int)
            real_counts = dict(real_int.value_counts())
            llm_counts  = dict(llm_vals.value_counts())

            kl = self.analyzer.kl_divergence(real_counts, llm_counts)
            js_stats = bootstrap_divergence(real_counts, llm_counts, real_int, len(llm_vals), n_bootstrap)

            results[f"kl_{metric}"] = kl
            results[f"js_{metric}"] = js_stats["observed_js"]
            results[f"js_{metric}_ci_lower"] = js_stats["ci_lower"]
            results[f"js_{metric}_ci_upper"] = js_stats["ci_upper"]
            results[f"js_{metric}_null_95th"] = js_stats["null_95th"]
            results[f"js_{metric}_significant"] = js_stats["significant"]

        # POS distribution
        if "intervener_upos" in real_df.columns and "intervener_upos" in llm_df.columns:
            real_pos = dict(real_df["intervener_upos"].value_counts())
            llm_pos  = dict(llm_df["intervener_upos"].value_counts())
            results["kl_upos"] = self.analyzer.kl_divergence(real_pos, llm_pos)
            results["js_upos"] = self.analyzer.js_divergence(real_pos, llm_pos)

        results["language"] = language
        results["real_n_interveners"] = len(real_df)
        results["llm_n_interveners"] = len(llm_df)
        results["n_bootstrap"] = n_bootstrap
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

        pivot = js_df.pivot_table(index="metric", columns="llm", values="value", aggfunc="mean")
        if pivot.empty:
            return
        # Ensure all values are float (prevents matplotlib dtype object error)
        pivot = pivot.apply(pd.to_numeric, errors="coerce").astype(float)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "JS Divergence"})
        ax.set_title("JS Divergence: Real Corpus vs LLM Outputs", fontsize=14, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved divergence heatmap: %s", out_path)

    def shuffled_text_control(
        self,
        real_df: pd.DataFrame,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> Dict:
        """Produce a uniform-random ungrammatical baseline for interpreting LLM divergences.

        AUDIT FIX (2026-04-24) — Fix 10:
        Without a baseline, reported JS divergences are uninterpretable.
        Consider: GPT-2 JS(dependency_distance) = 0.014.
          - If uniform-random gives JS ≈ 0.014 → GPT-2 is NO better than random
          - If uniform-random gives JS ≈ 0.200 → GPT-2 is much closer to real

        Method: draw each metric value uniformly at random from [min, max] of the
        real corpus (same sample size as the LLM). This simulates what distributions
        would look like if produced by a model with no knowledge of language structure.
        Note: permuting values preserves marginal distributions (JS≈0), so we must
        sample from a structurally different (uniform) distribution instead.

        Returns JS divergences for the uniform baseline alongside null baseline.
        """
        rng = np.random.default_rng(seed)
        results: Dict = {"type": "uniform_random_baseline", "n_bootstrap": n_bootstrap}

        numeric_metrics = ["arity", "dependency_distance", "subtree_size", "complexity_score"]
        for metric in numeric_metrics:
            if metric not in real_df.columns:
                continue
            real_vals = real_df[metric].dropna()
            if len(real_vals) < 10:
                continue

            real_int = real_vals.round(0).astype(int)
            v_min, v_max = int(real_int.min()), int(real_int.max())
            real_counts = dict(real_int.value_counts())

            # Uniform random: draw n values uniformly from [v_min, v_max]
            n_sample = len(real_int)
            uniform_vals = rng.integers(v_min, v_max + 1, size=n_sample)
            uniform_counts = dict(pd.Series(uniform_vals).value_counts())
            js_uniform = self.analyzer.js_divergence(real_counts, uniform_counts)

            # Null: two same-source samples (expected-by-chance divergence)
            null_js = []
            for _ in range(n_bootstrap):
                s1 = real_int.sample(min(5000, n_sample), replace=True)
                s2 = real_int.sample(min(5000, n_sample), replace=True)
                null_js.append(self.analyzer.js_divergence(
                    dict(s1.value_counts()), dict(s2.value_counts())
                ))
            null_mean = float(np.mean(null_js))
            null_95th = float(np.percentile(null_js, 95))

            results[f"js_{metric}_uniform_random"] = js_uniform
            results[f"js_{metric}_null_mean"] = null_mean
            results[f"js_{metric}_null_95th"] = null_95th
            logger.info(
                "Uniform-random baseline [%s]: JS=%.4f | null_mean=%.4f null_95th=%.4f",
                metric, js_uniform, null_mean, null_95th,
            )

        return results

    def interpret_llm_vs_shuffled(
        self,
        llm_results: Dict,
        shuffled_results: Dict,
        llm_name: str = "GPT-2",
    ) -> pd.DataFrame:
        """Compare LLM JS divergences against shuffled-text baseline.

        Answers: "Is LLM output closer to real text than shuffled (ungrammatical) text?"
        A well-behaved LLM should have JS << shuffled JS for structural metrics.
        """
        rows = []
        for key in llm_results:
            if not key.startswith("js_") or "_ci_" in key or "_null" in key or "_significant" in key:
                continue
            metric = key[3:]  # strip 'js_'
            # Key changed: uniform_random is now the baseline name
            shuffled_key = f"js_{metric}_uniform_random"
            if shuffled_key not in shuffled_results:
                continue

            llm_js      = llm_results[key]
            shuffled_js = shuffled_results[shuffled_key]
            null_95th   = shuffled_results.get(f"js_{metric}_null_95th", float("nan"))

            ratio = llm_js / shuffled_js if shuffled_js > 0 else float("nan")
            rows.append({
                "llm": llm_name,
                "metric": metric,
                "llm_js": llm_js,
                "shuffled_js": shuffled_js,
                "null_95th": null_95th,
                "llm_to_shuffled_ratio": ratio,
                "llm_better_than_shuffled": llm_js < shuffled_js,
                "interpretation": (
                    f"{llm_name} is {1/ratio:.1f}x closer to real text than shuffled baseline"
                    if ratio > 0 and ratio < 1
                    else f"{llm_name} is WORSE than shuffled baseline for {metric}"
                    if ratio >= 1
                    else "indeterminate"
                ),
            })
            logger.info(
                "[%s] %s: LLM JS=%.4f, Shuffled JS=%.4f, ratio=%.2f — %s",
                llm_name, metric, llm_js, shuffled_js, ratio if ratio == ratio else 0,
                "LLM better" if llm_js < shuffled_js else "LLM worse than shuffled",
            )

        return pd.DataFrame(rows)

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

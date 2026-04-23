"""
Publication-quality visualizations for intervener complexity analysis.
All plots saved as PNG (configurable DPI).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _save(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


class Visualizer:
    def __init__(self, cfg: Dict, plot_dir: str):
        self.cfg = cfg
        self.plot_dir = plot_dir
        vis_cfg = cfg.get("visualization", {})
        self.dpi = vis_cfg.get("dpi", 150)
        self.figsize = tuple(vis_cfg.get("figsize", [10, 6]))
        self.palette = vis_cfg.get("palette", "tab20")
        style = vis_cfg.get("style", "seaborn-v0_8-whitegrid")
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-whitegrid")

    def _path(self, language: str, name: str) -> str:
        return os.path.join(self.plot_dir, language, f"{name}.png")

    # ── Per-language plots ──────────────────────────────────────────────

    def arity_histogram(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        arities = df["arity"].dropna()
        ax.hist(arities, bins=range(0, int(arities.max()) + 2),
                color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Arity (number of dependents)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Arity Distribution — {language}", fontsize=14)
        _save(fig, self._path(language, "arity_histogram"), self.dpi)

    def pos_bar_chart(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        pos_counts = df["intervener_upos"].value_counts()
        pos_counts.plot(kind="bar", ax=ax, color=sns.color_palette(self.palette, len(pos_counts)))
        ax.set_xlabel("UPOS Tag", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Intervener POS Distribution — {language}", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        _save(fig, self._path(language, "pos_distribution"), self.dpi)

    def complexity_scatter(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        x = df["dependency_distance"].dropna()
        y = df["complexity_score"].dropna()
        idx = x.index.intersection(y.index)
        ax.scatter(x[idx], y[idx], alpha=0.3, s=10, color="steelblue")
        # Regression line
        if len(idx) > 10:
            z = np.polyfit(x[idx], y[idx], 1)
            p = np.poly1d(z)
            xr = np.linspace(x.min(), x.max(), 200)
            ax.plot(xr, p(xr), "r--", linewidth=2, label=f"slope={z[0]:.3f}")
            ax.legend()
        ax.set_xlabel("Dependency Distance", fontsize=12)
        ax.set_ylabel("Complexity Score", fontsize=12)
        ax.set_title(f"Dependency Distance vs Complexity — {language}", fontsize=14)
        _save(fig, self._path(language, "distance_vs_complexity"), self.dpi)

    def complexity_boxplot(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        df_plot = df[["intervener_upos", "complexity_score"]].dropna()
        order = df_plot.groupby("intervener_upos")["complexity_score"].median().sort_values(ascending=False).index
        sns.boxplot(data=df_plot, x="intervener_upos", y="complexity_score",
                    order=order, palette=self.palette, ax=ax)
        ax.set_xlabel("UPOS", fontsize=12)
        ax.set_ylabel("Complexity Score", fontsize=12)
        ax.set_title(f"Complexity by POS — {language}", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        _save(fig, self._path(language, "complexity_by_pos_boxplot"), self.dpi)

    def violin_complexity_by_direction(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        df_plot = df[["direction", "complexity_score"]].dropna()
        sns.violinplot(data=df_plot, x="direction", y="complexity_score",
                       palette=["#4C72B0", "#DD8452"], ax=ax)
        ax.set_title(f"Complexity by Dependency Direction — {language}", fontsize=14)
        ax.set_xlabel("Direction", fontsize=12)
        ax.set_ylabel("Complexity Score", fontsize=12)
        _save(fig, self._path(language, "complexity_by_direction_violin"), self.dpi)

    def ier_histogram(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        ier = df["efficiency_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(ier, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Intervener Efficiency Ratio (IER)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"IER Distribution — {language}", fontsize=14)
        _save(fig, self._path(language, "ier_histogram"), self.dpi)

    def depth_subtree_scatter(self, df: pd.DataFrame, language: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(df["depth"], df["subtree_size"], alpha=0.3, s=10, color="green")
        ax.set_xlabel("Depth in Tree", fontsize=12)
        ax.set_ylabel("Subtree Size", fontsize=12)
        ax.set_title(f"Depth vs Subtree Size — {language}", fontsize=14)
        _save(fig, self._path(language, "depth_vs_subtree"), self.dpi)

    def all_language_plots(self, df: pd.DataFrame, language: str):
        """Generate all per-language plots."""
        os.makedirs(os.path.join(self.plot_dir, language), exist_ok=True)
        if "arity" in df.columns:
            self.arity_histogram(df, language)
        if "intervener_upos" in df.columns:
            self.pos_bar_chart(df, language)
        if "dependency_distance" in df.columns and "complexity_score" in df.columns:
            self.complexity_scatter(df, language)
        if "intervener_upos" in df.columns and "complexity_score" in df.columns:
            self.complexity_boxplot(df, language)
        if "direction" in df.columns and "complexity_score" in df.columns:
            self.violin_complexity_by_direction(df, language)
        if "efficiency_ratio" in df.columns:
            self.ier_histogram(df, language)
        if "depth" in df.columns and "subtree_size" in df.columns:
            self.depth_subtree_scatter(df, language)

    # ── Global / cross-language plots ──────────────────────────────────

    def typology_heatmap(self, summary_df: pd.DataFrame, metric: str, out_path: str):
        """Heatmap of a metric grouped by typology and language."""
        pivot = summary_df.pivot_table(index="typology", columns="language", values=metric)
        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.6), 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title(f"{metric} by Typology and Language", fontsize=14)
        _save(fig, out_path, self.dpi)

    def cross_language_boxplot(self, features_df: pd.DataFrame, metric: str, out_path: str):
        """Boxplot of metric across all languages."""
        fig, ax = plt.subplots(figsize=(max(14, len(features_df["language"].unique()) * 0.7), 6))
        order = features_df.groupby("language")[metric].median().sort_values().index
        sns.boxplot(data=features_df, x="language", y=metric, order=order,
                    palette=self.palette, ax=ax)
        ax.set_xlabel("Language", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Cross-Language {metric} Distribution", fontsize=14)
        ax.tick_params(axis="x", rotation=90)
        _save(fig, out_path, self.dpi)

    def pca_plot(self, summary_df: pd.DataFrame, out_path: str):
        """PCA of language summary statistics, colored by typology."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        numeric_cols = ["avg_dependency_length", "avg_complexity", "avg_arity",
                        "avg_subtree_size", "avg_depth",
                        "percent_left_dependencies", "avg_efficiency_ratio"]
        avail = [c for c in numeric_cols if c in summary_df.columns]
        data = summary_df[avail].dropna()
        if len(data) < 3:
            return

        X = StandardScaler().fit_transform(data)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        typologies = summary_df.loc[data.index, "typology"] if "typology" in summary_df.columns else pd.Series(["?"] * len(data))
        langs = summary_df.loc[data.index, "language"] if "language" in summary_df.columns else data.index.astype(str)
        palette = sns.color_palette("tab10", len(typologies.unique()))
        color_map = {t: c for t, c in zip(sorted(typologies.unique()), palette)}

        for i, (x, y) in enumerate(coords):
            t = typologies.iloc[i] if i < len(typologies) else "?"
            ax.scatter(x, y, color=color_map.get(t, "gray"), s=80, zorder=5)
            ax.annotate(langs.iloc[i] if i < len(langs) else "", (x, y),
                        fontsize=7, ha="center", va="bottom")

        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=c, markersize=10, label=t)
                   for t, c in color_map.items()]
        ax.legend(handles=handles, title="Typology")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
        ax.set_title("PCA of Language Summary Features", fontsize=14)
        _save(fig, out_path, self.dpi)

    def typology_violin(self, features_df: pd.DataFrame, metric: str, out_path: str):
        """Violin plot of metric grouped by word-order typology."""
        if "typology" not in features_df.columns:
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.violinplot(data=features_df, x="typology", y=metric,
                       palette="Set2", ax=ax)
        ax.set_title(f"{metric} by Typology", fontsize=14)
        _save(fig, out_path, self.dpi)

    def correlation_heatmap(self, summary_df: pd.DataFrame, out_path: str):
        """Correlation matrix of summary metrics."""
        numeric = summary_df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Matrix of Language Summary Metrics", fontsize=14)
        _save(fig, out_path, self.dpi)

    def ml_results_heatmap(self, ml_df: pd.DataFrame, out_path: str):
        """Heatmap of F1 scores by language and model."""
        pivot = ml_df.pivot_table(index="language", columns="model_name", values="f1_score")
        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2),
                                        max(8, len(pivot) * 0.4)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax, linewidths=0.3)
        ax.set_title("ML Model F1-Score by Language", fontsize=14)
        _save(fig, out_path, self.dpi)

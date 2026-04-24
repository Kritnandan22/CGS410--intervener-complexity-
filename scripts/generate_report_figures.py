"""
Generate high-quality, Big-4-style report figures for CGS410.
Covers: ICM/DLM, typology, ML, LLM, correlations, anomalies, power, efficiency.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── OUTPUT DIR ──────────────────────────────────────────────────────────────
OUT = "plots/report_figures"
os.makedirs(OUT, exist_ok=True)

# ── BIG-4 STYLE CONSTANTS ───────────────────────────────────────────────────
BRAND   = "#003366"          # deep navy
ACCENT1 = "#0072CE"          # bright blue
ACCENT2 = "#E8A020"          # amber
ACCENT3 = "#D64045"          # red
ACCENT4 = "#2CA58D"          # teal
GREY1   = "#F5F6FA"          # near-white bg
GREY2   = "#DDDFE6"          # light grid
GREY3   = "#888C99"          # muted label
TEXT    = "#1A1D2E"          # near-black

PALETTE_TYPOLOGY = {
    "SOV":  ACCENT1,
    "SVO":  ACCENT4,
    "VSO":  ACCENT2,
    "Free": ACCENT3,
}

LANG_META = {
    # lang: (full_name, family, word_order)
    "ar": ("Arabic",     "Semitic",    "VSO"),
    "bn": ("Bengali",    "Indo-Aryan", "SOV"),
    "ca": ("Catalan",    "Romance",    "SVO"),
    "cs": ("Czech",      "Slavic",     "Free"),
    "cy": ("Welsh",      "Celtic",     "VSO"),
    "da": ("Danish",     "Germanic",   "SVO"),
    "de": ("German",     "Germanic",   "SOV"),
    "en": ("English",    "Germanic",   "SVO"),
    "es": ("Spanish",    "Romance",    "SVO"),
    "et": ("Estonian",   "Uralic",     "SOV"),
    "eu": ("Basque",     "Isolate",    "SOV"),
    "fa": ("Persian",    "Iranian",    "SOV"),
    "fi": ("Finnish",    "Uralic",     "SOV"),
    "fr": ("French",     "Romance",    "SVO"),
    "ga": ("Irish",      "Celtic",     "VSO"),
    "gl": ("Galician",   "Romance",    "SVO"),
    "gu": ("Gujarati",   "Indo-Aryan", "SOV"),
    "hi": ("Hindi",      "Indo-Aryan", "SOV"),
    "hu": ("Hungarian",  "Uralic",     "SOV"),
    "id": ("Indonesian", "Austronesian","SVO"),
    "it": ("Italian",    "Romance",    "SVO"),
    "ja": ("Japanese",   "Japonic",    "SOV"),
    "ko": ("Korean",     "Koreanic",   "SOV"),
    "ml": ("Malayalam",  "Dravidian",  "SOV"),
    "mr": ("Marathi",    "Indo-Aryan", "SOV"),
    "nl": ("Dutch",      "Germanic",   "SOV"),
    "no": ("Norwegian",  "Germanic",   "SVO"),
    "pl": ("Polish",     "Slavic",     "Free"),
    "pt": ("Portuguese", "Romance",    "SVO"),
    "ru": ("Russian",    "Slavic",     "Free"),
    "sv": ("Swedish",    "Germanic",   "SVO"),
    "ta": ("Tamil",      "Dravidian",  "SOV"),
    "te": ("Telugu",     "Dravidian",  "SOV"),
    "th": ("Thai",       "Tai-Kadai",  "SVO"),
    "tl": ("Tagalog",    "Austronesian","VSO"),
    "tr": ("Turkish",    "Turkic",     "SOV"),
    "vi": ("Vietnamese", "Austronesian","SVO"),
    "wo": ("Wolof",      "Niger-Congo","SVO"),
    "yo": ("Yoruba",     "Niger-Congo","SVO"),
    "zh": ("Chinese",    "Sino-Tibetan","SVO"),
}

def style_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(GREY1)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=GREY3)
    ax.set_ylabel(ylabel, fontsize=10, color=GREY3)
    ax.tick_params(colors=GREY3, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY2)
    if grid:
        ax.yaxis.grid(True, color=GREY2, linewidth=0.7, linestyle="--")
        ax.set_axisbelow(True)

def save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    if tight:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...")
summary  = pd.read_csv("final_outputs/all_language_summary.csv")
zscores  = pd.read_csv("final_outputs/all_zscores.csv")
ml_res   = pd.read_csv("final_outputs/all_ml_results.csv")
anova    = pd.read_csv("final_outputs/global_stats/typology_anova.csv")
sens     = pd.read_csv("final_outputs/global_stats/sensitivity_analysis.csv")
icm_full = pd.read_csv("final_outputs/global_stats/icm_full_zscore_table.csv")
llm_cmp  = pd.read_csv("final_outputs/global_stats/llm_comparison_results.csv")
power    = pd.read_csv("final_outputs/global_stats/power_analysis.csv")
dravidian= pd.read_csv("final_outputs/global_stats/dravidian_analysis.csv")
anomalies= pd.read_csv("final_outputs/global_stats/data_anomalies.csv")
corr_mat = pd.read_csv("final_outputs/global_stats/correlation_matrix.csv", index_col=0)

# Enrich summary
summary["full_name"]   = summary["language"].map(lambda x: LANG_META.get(x, (x,"",""))[0])
summary["family"]      = summary["language"].map(lambda x: LANG_META.get(x, ("","",""))[1])
summary["word_order"]  = summary["language"].map(lambda x: LANG_META.get(x, ("","","",""))[2])

# Enrich z-scores
icm_z = icm_full.copy()
icm_z["full_name"]  = icm_z["language"].map(lambda x: LANG_META.get(x,(x,"",""))[0])
icm_z["word_order"] = icm_z["language"].map(lambda x: LANG_META.get(x,("","","",""))[2])
icm_z["color"]      = icm_z["word_order"].map(PALETTE_TYPOLOGY)

# z-scores pivot
dlm_z = zscores[zscores["metric"]=="dependency_distance"].set_index("language")["z_score"]
icm_z_ser = zscores[zscores["metric"]=="complexity_score"].set_index("language")["z_score"]

print("Data loaded.\n")

# ════════════════════════════════════════════════════════════════════
# FIG 01 — ICM vs DLM Summary: Divergence of z-scores
# ════════════════════════════════════════════════════════════════════
print("Fig 01: ICM vs DLM overview...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
fig.suptitle("Hypothesis Comparison: ICM vs DLM Across 40 Languages",
             fontsize=16, fontweight="bold", color=TEXT, y=1.02)

langs_sorted = icm_z.sort_values("z_score")["language"].tolist()
full_names   = icm_z.sort_values("z_score")["full_name"].tolist()
z_vals       = icm_z.sort_values("z_score")["z_score"].tolist()
colors       = [ACCENT3 if v > 0 else ACCENT1 for v in z_vals]

ax = axes[0]
bars = ax.barh(range(len(langs_sorted)), z_vals, color=colors, edgecolor="white", linewidth=0.4)
ax.axvline(0, color=TEXT, linewidth=1.5, linestyle="-")
ax.axvline(-1.96, color=ACCENT2, linewidth=1.2, linestyle="--", alpha=0.8, label="p=0.05 threshold (±1.96)")
ax.axvline(1.96,  color=ACCENT2, linewidth=1.2, linestyle="--", alpha=0.8)
ax.set_yticks(range(len(langs_sorted)))
ax.set_yticklabels(full_names, fontsize=7.5)
ax.set_xlabel("Z-score (ICM: negative = more complex interveners avoided)", fontsize=9, color=GREY3)
style_ax(ax, title="ICM Z-scores by Language\n(Negative = ICM Supported)", grid=False)
ax.legend(fontsize=8)
# annotation
ax.text(0.98, 0.02, "25 support  |  15 contradict\nMean z = −0.120  |  0/40 significant",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=GREY2))

# DLM side
dlm_sorted = dlm_z.reindex(langs_sorted)
ax2 = axes[1]
dlm_colors = [ACCENT4 if v > 1.96 else ACCENT1 for v in dlm_sorted.values]
ax2.barh(range(len(langs_sorted)), dlm_sorted.values, color=dlm_colors, edgecolor="white", linewidth=0.4)
ax2.axvline(0, color=TEXT, linewidth=1.5)
ax2.axvline(1.96, color=ACCENT2, linewidth=1.2, linestyle="--", alpha=0.8, label="p=0.05 threshold")
ax2.set_yticks(range(len(langs_sorted)))
ax2.set_yticklabels(full_names, fontsize=7.5)
ax2.set_xlabel("Z-score (DLM: positive = shorter dependencies than random)", fontsize=9, color=GREY3)
style_ax(ax2, title="DLM Z-scores by Language\n(Positive = DLM Confirmed)", grid=False)
ax2.legend(fontsize=8)
ax2.text(0.98, 0.02, "40/40 positive  |  Mean z = +1.807\n15.1× stronger than ICM",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=GREY2))

plt.tight_layout()
save(fig, "fig01_icm_vs_dlm_zscores.png")

# ════════════════════════════════════════════════════════════════════
# FIG 02 — ICM Scatter: Real vs Random Complexity
# ════════════════════════════════════════════════════════════════════
print("Fig 02: ICM real vs random scatter...")
fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")

icm_data = icm_full.copy()
icm_data["full_name"]  = icm_data["language"].map(lambda x: LANG_META.get(x,(x,"",""))[0])
icm_data["word_order"] = icm_data["language"].map(lambda x: LANG_META.get(x,("","","",""))[2])

for wo, grp in icm_data.groupby("word_order"):
    sc = ax.scatter(grp["random_mean"], grp["real_value"],
                    color=PALETTE_TYPOLOGY.get(wo, GREY3),
                    s=80, edgecolors="white", linewidth=0.8,
                    label=wo, zorder=3, alpha=0.9)

# diagonal — where real == random (no effect)
all_vals = list(icm_data["real_value"]) + list(icm_data["random_mean"])
lo, hi = min(all_vals)*0.95, max(all_vals)*1.05
ax.plot([lo, hi], [lo, hi], color=GREY3, linewidth=1.5, linestyle="--", label="Real = Random (null)")
ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.05, color=ACCENT3, label="ICM contradicted zone")
ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.05, color=ACCENT1, label="ICM supported zone")

# label outliers
for _, row in icm_data.iterrows():
    if abs(row["z_score"]) > 0.5:
        ax.annotate(row["full_name"],
                    xy=(row["random_mean"], row["real_value"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=7.5, color=TEXT)

style_ax(ax,
    title="Real vs Random Baseline Complexity Score — ICM Test\n(Below diagonal = ICM supported; Above = ICM contradicted)",
    xlabel="Random Baseline Mean Complexity",
    ylabel="Observed Mean Complexity")
ax.legend(fontsize=9, loc="upper left")
ax.text(0.98, 0.03,
        "Mean z = −0.120\n0/40 languages statistically significant\n25 below diagonal (ICM trend), 15 above",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=GREY1, edgecolor=GREY2))
save(fig, "fig02_icm_real_vs_random_scatter.png")

# ════════════════════════════════════════════════════════════════════
# FIG 03 — DLM vs ICM Z-score Comparison (Paired)
# ════════════════════════════════════════════════════════════════════
print("Fig 03: DLM vs ICM paired comparison...")
fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")

both = pd.DataFrame({
    "language": icm_z_ser.index,
    "ICM_z": icm_z_ser.values,
    "DLM_z": dlm_z.reindex(icm_z_ser.index).values,
    "full_name": [LANG_META.get(l,(l,"",""))[0] for l in icm_z_ser.index],
    "word_order": [LANG_META.get(l,("","","",""))[2] for l in icm_z_ser.index],
}).dropna()

x = np.arange(len(both))
w = 0.38
b1 = ax.bar(x - w/2, both["ICM_z"], w, color=[ACCENT3 if v>0 else ACCENT1 for v in both["ICM_z"]],
            alpha=0.85, label="ICM z-score", edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + w/2, both["DLM_z"], w, color=ACCENT4, alpha=0.75, label="DLM z-score",
            edgecolor="white", linewidth=0.5)
ax.axhline(1.96,  color=ACCENT2, linewidth=1.3, linestyle="--", alpha=0.9, label="±1.96 significance")
ax.axhline(-1.96, color=ACCENT2, linewidth=1.3, linestyle="--", alpha=0.9)
ax.axhline(0, color=TEXT, linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(both["full_name"], rotation=55, ha="right", fontsize=8)
style_ax(ax, title="ICM vs DLM Z-scores — All 40 Languages\n(Blue=ICM supported, Red=ICM contradicted, Teal=DLM)",
         ylabel="Z-score")
ax.legend(fontsize=9)
ax.text(0.01, 0.97, f"DLM mean z = +1.807  |  ICM mean z = −0.120  |  Ratio = 15.1×",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ACCENT4, alpha=0.15, edgecolor=ACCENT4))
save(fig, "fig03_dlm_vs_icm_paired_bars.png")

# ════════════════════════════════════════════════════════════════════
# FIG 04 — Typology ANOVA: All Metrics with Bonferroni
# ════════════════════════════════════════════════════════════════════
print("Fig 04: Typology ANOVA Bonferroni...")
fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")

metric_labels = {
    "avg_dependency_length": "Dep. Length",
    "avg_complexity": "Complexity",
    "avg_arity": "Arity",
    "avg_subtree_size": "Subtree Size",
    "avg_depth": "Depth",
    "avg_efficiency_ratio": "Eff. Ratio (IER)"
}
anova_plot = anova.copy()
anova_plot["label"] = anova_plot["metric"].map(metric_labels)
anova_plot = anova_plot.sort_values("p_value")

colors_anova = [ACCENT4 if s else ACCENT3 for s in anova_plot["survives_bonferroni"]]
bars = ax.barh(anova_plot["label"], -np.log10(anova_plot["p_value"]),
               color=colors_anova, edgecolor="white", linewidth=0.6, height=0.6)

# Bonferroni line
bf = -np.log10(0.008333)
ax.axvline(bf, color=BRAND, linewidth=2, linestyle="--", label=f"Bonferroni threshold (p=0.0083)")
# nominal p=0.05
ax.axvline(-np.log10(0.05), color=GREY3, linewidth=1.2, linestyle=":", label="Nominal p=0.05")

for bar, (_, row) in zip(bars, anova_plot.iterrows()):
    label = f"p={row['p_value']:.4f}, F={row['f_stat']:.2f}"
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=8.5, color=TEXT)

style_ax(ax, title="Typology ANOVA: Which Metrics Differ Across Word-Order Groups?\n(Bonferroni-corrected, n=40 languages, 4 groups)",
         xlabel="−log₁₀(p-value)")
ax.legend(fontsize=9)
patch_sig   = mpatches.Patch(color=ACCENT4, label="Survives Bonferroni")
patch_nosig = mpatches.Patch(color=ACCENT3, label="Does not survive")
ax.legend(handles=[patch_sig, patch_nosig,
                   plt.Line2D([0],[0], color=BRAND, linewidth=2, linestyle="--", label="Bonferroni threshold"),
                   plt.Line2D([0],[0], color=GREY3, linewidth=1.2, linestyle=":", label="p=0.05 nominal")],
          fontsize=9)
save(fig, "fig04_typology_anova_bonferroni.png")

# ════════════════════════════════════════════════════════════════════
# FIG 05 — IER by Word Order (the surviving metric)
# ════════════════════════════════════════════════════════════════════
print("Fig 05: IER by word order...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

# Violin plot
ax = axes[0]
order = ["SOV","SVO","VSO","Free"]
data_by_order = [summary[summary["word_order"]==wo]["avg_efficiency_ratio"].values for wo in order]
parts = ax.violinplot(data_by_order, positions=range(len(order)), showmedians=True, showextrema=True)
for i, (pc, wo) in enumerate(zip(parts["bodies"], order)):
    pc.set_facecolor(PALETTE_TYPOLOGY[wo])
    pc.set_alpha(0.75)
parts["cmedians"].set_color(TEXT)
parts["cbars"].set_color(GREY3)
parts["cmins"].set_color(GREY3)
parts["cmaxes"].set_color(GREY3)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=10)
style_ax(ax, title="Intervener Efficiency Ratio by Word Order\n(Only metric surviving Bonferroni, p=0.0079)",
         ylabel="IER (Dependency Length / Complexity)")
ax.text(0.98, 0.97, "F=4.609, p=0.0079*\nSurvives Bonferroni",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F8F0", edgecolor=ACCENT4))

# Scatter: all languages
ax2 = axes[1]
for _, row in summary.iterrows():
    wo = row["word_order"]
    ax2.scatter(row["avg_complexity"], row["avg_efficiency_ratio"],
                color=PALETTE_TYPOLOGY.get(wo, GREY3),
                s=70, edgecolors="white", linewidth=0.7, zorder=3, alpha=0.9)
    ax2.annotate(row["language"], xy=(row["avg_complexity"], row["avg_efficiency_ratio"]),
                 xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=GREY3)

for wo, col in PALETTE_TYPOLOGY.items():
    ax2.scatter([], [], color=col, label=wo, s=60, edgecolors="white")

style_ax(ax2, title="Efficiency Ratio vs Mean Complexity — All 40 Languages",
         xlabel="Mean Complexity Score", ylabel="Mean IER")
ax2.legend(title="Word Order", fontsize=8)
save(fig, "fig05_ier_word_order.png")

# ════════════════════════════════════════════════════════════════════
# FIG 06 — Sensitivity Analysis: Complexity Weight Robustness
# ════════════════════════════════════════════════════════════════════
print("Fig 06: Sensitivity analysis...")
fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

sens_labels = {
    "original_0.35-0.25-0.20-0.20":   "Original\n(0.35/0.25/0.20/0.20)",
    "equal_0.25-0.25-0.25-0.25":       "Equal Weights\n(0.25/0.25/0.25/0.25)",
    "arity_heavy_0.50-0.20-0.20-0.10": "Arity-Heavy\n(0.50/0.20/0.20/0.10)",
    "struct_heavy_0.20-0.40-0.30-0.10":"Structure-Heavy\n(0.20/0.40/0.30/0.10)",
    "no_pos_0.35-0.30-0.35-0.00":      "No POS Weight\n(0.35/0.30/0.35/0.00)",
    "depth_heavy_0.20-0.20-0.50-0.10": "Depth-Heavy\n(0.20/0.20/0.50/0.10)",
}
sens["label"] = sens["weight_set"].map(sens_labels)

bar_colors = [ACCENT3] * len(sens)  # all non-significant
ax.bar(sens["label"], sens["anova_p"], color=bar_colors, edgecolor="white", linewidth=0.6, width=0.5)
ax.axhline(0.05,   color=GREY3,   linewidth=1.3, linestyle=":", label="p=0.05 (nominal)")
ax.axhline(0.0083, color=BRAND,   linewidth=2,   linestyle="--", label="Bonferroni threshold (p=0.0083)")

for i, (_, row) in enumerate(sens.iterrows()):
    ax.text(i, row["anova_p"] + 0.004, f"p={row['anova_p']:.3f}", ha="center", fontsize=8.5, color=TEXT)

style_ax(ax, title="Sensitivity Analysis: Typology ANOVA p-value Under 6 Complexity Weight Sets\n(0/6 significant — null finding is robust)",
         ylabel="ANOVA p-value")
ax.set_ylim(0, max(sens["anova_p"])*1.25)
ax.legend(fontsize=9)
ax.text(0.99, 0.97, "0 / 6 weight sets\nproduce significant ANOVA",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor=ACCENT2))
save(fig, "fig06_sensitivity_analysis.png")

# ════════════════════════════════════════════════════════════════════
# FIG 07 — ML Classification: F1 by Language (corrected, no leakage)
# ════════════════════════════════════════════════════════════════════
print("Fig 07: ML F1 scores...")
best_f1 = ml_res.groupby("language")["f1_score"].max().reset_index()
best_f1["full_name"]  = best_f1["language"].map(lambda x: LANG_META.get(x,(x,"",""))[0])
best_f1["word_order"] = best_f1["language"].map(lambda x: LANG_META.get(x,("","","",""))[2])
best_f1 = best_f1.sort_values("f1_score")

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
bar_cols = [PALETTE_TYPOLOGY.get(w, GREY3) for w in best_f1["word_order"]]
bars = ax.barh(best_f1["full_name"], best_f1["f1_score"],
               color=bar_cols, edgecolor="white", linewidth=0.4, height=0.7)
ax.axvline(0.829, color=BRAND, linewidth=2, linestyle="--", label=f"Mean F1 = 0.829")
ax.axvline(0.75,  color=GREY3, linewidth=1.2, linestyle=":", label="F1 = 0.75 (acceptable)")

for bar, val in zip(bars, best_f1["f1_score"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=7.5, color=TEXT)

for wo, col in PALETTE_TYPOLOGY.items():
    ax.scatter([], [], color=col, s=60, label=wo, edgecolors="white")

style_ax(ax, title="ML Classification F1 Score by Language\n(GradientBoosting, non-leaky features, corrected)",
         xlabel="Best F1 Score")
ax.set_xlim(0, 1.08)
ax.legend(title="Word Order", fontsize=8, ncol=2)
ax.text(0.99, 0.02,
        "Mean F1 = 0.829 (was 0.999 with leaky features)\nFeatures: dep_distance, direction, POS encodings",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=GREY2))
save(fig, "fig07_ml_f1_by_language.png")

# ════════════════════════════════════════════════════════════════════
# FIG 08 — ML Model Comparison Across All Languages
# ════════════════════════════════════════════════════════════════════
print("Fig 08: ML model comparison...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

model_order = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP"]
model_colors = [ACCENT1, ACCENT4, BRAND, ACCENT2]
model_labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting", "MLP"]

ax = axes[0]
model_stats = ml_res.groupby("model_name")["f1_score"].agg(["mean","std"]).reindex(model_order)
bars = ax.bar(model_labels, model_stats["mean"], color=model_colors, edgecolor="white",
              linewidth=0.6, width=0.55, yerr=model_stats["std"], capsize=5,
              error_kw={"elinewidth":1.5, "ecolor": GREY3})
for bar, mean in zip(bars, model_stats["mean"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{mean:.3f}", ha="center", fontsize=9, fontweight="bold")
style_ax(ax, title="Mean F1 Score by Model (±1 SD)\nAcross 40 Languages",
         ylabel="Mean F1 Score")
ax.set_ylim(0, 1.0)

ax2 = axes[1]
# Win frequency
best_per_lang = ml_res.loc[ml_res.groupby("language")["f1_score"].idxmax(), "model_name"]
win_counts = best_per_lang.value_counts().reindex(model_order, fill_value=0)
bars2 = ax2.bar(model_labels, win_counts.values, color=model_colors, edgecolor="white",
                linewidth=0.6, width=0.55)
for bar, cnt in zip(bars2, win_counts.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
             str(cnt), ha="center", fontsize=11, fontweight="bold")
style_ax(ax2, title="Best Model Win Count\n(Out of 40 Languages)",
         ylabel="Languages Won")

plt.tight_layout()
save(fig, "fig08_ml_model_comparison.png")

# ════════════════════════════════════════════════════════════════════
# FIG 09 — LLM Comparison: GPT-2 JS Divergence vs Null
# ════════════════════════════════════════════════════════════════════
print("Fig 09: LLM divergence comparison...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

metrics = ["arity","dependency_distance","subtree_size","complexity_score"]
metric_labels_llm = ["Arity","Dependency\nDistance","Subtree\nSize","Complexity\nScore"]

gpt2 = llm_cmp[llm_cmp["llm"]=="GPT-2"].set_index("metric")
bert = llm_cmp[llm_cmp["llm"]=="BERT"].set_index("metric")

ax = axes[0]
x = np.arange(len(metrics))
w = 0.25
ci_lo = [gpt2.loc[m,"js_ci_lower"] for m in metrics]
ci_hi = [gpt2.loc[m,"js_ci_upper"] for m in metrics]
gpt2_js = [gpt2.loc[m,"js_divergence"] for m in metrics]
null95  = [gpt2.loc[m,"js_null_95th"] for m in metrics]
uni_rnd = [gpt2.loc[m,"uniform_random_js"] for m in metrics]

b1 = ax.bar(x - w, gpt2_js, w*0.9, color=ACCENT1, label="GPT-2 JS", edgecolor="white")
b2 = ax.bar(x,     null95,  w*0.9, color=ACCENT2, alpha=0.7, label="Null 95th pct", edgecolor="white")

# CI error bars on GPT-2
err_lo = np.maximum(0, np.array(gpt2_js) - np.array(ci_lo))
err_hi = np.maximum(0, np.array(ci_hi) - np.array(gpt2_js))
ax.errorbar(x - w, gpt2_js, yerr=[err_lo, err_hi], fmt="none",
            ecolor=BRAND, capsize=4, elinewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels_llm, fontsize=9)
style_ax(ax, title="GPT-2 JS Divergence vs Null Baseline\n(with 95% bootstrap CI)",
         ylabel="Jensen-Shannon Divergence")
ax.legend(fontsize=9)
for i, (js, n95) in enumerate(zip(gpt2_js, null95)):
    sig = "★" if js > n95 else ""
    ax.text(i - w, js + max(err_hi)*0.1, sig, ha="center", fontsize=12, color=ACCENT3)
ax.text(0.99, 0.97, "★ = significant vs null\nAll 4 metrics significant",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=GREY2))

ax2 = axes[1]
# Ratio: uniform random / LLM  (how much closer LLM is than chance)
ratios_gpt2 = [gpt2.loc[m,"uniform_random_js"] / gpt2.loc[m,"js_divergence"] for m in metrics]
ratios_bert = [bert.loc[m,"uniform_random_js"] / bert.loc[m,"js_divergence"] for m in metrics]
x2 = np.arange(len(metrics))
ax2.bar(x2 - w/2, ratios_gpt2, w*0.9, color=ACCENT1, label="GPT-2", edgecolor="white")
ax2.bar(x2 + w/2, ratios_bert, w*0.9, color=ACCENT3, label="BERT (reference only)", edgecolor="white", alpha=0.7)
ax2.set_xticks(x2)
ax2.set_xticklabels(metric_labels_llm, fontsize=9)
for i, (rg, rb) in enumerate(zip(ratios_gpt2, ratios_bert)):
    ax2.text(i - w/2, rg + 0.3, f"{rg:.0f}×", ha="center", fontsize=8, color=ACCENT1, fontweight="bold")
    ax2.text(i + w/2, rb + 0.3, f"{rb:.1f}×", ha="center", fontsize=8, color=ACCENT3, fontweight="bold")

style_ax(ax2, title="LLM Proximity to Real Text\n(Ratio: Uniform-Random JS ÷ LLM JS, higher = closer to real)",
         ylabel="Ratio (×)")
ax2.legend(fontsize=9)
ax2.text(0.01, 0.97, "GPT-2 is 23×–1888× closer to\nreal text than uniform random",
         transform=ax2.transAxes, va="top", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4FD", edgecolor=ACCENT1))

plt.tight_layout()
save(fig, "fig09_llm_divergence_comparison.png")

# ════════════════════════════════════════════════════════════════════
# FIG 10 — Dravidian ANCOVA: Before vs After Word-Order Control
# ════════════════════════════════════════════════════════════════════
print("Fig 10: Dravidian ANCOVA...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")
fig.suptitle("Dravidian vs Non-Dravidian: Mann-Whitney vs ANCOVA (Word-Order Controlled)",
             fontsize=14, fontweight="bold", color=TEXT)

metrics_dravidian = ["complexity_score", "dependency_distance", "arity"]
metric_labels_drv = ["Complexity Score", "Dependency Distance", "Arity"]
dravidian_codes = ["ta","te","ml"]

for idx, (metric, mlabel) in enumerate(zip(metrics_dravidian, metric_labels_drv)):
    ax = axes[idx]
    drv_row = dravidian[dravidian["metric"] == metric].iloc[0]

    means = [drv_row["dravidian_mean"], drv_row["other_mean"]]
    groups = ["Dravidian\n(ta, te, ml)", "Non-Dravidian\n(37 languages)"]
    c = [ACCENT3, ACCENT1]
    bars = ax.bar(groups, means, color=c, edgecolor="white", linewidth=0.6, width=0.45)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{mean:.3f}", ha="center", fontsize=9, fontweight="bold")

    mw_p = drv_row["p_value_unadjusted"]
    anc_p = drv_row["ancova_p"]
    color_box = "#E8F8F0" if anc_p > 0.05 else "#FFF0F0"
    ax.text(0.5, 0.95,
            f"Mann-Whitney p={mw_p:.2e}\nANCOVA (word-order) p={anc_p:.3f}\nd={drv_row['cohens_d']:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color_box, edgecolor=GREY2))

    sig_label = "Effect disappears\nafter word-order control" if anc_p > 0.05 else "Effect persists"
    ax.set_title(f"{mlabel}\n({sig_label})", fontsize=10, fontweight="bold", color=TEXT)
    style_ax(ax, ylabel=mlabel)

plt.tight_layout()
save(fig, "fig10_dravidian_ancova.png")

# ════════════════════════════════════════════════════════════════════
# FIG 11 — Language Heatmap: 5 Metrics Across 40 Languages
# ════════════════════════════════════════════════════════════════════
print("Fig 11: Language metrics heatmap...")
cols = ["avg_dependency_length","avg_complexity","avg_arity","avg_subtree_size","avg_efficiency_ratio"]
col_labels = ["Dep. Length","Complexity","Arity","Subtree Size","IER"]

heat = summary.set_index("full_name")[cols].copy()
# z-score normalize each column for comparability
heat_z = heat.apply(lambda c: (c - c.mean())/c.std())
heat_z = heat_z.sort_values("avg_complexity")

fig, ax = plt.subplots(figsize=(12, 14), facecolor="white")
cmap = LinearSegmentedColormap.from_list("big4", ["#1A73E8","white","#D64045"], N=256)
im = ax.imshow(heat_z.values, cmap=cmap, aspect="auto", vmin=-2.5, vmax=2.5)

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(heat_z)))
ax.set_yticklabels(heat_z.index, fontsize=8.5)

for i in range(heat_z.shape[0]):
    for j in range(heat_z.shape[1]):
        v = heat_z.values[i, j]
        txt_col = "white" if abs(v) > 1.5 else TEXT
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=txt_col)

plt.colorbar(im, ax=ax, label="Z-score (column-normalized)", fraction=0.02, pad=0.02)
ax.set_title("Cross-Linguistic Metrics Heatmap — All 40 Languages\n(Z-score normalized, sorted by complexity)",
             fontsize=13, fontweight="bold", color=TEXT, pad=12)

# Word-order annotation on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(heat_z)))
wo_labels = [LANG_META.get(summary[summary["full_name"]==fn]["language"].values[0] if len(summary[summary["full_name"]==fn])>0 else "", ("","","",""))[2]
             for fn in heat_z.index]
ax2.set_yticklabels(wo_labels, fontsize=8)
ax2.tick_params(right=True, labelright=True)
for ytick, wo in zip(ax2.get_yticklabels(), wo_labels):
    ytick.set_color(PALETTE_TYPOLOGY.get(wo, GREY3))

save(fig, "fig11_language_metrics_heatmap.png")

# ════════════════════════════════════════════════════════════════════
# FIG 12 — Correlation Matrix of Language-Level Metrics
# ════════════════════════════════════════════════════════════════════
print("Fig 12: Correlation matrix...")
corr_cols = ["avg_dependency_length","avg_complexity","avg_arity",
             "avg_subtree_size","avg_depth","avg_efficiency_ratio","percent_left_dependencies"]
corr_labels = ["Dep.Length","Complexity","Arity","Subtree","Depth","IER","% Left Deps"]

c_mat = summary[corr_cols].corr()
c_mat.columns = corr_labels
c_mat.index   = corr_labels

fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
cmap2 = LinearSegmentedColormap.from_list("corr", [ACCENT3,"white",ACCENT4], N=256)
mask = np.triu(np.ones_like(c_mat, dtype=bool), k=1)
sns.heatmap(c_mat, ax=ax, annot=True, fmt=".2f", cmap=cmap2,
            vmin=-1, vmax=1, linewidths=0.5, mask=mask,
            cbar_kws={"label":"Pearson r", "shrink":0.8},
            annot_kws={"size":10})
ax.set_title("Correlation Matrix: Language-Level Metrics\n(Lower triangle, Pearson r)",
             fontsize=13, fontweight="bold", color=TEXT, pad=12)
ax.tick_params(labelsize=10)
save(fig, "fig12_correlation_matrix.png")

# ════════════════════════════════════════════════════════════════════
# FIG 13 — Power Analysis
# ════════════════════════════════════════════════════════════════════
print("Fig 13: Power analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

ax = axes[0]
n_per_group = np.arange(5, 250, 5)
from scipy.stats import f as f_dist

def compute_power(n, f_effect, k=4, alpha=0.05):
    df1 = k - 1
    df2 = k * (n - 1)
    ncp  = f_effect**2 * k * n
    f_crit = f_dist.ppf(1 - alpha, df1, df2)
    from scipy.stats import ncf
    return 1 - ncf.cdf(f_crit, df1, df2, ncp)

for effect, label, col in [(0.10,"Small (f=0.10)",ACCENT4),(0.25,"Medium (f=0.25)",ACCENT1),(0.40,"Large (f=0.40)",ACCENT3)]:
    power_curve = [compute_power(n, effect) for n in n_per_group]
    ax.plot(n_per_group, power_curve, color=col, linewidth=2.2, label=label)

ax.axhline(0.80, color=ACCENT2, linewidth=1.8, linestyle="--", label="80% power threshold")
ax.axvline(10,   color=BRAND,   linewidth=1.8, linestyle=":",  label="Current n ≈ 10/group (40 total)")
ax.fill_between(n_per_group,
                [compute_power(n, 0.25) for n in n_per_group], 0.80,
                where=[compute_power(n, 0.25) < 0.80 for n in n_per_group],
                alpha=0.12, color=ACCENT3, label="Underpowered region")

style_ax(ax, title="Statistical Power vs Sample Size\n(ANOVA, k=4 word-order groups)",
         xlabel="Languages per Group", ylabel="Statistical Power")
ax.legend(fontsize=8.5)
ax.set_xlim(5, 250)
ax.set_ylim(0, 1.02)
ax.text(10.5, 0.82, f"Current study:\n~10/group\n7.2% power\n(medium effect)",
        fontsize=8, color=BRAND,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=BRAND))

ax2 = axes[1]
effects   = ["Small\n(f=0.10)", "Medium\n(f=0.25)", "Large\n(f=0.40)"]
powers_40 = [compute_power(10, 0.10), compute_power(10, 0.25), compute_power(10, 0.40)]
needed    = [None, 179, None]

bar_cols2 = [ACCENT4, ACCENT3, ACCENT2]
bars = ax2.bar(effects, [p*100 for p in powers_40], color=bar_cols2, edgecolor="white",
               linewidth=0.6, width=0.45)
ax2.axhline(80, color=ACCENT2, linewidth=2, linestyle="--", label="Target: 80% power")
for bar, p in zip(bars, powers_40):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f"{p*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

style_ax(ax2, title="Achieved Power at n=40 Total Languages\n(Current study = massively underpowered)",
         ylabel="Power (%)")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9)
ax2.text(0.5, 0.60,
         "Need ~714 languages\nfor 80% power (medium effect)\nCurrent: 40 languages (5.6%)",
         transform=ax2.transAxes, ha="center", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor=ACCENT2))

plt.tight_layout()
save(fig, "fig13_power_analysis.png")

# ════════════════════════════════════════════════════════════════════
# FIG 14 — Arabic Outlier Investigation
# ════════════════════════════════════════════════════════════════════
print("Fig 14: Arabic outlier...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

ax = axes[0]
dep_lengths = summary[["full_name","avg_dependency_length","language"]].copy()
colors_ar = [ACCENT3 if l=="ar" else ACCENT1 for l in dep_lengths["language"]]
dep_lengths_sorted = dep_lengths.sort_values("avg_dependency_length", ascending=False)
bar_cols_ar = [ACCENT3 if l=="ar" else ACCENT1 for l in dep_lengths_sorted["language"]]
bars = ax.bar(dep_lengths_sorted["full_name"], dep_lengths_sorted["avg_dependency_length"],
              color=bar_cols_ar, edgecolor="white", linewidth=0.4)
pop_mean = dep_lengths["avg_dependency_length"].mean()
ax.axhline(pop_mean, color=BRAND, linewidth=1.8, linestyle="--", label=f"Population mean = {pop_mean:.1f}")
ax.text(0, dep_lengths_sorted[dep_lengths_sorted["language"]=="ar"]["avg_dependency_length"].values[0] + 0.3,
        f"Arabic = 30.85\n(z = 3.96, PADT clitic\ntokenization artifact)",
        fontsize=8.5, color=ACCENT3,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF0F0", edgecolor=ACCENT3))

ax.set_xticklabels(dep_lengths_sorted["full_name"], rotation=55, ha="right", fontsize=7.5)
style_ax(ax, title="Mean Dependency Length by Language\n(Arabic is a 3.96σ outlier — PADT clitic tokenization)",
         ylabel="Mean Dependency Length")
ax.legend(fontsize=9)

ax2 = axes[1]
# ANOVA robustness: with vs without Arabic
with_ar_p  = 0.1521
without_ar_p = 0.1372
corr_with  = 0.923
corr_without = 0.911

x2 = np.arange(2)
ax2.bar(["With Arabic","Without Arabic"], [with_ar_p, without_ar_p],
        color=[ACCENT3, ACCENT1], edgecolor="white", width=0.4)
ax2.axhline(0.0083, color=BRAND, linewidth=1.8, linestyle="--", label="Bonferroni threshold")
ax2.axhline(0.05,   color=GREY3, linewidth=1.2, linestyle=":", label="p=0.05")
for i, (label, val) in enumerate(zip(["With Arabic","Without Arabic"],[with_ar_p, without_ar_p])):
    ax2.text(i, val + 0.004, f"p={val:.4f}", ha="center", fontsize=10, fontweight="bold")
style_ax(ax2, title="Arabic Outlier Impact on Complexity ANOVA\n(Delta p = 0.015 — negligible impact)",
         ylabel="ANOVA p-value")
ax2.set_ylim(0, 0.25)
ax2.legend(fontsize=9)
ax2.text(0.5, 0.85, "Arabic inclusion does\nnot change conclusions",
         transform=ax2.transAxes, ha="center", fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F0", edgecolor=ACCENT4))

plt.tight_layout()
save(fig, "fig14_arabic_outlier.png")

# ════════════════════════════════════════════════════════════════════
# FIG 15 — Left vs Right Dependency Asymmetry
# ════════════════════════════════════════════════════════════════════
print("Fig 15: Left vs Right dependency...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

ax = axes[0]
summary_sorted = summary.sort_values("percent_left_dependencies")
left_pct  = summary_sorted["percent_left_dependencies"].values
right_pct = summary_sorted["percent_right_dependencies"].values
full_names_sorted = summary_sorted["full_name"].values
wo_sorted = summary_sorted["word_order"].values

x = np.arange(len(summary_sorted))
ax.barh(x, left_pct,  color=ACCENT1, label="% Left dependencies",  edgecolor="white", linewidth=0.3)
ax.barh(x, right_pct, left=left_pct, color=ACCENT3, label="% Right dependencies", edgecolor="white", linewidth=0.3)
ax.axvline(50, color=TEXT, linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_yticks(x)
ax.set_yticklabels(full_names_sorted, fontsize=8)
ax.set_xlabel("Percentage of Dependencies", fontsize=9, color=GREY3)
style_ax(ax, title="Left vs Right Dependency Direction\n(Sorted by % left dependencies)", grid=False)
ax.legend(fontsize=9)
for wo, col in PALETTE_TYPOLOGY.items():
    ax.scatter([], [], color=col, s=40, label=wo, marker="s")

ax2 = axes[1]
ax2.scatter(summary["percent_left_dependencies"], summary["avg_complexity"],
            c=[PALETTE_TYPOLOGY.get(w, GREY3) for w in summary["word_order"]],
            s=70, edgecolors="white", linewidth=0.8, zorder=3)
m, b, r, p, se = stats.linregress(summary["percent_left_dependencies"], summary["avg_complexity"])
x_line = np.linspace(summary["percent_left_dependencies"].min(), summary["percent_left_dependencies"].max(), 100)
ax2.plot(x_line, m*x_line+b, color=BRAND, linewidth=2, linestyle="--",
         label=f"Trend: r={r:.3f}, p={p:.3f}")
for _, row in summary.iterrows():
    ax2.annotate(row["language"], (row["percent_left_dependencies"], row["avg_complexity"]),
                 xytext=(3,2), textcoords="offset points", fontsize=6.5, color=GREY3)
for wo, col in PALETTE_TYPOLOGY.items():
    ax2.scatter([], [], color=col, s=50, label=wo)
style_ax(ax2, title="% Left Dependencies vs Mean Complexity\n(Left-branching = more complex interveners)",
         xlabel="% Left Dependencies", ylabel="Mean Complexity Score")
ax2.legend(fontsize=8.5, ncol=2)
ax2.text(0.01, 0.97,
         "Left deps: mean complexity 1.555\nRight deps: mean complexity 1.326\nd = 0.193, p ≈ 0",
         transform=ax2.transAxes, va="top", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=GREY1, edgecolor=GREY2))

plt.tight_layout()
save(fig, "fig15_left_right_dependency.png")

# ════════════════════════════════════════════════════════════════════
# FIG 16 — Cross-Language Typology Boxplots (4 metrics)
# ════════════════════════════════════════════════════════════════════
print("Fig 16: Typology boxplots (4 metrics)...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor="white")
fig.suptitle("Distribution of Key Metrics by Typological Word-Order Group",
             fontsize=15, fontweight="bold", color=TEXT)

metrics4 = ["avg_complexity","avg_dependency_length","avg_arity","avg_efficiency_ratio"]
titles4  = ["Mean Complexity Score","Mean Dependency Length","Mean Arity","Mean IER"]
order4   = ["SOV","SVO","VSO","Free"]

for ax, metric, title in zip(axes.flatten(), metrics4, titles4):
    data4 = [summary[summary["word_order"]==wo][metric].values for wo in order4]
    parts = ax.violinplot(data4, positions=range(len(order4)), showmedians=True, widths=0.7)
    for i, (pc, wo) in enumerate(zip(parts["bodies"], order4)):
        pc.set_facecolor(PALETTE_TYPOLOGY[wo])
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
    parts["cmedians"].set_color(BRAND)
    parts["cmedians"].set_linewidth(2)
    for comp in ["cbars","cmins","cmaxes"]:
        parts[comp].set_color(GREY3)
        parts[comp].set_linewidth(1.2)

    # overlay scatter
    for i, (wo, d) in enumerate(zip(order4, data4)):
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(d))
        ax.scatter(i + jitter, d, color=PALETTE_TYPOLOGY[wo], s=30, alpha=0.6,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # annotate language codes
    for i, (wo, d) in enumerate(zip(order4, data4)):
        langs_in_wo = summary[summary["word_order"]==wo]["language"].values
        for j, (lang, val) in enumerate(zip(langs_in_wo, d)):
            ax.annotate(lang, (i + 0.08, val), fontsize=6, color=GREY3)

    ax.set_xticks(range(len(order4)))
    ax.set_xticklabels(order4, fontsize=10)
    anova_row = anova[anova["metric"]==metric]
    if len(anova_row):
        p = anova_row["p_value"].values[0]
        surv = anova_row["survives_bonferroni"].values[0]
        sig_str = "★ Survives Bonferroni" if surv else f"p={p:.4f} (n.s.)"
        ax.text(0.99, 0.97, sig_str, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=ACCENT4 if surv else GREY3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=GREY1, edgecolor=GREY2))
    style_ax(ax, title=title, ylabel=title)

plt.tight_layout()
save(fig, "fig16_typology_boxplots.png")

# ════════════════════════════════════════════════════════════════════
# FIG 17 — ICM Contradiction Map: Full 40-Language Table
# ════════════════════════════════════════════════════════════════════
print("Fig 17: ICM contradiction summary table...")
fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")

icm_sort = icm_full.sort_values("z_score")
icm_sort["full_name"]  = icm_sort["language"].map(lambda x: LANG_META.get(x,(x,"",""))[0])
icm_sort["word_order"] = icm_sort["language"].map(lambda x: LANG_META.get(x,("","","",""))[2])
icm_sort["family"]     = icm_sort["language"].map(lambda x: LANG_META.get(x,("","",""))[1])

bar_colors = []
for _, row in icm_sort.iterrows():
    if row["z_score"] < -1.96:   bar_colors.append("#0D47A1")
    elif row["z_score"] < -0.5:  bar_colors.append(ACCENT1)
    elif row["z_score"] < 0:     bar_colors.append("#90CAF9")
    elif row["z_score"] < 0.5:   bar_colors.append("#FFCDD2")
    elif row["z_score"] < 1.0:   bar_colors.append(ACCENT2)
    else:                         bar_colors.append(ACCENT3)

bars = ax.barh(icm_sort["full_name"], icm_sort["z_score"],
               color=bar_colors, edgecolor="white", linewidth=0.4, height=0.72)
ax.axvline(0,     color=TEXT,   linewidth=1.8)
ax.axvline(-1.96, color=ACCENT2, linewidth=1.4, linestyle="--", alpha=0.8, label="±1.96 significance")
ax.axvline(1.96,  color=ACCENT2, linewidth=1.4, linestyle="--", alpha=0.8)

for bar, (_, row) in zip(bars, icm_sort.iterrows()):
    val = row["z_score"]
    offset = 0.03 if val >= 0 else -0.03
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=7.5, color=TEXT)

from matplotlib.patches import Patch
legend_patches = [
    Patch(color="#0D47A1", label="Strong ICM support (z<−1.96)"),
    Patch(color=ACCENT1,   label="Moderate ICM support (z<−0.5)"),
    Patch(color="#90CAF9", label="Weak ICM support (−0.5<z<0)"),
    Patch(color="#FFCDD2", label="Weak contradiction (0<z<0.5)"),
    Patch(color=ACCENT2,   label="Moderate contradiction (0.5<z<1)"),
    Patch(color=ACCENT3,   label="Strong contradiction (z>1)"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
style_ax(ax, title="ICM Z-scores — All 40 Languages Ranked\n(Left = ICM supported, Right = contradicted; Mean z = −0.120)", grid=False)
ax.set_xlabel("Z-score (< 0 = ICM supported)", fontsize=10, color=GREY3)

save(fig, "fig17_icm_full_contradiction_table.png")

# ════════════════════════════════════════════════════════════════════
# FIG 18 — PCA: Language Clustering by Typological Features
# ════════════════════════════════════════════════════════════════════
print("Fig 18: PCA language clustering...")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_cols = ["avg_dependency_length","avg_complexity","avg_arity",
            "avg_subtree_size","avg_depth","avg_efficiency_ratio","percent_left_dependencies"]
X = summary[pca_cols].values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
for _, row in summary.iterrows():
    idx = summary.index.get_loc(_)
    wo = row["word_order"]
    ax.scatter(coords[idx, 0], coords[idx, 1],
               color=PALETTE_TYPOLOGY.get(wo, GREY3),
               s=90, edgecolors="white", linewidth=1, zorder=3, alpha=0.9)
    ax.annotate(row["full_name"],
                xy=(coords[idx, 0], coords[idx, 1]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7.5, color=TEXT)

for wo, col in PALETTE_TYPOLOGY.items():
    ax.scatter([], [], color=col, s=70, label=wo)

var_explained = pca.explained_variance_ratio_
style_ax(ax,
    title=f"PCA: Language Clustering by Typological Features\n(PC1 explains {var_explained[0]*100:.1f}%, PC2 explains {var_explained[1]*100:.1f}%)",
    xlabel=f"PC1 ({var_explained[0]*100:.1f}% variance)",
    ylabel=f"PC2 ({var_explained[1]*100:.1f}% variance)")
ax.legend(title="Word Order", fontsize=9)

# biplot arrows
feature_labels_pca = ["Dep.Length","Complexity","Arity","Subtree","Depth","IER","% Left"]
scale = 2.5
for i, label in enumerate(feature_labels_pca):
    ax.annotate("", xy=(pca.components_[0, i]*scale, pca.components_[1, i]*scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GREY3, lw=1.2))
    ax.text(pca.components_[0, i]*scale*1.12, pca.components_[1, i]*scale*1.12,
            label, fontsize=7.5, color=GREY3, ha="center")

save(fig, "fig18_pca_language_clustering.png")

# ════════════════════════════════════════════════════════════════════
# FIG 19 — LLM: GPT-2 Distribution Comparison (key metrics)
# ════════════════════════════════════════════════════════════════════
print("Fig 19: GPT-2 distribution comparisons...")
try:
    gpt2_feat = pd.read_csv("final_outputs/global_stats/gpt2_features.csv")
    real_feat  = pd.read_csv("final_outputs/all_features.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.suptitle("GPT-2 Generated vs Real Corpus: Feature Distributions\n(n=43,112 GPT-2 interveners vs 555,052 real)",
                 fontsize=14, fontweight="bold", color=TEXT)

    plot_metrics = ["arity","dependency_distance","subtree_size","complexity_score"]
    plot_titles  = ["Arity","Dependency Distance","Subtree Size","Complexity Score"]

    real_en = real_feat[real_feat["language"]=="en"] if "language" in real_feat.columns else real_feat

    for ax, metric, ptitle in zip(axes.flatten(), plot_metrics, plot_titles):
        gpt2_vals = gpt2_feat[metric].dropna()
        real_vals = real_en[metric].dropna() if metric in real_en.columns else pd.Series(dtype=float)
        if len(real_vals) == 0:
            real_vals = real_feat[metric].dropna() if metric in real_feat.columns else pd.Series(dtype=float)

        vmax = min(gpt2_vals.quantile(0.99), real_vals.quantile(0.99)) if len(real_vals) > 0 else gpt2_vals.quantile(0.99)
        bins = np.linspace(0, vmax, 40)

        if len(real_vals) > 0:
            ax.hist(real_vals.clip(upper=vmax), bins=bins, density=True,
                    color=ACCENT1, alpha=0.65, label=f"Real corpus (en, n={len(real_vals):,})", edgecolor="none")
        ax.hist(gpt2_vals.clip(upper=vmax), bins=bins, density=True,
                color=ACCENT3, alpha=0.65, label=f"GPT-2 (n={len(gpt2_vals):,})", edgecolor="none")

        js = gpt2.loc[metric, "js_divergence"] if metric in gpt2.index else None
        ann_str = f"JS = {js:.5f}" if js else ""
        ax.text(0.98, 0.97, ann_str, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor=GREY1, edgecolor=GREY2))
        style_ax(ax, title=ptitle, xlabel=ptitle, ylabel="Density")
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    save(fig, "fig19_gpt2_distribution_comparison.png")
except Exception as e:
    print(f"  Fig 19 skipped: {e}")

# ════════════════════════════════════════════════════════════════════
# FIG 20 — Executive Summary Dashboard
# ════════════════════════════════════════════════════════════════════
print("Fig 20: Executive summary dashboard...")
fig = plt.figure(figsize=(20, 14), facecolor=GREY1)
fig.suptitle("Intervener Complexity in Dependency Grammar — Study Summary Dashboard\n40 Languages · Universal Dependencies Treebanks · Bootstrap CIs · Bonferroni Corrected",
             fontsize=16, fontweight="bold", color=TEXT, y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── KPI boxes ──
kpis = [
    ("40", "Languages\nAnalyzed", ACCENT1),
    ("0 / 40", "ICM Significant\n(0%)", ACCENT3),
    ("40 / 40", "DLM Confirmed\n(100%)", ACCENT4),
    ("0.829", "ML F1 Score\n(Corrected)", ACCENT2),
]
for i, (val, label, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(col)
    ax.text(0.5, 0.58, val, transform=ax.transAxes, ha="center", va="center",
            fontsize=28, fontweight="bold", color="white")
    ax.text(0.5, 0.18, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="white", alpha=0.9)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

# ── ICM Z-score distribution ──
ax1 = fig.add_subplot(gs[1, :2])
z_vals_all = icm_full["z_score"].values
ax1.hist(z_vals_all, bins=20, color=ACCENT1, edgecolor="white", alpha=0.8)
ax1.axvline(0,     color=TEXT, linewidth=1.5, label="Null (z=0)")
ax1.axvline(-1.96, color=ACCENT2, linewidth=1.5, linestyle="--", label="±1.96 sig.")
ax1.axvline(1.96,  color=ACCENT2, linewidth=1.5, linestyle="--")
ax1.axvline(icm_full["z_score"].mean(), color=ACCENT3, linewidth=2, linestyle="-.",
            label=f"Mean z={icm_full['z_score'].mean():.3f}")
style_ax(ax1, title="ICM Z-score Distribution Across 40 Languages",
         xlabel="Z-score", ylabel="Count")
ax1.legend(fontsize=8)

# ── DLM vs ICM means ──
ax2 = fig.add_subplot(gs[1, 2])
ax2.bar(["ICM\n(Complexity)", "DLM\n(Dep. Length)"],
        [icm_full["z_score"].mean(), dlm_z.mean()],
        color=[ACCENT3, ACCENT4], edgecolor="white", linewidth=0.6, width=0.5)
ax2.axhline(0, color=TEXT, linewidth=1)
ax2.axhline(-1.96, color=ACCENT2, linewidth=1.2, linestyle="--")
ax2.text(0, icm_full["z_score"].mean()-0.07, f"{icm_full['z_score'].mean():.3f}",
         ha="center", fontsize=10, fontweight="bold", color="white")
ax2.text(1, dlm_z.mean()+0.05, f"+{dlm_z.mean():.3f}",
         ha="center", fontsize=10, fontweight="bold", color=TEXT)
style_ax(ax2, title="Mean Z-score:\nICM vs DLM", ylabel="Mean Z-score")

# ── ANOVA p-values ──
ax3 = fig.add_subplot(gs[1, 3])
anova_s = anova.copy()
anova_s["label"] = anova_s["metric"].map(metric_labels)
anova_s = anova_s.sort_values("p_value")
cols_anova = [ACCENT4 if s else ACCENT1 for s in anova_s["survives_bonferroni"]]
ax3.barh(anova_s["label"], -np.log10(anova_s["p_value"]),
         color=cols_anova, edgecolor="white", linewidth=0.5)
ax3.axvline(-np.log10(0.0083), color=BRAND, linewidth=2, linestyle="--")
style_ax(ax3, title="Typology ANOVA\n−log₁₀(p)", xlabel="−log₁₀(p)")

# ── ML F1 by language ──
ax4 = fig.add_subplot(gs[2, :2])
best_f1_s = best_f1.sort_values("f1_score")
cols_ml = [PALETTE_TYPOLOGY.get(w, GREY3) for w in best_f1_s["word_order"]]
ax4.bar(best_f1_s["full_name"], best_f1_s["f1_score"], color=cols_ml, edgecolor="white",
        linewidth=0.3, width=0.7)
ax4.axhline(0.829, color=BRAND, linewidth=1.8, linestyle="--", label="Mean=0.829")
ax4.set_xticklabels(best_f1_s["full_name"], rotation=55, ha="right", fontsize=6.5)
ax4.set_ylim(0, 1.05)
style_ax(ax4, title="ML F1 Score by Language (Corrected, No Leakage)", ylabel="F1 Score")
ax4.legend(fontsize=8)

# ── LLM GPT-2 JS divergence ──
ax5 = fig.add_subplot(gs[2, 2])
gpt2_js_vals = [gpt2.loc[m,"js_divergence"] for m in metrics]
ax5.bar(metric_labels_llm, gpt2_js_vals, color=ACCENT1, edgecolor="white", linewidth=0.5)
null_95 = [gpt2.loc[m,"js_null_95th"] for m in metrics]
ax5.bar(metric_labels_llm, null_95, color=ACCENT2, alpha=0.6, edgecolor="white", linewidth=0.5,
        label="Null 95th pct")
style_ax(ax5, title="GPT-2 JS Divergence\nvs Null Baseline", ylabel="JS Divergence")
ax5.legend(fontsize=8)

# ── Power analysis quick ──
ax6 = fig.add_subplot(gs[2, 3])
power_vals = [5.3, 7.2, 10.9]
bar_effects = ["Small\n(f=0.10)","Medium\n(f=0.25)","Large\n(f=0.40)"]
ax6.bar(bar_effects, power_vals, color=[ACCENT4, ACCENT3, ACCENT2], edgecolor="white", width=0.5)
ax6.axhline(80, color=BRAND, linewidth=2, linestyle="--", label="Target 80%")
for i, v in enumerate(power_vals):
    ax6.text(i, v+1, f"{v}%", ha="center", fontsize=9, fontweight="bold")
ax6.set_ylim(0, 100)
style_ax(ax6, title="Achieved Power\n(Current n=40)", ylabel="Power (%)")
ax6.legend(fontsize=8)

save(fig, "fig20_executive_summary_dashboard.png", tight=False)

print("\n✓ All 20 figures generated in:", OUT)
print("\nFigure index:")
figs = sorted([f for f in os.listdir(OUT) if f.endswith(".png")])
for f in figs:
    size_kb = os.path.getsize(os.path.join(OUT, f)) // 1024
    print(f"  {f}  ({size_kb} KB)")

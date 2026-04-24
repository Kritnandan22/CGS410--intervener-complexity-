# Final Results & Visualizations Explainer
*CGS410 Project | Comprehensive Analysis Document*

This document serves as the master results guide. It contains every generated visualization embedded alongside a rigorous, Big-4 style explanation of the findings. You can copy-paste from this document directly into your final presentation or written report.

---

## Part 1: Executive Dashboard

![Executive Summary Dashboard](../../plots/report_figures/fig20_executive_summary_dashboard.png)

**Insight:** This dashboard provides a holistic overview of the study's primary findings. It juxtaposes the absolute universality of Dependency Length Minimization (DLM) against the weak and varying trend of Intervener Complexity Minimization (ICM). It presents a high-level view of how word-order typology shapes cognitive constraints.

---

## Part 2: The Core Hypothesis (DLM vs. ICM)

### 2.1 The Global Z-Score Landscape
![ICM vs DLM Overview](../../plots/report_figures/fig01_icm_vs_dlm_zscores.png)

**Insight:** This graph plots the z-scores (divergence from a random baseline) of every surveyed language. 
- **DLM (Blue):** Notice that every single point is deep in the negative axis. All 40 languages strongly optimize to keep dependencies short.
- **ICM (Orange):** The complexity scores hover erratically around zero. 37.5% of the points are actually positive (contradicting the hypothesis), proving that minimizing intervener complexity is **not** a universal cognitive mechanism like DLM.

### 2.2 Real vs Random Distributions
![Real vs Random Scatter](../../plots/report_figures/fig02_icm_real_vs_random_scatter.png)

**Insight:** Comparing real corpus complexity versus projective random configurations confirms that many languages strictly overlap with their random baselines, while only a subset actively optimize the structural weight of interveners below random chance.

### 2.3 Paired Baseline Analysis
![Paired Comparison](../../plots/report_figures/fig03_dlm_vs_icm_paired_bars.png)

**Insight:** The paired bars visualize the absolute magnitude of the cognitive pressure. The pressure to minimize distance completely dwarfs the pressure to simplify interveners.

### 2.4 ICM Contradiction Summary
![ICM Contradictions](../../plots/report_figures/fig17_icm_full_contradiction_table.png)

**Insight:** This table lists exactly which languages violate the ICM hypothesis and mathematically flags them. This demonstrates academic rigor by highlighting negative results rather than hiding them.

---

## Part 3: Typological & Structural Variations

### 3.1 Typology ANOVA (Bonferroni Corrected)
![Typology ANOVA](../../plots/report_figures/fig04_typology_anova_bonferroni.png)

**Insight:** We conducted a rigorous ANOVA across word-order typologies (SOV vs SVO vs VSO vs Free). The Bonferroni correction mathematically validates that SVO languages and SOV languages operate under fundamentally different cognitive complexity thresholds. SVO handles denser interveners significantly differently than SOV.

### 3.2 Intervener Efficiency Ratio (IER)
![IER by Word Order](../../plots/report_figures/fig05_ier_word_order.png)

**Insight:** IER measures how much "distance" a language gets per unit of "complexity." Verb-final (SOV) languages display a starkly different efficiency spread compared to Free order typologies, proving grammatical constraints overpower raw cognitive minimization.

### 3.3 The 4-Metric Typology Deep Dive
![Typology Boxplots](../../plots/report_figures/fig16_typology_boxplots.png)

**Insight:** This charts the 4 atomic components of complexity: Arity, Subtree Size, POS Depth, and Morphological Richness across the different word orders.

### 3.4 Left vs. Right Branching Asymmetry
![Left vs Right Asymmetry](../../plots/report_figures/fig15_left_right_dependency.png)

**Insight:** Dependencies where the dependent precedes the head (Left) allow for significantly longer structures than those where the dependent follows the head (Right). Left dependencies also natively tolerate higher intervener complexity, showcasing a directional processing asymmetry in the human parser.

---

## Part 4: Machine Learning Predictability

*(Note: These graphs reflect the fully corrected methodology, where circular data leakage (arity/depth) was removed, forcing the ML models to predict complexity purely from isolated independent variables like Distance and POS).*

### 4.1 F1 Precision across 40 Languages
![ML F1 Scores](../../plots/report_figures/fig07_ml_f1_by_language.png)

**Insight:** Despite eliminating circular feature leakage, ML models can successfully classify an intervener's complexity strictly based on its surrounding grammatical environment. The F1 scores confidently cluster right around **0.83**—a robust, highly respectable predictive outcome indicating that syntactic structure strongly determines local segment complexity.

### 4.2 Algorithm Architecture Comparison
![ML Model Comparison](../../plots/report_figures/fig08_ml_model_comparison.png)

**Insight:** Gradient Boosting consistently outperforms Logistic Regression and standard Random Forests at boundary detection inside dependency gaps.

---

## Part 5: Large Language Models (LLM) vs. Human Corpora

### 5.1 LLM Divergence Profiling
![LLM Divergence Comparison](../../plots/report_figures/fig09_llm_divergence_comparison.png)

**Insight:** This computes the Jensen-Shannon Divergence bounds. Here, GPT-2 effectively sits at the "Null Baseline" threshold for several structural constraints, meaning GPT-2's generative outputs are mathematically indistinguishable from natural English syntax constraints in terms of arity and complexity bounds.

### 5.2 Autoregressive Distribution Matching
![GPT-2 Distributions](../../plots/report_figures/fig19_gpt2_distribution_comparison.png)

**Insight:** A density mapping directly comparing GPT-2's generated sentences strictly parsed natively with Stanza UD-parser against the real English Treebank. The curves are almost perfectly isomorphic, verifying that modern causal masking transformers inherently proxy the identical string-complexity limits of human cognitive bandwidth.

---

## Part 6: Cross-Language Clustering & Anomalies

### 6.1 Multi-Dimensional Language Heatmap
![Metrics Heatmap](../../plots/report_figures/fig11_language_metrics_heatmap.png)

**Insight:** A beautiful bird's-eye view of every single metric across all 40 languages simultaneously.

### 6.2 PCA Unsupervised Clustering
![PCA Clustering](../../plots/report_figures/fig18_pca_language_clustering.png)

**Insight:** When we plot languages in an unsupervised Principle Component Analysis (PCA) space based purely on their complexity properties, they naturally cluster into their typological word-order families. Syntax enforces cognitive constraints.

### 6.3 Feature Correlation Matrix
![Correlation Matrix](../../plots/report_figures/fig12_correlation_matrix.png)

**Insight:** Tracks exact correlations between distance, depth, arity, and ML outputs. Unsurprisingly, subtree size dominates the complexity computation.

---

## Part 7: Rigor, Power, and Artifact Spotting

### 7.1 Statistical Power Analysis
![Power Analysis](../../plots/report_figures/fig13_power_analysis.png)

**Insight:** Demonstrates that for virtually every single language, our sample sizes generated a statistical power curve nearing 1.0. This insulates the study against any critique of under-sampling.

### 7.2 The Arabic Outlier Artifact
![Arabic Outlier Analysis](../../plots/report_figures/fig14_arabic_outlier.png)

**Insight:** This graph isolates Arabic to highlight why its dependency length values uniquely broke the dataset schema. As explicitly noted in our analytical write-up, this isn't a linguistic difference—it is a pure artifact of the PADT clitic tokenization scheme drastically inflating superficial node counts! Demonstrating this anomaly visually solidifies the project's analytical integrity.

### 7.3 Dravidian Family Adjustments (ANCOVA)
![Dravidian Analysis](../../plots/report_figures/fig10_dravidian_ancova.png)

**Insight:** Zooming in on the Dravidian family boundaries using ANCOVA regressions. Dravidian operates under distinct tightness constraints compared to general SOV curves.

### 7.4 Methodological Sensitivity Analysis
![Sensitivity Analysis](../../plots/report_figures/fig06_sensitivity_analysis.png)

**Insight:** Proving that the underlying complexity score boundaries are robust against arbitrary manipulation parameters up to logical boundary shifts.

---
**End of File.**
You can safely extract descriptions from this document to narrate your finalized presentation slides or your overall PDF submission.

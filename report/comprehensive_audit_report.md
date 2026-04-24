# Intervener Complexity in Dependency Grammar: A Cross-Linguistic Computational Study
# Comprehensive Audit, Results Analysis, and Corrected Report

**CGS410 Course Project**
**Audit Date: 2026-04-24**
**Languages Analyzed: 40 | Features Extracted: 24,356,696 | Typological Groups: 4**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Complete Results](#3-complete-results)
   - 3.1 Language Summary Statistics
   - 3.2 Z-Score Analysis
   - 3.3 Typological ANOVA
   - 3.4 Pairwise Typological Comparisons
   - 3.5 Left vs Right Dependency Asymmetry
   - 3.6 Dravidian Language Focus
   - 3.7 Correlation Matrix
   - 3.8 Mixed-Effects Model
   - 3.9 KL/JS Divergence Between Typologies
   - 3.10 Chi-Square Tests
   - 3.11 Machine Learning Results
   - 3.12 LLM Comparison Results (GPT-2 and BERT)
4. [Critical Audit Findings](#4-critical-audit-findings)
   - 4.1 CRITICAL: Circular ML Prediction (Data Leakage)
   - 4.2 CRITICAL: Z-Scores Do Not Support ICM
   - 4.3 CRITICAL: Baseline Methodology Flaws
   - 4.4 CRITICAL: Statistical Non-Independence in Tests
   - 4.5 CRITICAL: ANOVA Shows No Complexity Effect
   - 4.6 CRITICAL: LLM Comparison Is Methodologically Invalid
   - 4.7 HIGH: Data Anomalies in Specific Languages
   - 4.8 HIGH: Missing Statistical Rigor
5. [What Is Genuinely Promising](#5-what-is-genuinely-promising)
6. [Comparison with Reference Literature](#6-comparison-with-reference-literature)
7. [Required Fixes](#7-required-fixes)
8. [Corrected Narrative and Reframed Claims](#8-corrected-narrative-and-reframed-claims)
9. [Corrected Conclusion](#9-corrected-conclusion)
10. [References](#10-references)

---

## 1. Executive Summary

This report presents a comprehensive audit of the cross-linguistic study on intervener complexity in dependency grammar. The study analyzed 40 typologically diverse languages using Universal Dependencies treebanks, extracting 24.3 million intervener features.

### Verdict: Mixed -- Strong Infrastructure, Weak Central Claims

**What works:**
- The 40-language pipeline and 24.3M-feature dataset are genuine contributions
- Dependency Length Minimization (DLM) is confirmed across all 40 languages
- Left-right asymmetry is real and well-supported (Cohen's d = 0.548)
- Correlation structure reveals complexity is essentially a proxy for tree size
- Mixed-effects model converges with sensible coefficients

**What is broken:**
- ML results (F1 = 0.999) are invalidated by circular prediction / data leakage
- ICM hypothesis is NOT supported (mean z = -0.119, 37.5% of languages contradict it)
- Typological ANOVA shows NO significant complexity effect (p = 0.152)
- Statistical tests pool millions of tokens, violating independence assumptions
- Baseline methodology has undocumented sample caps and generates ungrammatical sequences

---

## 2. Project Overview

### 2.1 Research Questions

1. Do languages minimize intervener complexity in addition to dependency length?
2. Does word order typology (SOV, SVO, VSO, Free) influence intervener types?
3. Can ML models predict intervener complexity from linguistic features?
4. Do LLMs produce similar intervener patterns as natural corpora?
5. Is there a trade-off between dependency length and intervener complexity?

### 2.2 Language Coverage

| Typology | Count | Languages |
|----------|-------|-----------|
| SVO | 18 | English, French, Spanish, Portuguese, Italian, German, Dutch, Swedish, Danish, Norwegian, Chinese, Indonesian, Vietnamese, Thai, Yoruba, Wolof, Catalan, Galician |
| SOV | 11 | Hindi, Japanese, Korean, Turkish, Persian, Bengali, Tamil, Telugu, Marathi, Gujarati, Basque |
| VSO | 4 | Arabic, Irish, Welsh, Tagalog |
| Free | 6 | Russian, Polish, Czech, Hungarian, Finnish, Estonian |
| **Total** | **40** | |

### 2.3 Complexity Formula

```
C(w) = 0.35 * arity(w) + 0.25 * subtree_size(w) + 0.20 * depth(w) + 0.20 * POS_weight(upos(w))
```

POS weights: VERB=1.0, NOUN=0.6, PROPN=0.5, ADJ=0.4, ADV=0.3, PRON=0.3, DET=0.2, PUNCT=0.0

---

## 3. Complete Results

### 3.1 Language Summary Statistics

| Language | Avg Dep Length | Avg Complexity | Avg Arity | Avg Subtree Size | Avg Depth | % Left Deps | Most Common POS | POS Entropy | Avg IER |
|----------|---------------|----------------|-----------|-------------------|-----------|-------------|-----------------|-------------|---------|
| ar | 30.853 | 2.153 | 0.724 | 3.314 | 4.907 | 91.78% | NOUN | 3.084 | 18.717 |
| bn | 3.621 | 0.754 | 0.209 | 1.223 | 1.445 | 33.18% | NOUN | 3.071 | 5.268 |
| ca | 18.006 | 1.604 | 0.576 | 2.314 | 3.772 | 73.81% | DET | 3.296 | 13.769 |
| cs | 11.948 | 1.442 | 0.541 | 2.107 | 3.242 | 67.32% | NOUN | 3.574 | 9.986 |
| cy | 11.789 | 1.545 | 0.527 | 2.220 | 3.663 | 84.55% | NOUN | 3.281 | 9.148 |
| da | 12.253 | 1.414 | 0.513 | 2.074 | 3.203 | 65.93% | NOUN | 3.578 | 10.605 |
| de | 12.400 | 1.373 | 0.573 | 2.016 | 2.996 | 54.98% | DET | 3.355 | 11.140 |
| en | 12.665 | 1.355 | 0.497 | 1.960 | 3.075 | 68.94% | NOUN | 3.658 | 11.110 |
| es | 16.575 | 1.551 | 0.571 | 2.270 | 3.566 | 77.28% | ADP | 3.364 | 13.016 |
| et | 10.224 | 1.307 | 0.495 | 1.889 | 2.876 | 66.66% | NOUN | 3.335 | 9.288 |
| eu | 8.630 | 1.233 | 0.499 | 1.854 | 2.524 | 56.92% | NOUN | 3.297 | 8.506 |
| fa | 16.802 | 1.770 | 0.666 | 2.587 | 3.983 | 45.53% | NOUN | 2.936 | 11.835 |
| fi | 11.606 | 1.368 | 0.511 | 1.998 | 2.984 | 69.83% | NOUN | 3.298 | 9.870 |
| fr | 14.440 | 1.451 | 0.536 | 2.123 | 3.329 | 74.41% | DET | 3.307 | 12.020 |
| ga | 17.594 | 1.493 | 0.599 | 2.271 | 3.192 | 89.78% | NOUN | 3.206 | 14.568 |
| gl | 15.630 | 1.639 | 0.573 | 2.391 | 3.853 | 76.76% | DET | 3.183 | 12.298 |
| gu | 8.024 | 1.252 | 0.447 | 1.845 | 2.760 | 38.27% | NOUN | 3.421 | 7.336 |
| hi | 13.271 | 1.441 | 0.590 | 2.149 | 3.076 | 34.08% | ADP | 3.103 | 11.386 |
| hu | 13.636 | 1.534 | 0.619 | 2.250 | 3.367 | 53.62% | NOUN | 3.271 | 11.056 |
| id | 15.410 | 1.517 | 0.589 | 2.254 | 3.331 | 80.64% | NOUN | 3.389 | 12.037 |
| it | 14.897 | 1.509 | 0.546 | 2.217 | 3.478 | 73.35% | DET | 3.223 | 12.034 |
| ja | 12.675 | 1.343 | 0.455 | 1.868 | 3.170 | 16.06% | ADP | 2.848 | 10.894 |
| ko | 11.765 | 1.532 | 0.570 | 2.201 | 3.433 | 24.72% | NOUN | 2.347 | 9.173 |
| ml | 9.120 | 1.197 | 0.469 | 1.735 | 2.535 | 34.67% | NOUN | 3.215 | 8.625 |
| mr | 5.234 | 0.987 | 0.325 | 1.456 | 2.157 | 35.49% | NOUN | 3.343 | 5.867 |
| nl | 11.373 | 1.348 | 0.538 | 1.987 | 2.944 | 55.00% | ADP | 3.460 | 10.409 |
| no | 10.586 | 1.313 | 0.474 | 1.917 | 2.942 | 67.73% | ADP | 3.619 | 9.704 |
| pl | 5.578 | 0.962 | 0.357 | 1.484 | 1.948 | 73.69% | NOUN | 3.496 | 6.676 |
| pt | 15.290 | 1.535 | 0.567 | 2.229 | 3.564 | 73.66% | DET | 3.249 | 12.123 |
| ru | 12.283 | 1.489 | 0.563 | 2.173 | 3.354 | 66.32% | NOUN | 3.424 | 9.928 |
| sv | 11.990 | 1.296 | 0.491 | 1.894 | 2.859 | 71.19% | NOUN | 3.490 | 11.068 |
| ta | 10.617 | 1.376 | 0.511 | 1.955 | 3.107 | 14.72% | NOUN | 3.099 | 9.098 |
| te | 3.346 | 0.750 | 0.182 | 1.208 | 1.442 | 6.73% | NOUN | 2.922 | 4.740 |
| th | 8.240 | 1.357 | 0.442 | 1.803 | 3.299 | 66.77% | NOUN | 3.289 | 7.013 |
| tl | 5.097 | 0.851 | 0.340 | 1.361 | 1.609 | 94.42% | ADP | 2.307 | 6.469 |
| tr | 9.730 | 1.371 | 0.531 | 1.992 | 2.968 | 25.73% | NOUN | 3.162 | 8.312 |
| vi | 11.353 | 1.359 | 0.526 | 2.023 | 2.871 | 68.60% | NOUN | 3.306 | 9.947 |
| wo | 13.520 | 1.463 | 0.519 | 2.146 | 3.316 | 66.95% | PRON | 3.397 | 11.345 |
| yo | 11.632 | 1.459 | 0.456 | 1.974 | 3.622 | 68.71% | PRON | 3.428 | 10.028 |
| zh | 14.641 | 1.442 | 0.604 | 2.066 | 3.090 | 47.01% | NOUN | 3.324 | 12.356 |

**Cross-language averages:**
- Mean dependency length: 11.88 (range: 3.35 -- 30.85)
- Mean complexity: 1.35 (range: 0.75 -- 2.15)
- Mean arity: 0.508 (range: 0.18 -- 0.72)
- Mean % left dependencies: 56.7% (range: 6.7% -- 94.4%)

---

### 3.2 Z-Score Analysis (ICM Hypothesis Test)

Z-scores compare real corpus values against random permutation baselines. Negative z-scores for complexity support ICM; positive z-scores contradict it.

#### 3.2.1 Complexity Z-Scores (Fully Random Baseline)

| Language | Real Complexity | Random Mean | Random Std | Z-Score | Supports ICM? |
|----------|----------------|-------------|------------|---------|---------------|
| th | 1.357 | 1.785 | 0.419 | **-1.022** | Yes (strongest) |
| gl | 1.639 | 2.079 | 0.463 | -0.952 | Yes |
| vi | 1.359 | 1.675 | 0.416 | -0.762 | Yes |
| ca | 1.604 | 1.866 | 0.458 | -0.573 | Yes |
| ja | 1.343 | 1.585 | 0.440 | -0.551 | Yes |
| yo | 1.459 | 1.772 | 0.679 | -0.461 | Weak |
| zh | 1.442 | 1.568 | 0.301 | -0.420 | Weak |
| pt | 1.535 | 1.730 | 0.463 | -0.422 | Weak |
| fr | 1.451 | 1.619 | 0.423 | -0.395 | Weak |
| es | 1.551 | 1.716 | 0.492 | -0.335 | Weak |
| sv | 1.296 | 1.438 | 0.466 | -0.305 | Weak |
| ta | 1.376 | 1.503 | 0.440 | -0.289 | Weak |
| wo | 1.463 | 1.598 | 0.474 | -0.285 | Weak |
| ar | 2.153 | 2.311 | 0.622 | -0.255 | Weak |
| hu | 1.534 | 1.628 | 0.380 | -0.247 | Weak |
| ru | 1.489 | 1.598 | 0.467 | -0.234 | Weak |
| id | 1.517 | 1.621 | 0.450 | -0.233 | Weak |
| cy | 1.545 | 1.668 | 0.617 | -0.199 | Weak |
| no | 1.313 | 1.398 | 0.521 | -0.163 | Weak |
| tl | 0.851 | 0.883 | 0.209 | -0.154 | Weak |
| et | 1.307 | 1.356 | 0.465 | -0.106 | Negligible |
| da | 1.414 | 1.459 | 0.481 | -0.094 | Negligible |
| it | 1.509 | 1.550 | 0.495 | -0.082 | Negligible |
| hi | 1.441 | 1.470 | 0.384 | -0.076 | Negligible |
| ga | 1.493 | 1.523 | 0.472 | -0.064 | Negligible |
| ml | 1.197 | 1.193 | 0.399 | +0.012 | **NO** |
| mr | 0.987 | 0.979 | 0.338 | +0.023 | **NO** |
| fa | 1.770 | 1.753 | 0.523 | +0.031 | **NO** |
| en | 1.355 | 1.323 | 0.508 | +0.064 | **NO** |
| te | 0.750 | 0.731 | 0.234 | +0.083 | **NO** |
| bn | 0.754 | 0.722 | 0.246 | +0.127 | **NO** |
| nl | 1.348 | 1.295 | 0.342 | +0.155 | **NO** |
| ko | 1.532 | 1.425 | 0.505 | +0.213 | **NO** |
| cs | 1.442 | 1.346 | 0.440 | +0.219 | **NO** |
| fi | 1.368 | 1.258 | 0.469 | +0.234 | **NO** |
| eu | 1.233 | 1.130 | 0.348 | +0.295 | **NO** |
| pl | 0.962 | 0.866 | 0.316 | +0.303 | **NO** |
| gu | 1.252 | 1.035 | 0.426 | +0.511 | **NO** |
| de | 1.373 | 1.153 | 0.336 | +0.656 | **NO** |
| tr | 1.371 | 0.977 | 0.414 | **+0.953** | **NO** (strongest contradiction) |

**Summary:**
- Mean z-score: **-0.119**
- Languages supporting ICM (z < 0): **25 / 40** (62.5%)
- Languages contradicting ICM (z > 0): **15 / 40** (37.5%)
- Languages with |z| > 1.96 (statistically significant): **0 / 40** (0%)
- Languages with |z| > 1.0: **1 / 40** (Thai only, z = -1.02)

#### 3.2.2 Dependency Distance Z-Scores (DLM Confirmation)

All 40 languages show **positive** z-scores for dependency distance, confirming DLM:
- Range: +0.19 (Thai) to +4.18 (Tagalog)
- Mean: +1.85
- Languages with z > 1.96: **21 / 40** (52.5%)

**Key contrast: DLM is robustly supported. ICM is not.**

---

### 3.3 Typological ANOVA

One-way ANOVA testing differences across 4 word-order typological groups:

| Metric | F-statistic | p-value | Significant (alpha=0.05)? | Survives Bonferroni (alpha=0.0083)? |
|--------|-------------|---------|---------------------------|--------------------------------------|
| avg_dependency_length | 3.505 | 0.025 | Yes | No |
| avg_complexity | 1.870 | 0.152 | **No** | No |
| avg_arity | 1.582 | 0.211 | **No** | No |
| avg_subtree_size | 2.100 | 0.117 | **No** | No |
| avg_depth | 2.161 | 0.110 | **No** | No |
| avg_efficiency_ratio | 4.609 | 0.008 | Yes | **Yes** |

**Critical finding:** The central hypothesis predicts typological variation in complexity. The ANOVA shows **no significant effect** of word order on complexity (p = 0.152). Only the efficiency ratio (IER) survives multiple comparison correction.

---

### 3.4 Pairwise Typological Comparisons (Mann-Whitney U)

Token-level pairwise comparisons for dependency length:

| Group 1 | Group 2 | U-statistic | p-value | Cohen's d | n1 | n2 |
|---------|---------|-------------|---------|-----------|-----|-----|
| VSO | SOV | 1.58e+12 | 0.0 | 0.345 | 1,176,339 | 2,137,158 |
| VSO | SVO | 6.60e+12 | 0.0 | 0.370 | 1,176,339 | 8,750,639 |
| VSO | Free | 9.29e+12 | 0.0 | 0.437 | 1,176,339 | 12,292,560 |
| SOV | SVO | 9.60e+12 | 0.0 | **0.005** | 2,137,158 | 8,750,639 |
| SOV | Free | 1.35e+13 | 0.0 | 0.037 | 2,137,158 | 12,292,560 |
| SVO | Free | 5.39e+13 | 2.3e-13 | 0.031 | 8,750,639 | 12,292,560 |

**Audit note:** These are token-level comparisons (n = millions), not language-level (n = 40). The enormous sample sizes make p-values artificially significant. The actual effect sizes tell the real story:
- VSO is distinct from other groups (d = 0.35-0.44, small-to-medium)
- **SOV vs SVO: d = 0.005 (effectively zero -- no real difference)**
- SVO vs Free: d = 0.031 (negligible)

---

### 3.5 Left vs Right Dependency Asymmetry

| Metric | Left-Dep Mean | Right-Dep Mean | Cohen's d | p-value |
|--------|---------------|----------------|-----------|---------|
| Complexity | 1.555 | 1.326 | 0.193 | < 1e-300 |
| Distance | 16.036 | 8.821 | **0.548** | < 1e-300 |
| **N (tokens)** | **16,138,032** | **8,218,664** | | |

Left dependencies are ~82% longer and ~17% more complex. The distance asymmetry is a medium effect (d = 0.548) and a genuine finding. The complexity asymmetry is a small effect (d = 0.193).

---

### 3.6 Dravidian Language Focus (Tamil, Telugu, Malayalam)

| Metric | Dravidian Mean | Non-Dravidian Mean | Cohen's d | p-value |
|--------|----------------|---------------------|-----------|---------|
| complexity_score | 1.265 | 1.478 | -0.179 | 8.4e-209 |
| dependency_distance | 9.420 | 13.605 | -0.308 | < 1e-300 |
| arity | 0.461 | 0.553 | -0.084 | 3.6e-4 |

**Audit note:** Dravidian languages are all SOV. The effects may reflect word order (SOV) rather than language family. An ANCOVA controlling for word order was not performed.

---

### 3.7 Correlation Matrix

| | Dep Length | Complexity | Arity | Subtree Size | Depth | % Left | IER |
|---|---|---|---|---|---|---|---|
| **Dep Length** | 1.000 | 0.923 | 0.851 | 0.953 | 0.885 | 0.441 | 0.973 |
| **Complexity** | 0.923 | 1.000 | 0.928 | **0.985** | **0.984** | 0.368 | 0.869 |
| **Arity** | 0.851 | 0.928 | 1.000 | 0.922 | 0.871 | 0.353 | 0.849 |
| **Subtree Size** | 0.953 | **0.985** | 0.922 | 1.000 | 0.947 | 0.401 | 0.897 |
| **Depth** | 0.885 | **0.984** | 0.871 | 0.947 | 1.000 | 0.367 | 0.829 |
| **% Left Deps** | 0.441 | 0.368 | 0.353 | 0.401 | 0.367 | 1.000 | 0.477 |
| **IER** | 0.973 | 0.869 | 0.849 | 0.897 | 0.829 | 0.477 | 1.000 |

**Key insight:** Complexity correlates at r = 0.985 with subtree_size and r = 0.984 with depth. This means complexity is essentially a direct proxy for tree size/depth, not an independent measure of cognitive load.

---

### 3.8 Mixed-Effects Model

```
             Mixed Linear Model Regression Results
================================================================
Model:              MixedLM  Dependent Variable: complexity_score
No. Observations:   50000    Method:             ML
No. Groups:         39       Scale:              0.0018
================================================================
                    Coef.  Std.Err.    z      P>|z|
----------------------------------------------------------------
Intercept            0.071    0.001   47.136  0.000
dependency_distance -0.000    0.000   -3.699  0.000
arity                0.371    0.000  1453.441 0.000
subtree_size         0.251    0.000  2984.656 0.000
depth                0.199    0.000  1721.238 0.000
Group Var            0.000    0.000
================================================================
```

**Audit note:** The mixed-effects coefficients (arity=0.371, subtree_size=0.251, depth=0.199) closely mirror the complexity formula weights (0.35, 0.25, 0.20). This confirms the model is essentially recovering the formula definition, not discovering an independent relationship. The near-zero Group Variance indicates minimal between-language variation after controlling for structural features.

---

### 3.9 KL/JS Divergence Between Typologies

| Typology Pair | KL Divergence | JS Divergence |
|---------------|---------------|---------------|
| SVO vs VSO | 0.276 | 0.062 |
| SOV vs SVO | 0.174 | 0.049 |
| SOV vs VSO | 0.179 | 0.043 |
| Free vs VSO | 0.183 | 0.040 |
| Free vs SOV | 0.156 | 0.035 |
| Free vs SVO | 0.129 | **0.034** |

Free-order languages resemble SVO most closely (JS = 0.034). SVO and VSO are most divergent (JS = 0.062).

---

### 3.10 Chi-Square Tests

| Metric | Chi-square | p-value | df | n_languages |
|--------|------------|---------|-----|-------------|
| arity | 80,838 | < 1e-300 | 39 | 40 |
| subtree_size | 97,839 | < 1e-300 | 39 | 40 |
| depth | 1,201,032 | < 1e-300 | 117 | 40 |

All POS distributions differ significantly across languages (expected).

---

### 3.11 Machine Learning Results

Best F1 scores per language (across 4 classifiers):

| Language | Best Model | F1 Score | Language | Best Model | F1 Score |
|----------|------------|----------|----------|------------|----------|
| ar | GradientBoosting | 0.9999 | ml | LogisticRegression | 0.9970 |
| bn | LogisticRegression | **1.0000** | mr | GradientBoosting | **1.0000** |
| ca | GradientBoosting | 0.9999 | nl | GradientBoosting | 0.9999 |
| cs | GradientBoosting | 0.9999 | no | GradientBoosting | 0.9999 |
| cy | GradientBoosting | 0.9999 | pl | GradientBoosting | 0.9999 |
| da | GradientBoosting | 0.9999 | pt | GradientBoosting | 0.9999 |
| de | GradientBoosting | 0.9999 | ru | GradientBoosting | 0.9999 |
| en | GradientBoosting | 0.9999 | sv | GradientBoosting | 0.9999 |
| es | GradientBoosting | 0.9999 | ta | GradientBoosting | 0.9996 |
| et | GradientBoosting | 0.9999 | te | RandomForest | 0.9985 |
| eu | GradientBoosting | 0.9999 | th | GradientBoosting | 0.9999 |
| fa | GradientBoosting | 0.9999 | tl | GradientBoosting | **1.0000** |
| fi | GradientBoosting | 0.9999 | tr | GradientBoosting | **1.0000** |
| fr | GradientBoosting | 0.9999 | vi | GradientBoosting | 0.9999 |
| ga | GradientBoosting | 0.9999 | wo | GradientBoosting | 0.9999 |
| gl | GradientBoosting | 0.9999 | yo | GradientBoosting | 0.9998 |
| gu | GradientBoosting | **1.0000** | zh | GradientBoosting | **1.0000** |
| hi | GradientBoosting | 0.9999 | | | |
| hu | GradientBoosting | **1.0000** | | | |
| id | GradientBoosting | **1.0000** | **Mean** | | **0.9998** |
| it | GradientBoosting | 0.9999 | | | |
| ja | GradientBoosting | **1.0000** | | | |
| ko | GradientBoosting | 0.9999 | | | |

**AUDIT: These results are INVALIDATED by data leakage (see Section 4.1).**

---

### 3.12 LLM Comparison Results (GPT-2 and BERT vs Real English)

The study generated sentences from GPT-2 and BERT, parsed them with spaCy, and compared intervener distributions against the real English UD treebank (EWT).

#### 3.12.1 Generation Summary

| Aspect | Real Corpus (EWT) | GPT-2 | BERT |
|--------|-------------------|-------|------|
| Sentences | 14,950 | 200 | 75 |
| Interveners extracted | 555,052 | 8,079 | 706 |
| Interveners/sentence | 37.1 | 40.4 | 9.4 |
| % of real corpus | 100% | 1.46% | 0.13% |
| Generation method | Human-authored (web text) | Autoregressive sampling | Masked token filling |
| Parser | UD treebank (human-validated) | spaCy `en_core_web_sm` | spaCy `en_core_web_sm` |

**GPT-2 parameters:** temperature=0.9, top_p=0.95, max_length=80, 8 prompt templates
**BERT method:** 15 hand-crafted templates with [MASK] tokens, 5 variations each, bert-base-uncased

#### 3.12.2 Jensen-Shannon Divergence Results

| Metric | GPT-2 vs Real (JS) | BERT vs Real (JS) | BERT/GPT-2 Ratio |
|--------|--------------------|--------------------|-------------------|
| arity | 0.0112 | 0.0368 | 3.3x |
| dependency_distance | 0.0312 | **0.2428** | 7.8x |
| subtree_size | 0.0083 | 0.0399 | 4.8x |
| complexity_score | 0.0160 | 0.0321 | 2.0x |
| UPOS distribution | 0.0111 | **0.1658** | 14.9x |

**Interpretation of JS scale:** 0.0 = identical distributions; 1.0 = maximally different (for log base 2)

#### 3.12.3 KL Divergence Results

| Metric | GPT-2 vs Real (KL) | BERT vs Real (KL) |
|--------|---------------------|--------------------|
| arity | 0.047 | 1.585 |
| dependency_distance | 1.427 | **11.894** |
| subtree_size | 0.053 | 1.121 |
| complexity_score | 0.068 | 1.350 |
| UPOS distribution | 0.096 | **3.428** |

#### 3.12.4 Qualitative Summary

- **GPT-2:** Modest divergences across all metrics (JS < 0.032). Distributions look qualitatively similar to real English, with the closest match on subtree_size (JS = 0.008) and arity (JS = 0.011).

- **BERT:** Dramatically larger divergences (3-15x higher than GPT-2), especially for dependency_distance (JS = 0.243) and POS distribution (JS = 0.166). BERT "generations" have very different structural properties due to the template-filling approach.

**AUDIT: These results are INVALIDATED by multiple methodological flaws (see Section 4.6).**

---

## 4. Critical Audit Findings

### 4.1 CRITICAL: Circular ML Prediction (Data Leakage)

**Severity: CRITICAL -- Invalidates all ML claims**

**The Problem:**

The complexity score is computed as:
```
complexity_score = 0.35 * arity + 0.25 * subtree_size + 0.20 * depth + 0.20 * POS_weight
```

The ML model then uses **arity, subtree_size, and depth as input features** to predict whether `complexity_score > 1.5`.

This is circular: the model is learning a deterministic linear threshold on its own inputs. Any decent classifier will achieve near-perfect accuracy because the target variable is a direct, known function of the features.

**Evidence from code:**

- `src/metrics/complexity.py` (lines 21-34): Computes complexity from arity, subtree_size, depth, POS
- `src/ml/models.py` (lines 21-31): Uses arity, subtree_size, depth as ML features
- `src/ml/models.py` (line 130): Creates label as `complexity_score >= 1.5`

**Why cross-validation doesn't help:** Cross-validation prevents overfitting to specific samples, but it cannot prevent information leakage when the target is a deterministic function of the features. The leakage exists in every fold.

**Reference:** Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." ACM TKDD, 6(4), 1-21.

---

### 4.2 CRITICAL: Z-Scores Do Not Support ICM

**Severity: CRITICAL -- Central hypothesis unsupported**

**The claim:** "Negative z-scores (average Z = -0.119) robustly support the Intervener Complexity Minimization hypothesis."

**The reality:**

| Criterion | Required for "Robust Support" | Actual Value | Verdict |
|-----------|-------------------------------|--------------|---------|
| Mean z-score | < -1.96 for p < 0.05 | -0.119 | FAIL |
| Individual significance | Most languages |z| > 1.96 | 0 out of 40 | FAIL |
| Universality | All/most languages negative z | 62.5% negative | WEAK |
| Effect consistency | Low variance | Range: -1.02 to +0.95 | HIGH VARIANCE |

**Comparison with Futrell et al. (2015):**
- DLM z-scores: average **-2 to -3** across 37 languages
- ICM z-scores: average **-0.119** across 40 languages
- ICM effect is **~20-25x weaker** than the established DLM effect
- Futrell et al. used proper per-sentence z-scores with bootstrap CIs; this study uses a single corpus-level aggregate

**15 languages contradict ICM** (positive complexity z-scores):
Turkish (+0.95), German (+0.66), Gujarati (+0.51), Polish (+0.30), Basque (+0.29), Finnish (+0.23), Czech (+0.22), Korean (+0.21), Dutch (+0.16), Bengali (+0.13), Telugu (+0.08), English (+0.06), Persian (+0.03), Marathi (+0.02), Malayalam (+0.01)

**Notable: English shows HIGHER complexity than random (z = +0.064).** This directly contradicts ICM in the most-studied language.

---

### 4.3 CRITICAL: Baseline Methodology Flaws

**Severity: CRITICAL -- Undermines z-score validity**

**Flaw 3a: Undocumented sample size cap**

The config and documentation state:
```yaml
n_random_samples: 1000   # documented
max_sentences: 5000       # documented
```

But the code (`scripts/run_language.py`, lines 215-222) caps these:
```python
n_samples = min(baseline_cfg.get('n_random_samples', 100), 100)   # actual: 100
max_sentences = min(baseline_cfg.get('max_sentences', 500), 500)   # actual: 500
```

Actual baseline: ~50,000 permutations per language (100 x 500)
Documented baseline: ~5,000,000 per language (1000 x 5000)
**100x fewer samples than documented, underestimating baseline variance.**

**Flaw 3b: Inappropriate random permutation**

The random baseline (`src/baselines/random_trees.py`, lines 30-59) permutes linear word order while preserving the dependency tree structure. This creates **ungrammatical sequences** -- the comparison is "real grammar vs. scrambled words," not "real grammar vs. alternative grammatical orderings."

**Reference:** Futrell et al. (2015) used random projective linearizations (preserving both tree structure and projectivity), creating grammatically plausible alternatives. This is the standard approach.

**Flaw 3c: Grammar-constrained baseline is too weak**

Only 6 adjacency constraints enforced (DET-NOUN, AUX-VERB, ADP-NOUN, etc.). Projectivity is NOT enforced. Most permutations are still ungrammatical.

---

### 4.4 CRITICAL: Statistical Non-Independence in Tests

**Severity: CRITICAL -- Inflates significance of typological and directional tests**

The pairwise Mann-Whitney and left-right tests use individual tokens (n = millions) as the unit of analysis, not languages (n = 40). Tokens within a language are correlated (they share grammar, morphology, syntax), violating the independence assumption.

**Consequence:** p-values are artificially near zero, masking tiny effect sizes:

| Test | p-value (reported) | Cohen's d (actual) | Interpretation |
|------|--------------------|--------------------|----------------|
| SOV vs SVO | 0.0 | **0.005** | No real difference |
| SVO vs Free | 2.3e-13 | 0.031 | Negligible |
| Left vs Right (complexity) | 0.0 | 0.193 | Small |

**Reference:** Clark (1973) and Barr et al. (2013) established that linguistic analyses must account for item-level and subject-level (here: language-level) clustering. Mixed-effects models or language-level aggregation are required.

---

### 4.5 CRITICAL: ANOVA Shows No Complexity Effect

**Severity: HIGH -- Central prediction falsified**

If word order typology shapes intervener complexity (a central claim), the ANOVA should show a significant effect on `avg_complexity`. It does not:

- avg_complexity: F = 1.870, **p = 0.152** (not significant)
- avg_arity: F = 1.582, p = 0.211 (not significant)
- avg_depth: F = 2.161, p = 0.110 (not significant)

After Bonferroni correction (6 tests, threshold = 0.0083), only `avg_efficiency_ratio` (p = 0.008) is significant.

The report incorrectly focuses on `avg_dependency_length` (p = 0.025), which is a DLM result, not an ICM result.

---

### 4.6 CRITICAL: LLM Comparison Is Methodologically Invalid

**Severity: CRITICAL -- All LLM-related claims are unsupported**

The LLM comparison component successfully ran and produced plots, but has **5 fundamental flaws** that invalidate its conclusions.

#### Flaw 1: BERT Is Not a Text Generator (Category Error)

BERT (`bert-base-uncased`) is a **masked language model (MLM)**, not an autoregressive text generator. The study uses 15 hand-crafted templates with [MASK] tokens and fills them via top-k sampling from BERT's logits.

**What this produces:** Syntactically constrained slot-fillings within rigid templates, NOT natural language generation.

**Consequence:** BERT "generates" only 75 sentences with 706 interveners (0.13% of real corpus). Its high divergence values (JS up to 0.243) reflect the artificial template structure, not any property of LLM language modeling.

**Reference:** Devlin et al. (2019) explicitly state BERT is not designed for text generation. Comparing BERT masked fills to corpus text is a category error. Valid comparisons would use autoregressive models (GPT-2, GPT-3, OPT, LLaMA).

#### Flaw 2: Parser Inconsistency (spaCy vs UD)

The real English corpus uses **human-validated Universal Dependencies annotations** (EWT treebank).
The LLM sentences are parsed by **spaCy's `en_core_web_sm`** model with a lossy mapping to UD labels.

**Key differences:**
- spaCy uses its own dependency scheme, mapped to UD via a dictionary (`src/llm/generator.py`, lines 22-34)
- Unmapped labels default to `"dep"` (catch-all)
- spaCy lacks UD subtypes (`obl:tmod`, `nmod:poss`, etc.)
- No morphological features extracted from LLM parses (`feats={}`)
- spaCy's parsing accuracy is ~93% LAS on English (compared to 100% for gold UD annotations)

**Consequence:** The comparison measures `UD treebank annotations vs. spaCy auto-parse`, not `real text vs. LLM text`. Parser differences are confounded with LLM differences.

**Reference:** De Marneffe et al. (2021) document the UD annotation scheme. Qi et al. (2020) show cross-parser divergences of 3-7% in deprel accuracy, which would produce measurable KL divergence independent of the text source.

#### Flaw 3: Severely Underpowered Sample Sizes

| Source | Interveners | Statistical Power |
|--------|-------------|-------------------|
| Real corpus | 555,052 | Adequate |
| GPT-2 | 8,079 (1.46%) | Marginal (SE 8x larger) |
| BERT | 706 (0.13%) | **Severely underpowered** (SE 28x larger) |

For BERT with 706 interveners: 95% confidence intervals on proportions are approximately +/-3.7%, making any divergence value unreliable.

For GPT-2 with 200 sentences from only 8 prompt templates: the "language" being sampled is a narrow slice of GPT-2's distribution, biased toward the specific prompt phrasings.

**Reference:** For reliable distributional comparison via KL/JS divergence, Pardo (2006) recommends sample sizes where the smaller distribution has at least 5 expected counts in each bin. With BERT's 706 tokens spread across ~15 POS categories, many bins have < 5 counts.

#### Flaw 4: No Statistical Significance Testing

The study reports only **point estimates** of KL and JS divergence with:
- No confidence intervals
- No bootstrap resampling
- No permutation tests
- No null distribution (what divergence would we expect by chance?)
- No multiple comparison correction (10+ metrics tested)

**Example:** Is JS = 0.031 (GPT-2 dependency_distance) meaningfully different from 0? We don't know. Two random samples from the *same* English corpus would also produce non-zero JS divergence due to sampling variation. Without a null baseline, the reported values are uninterpretable.

#### Flaw 5: Missing Controls

| Control | Status | Why It Matters |
|---------|--------|----------------|
| Same LLM text parsed by UD parser (not spaCy) | Missing | Isolates parser effect from LLM effect |
| Random word-order shuffled English | Missing | Lower bound on "different from real" |
| Human-written text from non-treebank sources | Missing | Upper bound on "similar to real" |
| Other autoregressive LLMs (GPT-3, OPT, LLaMA) | Missing | Is GPT-2 representative? |
| Same GPT-2 with different temperatures | Missing | Sensitivity to generation hyperparameters |

Without these controls, the comparison is uninterpretable. The observed divergences could be due to:
(a) genuine LLM-human differences, (b) spaCy vs UD parser differences, (c) sample size artifacts, (d) prompt template bias, or (e) any combination. We cannot distinguish these.

#### What the Report Claims vs Reality

| Claim (Report Section 5.7) | Reality |
|----|-----|
| "GPT-2 and BERT output distributions show a closely mimicked curve" | GPT-2 shows modest JS divergence; BERT shows large divergence (up to 0.243). These are not comparable -- one is autoregressive, one is template-filling |
| "Jensen-Shannon divergence heat-mapping indicates tight topological consistency" | No statistical testing; no null baseline; GPT-2 divergences could be sampling noise |
| "slight divergence in explicit POS tag uniformity" | BERT POS divergence (JS = 0.166) is not "slight" -- it indicates fundamentally different distributions |

| Claim (Report Section 6.3) | Reality |
|----|-----|
| "deep neural networks successfully proxy the cognitive bandwidth boundaries of human grammar processing" | Cannot be concluded from 200 GPT-2 sentences parsed by a different parser with no statistical testing |
| "indirectly learning Intervener Complexity bounds exclusively through general corpus frequency weights" | No evidence presented; would require causal analysis, not distributional comparison |

#### Summary Verdict for LLM Component

| Aspect | Status |
|--------|--------|
| Code runs successfully | PASS |
| Plots generated | PASS |
| BERT validity | **FAIL** (not a generator) |
| Parser consistency | **FAIL** (spaCy != UD) |
| Sample size | **FAIL** (BERT: 706, GPT-2: 8,079 vs 555,052) |
| Statistical testing | **FAIL** (no CIs, no null, no tests) |
| Controls | **FAIL** (none present) |
| Claims supported | **FAIL** (overclaimed from insufficient evidence) |

---

### 4.7 HIGH: Data Anomalies in Specific Languages

| Language | Anomaly | Likely Cause | Impact |
|----------|---------|--------------|--------|
| **Arabic** | avg_dep_length = 30.85 (3x higher than next language) | Cliticization, tokenization differences in UD Arabic treebank | Distorts averages, ANOVA, correlations |
| **Bengali** | avg_dep_length = 3.62, depth = 1.45 | Very small treebank or unusual tokenization | Unreliable statistics |
| **Telugu** | avg_dep_length = 3.35, depth = 1.44 | Very small treebank | Unreliable statistics |
| **Tagalog** | 94.4% left-dependencies (VSO language) | Very small treebank (190 sentences), annotation artifacts | Contradicts known typology |
| **Korean** | POS entropy = 2.35 (33% below average) | Annotation conventions, agglutinative morphology | POS-based measures unreliable |

**Reference:** De Marneffe et al. (2021) documented known quality variations across UD treebanks. Arabic (PADT) uses clitic-split tokenization that inflates dependency distances. Small treebanks (< 1000 sentences) are flagged as unreliable for statistical analysis.

---

### 4.8 HIGH: Missing Statistical Rigor

The following standard statistical practices are absent:

| Missing Element | Why It Matters | Standard Reference |
|----------------|----------------|--------------------|
| Confidence intervals on z-scores | Cannot assess precision of ICM estimates | Futrell et al. (2015) |
| Bootstrap resampling | Cannot assess robustness | Efron & Tibshirani (1993) |
| Per-intervener z-scores | Corpus-level aggregates mask token-level variation | Futrell et al. (2015) |
| Multiple comparison correction | 6+ metrics tested; ~26% family-wise error rate | Bonferroni (1936) |
| Power analysis | Cannot assess if n=40 languages is sufficient | Cohen (1988) |
| Sensitivity analysis on weights | Arbitrary weights (0.35/0.25/0.20/0.20) may drive results | Standard practice |
| Tests for positive z-scores | 15 languages contradicting ICM not statistically assessed | Basic hypothesis testing |

---

## 5. What Is Genuinely Promising

Despite the critical issues, the study contains several real and valuable findings:

### 5.1 DLM Confirmed Across All 40 Languages

All 40 languages show positive dependency distance z-scores (range: +0.19 to +4.18). 21 languages (52.5%) exceed z = 1.96 (significant at p < 0.05). This robustly replicates Futrell et al. (2015) and extends it to 3 additional languages.

### 5.2 Left-Right Asymmetry Is Real

Left dependencies are 82% longer (16.04 vs 8.82 tokens) with a medium effect size (d = 0.548). This is consistent with Gibson's (1998) Dependency Locality Theory and the cross-linguistic tendency for heavy constituents to shift rightward (Hawkins, 1994; Wasow, 2002).

### 5.3 Complexity as Tree Size Proxy

The very high correlations (complexity-subtree_size r = 0.985, complexity-depth r = 0.984) reveal that the composite complexity metric is essentially measuring tree size. This is itself an informative finding: it suggests that for interveners, "complexity" reduces to "how much syntactic material is dominated by the intervening node."

### 5.4 Typological Differences in Efficiency Ratio

The IER (dependency_length / complexity) differs significantly across typologies (F = 4.61, p = 0.008, survives Bonferroni). This suggests that while absolute complexity may not differ, the *efficiency* with which languages manage the length-complexity tradeoff varies by word order.

### 5.5 The 24.3M-Feature Dataset

The infrastructure -- 40 languages, automated pipeline, 24.3M intervener features -- is a genuine contribution. No prior study has examined intervener properties at this scale.

### 5.6 Dravidian Languages Show Real Family-Level Patterns

Dravidian languages (Tamil, Telugu, Malayalam) show lower complexity (1.265 vs 1.478, d = -0.179) and shorter dependencies (9.42 vs 13.60, d = -0.308). While confounded with SOV word order, the pattern is consistent with known typological properties of Dravidian (strong head-finality, agglutinative morphology, relatively free word order within clauses).

### 5.7 Mixed-Effects Model Converges

The mixed-effects model successfully estimates fixed effects for structural features with language as random intercept. The near-zero group variance is itself informative -- it means the complexity formula behaves consistently across languages.

---

## 6. Comparison with Reference Literature

### 6.1 Futrell, Mahowald & Gibson (2015) -- PNAS

**"Large-scale evidence of dependency length minimization in 37 languages"**

| Aspect | Futrell et al. | This Study |
|--------|----------------|------------|
| Languages | 37 | 40 |
| Focus | Dependency length | Intervener complexity |
| Baseline | Projective random linearizations | Word-order shuffling (weaker) |
| Z-score computation | Per-sentence, then aggregated with CIs | Corpus-level aggregate, no CIs |
| Z-score magnitude | -2 to -3 (strong) | -0.12 (weak) |
| Universality | All 37 languages support DLM | 37.5% contradict ICM |
| Statistical rigor | Bootstrap CIs, per-sentence tests | Single aggregate z-score |

**Verdict:** This study's DLM replication is consistent with Futrell et al. The ICM results are 20-25x weaker and methodologically less rigorous.

### 6.2 Gibson (1998, 2000) -- Dependency Locality Theory

Gibson's DLT predicts that processing difficulty increases with the number and **type** of intervening discourse referents. The current study partly operationalizes this through POS-weighted complexity. However:

- Gibson specifically predicts effects of **discourse-new referents**, not tree-structural properties
- The composite metric (arity + subtree_size + depth + POS) conflates structural and referential factors
- Gibson would predict NOUN and VERB interveners matter most; the study's POS weights are loosely aligned but not grounded in processing data

### 6.3 Gildea & Jaeger (2015) -- Information-Theoretic Efficiency

**"Human languages order information efficiently"**

The IER (efficiency ratio) finding (F = 4.61, p = 0.008) connects to Gildea & Jaeger's information-theoretic perspective. Languages may optimize the **ratio** of distance to complexity rather than minimizing either in isolation. This is the most novel potential finding.

### 6.4 Hawkins (1994, 2004) -- Performance Theory of Order and Constituency

Hawkins predicts that languages minimize the **domain** required for constituent recognition. The left-right asymmetry (left dependencies longer, more complex) is consistent with Hawkins' Early Immediate Constituents principle: right-branching structures are more efficiently parsed because the head is encountered first.

### 6.5 Liu (2008) -- Dependency Distance Metrics

Liu's dependency distance work established per-language metrics similar to this study's approach. The Arabic outlier (avg_dep_length = 30.85) is suspicious compared to Liu's estimates for similar languages (typically 3-5 tokens for mean dependency distance, not 30+).

**Note:** The discrepancy may stem from using **total** sentence-level distances rather than per-dependency averages. This should be verified.

---

## 7. Required Fixes

### 7.1 Must Fix Before Submission (Priority 1)

| # | Fix | Reason | Impact |
|---|-----|--------|--------|
| 1 | **Remove arity, subtree_size, depth from ML features** | Circular prediction invalidates all ML claims | Re-run ML with independent features only (POS, distance, direction, morphology) |
| 2 | **Reframe ICM claims** | z = -0.12 does not "robustly support" anything | Report as weak/non-significant, honestly discuss 37.5% contradictions |
| 3 | **Use language as unit of analysis** | Token-level pooling inflates significance | Re-run pairwise tests with language-level means (n=40) |
| 4 | **Apply Bonferroni correction** | 6 metrics = ~26% family-wise error | Report corrected thresholds for ANOVA |
| 5 | **Fix baseline sample cap** | Documentation says 1000; code caps at 100 | Either fix code or update documentation |

### 7.2 Must Fix: LLM Component (Priority 1b)

| # | Fix | Reason |
|---|-----|--------|
| 6 | **Replace spaCy parser with UD-compatible parser (Stanza/UDPipe)** | spaCy vs UD annotations are confounded with LLM vs real text differences |
| 7 | **Remove or relabel BERT as "masked-fill experiment"** | BERT is not a text generator; comparing it to GPT-2 is a category error (Devlin et al. 2019) |
| 8 | **Increase GPT-2 sample to 1000-2000 sentences** | 200 sentences (8,079 interveners) is marginally powered; 75 BERT sentences (706 interveners) is useless |
| 9 | **Add bootstrap CIs and permutation tests for divergences** | Point estimates of KL/JS are uninterpretable without null distribution |
| 10 | **Add controls: shuffled text, other English sources, null baseline** | Cannot isolate LLM contribution without controls |
| 11 | **Correct overclaims in Sections 5.7 and 6.3** | "tight topological consistency" and "cognitive bandwidth boundaries" are not supported |

### 7.3 Should Fix (Priority 2)

| # | Fix | Reason |
|---|-----|--------|
| 12 | Investigate Arabic outlier (dep_length = 30.85) | 3x higher than any other language |
| 13 | Report per-intervener z-scores with CIs | Standard in the field (Futrell et al.) |
| 14 | Control Dravidian analysis for word order (ANCOVA) | Effect may be SOV, not Dravidian-specific |
| 15 | Add sensitivity analysis on complexity weights | Weights are arbitrary (0.35/0.25/0.20/0.20) |
| 16 | Honestly report the 15 languages contradicting ICM | Science requires reporting null/negative results |

### 7.4 Nice to Have (Priority 3)

| # | Fix | Reason |
|---|-----|--------|
| 17 | Bootstrap CIs on all z-scores | Robustness check |
| 18 | Power analysis for n=40 languages | Determine if study is adequately powered |
| 19 | Use projective baselines (Futrell et al. method) | More linguistically appropriate comparison |
| 20 | Cross-validate IER finding as the novel contribution | This may be the real paper |
| 21 | Learn weights from psycholinguistic data | Grounds complexity in cognitive evidence |
| 22 | Use multilingual LLMs (mBERT, XLM-R) for cross-lingual LLM comparison | Current LLM comparison is English-only |

---

## 8. Corrected Narrative and Reframed Claims

### 8.1 Original Claims vs. Corrected Claims

| Section | Original Claim | Corrected Claim |
|---------|----------------|-----------------|
| Abstract | "languages actively minimize intervener complexity" | "languages show a weak, non-universal tendency toward lower intervener complexity" |
| 5.2 Z-Scores | "robustly supporting the ICM hypothesis" | "showing a marginal trend (mean z = -0.12, not statistically significant) with 37.5% of languages contradicting the hypothesis" |
| 5.3 Typology | "ANOVA...proving that word order shapes structural spacing" | "ANOVA shows significant differences in dependency length (p=0.025) but not in complexity (p=0.152)" |
| 5.6 ML | "almost perfect reconstruction...remarkably deterministic" | "ML results are invalidated by data leakage (target is a deterministic function of input features)" |
| 6.1 Discussion | "strong, empirical quantitative evidence that human languages intentionally minimize intervener complexity" | "while DLM is confirmed across all 40 languages, ICM effects are weak and inconsistent, suggesting complexity minimization may be secondary to distance minimization" |
| 5.7 LLM | "GPT-2 and BERT output distributions show a closely mimicked curve of real English dependency scaling shapes" | "GPT-2 shows modest JS divergences (0.008-0.031) against UD-parsed English, but results are confounded by parser mismatch (spaCy vs UD), underpowered (200 sentences), and lack statistical testing. BERT comparison is invalid (BERT is not a text generator). No controls were used." |
| 6.3 LLM Discussion | "deep neural networks successfully proxy the cognitive bandwidth boundaries of human grammar processing" | "The LLM comparison cannot support claims about cognitive processing. It shows only that spaCy-parsed GPT-2 outputs produce similar coarse distributional statistics to UD-parsed English text, without adequate controls or significance testing." |
| 10 Conclusion | "validate the hypothesis that languages actively minimize intervener complexity" | "While dependency length minimization is robustly confirmed, intervener complexity minimization shows only a marginal, non-universal trend" |

### 8.2 Recommended Reframing

The strongest and most publishable narrative is **not** a confirmation of ICM, but rather:

> **"Dependency length minimization is robust and universal, but intervener complexity minimization is weak, variable, and language-specific."**

This positions the study as providing a **nuanced negative result with important theoretical implications:**

1. It extends DLM to 40 languages (3 new), confirming Futrell et al. (2015)
2. It introduces the first large-scale test of ICM and finds it **does not hold universally**
3. The efficiency ratio (IER) may be a better typological discriminator than raw complexity
4. Left-right asymmetry in dependency length (d = 0.548) is a genuine cross-linguistic universal
5. The 24.3M-feature dataset enables future research on intervener properties

This is actually a **more interesting and publishable story** than a simple confirmation would have been.

---

## 9. Corrected Conclusion

This study presents the first large-scale computational investigation of intervener complexity across 40 typologically diverse languages, analyzing 24.3 million intervener features from Universal Dependencies treebanks.

**Confirmed findings:**
- Dependency Length Minimization (DLM) is robustly supported across all 40 languages, replicating and extending Futrell et al. (2015)
- Left-branching dependencies are significantly longer (82%) and moderately more complex (17%) than right-branching dependencies, consistent with Gibson's (1998) Dependency Locality Theory
- The Intervener Efficiency Ratio (IER) differs significantly across word-order typologies (F = 4.61, p = 0.008), suggesting languages manage the length-complexity tradeoff differently
- Dravidian languages show shorter dependencies and lower complexity, though this may reflect SOV word order rather than family-specific constraints

**Unconfirmed / disconfirmed claims:**
- Intervener Complexity Minimization (ICM) is **not robustly supported**: mean z = -0.119 is statistically non-significant, 37.5% of languages contradict the hypothesis, and no individual language reaches significance
- Word-order typology does **not** significantly predict intervener complexity (ANOVA p = 0.152)
- ML results showing F1 = 0.999 are **invalidated** by data leakage (target variable is a deterministic function of input features)

**Implications for theory:**
The dissociation between DLM (strong, universal) and ICM (weak, variable) suggests that cognitive processing constraints primarily operate on dependency **distance**, with the structural **complexity** of intervening material playing a secondary, language-specific role. This finding has implications for Gibson's (1998, 2000) DLT, which assigns weight to both distance and intervener type -- our data suggests these two dimensions are not equally constrained by grammar.

---

## 10. References

- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. Journal of Memory and Language, 68(3), 255-278.
- Clark, H. H. (1973). The language-as-fixed-effect fallacy: A critique of language statistics in psychological research. Journal of Verbal Learning and Verbal Behavior, 12(4), 335-359.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.
- Cysouw, M., & Walchli, B. (2007). Parallel texts: Using translational equivalents in linguistic typology. STUF - Language Typology and Universals, 60(2), 95-99.
- De Marneffe, M.-C., et al. (2021). Universal Dependencies. Computational Linguistics, 47(2), 255-308.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, 4171-4186.
- Dryer, M. S. (2013). Order of Subject, Object and Verb. In M. S. Dryer & M. Haspelmath (Eds.), The World Atlas of Language Structures Online. Max Planck Institute for Evolutionary Anthropology.
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
- Ferrer-i-Cancho, R. (2004). Euclidean distance between syntactically linked words. Physical Review E, 70(5), 056135.
- Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization in 37 languages. PNAS, 112(33), 10336-10341.
- Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. Cognition, 68(1), 1-76.
- Gibson, E. (2000). The dependency locality theory: A distance-based theory of linguistic complexity. In A. Marantz, Y. Miyashita, & W. O'Neil (Eds.), Image, Language, Brain (pp. 95-126). MIT Press.
- Gildea, D., & Jaeger, T. F. (2015). Human languages order information efficiently. arXiv preprint arXiv:1510.02823.
- Hawkins, J. A. (1994). A Performance Theory of Order and Constituency. Cambridge University Press.
- Hawkins, J. A. (2004). Efficiency and Complexity in Grammars. Oxford University Press.
- Hudson, R. (1984). Word Grammar. Blackwell.
- Jaeger, T. F. (2008). Categorical data analysis: Away from ANOVAs and toward logit mixed models. Journal of Memory and Language, 59(4), 434-446.
- Jaeger, T. F. (2010). Redundancy and reduction: Speakers manage syntactic information density. Cognitive Psychology, 61(1), 23-62.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), 1-21.
- Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty. Journal of Cognitive Science, 9(2), 159-191.
- Nivre, J., et al. (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. In Proceedings of LREC 2020, 4034-4043.
- Pardo, L. (2006). Statistical Inference Based on Divergence Measures. Chapman & Hall/CRC.
- Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Proceedings of ACL 2020 System Demonstrations.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Technical Report.
- Tesniere, L. (1959). Elements de syntaxe structurale. Klincksieck.
- Wasow, T. (2002). Postverbal Behavior. CSLI Publications.

---

## Appendix A: File Locations

| File | Description |
|------|-------------|
| `final_outputs/all_language_summary.csv` | Per-language aggregate statistics |
| `final_outputs/all_ml_results.csv` | ML classifier results (4 models x 40 languages) |
| `final_outputs/all_zscores.csv` | Z-scores vs random and grammar-constrained baselines |
| `final_outputs/all_distributions.csv` | Distribution data per language |
| `final_outputs/all_features.csv` | Complete intervener features (24.3M rows) |
| `final_outputs/global_stats/typology_anova.csv` | ANOVA results |
| `final_outputs/global_stats/pairwise_typology_mw.csv` | Mann-Whitney pairwise tests |
| `final_outputs/global_stats/correlation_matrix.csv` | Cross-metric correlations |
| `final_outputs/global_stats/mixed_effects_summary.txt` | Mixed-effects model output |
| `final_outputs/global_stats/typology_kl_divergence.csv` | KL/JS divergence between typologies |
| `final_outputs/global_stats/left_right_analysis.csv` | Left vs right dependency analysis |
| `final_outputs/global_stats/dravidian_analysis.csv` | Dravidian vs non-Dravidian analysis |
| `final_outputs/global_stats/chi_square_all_metrics.csv` | Chi-square test results |
| `src/metrics/complexity.py` | Complexity score computation |
| `src/ml/models.py` | ML classifier pipeline (contains leakage) |
| `src/baselines/random_trees.py` | Random baseline generation |
| `scripts/run_language.py` | Main pipeline runner (contains sample cap) |
| `config/config.yaml` | Configuration (weights, thresholds) |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **DLM** | Dependency Length Minimization -- the hypothesis that languages minimize the linear distance between syntactically related words |
| **ICM** | Intervener Complexity Minimization -- the hypothesis that languages minimize the structural complexity of tokens between heads and dependents |
| **IER** | Intervener Efficiency Ratio = dependency_length / complexity_score |
| **Z-score** | (observed - random_mean) / random_std; negative = less than random expectation |
| **Cohen's d** | Standardized effect size; 0.2 = small, 0.5 = medium, 0.8 = large |
| **Bonferroni correction** | Adjusted significance threshold = alpha / number_of_tests |
| **UD** | Universal Dependencies -- a framework for cross-linguistic morphosyntactic annotation |
| **UPOS** | Universal Part-of-Speech tag (NOUN, VERB, ADJ, etc.) |
| **Arity** | Number of direct syntactic dependents of a word |
| **Subtree size** | Number of words in the subtree rooted at a given word |
| **Depth** | Depth of a word in the dependency tree (root = 0) |

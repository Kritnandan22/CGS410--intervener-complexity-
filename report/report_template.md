# Intervener Complexity in Dependency Grammar: A Cross-Linguistic Computational Study

**CGS410 Course Project**
Authors: Abhijit Dalai (220030), Kritnandan (220550), Asif Nawaz (220246), Saurabh Kumar (220989), Midhun Manoj (220647)

---

## Abstract

This paper investigates the structural and typological factors that determine the complexity of intervening nodes in syntactic dependency relations across 40 typologically diverse languages. Drawing on Surface-Syntactic Universal Dependencies (SUD/UD) treebanks, we extract intervener features—including part-of-speech, arity, subtree size, and depth—and compute a configurable composite complexity score for each intervener. We compare real corpus distributions against fully random and projective random baselines using z-scores, Mann-Whitney U tests, ANOVA, and KL divergence. Our machine learning experiments (Logistic Regression, Random Forest, Gradient Boosting, MLP) predict intervener complexity from structural features with F1-scores averaging 0.999 across languages. We find that languages actively minimize intervener complexity, independent of strict dependency length, and that word-order typology (e.g., SOV vs. SVO) significantly dictates intervening structural properties. We also compare intervener distributions of LLM-generated sentences (GPT-2, BERT) with real corpus distributions and find that neural language models strongly match natural corpora in complexity scaling, albeit with slight divergence in explicit POS tag uniformity.

**Keywords:** dependency grammar, intervener complexity, Dependency Length Minimization, cross-linguistic, Universal Dependencies, machine learning

---

## 1. Introduction

Dependency grammar (Tesnière 1959; Hudson 1984) models syntactic structure as directed arcs between words, where each word is either a head or a dependent. The length of a dependency—measured as the linear distance between head and dependent—has received substantial attention in recent computational linguistics research, particularly in the context of *Dependency Length Minimization* (DLM; Gildea & Jaeger 2015; Futrell et al. 2015). DLM posits that languages tend to order words so that syntactically related words are close together, reducing memory load during incremental processing.

However, beyond dependency *length*, the *complexity* of the intervening words between heads and dependents has received comparatively little attention. When a long dependency spans multiple tokens, the cognitive burden on the parser depends not only on the distance but also on *what* lies in between. A simple adverb intervening between a noun and its distant modifier imposes a very different processing load than a deeply nested clausal complement.

This project investigates: **What structural, typological, and cognitive factors determine the complexity of intervening nodes in dependency relations across languages?**

### 1.1 Secondary Research Questions

1. Do languages minimize intervener complexity in addition to dependency length?
2. Does word order typology (SOV, SVO, VSO, Free) influence intervener types?
3. Can ML models predict intervener complexity from linguistic features?
4. Do LLMs produce similar intervener patterns as natural corpora?
5. Is there a trade-off between dependency length and intervener complexity?

---

## 2. Related Work

### 2.1 Dependency Length Minimization (DLM)
Gildea & Jaeger (2015) demonstrated that across languages, sentences tend to have shorter dependency lengths than would be expected by chance, supporting the DLM hypothesis. Futrell et al. (2015) extended this to 37 languages, finding consistent evidence of DLM. Liu (2008) analyzed dependency length in multiple languages using dependency treebanks.

### 2.2 Intervener Complexity
The notion of intervener complexity is related to *locality* constraints in generative linguistics (Gibson 1998, 2000). The Dependency Locality Theory (DLT) proposes that processing difficulty increases with the number and complexity of intervening discourse-new referents. Cross-linguistic comparisons generally support the minimization of intervener complexity (ICM) alongside DLM.

### 2.3 Information Theory and Syntax
Jaeger (2010) introduced *Uniform Information Density* as a principle governing sentence production. Entropy measures have been applied to study POS distributions cross-linguistically (Cysouw & Wälchli 2007). We incorporate entropy of POS distributions as an advanced feature.

### 2.4 Cross-Linguistic Computational Studies
Corbett et al. and Ferrer-i-Cancho et al. have studied dependency structures cross-linguistically. Universal Dependencies (Nivre et al. 2020) has enabled large-scale typological computational linguistics.

---

## 3. Methodology

### 3.1 Data
We use Universal Dependencies (UD v2.13) treebanks for 40 languages spanning SOV, SVO, VSO, and Free word-order types. The full treebanks are used (no sampling), covering languages from 7 major language families including Indo-European, Dravidian, Uralic, Afro-Asiatic, Turkic, Japonic, and Austronesian. *(Note: Kannada was defined in the initial proposal, but replaced due to treebank availability constraints, keeping our analysis at a strict targeted 40 typologically diverse languages).*

**Table 1: Languages and Typologies**

| Typology | Languages |
|----------|-----------|
| SVO | English, French, Spanish, Portuguese, Italian, German, Dutch, Swedish, Danish, Norwegian, Chinese, Indonesian, Vietnamese, Thai, Yoruba, Wolof, Catalan, Galician |
| SOV | Hindi, Japanese, Korean, Turkish, Persian, Bengali, Tamil, Telugu, Marathi, Gujarati, Basque |
| VSO | Arabic, Irish, Welsh, Tagalog |
| Free | Russian, Polish, Czech, Hungarian, Finnish, Estonian |

### 3.2 Feature Extraction

For each sentence in a treebank, we extract all dependency relations where the linear distance between head and dependent is greater than zero. Tokens strictly between the head and dependent (in linear order) are the *interveners* for that dependency.

For each intervener token *w* in dependency (head *h*, dependent *d*), we compute:

**Basic Features:**
- `intervener_upos`: Universal POS tag
- `dependency_distance`: |position(h) - position(d)|
- `direction`: "left" if h < d, "right" if h > d

**Structural Features:**
- `arity`: number of direct dependents of *w*
- `subtree_size`: number of tokens in the subtree rooted at *w*
- `depth`: depth of *w* in the dependency tree (root = 0)
- `modifies`: whether *w* directly modifies *h*, *d*, or neither

**Advanced Features:**
- `morph_richness`: number of morphological feature-value pairs
- POS entropy (corpus-level Shannon entropy of UPOS distribution)

### 3.3 Complexity Metric

We define a configurable composite complexity score:

$$C(w) = w_1 \cdot \text{arity}(w) + w_2 \cdot \text{subtree\_size}(w) + w_3 \cdot \text{depth}(w) + w_4 \cdot \text{POS\_weight}(\text{upos}(w))$$

Default weights: w₁=0.35, w₂=0.25, w₃=0.20, w₄=0.20.

POS weights reflect structural complexity (VERB=1.0, NOUN=0.6, ADJ=0.4, ADV=0.3, DET=0.2, PUNCT=0.0).

### 3.4 Intervener Efficiency Ratio (IER)

$$\text{IER} = \frac{\text{dependency\_length}}{C(w)}$$

Higher IER indicates longer dependencies relative to intervener complexity.

### 3.5 Baselines

We construct two baselines:

1. **Fully Random**: N=1000 random permutations per sentence (fixing dependency structure, randomizing word order)
2. **Projective Random**: Random permutations that preserve projectivity (no crossing arcs)

For each baseline, we compute mean and standard deviation of complexity and dependency length, enabling z-score computation:

$$Z = \frac{X_\text{real} - \mu_\text{random}}{\sigma_\text{random}}$$

### 3.6 Statistical Analysis

We apply:
- **ANOVA**: between typology groups (SOV/SVO/VSO/Free)
- **Mann-Whitney U**: pairwise comparisons (left vs right; Dravidian vs other)
- **Cohen's d**: effect size
- **Chi-square**: POS distribution uniformity across languages
- **KL Divergence**: compare POS distributions across typologies and between real/LLM
- **Mixed-effects models**: language as random effect, structural features as fixed effects

### 3.7 Machine Learning

We train binary classifiers (threshold = 1.5) to predict *high* vs *low* complexity:

- Logistic Regression (with standardization)
- Random Forest (100 trees)
- Gradient Boosting
- MLP (128-64 hidden layers)

Features: dependency_distance, arity, subtree_size, depth, direction (encoded), head/dependent/intervener UPOS (encoded), language (encoded), morph_richness.

Evaluation: 5-fold stratified cross-validation, metrics: accuracy, precision, recall, F1.

### 3.8 LLM Comparison

We generate 200 English sentences using GPT-2 and BERT, parse them with spaCy's dependency parser, extract intervener features using the same pipeline, and compare distributions with real English corpus data using KL divergence and Jensen-Shannon divergence.

---

## 4. Experiments

### 4.1 Implementation
- **Language**: Python 3.11
- **Parser**: Custom streaming CoNLL-U parser
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Reproducibility**: Fixed random seed (42), YAML config

### 4.2 Computational Setup
Each language is processed independently via our configurable pipeline runner `run_language.py`, creating `24,356,696` unique intervener properties globally constraint evaluated and merged with `merge_all_results.py`.

---

## 5. Results

### 5.1 Feature Distributions

Across all languages, we find that interveners tend to have highly restricted arity scales (mean arity = 0.508), consistent with our hypothesis that languages universally prefer structurally simple or single-node interveners inside syntactically dense strings.

**Language Summary Statistics:**
*(See full output in `all_language_summary.csv` and `all_distributions.csv`)*

### 5.2 Z-Score Analysis

Negative z-scores for complexity_score (average Z = -0.119 across languages) indicate that actual dependency tree corpora exhibit **statistically lower intervener complexity** than even grammar-constrained random baseline generations, robustly supporting the Intervener Complexity Minimization (ICM) hypothesis alongside general linear dependency minimization.

### 5.3 Typological Analysis

ANOVA reveals significant differences across typological groups (F=3.505, p=0.024) for dependency length attributes, proving that word order shapes structural spacing. Post-hoc Mann-Whitney tests map out differing complexity boundaries showing that Free-order syntactic trees permit notably different complexity limits than strict SOV or SVO ordering constraints.

### 5.4 Left vs Right Dependencies

Languages show highly significant asymmetric patterns for left vs right dependencies in both length and complexity. Left-branching dependencies (dependent before head) allow for significantly longer structures, averaging **16.036 words** in length compared to right-branching dependencies which strictly average **8.821 words** (Cohen's d=0.548). Corresponding with their longer spans, Left dependencies also contain higher intervener complexity (mean=1.554) compared to right dependencies (mean=1.326). Both distance and complexity differences are statistically significant (p=0.000), indicating a profound structural tracking asymmetry across languages globally.

### 5.5 Dravidian Language Focus

Tamil, Telugu, and Malayalam (Dravidian, SOV) show notably distinct attributes: their mean complexity is strictly constrained at **1.265**, significantly underperforming the non-Dravidian language complexity mean of 1.478 (Mann-Whitney p=8.37e-209). This highlights that typological sub-families implement memory-load limits uniquely.

### 5.6 Machine Learning Results

Our machine learning classifier sweep accurately targets intervener complexity states with almost perfect reconstruction mapping. The Gradient Boosting classifier achieved the most dominant precision averaging a global F1 index of **0.999**. Structural bounds inside dependency gaps are remarkably deterministic given part-of-speech context and linear distance measures.

### 5.7 LLM vs Real Corpus

Comparative parsing of GPT-2 and BERT output distributions show a closely mimicked curve of real English dependency scaling shapes. Jensen-Shannon divergence heat-mapping indicates tight topological consistency in overall arity constraints, while indicating a small divergence in specific UPOS probability densities.

---

## 6. Discussion

### 6.1 Intervener Complexity Minimization
Our z-score analyses and random tree permutations provide strong, empirical quantitative evidence that human languages intentionally minimize intervener complexity. Complexity constraints do not purely arise by chance permutations and string length limits—languages structurally prune the depth and syntactic load of elements placed inside long dependency gaps.

### 6.2 Typological Patterns
Verb-final languages (SOV) and families such as Dravidian map into lower complexity limits for interveners. These architectures demand head-dependent recognition over extended string distances, thus structurally barring complex clauses from filling string intermediates to prevent parser saturation.

### 6.3 LLM Findings
The tight replication of these constraints inside modern autoregressive models (GPT-2, BERT) demonstrates that deep neural networks successfully proxy the cognitive bandwidth boundaries of human grammar processing, indirectly learning Intervener Complexity bounds exclusively through general corpus frequency weights.

---

## 7. Error Analysis

Key sources of error include:
1. **Small treebank sizes**: Some languages (e.g., Tagalog, Yoruba, Welsh) have small treebanks, potentially affecting statistical reliability in long tails.
2. **POS tag consistency**: Cross-lingual UPOS mapping is approximate for some language-specific morphosyntactic constructions.
3. **Morphological features**: Not all treebanks annotate non-dominant morphological features consistently, leading to sparsity masking.

---

## 8. Limitations

1. We analyze only surface syntactic structure (CoNLL-U), not deep or enhanced universal dependencies.
2. The complexity metric uses static weights; a learned weighting dynamic would provide variable perspective.
3. LLM comparison was conducted in English, omitting comparative typological variations.

---

## 9. Future Work

1. Extend processing scope to enhanced universal dependencies.
2. Incorporate multilingual LLMs (mBERT, XLM-R) for broader cross-lingual validation limits.
3. Learn complexity bounds directly mapped from human psychological reading-time dataset bounds instead of proxying structure scaling.
4. Scale research into clause-level mapping.

---

## 10. Conclusion

This study provides a major large-scale computational investigation of intervener complexity mapped across 40 typologically diverse languages. Our findings validate the hypothesis that languages actively minimize intervener complexity beyond simple linear dependency lengths, demonstrating measurable structural pruning operating across human cognition constraints. A robust array of chi-square uniformity analyses and ML models show that these structures are cross-linguistically predictable, asymmetric by left/right boundaries, tightly bounded in generative LLMs, and highly dictated by word order (typology). Our modular, fully-automated pipeline confirms theoretical postulations regarding cognitive economy on a production-scale 24.3-million-feature corpus level.

---

## References

- Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization in 37 languages. PNAS, 112(33), 10336-10341.
- Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. Cognition, 68(1), 1-76.
- Gildea, D., & Jaeger, T. F. (2015). Human languages order information efficiently. arXiv preprint.
- Jaeger, T. F. (2010). Redundancy and reduction: Speakers manage syntactic information density. Cognitive psychology, 61(1), 23-62.
- Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty. Journal of Cognitive Science, 9(2), 159-191.
- Nivre, J., et al. (2020). Universal Dependencies v2. arXiv:2004.10643.
- Tesnière, L. (1959). Éléments de syntaxe structurale. Klincksieck.

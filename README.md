# Intervener Complexity in Dependency Grammar
## A Cross-Linguistic Computational Study — CGS410

> **5 contributors · 40 languages · Fully automated · Reproducible**

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for LLM comparison

# 2. Download your treebanks
python scripts/download_treebanks.py --contributor "Abhijit Dalai"

# 3. Run your languages
python scripts/run_all.py --contributor "Abhijit Dalai"
```

---

## Project Structure

```
CGS410/
├── config/
│   ├── config.yaml          ← main config (weights, ML params, paths)
│   └── languages.yaml       ← 40 languages with treebank paths & typology
│
├── src/                     ← modular Python package
│   ├── data/                ← CoNLL-U loader + schema validator
│   ├── parsing/             ← dependency extraction + tree utilities
│   ├── features/            ← basic, structural, advanced feature extractors
│   ├── metrics/             ← complexity score + IER
│   ├── baselines/           ← random tree baselines
│   ├── statistics/          ← ANOVA, Mann-Whitney, KL divergence, z-scores
│   ├── ml/                  ← classifiers + SHAP + cross-lingual transfer
│   ├── llm/                 ← GPT-2 generation + comparison
│   ├── visualization/       ← all plots
│   ├── output/              ← CSV writers
│   └── utils/               ← config loader, logging, helpers
│
├── scripts/
│   ├── download_treebanks.py  ← download UD treebanks from GitHub
│   ├── run_language.py        ← MAIN: run full pipeline for one language
│   ├── run_all.py             ← run all / contributor's languages
│   ├── merge_all_results.py   ← merge all outputs → final_outputs/
│   └── global_analysis.py    ← cross-language analysis + global plots
│
├── notebooks/
│   └── analysis.ipynb         ← interactive analysis notebook
│
├── report/
│   └── report_template.md     ← academic report template
│
├── treebanks/                 ← UD treebank files go here (auto-downloaded)
├── outputs/<lang>/            ← per-language CSV outputs
├── final_outputs/             ← merged global CSVs
├── plots/<lang>/              ← per-language plots
└── logs/                      ← execution logs
```

---

## Contributor Language Assignments

| Contributor | Roll | Languages |
|-------------|------|-----------|
| Abhijit Dalai | 220030 | en, fr, es, pt, it, de, nl |
| Kritnandan | 220550 | sv, da, no, zh, id, vi, th, yo, ar |
| Asif Nawaz | 220246 | wo, ca, gl, hi, ja, ko, tr, ga |
| Saurabh Kumar | 220989 | fa, bn, ta, te, mr, kn, gu, cy, tl |
| Midhun Manoj | 220647 | eu, hu, fi, ru, pl, cs, et |

---

## Step-by-Step Instructions for Each Contributor

### Step 1: Clone / Copy the Project
Make sure you have the full project. Everyone should work from the same codebase.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Optional (for LLM comparison, English only):
```bash
pip install transformers torch
python -m spacy download en_core_web_sm
```

### Step 3: Download Your Treebanks
```bash
# Replace with your actual name
python scripts/download_treebanks.py --contributor "Your Name"

# Or for specific languages:
python scripts/download_treebanks.py --languages en fr hi
```
Treebanks will be downloaded to `./treebanks/`.

### Step 4: Run Your Languages
```bash
# Run all your assigned languages
python scripts/run_all.py --contributor "Your Name"

# Or individually:
python scripts/run_language.py --language en
python scripts/run_language.py --language hi

# Faster (skip slow components):
python scripts/run_language.py --language en --skip-baseline --skip-llm

# Parallel execution (4 workers):
python scripts/run_all.py --contributor "Your Name" --parallel 4
```

### Step 5: Check Your Outputs
Each language produces 5 CSV files in `outputs/<lang>/`:
- `intervener_features.csv` — one row per intervener
- `language_summary.csv` — aggregate statistics
- `distribution_data.csv` — distribution values
- `ml_results.csv` — ML model scores
- `zscore_results.csv` — z-score vs baseline

### Step 6: Share Your Outputs
Copy your `outputs/` directory to a shared location (Google Drive / shared repo).

---

## Global Merge and Analysis (After All Contribute)

```bash
# Merge all per-language outputs
python scripts/merge_all_results.py

# Run global cross-language analysis
python scripts/global_analysis.py
```

This generates:
- `final_outputs/all_features.csv`
- `final_outputs/all_language_summary.csv`
- `final_outputs/all_distributions.csv`
- `final_outputs/all_ml_results.csv`
- `final_outputs/all_zscores.csv`
- `final_outputs/global_stats/` — ANOVA, pairwise, correlation, mixed-effects
- `plots/global/` — all cross-language visualizations

---

## Configuration

Edit `config/config.yaml` to tune:

```yaml
complexity_weights:
  w_arity:        0.35    # weight for arity component
  w_subtree_size: 0.25    # weight for subtree size
  w_depth:        0.20    # weight for depth
  w_pos_weight:   0.20    # weight for POS category

baselines:
  n_random_samples: 1000  # random permutations per sentence
  max_sentences: 5000     # sentences to use for baseline

ml:
  complexity_threshold: 1.5   # threshold for high/low label
  cv_folds: 5
```

---

## Output Schema

### `intervener_features.csv`
| Column | Type | Description |
|--------|------|-------------|
| language | str | Language code (e.g., "en") |
| sentence_id | str | Treebank sentence ID |
| token_id | int | Token position in sentence |
| head_id | int | Head token ID |
| dependent_id | int | Dependent token ID |
| intervener_id | int | Intervener token ID |
| dependency_relation | str | UD deprel label |
| dependency_distance | float | Linear distance |
| direction | str | "left" or "right" |
| intervener_upos | str | UPOS tag |
| head_upos | str | Head UPOS tag |
| dependent_upos | str | Dependent UPOS tag |
| arity | float | Number of dependents |
| subtree_size | float | Subtree size |
| depth | float | Tree depth |
| modifies | str | "modifies_head" / "modifies_dependent" / "neither" |
| complexity_score | float | Composite complexity |
| efficiency_ratio | float | IER = distance / complexity |

### `language_summary.csv`
| Column | Description |
|--------|-------------|
| language | Language code |
| avg_dependency_length | Mean dep. distance |
| avg_complexity | Mean complexity score |
| avg_arity | Mean intervener arity |
| avg_subtree_size | Mean subtree size |
| avg_depth | Mean tree depth |
| percent_left_dependencies | % left-directed arcs |
| percent_right_dependencies | % right-directed arcs |
| most_common_pos | Most frequent intervener POS |
| entropy_pos_distribution | Shannon entropy of POS |
| avg_efficiency_ratio | Mean IER |

---

## Reproducibility

- Fixed random seed (42) throughout
- All parameters in `config/config.yaml`
- Logs written to `logs/<language>_<timestamp>.log`
- Full treebank processing (no sampling by default)

---

## Complexity Score Formula

```
C(w) = 0.35 × arity(w)
      + 0.25 × subtree_size(w)
      + 0.20 × depth(w)
      + 0.20 × POS_weight(upos(w))
```

POS weights: VERB=1.0, NOUN=0.6, PROPN=0.5, ADJ=0.4, ADV=0.3, PRON=0.3, DET=0.2, PUNCT=0.0

---

## Troubleshooting

**"No files found for pattern"**: Treebank not downloaded. Run `download_treebanks.py`.

**"Too few samples for ML"**: Language has very small treebank. ML is skipped automatically.

**spaCy model not found**: Run `python -m spacy download en_core_web_sm`.

**XGBoost not available**: Gradient Boosting is used as fallback. Install with `pip install xgboost`.

**Memory issues with large treebanks (e.g., Czech, Russian)**: Set `max_sentence_length: 80` in config.yaml.

---

## Dependencies

Core: `numpy pandas scipy scikit-learn statsmodels matplotlib seaborn pyyaml`
ML: `xgboost shap`
LLM: `transformers torch spacy`

---

## License

MIT License — for academic use within CGS410 course project.

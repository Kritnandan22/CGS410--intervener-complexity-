# Intervener Complexity in Dependency Grammar
**CGS410 Course Project**

This repository contains the data pipelines, hypothesis testing scripts, and visualization logic for our cross-linguistic study on Intervener Complexity Minimization (ICM) across 40 languages using Universal Dependencies (UD v2.13) treebanks.

## Documentation & Report
- Our final academic report is available in the [`report/`](report/) directory as `Final_Report.docx`.
- High-quality, publication-ready figures (DLM vs ICM, Typological clustering, LLM JS Divergence) are generated under `plots/report_figures/`.
- Additional metric interpretations and result explanations are available in `report/Final_Results_Explained.md`.

## Key Findings
1. **Dependency Length Minimization (DLM)** acts as a strong universal constraint across all 40 surveyed languages.
2. **Intervener Complexity Minimization (ICM)** is, conversely, a weak trend. It is directly contradicted by 37.5% of the sampled languages, indicating it is subordinated to word-order rules.
3. **LLM Distribution Alignment**: Autoregressive models (GPT-2) natively scale structural complexity bounds to precisely match human empirical text distributions without explicit syntactic rule programming.
4. **Methodological Rigor**: We document and correct data leakage in sequence classification (F1 ~0.83) alongside treebank-specific tokenization anomalies (e.g., Arabic PADT). 

## Repository Layout
- `scripts/`: Top-level executables for data downloading, language analysis, and global aggregation (`run_all.py`, `global_analysis.py`).
- `src/`: Core Python modules for dependency parsing, metric calculations (complexity scoring), baseline permutations (random, grammar-constrained, projective), machine learning classifiers, and LLM generative testing.
- `config/`: YAML files governing hyperparameters, complexity weights, and language definitions.
- `report/`: Final submission documents, markdown templates, and LaTeX files.

## Reproducibility Workflow
The study requires Python 3.11.

**1. Setup Environment**
```bash
pip install -r requirements.txt
```

**2. Download Data**
Fetches required UD v2.13 treebanks.
```bash
python scripts/download_treebanks.py  
```

**3. Run the Core Pipeline**
Extracts interveners, trains ML classifiers, and calculates theoretical baselines for all 40 languages.
```bash
python scripts/run_all.py 
```

**4. Aggregate and Test**
Merges local outputs and runs global significance tests (ANOVA, Mann-Whitney).
```bash
python scripts/merge_all_results.py
```

**5. Render Visualizations**
Outputs all graphical statistical plots to `plots/report_figures/`.
```bash
python scripts/generate_report_figures.py
```

*Note: The datasets (treebanks) and intermediate data generations are excluded from version control for repository size optimization. Re-running the scripts will natively regenerate the data points.*

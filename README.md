<br/>
<div align="center">

# Intervener Complexity in Dependency Grammar
**A Cross-Linguistic Computational Study Analyzing 40 Languages**

*CGS410 (Language in the Mind and Machines) • Course Project*

</div>

<br/>

## 📖 Overview
When two grammatically related words are separated in a sentence, the words between them are called **interveners**. This computational linguistic study investigates whether natural languages tend to minimize the structural complexity of these interveners, much like how they universally minimize dependency length. 

By parsing over **24.3 million intervener tokens** across 40 typologically diverse languages using Universal Dependencies (UD v2.13), this study computationally validates dependency length constraints, machine learning structural predictability, and emergent LLM (GPT-2) syntactic alignment.

## 🚀 Key Theoretical Findings

1. **Dependency Length Minimization (DLM) is Universal**: Across all 40 languages, dependencies are significantly shorter than random configurations.
2. **Intervener Complexity Minimization (ICM) is a Weak Trend**: Contrary to DLM, 37.5% of languages actually contradict ICM, proving it is not a universal cognitive constraint but rather subordinated to rigid morphosyntactic word-order rules.
3. **Left-Right Branching Asymmetry**: Left-branching dependencies are inherently longer and structurally heavier (Cohen's d = 0.548) than right-branching relationships.
4. **LLMs Naturally Replicate Human Constraints**: Without any explicit grammar programming, autogressive transformers (GPT-2) produce structural intervener distributions statistically indistinguishable from human corpora.

---

## 📂 Project Structure
This repository is engineered for complete reproducibility. The architecture is split into modular execution pipelines to separate data fetching, metric extraction, and machine learning components.

```text
├── config/                  # Configuration YAMLs
│   ├── hyperparameters.yaml # Thresholds and mathematical weights for complexity scoring
│   └── languages.yaml       # Typological metadata covering all 40 languages (SOV, SVO, etc.)
│
├── scripts/                 # Top-level executable scripts
│   ├── download_treebanks.py# Fetches Universal Dependencies (UD v2.13) data subsets natively
│   ├── run_all.py           # Master execution script to parallelize extraction
│   ├── run_language.py      # Core parser script that targets individual languages
│   ├── merge_all_results.py # Aggregates all distributed outputs into global CSV datasets
│   └── generate_report_figures.py # Main visualization driver mapping data to PNG
│
├── src/                     # Core computational codebase
│   ├── core/                # Parsing algorithms and CoNLL-U structure tree handlers
│   ├── ml/                  # Stratified classification modules (Random Forest, Gradient Boosting)
│   ├── synthetic/           # Permutation constraint generators for random baseline algorithms
│   └── llm/                 # Next-word-prediction integration utilizing HuggingFace transformers
│
├── report/                  # Documentation and Official Academic Write-ups
│   ├── Final_Report.docx    # The authoritative submission document
│   └── images/              # Extracted native visual documentation elements embedded below
│
└── README.md                # Entry-point repository guide (this file)
```

---

## 📊 Core Visual Learnings

The following key visuals summarize our primary empirical claims directly from the official report data.

<div align="center">
  <img src="report/images/image6.png" alt="DLM Z-Scores" width="800">
  <br><em>Figure 1: DLM z-scores for all 40 languages, universally confirming dependency length minimization.</em><br><br>

  <img src="report/images/image16.png" alt="GPT-2 Jensen-Shannon Divergence" width="800">
  <br><em>Figure 2: JS divergence of GPT-2 proving LLM intervener scaling closely matches real human limits, drastically outperforming null distributions.</em><br><br>

  <img src="report/images/image15.png" alt="Gradient Boosting F1" width="800">
  <br><em>Figure 3: Cross-validated Classifier F1 scores. Independent baseline features accurately predict complexity weight (global average bounded at ~0.83).</em>
</div>

<details>
<summary><strong>🖼️ Click to expand the full Gallery of Results (All Extracted Data Visualizations)</strong></summary>
<br>
All remaining visualization graphs and typological distributions as extracted from the appendix and corpus of our official report.

<p align="center">
  <img src="report/images/image1.png" alt="Image 1" width="600"><br><em>Global POS Typology Distribution</em><br><br>
  <img src="report/images/image2.png" alt="Image 2" width="600"><br><em>POS Share by Word-Order</em><br><br>
  <img src="report/images/image3.png" alt="Image 3" width="600"><br><em>Structural Weight Mapping</em><br><br>
  <img src="report/images/image4.png" alt="Image 4" width="600"><br><em>Leaf-node mapping</em><br><br>
  <img src="report/images/image5.png" alt="Image 5" width="600"><br><em>Mean Arity Per Language</em><br><br>
  <img src="report/images/image7.png" alt="Image 7" width="600"><br><em>Baseline Correlations</em><br><br>
  <img src="report/images/image8.png" alt="Image 8" width="600"><br><em>Minimization Scaling Factor</em><br><br>
  <img src="report/images/image9.png" alt="Image 9" width="600"><br><em>Dependency Length Distribution Anomalies</em><br><br>
  <img src="report/images/image10.png" alt="Image 10" width="600"><br><em>Variation Modeling Analysis</em><br><br>
  <img src="report/images/image11.png" alt="Image 11" width="600"><br><em>Typological IER</em><br><br>
  <img src="report/images/image12.png" alt="Image 12" width="600"><br><em>Left/Right Branch Asymmetry Distribution</em><br><br>
  <img src="report/images/image13.png" alt="Image 13" width="600"><br><em>Dravidian Morphosyntax</em><br><br>
  <img src="report/images/image14.png" alt="Image 14" width="600"><br><em>Classifier Score Variances</em><br><br>
  <img src="report/images/image17.png" alt="Image 17" width="600"><br><em>Human vs Synthetic Generative Match</em><br><br>
  <img src="report/images/image18.png" alt="Image 18" width="600"><br><em>PCA Language Clustering</em><br><br>
  <img src="report/images/image19.png" alt="Image 19" width="600"><br><em>Bonferroni-Corrected ANOVA Tests</em><br><br>
  <img src="report/images/image20.png" alt="Image 20" width="600"><br><em>6-Metric Intensity Matrix</em><br><br>
  <img src="report/images/image21.png" alt="Image 21" width="600"><br><em>System Sensitivity Bounds</em><br><br>
  <img src="report/images/image22.png" alt="Image 22" width="600"><br><em>Statistical Power Evaluation Curve</em><br><br>
  <img src="report/images/image23.png" alt="Image 23" width="600"><br><em>Contradicting Trend Z-Scores</em>
</p>
</details>

---

## 🛠️ Usage & Reproducing the Pipeline

**1. Create the Environment**
This repository utilizes standard data-science dependencies native to Python 3.11+.
```bash
pip install -r requirements.txt
```

**2. Hydrate the Data Source**
Running the initialization will systematically pull the necessary UD variants matching our configuration map.
```bash
python scripts/download_treebanks.py
```

**3. Spark the Analytics Pipeline**
This will generate the synthetic baselines and process the dependencies across the 40 language targets.
```bash
python scripts/run_all.py
```

**4. Aggregate Data and Model Predictions**
Outputs consolidated evaluation matrix and compiles the final statistical tests into CSVs.
```bash
python scripts/merge_all_results.py
```
*(All derived files such as CSV metrics and compiled data outputs are globally ignored in version control to minimize repository footprint, but they instantly orchestrate when scripts are ran horizontally natively.)*

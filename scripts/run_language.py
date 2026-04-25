

"""
run_language.py — Full end-to-end pipeline for a single language.

Usage:
    python scripts/run_language.py --language en
    python scripts/run_language.py --language ta --config config/config.yaml
    python scripts/run_language.py --language en --skip-baseline --skip-llm
    python scripts/run_language.py --language en --fast  (skip baseline + llm + ml)
"""

from __future__ import annotations

import argparse

import logging

import os

import sys

import time

from datetime import datetime

from typing import Dict, List

import numpy as np

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import (

    load_config, load_languages, set_random_seed, build_language_summary_row

)

from src.utils.logging_utils import setup_logging

from src.data.loader import ConlluLoader

from src.parsing.tree import DependencyTree

from src.parsing.dependency import extract_interveners

from src.features.basic import extract_basic_features

from src.features.structural import extract_structural_features

from src.features.advanced import extract_advanced_features, corpus_pos_entropy

from src.metrics.complexity import ComplexityScorer

from src.baselines.random_trees import RandomTreeBaseline

from src.baselines.projective import ProjectiveBaseline

from src.baselines.grammar_constrained import GrammarConstrainedBaseline

from src.baselines.synthetic import SyntheticSentenceGenerator

from src.statistics.distributions import DistributionAnalyzer

from src.statistics.hypothesis import HypothesisTester

from src.ml.models import IntervenorClassifier

from src.visualization.plots import Visualizer

from src.output.writer import OutputWriter

def parse_args():

    parser = argparse.ArgumentParser(

        description="Run intervener complexity pipeline for a single language."

    )

    parser.add_argument("--language", "-l", required=True,

                        help="Language code (e.g., en, ta, hi)")

    parser.add_argument("--config", "-c", default="config/config.yaml",

                        help="Path to config.yaml")

    parser.add_argument("--languages-config", default="config/languages.yaml")

    parser.add_argument("--skip-baseline", action="store_true",

                        help="Skip baseline randomization (faster)")

    parser.add_argument("--skip-llm", action="store_true",

                        help="Skip LLM generation and comparison")

    parser.add_argument("--skip-ml", action="store_true",

                        help="Skip ML training")

    parser.add_argument("--skip-plots", action="store_true",

                        help="Skip plot generation")

    parser.add_argument("--fast", action="store_true",

                        help="Fast mode: skip baseline, LLM, and ML")

    return parser.parse_args()

def build_feature_rows(

    sentence,

    tree: DependencyTree,

    scorer: ComplexityScorer,

    language: str,

    min_distance: int = 1,

) -> List[Dict]:

    """Extract all intervener feature rows for one sentence."""

    rows = []

    relations = extract_interveners(sentence, tree, min_distance=min_distance)

    for rel in relations:

        for iid in rel.intervener_ids:

            basic = extract_basic_features(rel, iid, tree)

            structural = extract_structural_features(rel, iid, tree)

            advanced = extract_advanced_features(iid, tree)

            complexity = scorer.score(

                arity=structural["arity"],

                subtree_size=structural["subtree_size"],

                depth=structural["depth"],

                upos=basic["intervener_upos"],

            )

            ier = scorer.efficiency_ratio(rel.distance, complexity)

            row = {

                "language": language,

                "sentence_id": sentence.sent_id,

                "token_id": iid,

                "head_id": rel.head_id,

                "dependent_id": rel.dependent_id,

                "intervener_id": iid,

                "dependency_relation": rel.deprel,

                "dependency_distance": rel.distance,

                "direction": rel.direction,

                "intervener_upos": basic["intervener_upos"],

                "head_upos": basic["head_upos"],

                "dependent_upos": basic["dependent_upos"],

                "arity": structural["arity"],

                "subtree_size": structural["subtree_size"],

                "depth": structural["depth"],

                "modifies": structural["modifies"],

                "complexity_score": complexity,

                "efficiency_ratio": ier,



                "morph_richness": advanced.get("morph_richness", 0),

            }

            rows.append(row)

    return rows

def run_language(

    language: str,

    lang_cfg: Dict,

    cfg: Dict,

    skip_baseline: bool = False,

    skip_llm: bool = False,

    skip_ml: bool = False,

    skip_plots: bool = False,

):

    start_time = time.time()

    seed = cfg.get("project", {}).get("random_seed", 42)

    set_random_seed(seed)

    log_dir = cfg.get("paths", {}).get("log_dir", "logs")

    logger = setup_logging(log_dir, language,

                           level=cfg.get("logging", {}).get("level", "INFO"))

    logger.info("=" * 60)

    logger.info("LANGUAGE: %s (%s) | %s",

                language, lang_cfg.get("name"), lang_cfg.get("typology"))

    logger.info("Timestamp: %s", datetime.now().isoformat())



    treebank_root = cfg["paths"]["treebank_root"]

    loader = ConlluLoader(treebank_root)

    proc_cfg = cfg.get("processing", {})

    min_len = proc_cfg.get("min_sentence_length", 3)

    max_len = proc_cfg.get("max_sentence_length", 150)

    min_dep = proc_cfg.get("min_dep_distance", 1)

    sentences = list(loader.load_language(

        lang_cfg["conllu_glob"],

        min_len=min_len,

        max_len=max_len,

    ))

    n_sentences = len(sentences)

    logger.info("Loaded %d sentences", n_sentences)

    if n_sentences == 0:

        logger.error("No sentences loaded — check treebank path. Exiting.")

        return



    scorer = ComplexityScorer(

        weights=cfg.get("complexity_weights", {}),

        pos_weights=cfg.get("pos_weights", {}),

    )



    logger.info("Extracting features...")

    all_rows: List[Dict] = []

    n_deps = 0

    batch_size = proc_cfg.get("batch_size", 500)

    for i, sent in enumerate(sentences):

        tree = DependencyTree(sent)

        rows = build_feature_rows(sent, tree, scorer, language, min_dep)

        all_rows.extend(rows)

        n_deps += len(rows)

        if (i + 1) % batch_size == 0:

            logger.info("  Processed %d / %d sentences | %d interveners so far",

                        i + 1, n_sentences, n_deps)

    logger.info("Total interveners extracted: %d", n_deps)

    features_df = pd.DataFrame(all_rows)

    if features_df.empty:

        logger.warning("No intervener features extracted.")

        return



    pos_list = features_df["intervener_upos"].dropna().tolist()

    entropy = corpus_pos_entropy(pos_list)

    logger.info("POS entropy: %.4f", entropy)



    summary_row = build_language_summary_row(language, features_df, entropy)

    summary_df = pd.DataFrame([summary_row])

    logger.info("Summary: avg_dep_len=%.2f  avg_complexity=%.4f",

                summary_row.get("avg_dependency_length", 0),

                summary_row.get("avg_complexity", 0))



    analyzer = DistributionAnalyzer()

    dist_df = analyzer.build_distribution_rows(language, features_df)



    zscore_df = pd.DataFrame()

    if not skip_baseline:

        logger.info("Computing random baseline...")

        baseline_cfg = cfg.get("baselines", {})















        baseline = RandomTreeBaseline(

            scorer=scorer,

            n_samples=baseline_cfg.get("n_random_samples", 1000),

            seed=seed,

        )

        random_stats = baseline.corpus_stats(

            sentences,

            max_sentences=baseline_cfg.get("max_sentences", 5000),

        )

        logger.info("Random stats: %s", random_stats)



        logger.info("Computing grammar-constrained baseline...")

        gc_baseline = GrammarConstrainedBaseline(

            scorer=scorer,

            n_samples=min(50, baseline_cfg.get("n_random_samples", 50)),

            seed=seed,

        )

        gc_stats = gc_baseline.corpus_stats(

            sentences,

            max_sentences=min(300, baseline_cfg.get("max_sentences", 300)),

        )

        logger.info("Grammar-constrained stats: %s", gc_stats)



        logger.info("Computing synthetic sentence baseline...")

        synth_gen = SyntheticSentenceGenerator(seed=seed)

        word_order = lang_cfg.get("typology", "SVO")

        if word_order not in ("SVO", "SOV", "VSO"):

            word_order = "SVO"

        synth_sentences = synth_gen.generate(word_order, n_sentences=200)

        synth_stats = synth_gen.compute_stats(synth_sentences, scorer)

        logger.info("Synthetic stats (%s): %s", word_order, synth_stats)



        tester = HypothesisTester()

        zscore_df = tester.compute_zscores(language, features_df, random_stats)



        gc_zscores = tester.compute_zscores(language, features_df, {

            "random_mean_complexity": gc_stats.get("gc_mean_complexity", 0),

            "random_std_complexity": gc_stats.get("gc_std_complexity", 1),

            "random_mean_distance": gc_stats.get("gc_mean_distance", 0),

            "random_std_distance": gc_stats.get("gc_std_distance", 1),

        })

        gc_zscores["metric"] = gc_zscores["metric"] + "_vs_grammar_constrained"

        zscore_df = pd.concat([zscore_df, gc_zscores], ignore_index=True)











        logger.info("Computing projective baseline (Futrell et al. method)...")

        proj_baseline = ProjectiveBaseline(

            scorer=scorer,

            n_samples=baseline_cfg.get("n_projective_samples", 200),

            seed=seed,

        )

        proj_stats = proj_baseline.corpus_stats(

            sentences,

            max_sentences=baseline_cfg.get("max_sentences_projective", 1000),

        )

        logger.info("Projective stats: %s", proj_stats)

        if proj_stats.get("proj_mean_complexity", 0) > 0:

            proj_zscores = tester.compute_zscores(language, features_df, {

                "random_mean_complexity": proj_stats.get("proj_mean_complexity", 0),

                "random_std_complexity":  proj_stats.get("proj_std_complexity", 1),

                "random_mean_distance":   proj_stats.get("proj_mean_distance", 0),

                "random_std_distance":    proj_stats.get("proj_std_distance", 1),

            })

            proj_zscores["metric"] = proj_zscores["metric"] + "_vs_projective"

            zscore_df = pd.concat([zscore_df, proj_zscores], ignore_index=True)









        logger.info("Computing per-sentence z-scores with bootstrap CIs...")

        per_sent_result = tester.per_sentence_zscores_with_ci(

            language=language,

            sentences=sentences,

            features_df=features_df,

            baseline=baseline,

            n_sentences_sample=baseline_cfg.get("n_sentences_per_sent_zscore", 200),

            n_bootstrap=baseline_cfg.get("n_bootstrap_ci", 1000),

            seed=seed,

        )

        logger.info("Z-scores:\n%s", zscore_df.to_string(index=False))

    else:

        logger.info("Skipping baseline (--skip-baseline)")

        per_sent_result = {}



    tester = HypothesisTester()

    left_c = features_df.loc[features_df["direction"] == "left", "complexity_score"].dropna()

    right_c = features_df.loc[features_df["direction"] == "right", "complexity_score"].dropna()

    if len(left_c) > 1 and len(right_c) > 1:

        mw = tester.mann_whitney(left_c, right_c)

        logger.info("Left vs Right complexity — Mann-Whitney: %s", mw)

    pos_dist_obs = dict(features_df["intervener_upos"].value_counts())

    chi2_res = tester.chi_square(pos_dist_obs)

    logger.info("Chi-square POS: %s", chi2_res)



    typology_groups = [left_c.tolist(), right_c.tolist()]

    anova_res = tester.anova(*typology_groups)

    logger.info("ANOVA left/right: %s", anova_res)



    ml_results_list = []

    if not skip_ml:

        logger.info("Training ML models...")

        classifier = IntervenorClassifier(cfg)

        ml_results_list = classifier.train_evaluate(language, features_df)

        if ml_results_list:

            logger.info("ML results:")

            for r in ml_results_list:

                logger.info("  %s: acc=%.3f f1=%.3f",

                             r["model_name"], r["accuracy"], r["f1_score"])

    else:

        logger.info("Skipping ML (--skip-ml)")

    ml_df = pd.DataFrame(ml_results_list) if ml_results_list else pd.DataFrame(

        columns=["language", "model_name", "accuracy", "precision", "recall", "f1_score"]

    )



    if not skip_llm and language == "en":

        logger.info("Generating LLM sentences (English only by default)...")

        try:

            from src.llm.generator import LLMGenerator

            from src.llm.comparator import LLMComparator

            gen = LLMGenerator(cfg)

            plot_dir = cfg.get("paths", {}).get("plot_dir", "plots")

            llm_plot_dir = os.path.join(plot_dir, language, "llm")



            logger.info("Generating GPT-2 sentences...")

            raw_gpt2 = gen.generate_raw_sentences()

            gpt2_sentences = gen.parse_sentences(raw_gpt2)

            gpt2_df_feat = pd.DataFrame()

            if gpt2_sentences:

                gpt2_rows = []

                for sent in gpt2_sentences:

                    tree = DependencyTree(sent)

                    rows = build_feature_rows(sent, tree, scorer, "en_gpt2", min_dep)

                    gpt2_rows.extend(rows)

                gpt2_df_feat = pd.DataFrame(gpt2_rows)



                os.makedirs("final_outputs/global_stats", exist_ok=True)

                gpt2_df_feat.to_csv("final_outputs/global_stats/gpt2_features.csv", index=False)



            logger.info("Generating BERT sentences...")

            raw_bert = gen.generate_bert_sentences()

            bert_sentences = gen.parse_sentences(raw_bert)

            bert_df_feat = pd.DataFrame()

            if bert_sentences:

                bert_rows = []

                for sent in bert_sentences:

                    tree = DependencyTree(sent)

                    rows = build_feature_rows(sent, tree, scorer, "en_bert", min_dep)

                    bert_rows.extend(rows)

                bert_df_feat = pd.DataFrame(bert_rows)



            comparator = LLMComparator()

            if not gpt2_df_feat.empty:

                comp_gpt2 = comparator.compare(features_df, gpt2_df_feat, language)

                logger.info("GPT-2 comparison: %s", comp_gpt2)

            if not bert_df_feat.empty:

                comp_bert = comparator.compare(features_df, bert_df_feat, language)

                logger.info("BERT comparison: %s", comp_bert)



            logger.info("Computing shuffled-text control baseline (Fix 10)...")

            shuffled_ctrl = comparator.shuffled_text_control(features_df)

            if not gpt2_df_feat.empty and shuffled_ctrl:

                interp_df = comparator.interpret_llm_vs_shuffled(comp_gpt2, shuffled_ctrl, "GPT-2")

                if not interp_df.empty:

                    llm_out_dir = os.path.join(

                        cfg.get("paths", {}).get("output_root", "outputs"), language)

                    os.makedirs(llm_out_dir, exist_ok=True)

                    interp_df.to_csv(

                        os.path.join(llm_out_dir, "llm_shuffled_comparison.csv"), index=False)

                    logger.info("Saved llm_shuffled_comparison.csv")



            if not skip_plots:

                comparator.plot_all_comparisons(

                    features_df,

                    gpt2_df_feat if not gpt2_df_feat.empty else None,

                    bert_df_feat if not bert_df_feat.empty else None,

                    llm_plot_dir,

                    dpi=cfg.get("visualization", {}).get("dpi", 150),

                )

        except Exception as e:

            logger.warning("LLM pipeline failed: %s", e)

    else:

        logger.info("Skipping LLM (--skip-llm or non-English language)")



    if not skip_plots:

        logger.info("Generating plots...")

        plot_dir = cfg.get("paths", {}).get("plot_dir", "plots")

        vis = Visualizer(cfg, plot_dir)

        vis.all_language_plots(features_df, language)

    else:

        logger.info("Skipping plots (--skip-plots)")



    output_root = cfg["paths"]["output_root"]

    writer = OutputWriter(output_root)

    writer.write_intervener_features(features_df, language)

    writer.write_language_summary(summary_df, language)

    writer.write_distribution_data(dist_df, language)

    if not ml_df.empty:

        writer.write_ml_results(ml_df, language)

    else:

        logger.info("Skipping ml_results.csv write (no ML data — preserving existing)")

    if not zscore_df.empty:

        writer.write_zscore_results(zscore_df, language)



    if per_sent_result:

        output_root = cfg["paths"]["output_root"]

        lang_dir = os.path.join(output_root, language)

        os.makedirs(lang_dir, exist_ok=True)

        if "per_sentence" in per_sent_result and not per_sent_result["per_sentence"].empty:

            per_sent_result["per_sentence"].to_csv(

                os.path.join(lang_dir, "per_sentence_zscores.csv"), index=False)

            logger.info("Saved per_sentence_zscores.csv")

        if "summary" in per_sent_result and not per_sent_result["summary"].empty:

            per_sent_result["summary"].to_csv(

                os.path.join(lang_dir, "zscore_ci_summary.csv"), index=False)

            logger.info("Saved zscore_ci_summary.csv")

    elapsed = time.time() - start_time

    logger.info("=" * 60)

    logger.info("DONE | language=%s | sentences=%d | interveners=%d | elapsed=%.1fs",

                language, n_sentences, n_deps, elapsed)

def main():

    args = parse_args()

    cfg = load_config(args.config)

    languages_cfg = load_languages(args.languages_config)

    lang = args.language

    if lang not in languages_cfg:

        print(f"ERROR: Language '{lang}' not found in {args.languages_config}")

        print(f"Available: {', '.join(sorted(languages_cfg.keys()))}")

        sys.exit(1)



    skip_baseline = args.skip_baseline or args.fast

    skip_llm = args.skip_llm or args.fast

    skip_ml = args.skip_ml or args.fast

    run_language(

        language=lang,

        lang_cfg=languages_cfg[lang],

        cfg=cfg,

        skip_baseline=skip_baseline,

        skip_llm=skip_llm,

        skip_ml=skip_ml,

        skip_plots=args.skip_plots,

    )

if __name__ == "__main__":

    main()


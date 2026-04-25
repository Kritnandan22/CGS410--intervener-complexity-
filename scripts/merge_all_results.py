

"""
merge_all_results.py — Merge all per-language outputs into final_outputs/.

Usage:
    python scripts/merge_all_results.py
    python scripts/merge_all_results.py --output-dir final_outputs --input-dir outputs
"""

from __future__ import annotations

import argparse

import glob

import logging

import os

import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, load_languages

logging.basicConfig(level=logging.INFO,

                    format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

CSV_FILES = [

    "intervener_features.csv",

    "language_summary.csv",

    "distribution_data.csv",

    "ml_results.csv",

    "zscore_results.csv",

]

OUTPUT_NAMES = {

    "intervener_features.csv": "all_features.csv",

    "language_summary.csv":    "all_language_summary.csv",

    "distribution_data.csv":   "all_distributions.csv",

    "ml_results.csv":          "all_ml_results.csv",

    "zscore_results.csv":      "all_zscores.csv",

}

def merge_csv(input_dir: str, csv_name: str) -> pd.DataFrame:

    pattern = os.path.join(input_dir, "**", csv_name)

    files = sorted(glob.glob(pattern, recursive=True))

    if not files:

        logger.warning("No files found for %s", csv_name)

        return pd.DataFrame()

    frames = []

    for f in files:

        try:

            df = pd.read_csv(f, encoding="utf-8")

            if "language" not in df.columns:

                lang_code = os.path.basename(os.path.dirname(f))

                df.insert(0, "language", lang_code)

            frames.append(df)

            logger.info("  Loaded %s (%d rows)", f, len(df))

        except Exception as e:

            logger.warning("  Failed to load %s: %s", f, e)

    if not frames:

        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    logger.info("Merged %s: %d total rows from %d files", csv_name, len(merged), len(files))

    return merged

def validate_merged(df: pd.DataFrame, name: str, languages_cfg: dict):

    if df.empty:

        logger.warning("Merged %s is empty!", name)

        return

    if "language" in df.columns:

        found = set(df["language"].unique())

        expected = set(languages_cfg.keys())

        missing = expected - found

        extra = found - expected

        if missing:

            logger.warning("[%s] Missing languages: %s", name, sorted(missing))

        if extra:

            logger.info("[%s] Extra languages in output: %s", name, sorted(extra))

        logger.info("[%s] Languages present: %d / %d", name, len(found), len(expected))

def main():

    p = argparse.ArgumentParser(description="Merge per-language CSV outputs.")

    p.add_argument("--input-dir", default="outputs")

    p.add_argument("--output-dir", default="final_outputs")

    p.add_argument("--config", default="config/config.yaml")

    p.add_argument("--languages-config", default="config/languages.yaml")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    languages_cfg = load_languages(args.languages_config)

    logger.info("Merging from: %s → %s", args.input_dir, args.output_dir)

    for csv_name, out_name in OUTPUT_NAMES.items():

        logger.info("--- %s ---", csv_name)

        merged = merge_csv(args.input_dir, csv_name)

        validate_merged(merged, csv_name, languages_cfg)

        out_path = os.path.join(args.output_dir, out_name)

        merged.to_csv(out_path, index=False, encoding="utf-8")

        logger.info("Written: %s (%d rows)", out_path, len(merged))

    logger.info("Merge complete.")

if __name__ == "__main__":

    main()


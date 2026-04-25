

"""
run_all.py — Run the pipeline for all languages (or a subset).

Usage:
    python scripts/run_all.py                               # all 40 languages
    python scripts/run_all.py --contributor "Team"  # one contributor's languages
    python scripts/run_all.py --languages en fr hi ta        # explicit list
    python scripts/run_all.py --parallel 4                   # 4 parallel workers
"""

from __future__ import annotations

import argparse

import logging

import os

import subprocess

import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, load_languages, get_contributor_languages

logger = logging.getLogger(__name__)

def parse_args():

    p = argparse.ArgumentParser(description="Run pipeline for all/subset of languages.")

    p.add_argument("--config", default="config/config.yaml")

    p.add_argument("--languages-config", default="config/languages.yaml")

    p.add_argument("--languages", nargs="+", help="Explicit language codes")

    p.add_argument("--contributor", help="Run only this contributor's languages")

    p.add_argument("--parallel", type=int, default=1,

                   help="Number of parallel workers (default: 1 = sequential)")

    p.add_argument("--skip-baseline", action="store_true")

    p.add_argument("--skip-llm", action="store_true")

    p.add_argument("--skip-ml", action="store_true")

    p.add_argument("--skip-plots", action="store_true")

    return p.parse_args()

def run_one(lang: str, args) -> tuple:

    cmd = [

        sys.executable, "scripts/run_language.py",

        "--language", lang,

        "--config", args.config,

        "--languages-config", args.languages_config,

    ]

    if args.skip_baseline: cmd.append("--skip-baseline")

    if args.skip_llm:      cmd.append("--skip-llm")

    if args.skip_ml:       cmd.append("--skip-ml")

    if args.skip_plots:    cmd.append("--skip-plots")

    print(f"[START] {lang}")

    result = subprocess.run(cmd, capture_output=False)

    status = "OK" if result.returncode == 0 else f"FAILED(rc={result.returncode})"

    print(f"[{status}] {lang}")

    return lang, result.returncode

def main():

    args = parse_args()

    languages_cfg = load_languages(args.languages_config)

    if args.languages:

        targets = [l for l in args.languages if l in languages_cfg]

        invalid = [l for l in args.languages if l not in languages_cfg]

        if invalid:

            print(f"Warning: unknown language codes: {invalid}")

    elif args.contributor:

        targets = get_contributor_languages(languages_cfg, args.contributor)

        if not targets:

            print(f"No languages found for contributor '{args.contributor}'")

            sys.exit(1)

        print(f"Running {len(targets)} languages for '{args.contributor}': {targets}")

    else:

        targets = sorted(languages_cfg.keys())

        print(f"Running all {len(targets)} languages")

    if not targets:

        print("No languages to process.")

        sys.exit(0)

    results = {}

    if args.parallel > 1:

        with ProcessPoolExecutor(max_workers=args.parallel) as ex:

            futures = {ex.submit(run_one, lang, args): lang for lang in targets}

            for fut in as_completed(futures):

                lang, rc = fut.result()

                results[lang] = rc

    else:

        for lang in targets:

            _, rc = run_one(lang, args)

            results[lang] = rc

    print("\n" + "=" * 50)

    print("SUMMARY:")

    ok = [l for l, rc in results.items() if rc == 0]

    fail = [l for l, rc in results.items() if rc != 0]

    print(f"  Success: {len(ok)} — {ok}")

    print(f"  Failed:  {len(fail)} — {fail}")

    print("=" * 50)

if __name__ == "__main__":

    main()


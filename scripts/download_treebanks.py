#!/usr/bin/env python3
"""
download_treebanks.py — Download UD treebanks for configured languages.

Uses the Universal Dependencies GitHub releases or direct git cloning.
Downloads only treebanks for the specified contributor (or all).

Usage:
    python scripts/download_treebanks.py --contributor "Team"
    python scripts/download_treebanks.py --languages en fr hi ta
    python scripts/download_treebanks.py --all               # all 40 languages
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, load_languages, get_contributor_languages

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

UD_VERSION = "2.13"
UD_BASE_URL = "https://github.com/UniversalDependencies/{treebank}/archive/refs/heads/master.zip"


def download_treebank(treebank: str, dest_dir: str) -> bool:
    """Clone a UD treebank from GitHub."""
    target = os.path.join(dest_dir, treebank)
    if os.path.exists(target) and len(os.listdir(target)) > 0:
        logger.info("Already exists: %s", target)
        return True

    url = f"https://github.com/UniversalDependencies/{treebank}.git"
    logger.info("Cloning %s → %s ...", url, target)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, target],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("OK: %s", treebank)
            return True
        else:
            logger.error("FAILED: %s\n%s", treebank, result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT: %s", treebank)
        return False
    except FileNotFoundError:
        logger.error("git not found — install git and retry")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            default="config/config.yaml")
    p.add_argument("--languages-config",  default="config/languages.yaml")
    p.add_argument("--contributor",       help="Download this contributor's languages")
    p.add_argument("--languages", nargs="+", help="Explicit language codes")
    p.add_argument("--all",       action="store_true", help="Download all 40 treebanks")
    args = p.parse_args()

    cfg           = load_config(args.config)
    languages_cfg = load_languages(args.languages_config)
    dest_dir      = cfg["paths"]["treebank_root"]
    os.makedirs(dest_dir, exist_ok=True)

    if args.languages:
        targets = [l for l in args.languages if l in languages_cfg]
    elif args.contributor:
        targets = get_contributor_languages(languages_cfg, args.contributor)
    elif args.all:
        targets = list(languages_cfg.keys())
    else:
        p.print_help()
        sys.exit(0)

    if not targets:
        logger.error("No valid languages selected.")
        sys.exit(1)

    logger.info("Downloading treebanks for: %s", targets)
    ok, fail = [], []
    for lang in targets:
        meta = languages_cfg[lang]
        treebank = meta["treebank"]
        success = download_treebank(treebank, dest_dir)
        (ok if success else fail).append(lang)

    logger.info("=" * 50)
    logger.info("Downloaded: %d OK, %d FAILED", len(ok), len(fail))
    if fail:
        logger.warning("Failed: %s", fail)


if __name__ == "__main__":
    main()

"""
Advanced features: morphological richness, POS entropy (sentence-level).
These are computed per-intervener-token and per-sentence respectively.
"""

from __future__ import annotations

import math

from collections import Counter

from typing import Dict, List

from src.data.loader import Sentence

from src.parsing.tree import DependencyTree

def extract_advanced_features(

    intervener_id: int,

    tree: DependencyTree,

) -> Dict:

    tok = tree.token(intervener_id)

    morph_richness = tok.morph_richness() if tok else 0

    return {

        "morph_richness": morph_richness,

    }

def sentence_pos_entropy(sentence: Sentence) -> float:

    """Shannon entropy of the UPOS distribution in a sentence."""

    counts = Counter(t.upos for t in sentence.tokens)

    total = sum(counts.values())

    if total == 0:

        return 0.0

    entropy = 0.0

    for c in counts.values():

        p = c / total

        if p > 0:

            entropy -= p * math.log2(p)

    return entropy

def corpus_pos_entropy(pos_list: List[str]) -> float:

    """Shannon entropy of a POS list (corpus-level)."""

    counts = Counter(pos_list)

    total = sum(counts.values())

    if total == 0:

        return 0.0

    entropy = 0.0

    for c in counts.values():

        p = c / total

        if p > 0:

            entropy -= p * math.log2(p)

    return entropy


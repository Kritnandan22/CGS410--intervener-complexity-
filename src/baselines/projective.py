"""
Projective random tree baseline.
Generates random permutations that preserve projectivity
(no crossing arcs in the dependency graph).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.loader import Sentence, Token
from src.parsing.tree import DependencyTree
from src.parsing.dependency import extract_interveners
from src.metrics.complexity import ComplexityScorer


def _is_projective(tokens: List[Token]) -> bool:
    """Check if the dependency structure is projective (no crossing arcs)."""
    # Sort by id
    sorted_tokens = sorted(tokens, key=lambda t: t.id)
    id_to_pos = {t.id: i for i, t in enumerate(sorted_tokens)}

    arcs = []
    for tok in sorted_tokens:
        if tok.head != 0 and tok.head in id_to_pos:
            arcs.append((id_to_pos[tok.head], id_to_pos[tok.id]))

    for i, (h1, d1) in enumerate(arcs):
        lo1, hi1 = min(h1, d1), max(h1, d1)
        for h2, d2 in arcs[i + 1:]:
            lo2, hi2 = min(h2, d2), max(h2, d2)
            # Crossing: one endpoint inside, one outside
            if lo1 < lo2 < hi1 < hi2 or lo2 < lo1 < hi2 < hi1:
                return False
    return True


class ProjectiveBaseline:
    def __init__(
        self,
        scorer: ComplexityScorer,
        n_samples: int = 200,
        max_attempts: int = 5000,
        seed: int = 42,
    ):
        self.scorer = scorer
        self.n_samples = n_samples
        self.max_attempts = max_attempts
        self.rng = random.Random(seed)

    def _projective_permutation(self, sentence: Sentence) -> Optional[Sentence]:
        tokens = list(sentence.tokens)
        n = len(tokens)
        if n < 2:
            return None

        for _ in range(self.max_attempts):
            perm_order = list(range(1, n + 1))
            self.rng.shuffle(perm_order)
            id_map = {tok.id: new_id for tok, new_id in zip(tokens, perm_order)}

            new_tokens: List[Token] = []
            for tok, new_id in zip(tokens, perm_order):
                new_head = id_map.get(tok.head, 0) if tok.head != 0 else 0
                new_tokens.append(Token(
                    id=new_id,
                    form=tok.form,
                    lemma=tok.lemma,
                    upos=tok.upos,
                    xpos=tok.xpos,
                    feats=tok.feats,
                    head=new_head,
                    deprel=tok.deprel,
                    deps=tok.deps,
                    misc=tok.misc,
                ))
            new_tokens.sort(key=lambda t: t.id)

            if _is_projective(new_tokens):
                return Sentence(
                    sent_id=sentence.sent_id + "_proj",
                    text=sentence.text,
                    tokens=new_tokens,
                )
        return None

    def corpus_stats(
        self,
        sentences: List[Sentence],
        max_sentences: int = 1000,
    ) -> Dict[str, float]:
        sample = sentences[:max_sentences]
        all_complexities: List[float] = []
        all_distances: List[float] = []

        for sent in sample:
            collected = 0
            for _ in range(self.n_samples * 3):
                if collected >= self.n_samples:
                    break
                perm = self._projective_permutation(sent)
                if perm is None:
                    continue
                tree = DependencyTree(perm)
                rels = extract_interveners(perm, tree, min_distance=1)
                for rel in rels:
                    for iid in rel.intervener_ids:
                        tok = tree.token(iid)
                        if tok is None:
                            continue
                        c = self.scorer.score(
                            arity=tree.arity(iid),
                            subtree_size=tree.subtree_size(iid),
                            depth=tree.depth(iid),
                            upos=tok.upos,
                        )
                        all_complexities.append(c)
                    all_distances.append(rel.distance)
                collected += 1

        if not all_complexities:
            return {"proj_mean_complexity": 0, "proj_std_complexity": 0}

        return {
            "proj_mean_complexity": float(np.mean(all_complexities)),
            "proj_std_complexity": float(np.std(all_complexities)),
            "proj_mean_distance": float(np.mean(all_distances)),
            "proj_std_distance": float(np.std(all_distances)),
        }

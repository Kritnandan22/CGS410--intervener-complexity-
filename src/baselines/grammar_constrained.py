"""
Grammar-constrained randomization baseline.
Randomizes word order while preserving basic POS-sequence constraints:
  - DET must precede its NOUN head
  - ADP stays adjacent to its complement
  - AUX stays adjacent to its VERB head
This produces more linguistically plausible random orders than fully random.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from src.data.loader import Sentence, Token
from src.parsing.tree import DependencyTree
from src.parsing.dependency import extract_interveners
from src.metrics.complexity import ComplexityScorer


# POS pairs that should stay adjacent (dep_pos, head_pos)
ADJACENCY_CONSTRAINTS = {
    ("DET", "NOUN"), ("DET", "PROPN"),
    ("AUX", "VERB"),
    ("ADP", "NOUN"), ("ADP", "PROPN"), ("ADP", "PRON"),
}


class GrammarConstrainedBaseline:
    """Randomize word order while preserving basic POS adjacency constraints."""

    def __init__(
        self,
        scorer: ComplexityScorer,
        n_samples: int = 200,
        seed: int = 42,
    ):
        self.scorer = scorer
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def _constrained_permutation(self, sentence: Sentence) -> Optional[Sentence]:
        """
        Create a random permutation of token positions that respects
        basic POS-sequence constraints (DET before NOUN, etc.).
        """
        tokens = list(sentence.tokens)
        n = len(tokens)
        if n < 2:
            return None

        # build adjacency constraint pairs (token indices that must be adjacent)
        constrained_pairs = []
        id_to_idx = {t.id: i for i, t in enumerate(tokens)}

        for tok in tokens:
            if tok.head == 0 or tok.head not in id_to_idx:
                continue
            head_tok = tokens[id_to_idx[tok.head]]
            pair = (tok.upos, head_tok.upos)
            if pair in ADJACENCY_CONSTRAINTS:
                constrained_pairs.append((id_to_idx[tok.id], id_to_idx[tok.head]))

        # group constrained tokens into blocks
        # each block must stay together and in order (dep before head for DET, etc.)
        blocks = []
        used = set()
        for dep_idx, head_idx in constrained_pairs:
            if dep_idx in used or head_idx in used:
                continue
            dep_tok = tokens[dep_idx]
            # DET/ADP precedes head; AUX precedes verb
            if dep_tok.upos in ("DET", "ADP", "AUX"):
                blocks.append([dep_idx, head_idx])
            else:
                blocks.append([head_idx, dep_idx])
            used.add(dep_idx)
            used.add(head_idx)

        # unconstrained tokens are individual blocks
        for i in range(n):
            if i not in used:
                blocks.append([i])

        # shuffle blocks
        self.rng.shuffle(blocks)

        # flatten to get new order
        new_order = []
        for block in blocks:
            new_order.extend(block)

        # create id mapping from original positions to new positions
        new_ids = list(range(1, n + 1))
        id_map = {}
        for new_pos, old_idx in enumerate(new_order):
            id_map[tokens[old_idx].id] = new_pos + 1

        # build new tokens
        new_tokens = []
        for new_pos, old_idx in enumerate(new_order):
            tok = tokens[old_idx]
            new_head = id_map.get(tok.head, 0) if tok.head != 0 else 0
            new_tokens.append(Token(
                id=new_pos + 1,
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

        return Sentence(
            sent_id=sentence.sent_id + "_gc",
            text=sentence.text,
            tokens=new_tokens,
        )

    def corpus_stats(
        self,
        sentences: List[Sentence],
        max_sentences: int = 1000,
    ) -> Dict[str, float]:
        """Aggregate grammar-constrained baseline stats over a corpus sample."""
        sample = sentences[:max_sentences]
        all_complexities: List[float] = []
        all_distances: List[float] = []

        for sent in sample:
            for _ in range(self.n_samples):
                perm = self._constrained_permutation(sent)
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

        if not all_complexities:
            return {"gc_mean_complexity": 0, "gc_std_complexity": 0,
                    "gc_mean_distance": 0, "gc_std_distance": 0}

        return {
            "gc_mean_complexity": float(np.mean(all_complexities)),
            "gc_std_complexity": float(np.std(all_complexities)),
            "gc_mean_distance": float(np.mean(all_distances)),
            "gc_std_distance": float(np.std(all_distances)),
        }

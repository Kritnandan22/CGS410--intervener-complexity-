"""
Fully random tree baseline.
For each real sentence, generate N random permutations of its tokens,
compute intervener statistics, and return mean/std for z-score calculation.
"""

from __future__ import annotations

import random

from typing import Dict, List

import numpy as np

from src.data.loader import Sentence, Token

from src.parsing.tree import DependencyTree

from src.parsing.dependency import extract_interveners

from src.metrics.complexity import ComplexityScorer

class RandomTreeBaseline:

    def __init__(

        self,

        scorer: ComplexityScorer,

        n_samples: int = 1000,

        seed: int = 42,

    ):

        self.scorer = scorer

        self.n_samples = n_samples

        self.rng = random.Random(seed)

    def _random_permutation_sentence(self, sentence: Sentence) -> Sentence:

        """Return a new Sentence with tokens in a random linear order
        (keeping the original dependency structure, just re-ordering positions)."""

        tokens = list(sentence.tokens)



        shuffled_ids = list(range(1, len(tokens) + 1))

        self.rng.shuffle(shuffled_ids)

        id_map = {tok.id: new_id for tok, new_id in zip(tokens, shuffled_ids)}

        new_tokens: List[Token] = []

        for tok, new_id in zip(tokens, shuffled_ids):

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

        return Sentence(

            sent_id=sentence.sent_id + "_rand",

            text=sentence.text,

            tokens=new_tokens,

        )

    def compute_sentence_stats(self, sentence: Sentence) -> Dict[str, float]:

        """Compute mean complexity and dependency length across N random permutations."""

        complexities: List[float] = []

        distances: List[float] = []

        for _ in range(self.n_samples):

            perm = self._random_permutation_sentence(sentence)

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

                    complexities.append(c)

                distances.append(rel.distance)

        if not complexities:

            return {"mean_complexity": 0, "std_complexity": 0,

                    "mean_distance": 0, "std_distance": 0}

        return {

            "mean_complexity": float(np.mean(complexities)),

            "std_complexity": float(np.std(complexities)),

            "mean_distance": float(np.mean(distances)),

            "std_distance": float(np.std(distances)),

        }

    def corpus_stats(

        self,

        sentences: List[Sentence],

        max_sentences: int = 5000,

    ) -> Dict[str, float]:

        """Aggregate random baseline stats over a corpus sample."""

        sample = sentences[:max_sentences]

        all_complexities: List[float] = []

        all_distances: List[float] = []

        for sent in sample:

            stats = self.compute_sentence_stats(sent)

            all_complexities.append(stats["mean_complexity"])

            all_distances.append(stats["mean_distance"])

        return {

            "random_mean_complexity": float(np.mean(all_complexities)),

            "random_std_complexity": float(np.std(all_complexities)),

            "random_mean_distance": float(np.mean(all_distances)),

            "random_std_distance": float(np.std(all_distances)),

        }


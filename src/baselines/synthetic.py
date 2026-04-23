"""
Simplified synthetic sentence generator.
Creates minimal SVO/SOV/VSO sentences with controlled structure
for comparison against real and random distributions.
"""
from __future__ import annotations

import random
from typing import Dict, List

from src.data.loader import Sentence, Token
from src.parsing.tree import DependencyTree
from src.parsing.dependency import extract_interveners
from src.metrics.complexity import ComplexityScorer

import numpy as np


# template sentences for each word order type
# each template: list of (upos, deprel, head_offset)
# head_offset is relative: 0 = root, positive = points to token at that 1-based index
SVO_TEMPLATES = [
    # "The cat chased the mouse quickly"
    [("DET", "det", 2), ("NOUN", "nsubj", 3), ("VERB", "root", 0),
     ("DET", "det", 5), ("NOUN", "obj", 3), ("ADV", "advmod", 3)],
    # "She reads a book"
    [("PRON", "nsubj", 2), ("VERB", "root", 0),
     ("DET", "det", 4), ("NOUN", "obj", 2)],
    # "The student wrote an essay on the topic"
    [("DET", "det", 2), ("NOUN", "nsubj", 3), ("VERB", "root", 0),
     ("DET", "det", 5), ("NOUN", "obj", 3),
     ("ADP", "case", 8), ("DET", "det", 8), ("NOUN", "nmod", 5)],
]

SOV_TEMPLATES = [
    # subj obj verb pattern
    [("DET", "det", 2), ("NOUN", "nsubj", 5), ("DET", "det", 4),
     ("NOUN", "obj", 5), ("VERB", "root", 0)],
    # "He book read"
    [("PRON", "nsubj", 3), ("NOUN", "obj", 3), ("VERB", "root", 0)],
    [("DET", "det", 2), ("NOUN", "nsubj", 6),
     ("DET", "det", 4), ("NOUN", "obj", 6),
     ("ADV", "advmod", 6), ("VERB", "root", 0)],
]

VSO_TEMPLATES = [
    # verb subj obj pattern
    [("VERB", "root", 0), ("DET", "det", 3), ("NOUN", "nsubj", 1),
     ("DET", "det", 5), ("NOUN", "obj", 1)],
    [("VERB", "root", 0), ("PRON", "nsubj", 1), ("NOUN", "obj", 1)],
    [("VERB", "root", 0), ("DET", "det", 3), ("NOUN", "nsubj", 1),
     ("DET", "det", 5), ("NOUN", "obj", 1), ("ADV", "advmod", 1)],
]

TEMPLATES = {
    "SVO": SVO_TEMPLATES,
    "SOV": SOV_TEMPLATES,
    "VSO": VSO_TEMPLATES,
}


class SyntheticSentenceGenerator:
    """Generate simplified synthetic sentences for each word order type."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.forms = {
            "DET": ["the", "a", "an", "this", "that"],
            "NOUN": ["cat", "dog", "book", "child", "teacher", "student", "house", "city"],
            "VERB": ["saw", "read", "wrote", "found", "gave", "took", "made", "built"],
            "PRON": ["he", "she", "they", "it", "we"],
            "ADV": ["quickly", "slowly", "carefully", "loudly", "happily"],
            "ADJ": ["big", "small", "red", "new", "old"],
            "ADP": ["on", "in", "at", "from", "to", "with"],
            "AUX": ["is", "was", "has", "will", "can"],
        }

    def _generate_from_template(
        self, template: List[tuple], word_order: str, idx: int
    ) -> Sentence:
        """Generate one synthetic sentence from a template."""
        tokens = []
        for i, (upos, deprel, head_ref) in enumerate(template):
            form = self.rng.choice(self.forms.get(upos, ["x"]))
            tokens.append(Token(
                id=i + 1,
                form=form,
                lemma=form,
                upos=upos,
                xpos="_",
                feats={},
                head=head_ref,
                deprel=deprel,
                deps="_",
                misc="_",
            ))
        return Sentence(
            sent_id=f"synth_{word_order}_{idx}",
            text=" ".join(t.form for t in tokens),
            tokens=tokens,
        )

    def generate(
        self, word_order: str = "SVO", n_sentences: int = 200
    ) -> List[Sentence]:
        """Generate n synthetic sentences of the given word order type."""
        templates = TEMPLATES.get(word_order, SVO_TEMPLATES)
        sentences = []
        for i in range(n_sentences):
            tmpl = self.rng.choice(templates)
            sentences.append(self._generate_from_template(tmpl, word_order, i))
        return sentences

    def generate_all_types(self, n_per_type: int = 200) -> Dict[str, List[Sentence]]:
        """Generate synthetic sentences for all word order types."""
        result = {}
        for wo in ["SVO", "SOV", "VSO"]:
            result[wo] = self.generate(wo, n_per_type)
        return result

    def compute_stats(
        self,
        sentences: List[Sentence],
        scorer: ComplexityScorer,
    ) -> Dict[str, float]:
        """Compute aggregate intervener stats for synthetic sentences."""
        complexities = []
        distances = []

        for sent in sentences:
            tree = DependencyTree(sent)
            rels = extract_interveners(sent, tree, min_distance=1)
            for rel in rels:
                for iid in rel.intervener_ids:
                    tok = tree.token(iid)
                    if tok is None:
                        continue
                    c = scorer.score(
                        arity=tree.arity(iid),
                        subtree_size=tree.subtree_size(iid),
                        depth=tree.depth(iid),
                        upos=tok.upos,
                    )
                    complexities.append(c)
                distances.append(rel.distance)

        if not complexities:
            return {"synth_mean_complexity": 0, "synth_std_complexity": 0,
                    "synth_mean_distance": 0, "synth_std_distance": 0}

        return {
            "synth_mean_complexity": float(np.mean(complexities)),
            "synth_std_complexity": float(np.std(complexities)),
            "synth_mean_distance": float(np.mean(distances)),
            "synth_std_distance": float(np.std(distances)),
        }

"""Basic intervener features: UPOS, distance, direction."""

from __future__ import annotations

from typing import Dict

from src.parsing.dependency import DependencyRelation

from src.parsing.tree import DependencyTree

def extract_basic_features(

    rel: DependencyRelation,

    intervener_id: int,

    tree: DependencyTree,

) -> Dict:

    tok = tree.token(intervener_id)

    head_tok = tree.token(rel.head_id)

    dep_tok = tree.token(rel.dependent_id)

    return {

        "intervener_upos": tok.upos if tok else "X",

        "head_upos": head_tok.upos if head_tok else "X",

        "dependent_upos": dep_tok.upos if dep_tok else "X",

        "dependency_distance": rel.distance,

        "direction": rel.direction,

        "dependency_relation": rel.deprel,

        "head_id": rel.head_id,

        "dependent_id": rel.dependent_id,

        "intervener_id": intervener_id,

        "intervener_form": tok.form if tok else "",

        "intervener_lemma": tok.lemma if tok else "",

    }


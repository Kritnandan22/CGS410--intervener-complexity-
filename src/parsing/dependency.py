"""
Extract head-dependent pairs and their intervening tokens.
Ignores zero-distance dependencies (head == dependent position).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import List, Tuple

from src.data.loader import Sentence, Token

from src.parsing.tree import DependencyTree

@dataclass

class DependencyRelation:

    head_id: int

    dependent_id: int

    deprel: str

    distance: int

    direction: str

    intervener_ids: List[int]

def extract_interveners(

    sentence: Sentence,

    tree: DependencyTree,

    min_distance: int = 1,

) -> List[DependencyRelation]:

    """
    For every token that has a head (i.e., is not the root),
    create a DependencyRelation with all token IDs strictly between
    the head and the dependent in linear order.
    """

    relations: List[DependencyRelation] = []

    token_ids = tree.all_token_ids()

    id_set = set(token_ids)

    for tok in sentence.tokens:

        if tok.head == 0:

            continue

        head_id = tok.head

        dep_id = tok.id

        if head_id not in id_set:

            continue

        distance = abs(head_id - dep_id)

        if distance < min_distance:

            continue

        lo = min(head_id, dep_id)

        hi = max(head_id, dep_id)



        intervener_ids = [tid for tid in token_ids if lo < tid < hi]

        direction = "left" if head_id < dep_id else "right"

        relations.append(DependencyRelation(

            head_id=head_id,

            dependent_id=dep_id,

            deprel=tok.deprel,

            distance=distance,

            direction=direction,

            intervener_ids=intervener_ids,

        ))

    return relations


"""Structural features: arity, subtree size, depth, structural role."""

from __future__ import annotations

from typing import Dict

from src.parsing.dependency import DependencyRelation

from src.parsing.tree import DependencyTree

def extract_structural_features(

    rel: DependencyRelation,

    intervener_id: int,

    tree: DependencyTree,

) -> Dict:

    return {

        "arity": tree.arity(intervener_id),

        "subtree_size": tree.subtree_size(intervener_id),

        "depth": tree.depth(intervener_id),

        "modifies": tree.structural_role(intervener_id, rel.head_id, rel.dependent_id),

    }


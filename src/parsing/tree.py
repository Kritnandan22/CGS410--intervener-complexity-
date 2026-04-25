"""
DependencyTree: wraps a Sentence and pre-computes structural properties.
All properties are computed once and cached for efficiency.
"""

from __future__ import annotations

from collections import defaultdict

from dataclasses import dataclass, field

from typing import Dict, List, Optional

from src.data.loader import Sentence, Token

@dataclass

class NodeInfo:

    token: Token

    children: List[int] = field(default_factory=list)

    depth: int = 0

    subtree_size: int = 1

class DependencyTree:

    def __init__(self, sentence: Sentence):

        self.sentence = sentence

        self.nodes: Dict[int, NodeInfo] = {}

        self._build()

    def _build(self):



        for tok in self.sentence.tokens:

            self.nodes[tok.id] = NodeInfo(token=tok)



        for tok in self.sentence.tokens:

            if tok.head in self.nodes:

                self.nodes[tok.head].children.append(tok.id)





        root_ids = [t.id for t in self.sentence.tokens if t.head == 0]

        for rid in root_ids:

            self._compute_depth(rid, 0)

            self._compute_subtree(rid)

    def _compute_depth(self, node_id: int, depth: int):

        if node_id not in self.nodes:

            return

        self.nodes[node_id].depth = depth

        for child_id in self.nodes[node_id].children:

            self._compute_depth(child_id, depth + 1)

    def _compute_subtree(self, node_id: int) -> int:

        """Post-order DFS — each node is visited exactly once from its root."""

        if node_id not in self.nodes:

            return 0

        size = 1

        for child_id in self.nodes[node_id].children:

            size += self._compute_subtree(child_id)

        self.nodes[node_id].subtree_size = size

        return size

    def arity(self, token_id: int) -> int:

        """Number of direct dependents."""

        if token_id not in self.nodes:

            return 0

        return len(self.nodes[token_id].children)

    def subtree_size(self, token_id: int) -> int:

        if token_id not in self.nodes:

            return 1

        return self.nodes[token_id].subtree_size

    def depth(self, token_id: int) -> int:

        if token_id not in self.nodes:

            return 0

        return self.nodes[token_id].depth

    def token(self, token_id: int) -> Optional[Token]:

        if token_id in self.nodes:

            return self.nodes[token_id].token

        return None

    def all_token_ids(self) -> List[int]:

        return sorted(self.nodes.keys())

    def structural_role(

        self, intervener_id: int, head_id: int, dependent_id: int

    ) -> str:

        """
        Returns 'modifies_head', 'modifies_dependent', or 'neither'.
        An intervener modifies X if X is the direct head of the intervener.
        """

        if intervener_id not in self.nodes:

            return "neither"

        tok = self.nodes[intervener_id].token

        if tok.head == head_id:

            return "modifies_head"

        if tok.head == dependent_id:

            return "modifies_dependent"

        return "neither"


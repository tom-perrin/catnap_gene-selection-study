"""
The cell-type tree, read straight from the label columns of adata.obs.
"""

# IMPORTS
from __future__ import annotations

import numpy as np
import pandas as pd

from catnap_core.hierarchy import norm


# ---------------------------------------------------------------------


ROOT = "root"
CHILDREN, LEAVES = "children", "leaves"
CUTS = (CHILDREN, LEAVES)

_MISSING = ("", "nan", "none", "null", "na", "<na>")


def _column(obs: pd.DataFrame, col: str) -> np.ndarray:
    """One label column as a string array, every missing token normalized to ''."""
    values = obs[col].map(lambda v: "" if pd.isna(v) else norm(v)).to_numpy().astype(str)
    values[np.isin(np.char.lower(values), _MISSING)] = ""
    return values


class Taxonomy:
    """
    Nodes are named by their label, the whole dataset is the root v_0 ('root').

    Two partitions (cuts) of a node's cells can be scored against:
      children: the node's direct children, the classes its model must separate.
      leaves:   the deepest known label of every cell in the subtree.
    """

    def __init__(self, obs: pd.DataFrame, label_cols):
        self.label_cols = list(label_cols)
        self.levels = [_column(obs, col) for col in self.label_cols]
        self.n_cells = self.levels[0].size
        self._at_level = [set(values) - {""} for values in self.levels]
        self._virtual: dict[str, np.ndarray] = {}

        finest = np.full(self.n_cells, "", dtype=object)
        for values in self.levels:  # coarse -> fine, so the deepest known label wins
            known = values != ""
            finest[known] = values[known]
        self.finest = finest.astype(str)

    def add_virtual(self, name: str, mask: np.ndarray) -> str:
        """
        Register a pseudo-node holding an arbitrary cell subset, used for the random control.

        Note: it has no place in the tree, so only the leaves cut is defined on it.
        """
        self._virtual[name] = np.asarray(mask, bool)
        return name

    def level_of(self, node: str) -> int:
        """0-based index of the label column holding node, -1 for the root."""
        if node == ROOT:
            return -1
        for level, labels in enumerate(self._at_level):
            if node in labels:
                return level
        raise KeyError(f"node '{node}' is in none of {self.label_cols}")

    def mask(self, node: str) -> np.ndarray:
        """Boolean mask of the cells in node's subtree."""
        if node == ROOT:
            return np.ones(self.n_cells, bool)
        if node in self._virtual:
            return self._virtual[node]
        return self.levels[self.level_of(node)] == node

    def size(self, node: str) -> int:
        return int(self.mask(node).sum())

    def children(self, node: str) -> list[str]:
        """Distinct child labels one level below node, a self-named child is a class."""
        level = self.level_of(node)
        if level + 1 >= len(self.levels):
            return []
        return sorted(str(c) for c in set(self.levels[level + 1][self.mask(node)]) - {""})

    def internal_nodes(self, level: int) -> list[str]:
        """Nodes of level, 0-based over label_cols, that split into two children or more."""
        return sorted(str(n) for n in self._at_level[level] if len(self.children(n)) >= 2)

    def partition(self, node: str, cut: str) -> tuple[np.ndarray, np.ndarray]:
        """(cell mask, class labels) scored at node under cut, classes align to the mask."""
        if cut not in CUTS:
            raise ValueError(f"cut must be one of {CUTS}, got {cut!r}")
        subtree = self.mask(node)

        if cut == LEAVES:
            labels = self.finest
        else:
            if node in self._virtual:
                raise ValueError(f"virtual node '{node}' has no children cut")
            level = self.level_of(node)
            if level + 1 >= len(self.levels):
                raise ValueError(f"node '{node}' sits at the finest level: no children cut")
            labels = self.levels[level + 1]

        keep = subtree & (labels != "")
        return keep, labels[keep]

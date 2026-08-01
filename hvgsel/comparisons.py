"""
Score objects name one per-gene score vector, Comparison objects name two.

BUILDERS contains the named experiments of the study.
"""

# IMPORTS
from __future__ import annotations

from dataclasses import dataclass

from hvgsel.scores import SUPERVISED
from hvgsel.taxonomy import CHILDREN, CUTS, LEAVES, ROOT


# ---------------------------------------------------------------------


TAG = {"f_stat": "F", "kw": "KW", "vst": "VST"}
ALL_NR = ("F-NR", "KW-NR", "VST(v)-NR", "VST(v_0)-NR")
ALL_NN = ("F-NN", "KW-NN", "VST(v_0)-NN")


@dataclass(frozen=True)
class Score:
    """
    One per-gene score vector: method evaluated on a node's cells.

    Supervised methods ('f_stat', 'kw') use cut='children' or 'leaves'.
    VST use cut=None and loess_from='node_name' (or None by default)
    """

    method: str
    node: str = ROOT
    cut: str | None = None
    loess_from: str | None = None

    def __post_init__(self):
        if self.method in SUPERVISED:
            if self.loess_from is not None:
                raise ValueError(f"{self.method} takes no loess_from")
            object.__setattr__(self, "cut", self.cut or CHILDREN)
            if self.cut not in CUTS:
                raise ValueError(f"cut must be one of {CUTS}, got {self.cut!r}")
        elif self.method == "vst":
            if self.cut is not None:
                raise ValueError("vst is unsupervised and takes no cut")
            if self.loess_from == self.node:  # the local fit, under another name
                object.__setattr__(self, "loess_from", None)
        else:
            raise ValueError(f"unknown method {self.method!r}")

    @property
    def spec(self) -> dict:
        """Canonical description, used both as the cache key and as stored metadata."""
        return {"method": self.method, "node": self.node, "cut": self.cut,
                "loess_from": self.loess_from}

    @property
    def name(self) -> str:
        qualifier = self.cut if self.method in SUPERVISED else (self.loess_from or "local")
        return f"{TAG[self.method]}[{self.node} | {qualifier}]"

    def latex(self, symbol: str = "v") -> str:
        """LaTeX for the symbol of this score's node (v, v' or v_0)."""
        if self.method in SUPERVISED:
            stat = "F" if self.method == "f_stat" else "H"
            suffix = r"(\mathrm{leaves})" if self.cut == LEAVES else ""
            return rf"{stat}_{{g,{symbol}}}{suffix}"
        loess = symbol if self.loess_from is None else ("v_0" if self.loess_from == ROOT else "v''")
        return rf"\mathrm{{VST}}_{{g,{symbol}}}^{{(\widetilde{{f}}_{{{loess}}})}}"


@dataclass(frozen=True)
class Comparison:
    """
    Two scores put side by side, plus the symbols each node is written with in the figures.
    """

    key: str
    a: Score
    b: Score
    sym_a: str = "v"
    sym_b: str = "v_0"

    @property
    def latex_a(self) -> str:
        return self.a.latex(self.sym_a)

    @property
    def latex_b(self) -> str:
        return self.b.latex(self.sym_b)

    @property
    def nodes(self) -> str:
        seen = {}
        for symbol, score in ((self.sym_a, self.a), (self.sym_b, self.b)):
            seen.setdefault(symbol, score.node)
        return ", ".join(rf"${symbol}=$'{node}'" for symbol, node in seen.items())

    @property
    def title(self) -> str:
        return rf"[{self.key}]  ${self.latex_a}$  vs  ${self.latex_b}$ | {self.nodes}"

    @property
    def spec(self) -> dict:
        return {"key": self.key, "a": self.a.spec, "b": self.b.spec}


# --- Experiments -----------------------------------------------------

# NR: node vs root, node scores on children and root scores on leaves for supervised methods.
# NN: node vs node, both nodes score on children for supervised methods.
# VST(v_0): share_loess=True, keep root's mean-variance LOESS at the node.
# VST(v): share_loess=False, refit at the node.

# Note: control is a virtual node with no children and is summarised on the leaf cut instead.

BUILDERS = {
    "F-NR":        lambda v, w, cut: Comparison("F-NR", Score("f_stat", v, cut),
                                                Score("f_stat", ROOT, LEAVES)),
    "KW-NR":       lambda v, w, cut: Comparison("KW-NR", Score("kw", v, cut),
                                                Score("kw", ROOT, LEAVES)),
    "VST(v)-NR":   lambda v, w, cut: Comparison("VST(v)-NR", Score("vst", v),
                                                Score("vst", ROOT)),
    "VST(v_0)-NR": lambda v, w, cut: Comparison("VST(v_0)-NR", Score("vst", v, loess_from=ROOT),
                                                Score("vst", ROOT)),
    "F-NN":        lambda v, w, cut: Comparison("F-NN", Score("f_stat", v, cut),
                                                Score("f_stat", w, cut), "v", "v'"),
    "KW-NN":       lambda v, w, cut: Comparison("KW-NN", Score("kw", v, cut),
                                                Score("kw", w, cut), "v", "v'"),
    "VST(v_0)-NN": lambda v, w, cut: Comparison("VST(v_0)-NN", Score("vst", v, loess_from=ROOT),
                                                Score("vst", w, loess_from=ROOT), "v", "v'"),
}


def keys_for(methods, keys=ALL_NR) -> tuple[str, ...]:
    """Experiments of keys a dataset can afford, given its Dataset.methods."""
    return tuple(key for key in keys if BUILDERS[key](ROOT, ROOT, CHILDREN).a.method in methods)


def build(keys, node: str, other: str = ROOT, cut: str = CHILDREN) -> list[Comparison]:
    """Named comparisons keys for node (and other, where the experiment needs two)."""
    unknown = set(keys) - set(BUILDERS)
    if unknown:
        raise KeyError(f"no builder for {sorted(unknown)}; have {sorted(BUILDERS)}")
    return [BUILDERS[key](node, other, cut) for key in keys]


def node_vs_root(node: str, keys=ALL_NR, cut: str = CHILDREN) -> list[Comparison]:
    """Every node-vs-root comparison of keys, for one node."""
    return build(keys, node, cut=cut)


def node_vs_node(node: str, other: str, keys=ALL_NN, cut: str = CHILDREN) -> list[Comparison]:
    """Every node-vs-node comparison of keys, for one node pair."""
    return build(keys, node, other, cut)


def cl(node: str = ROOT, method: str = "f_stat") -> Comparison:
    """F-CL / KW-CL: children cut vs leaf cut at one node."""
    symbol = "v_0" if node == ROOT else "v"
    return Comparison(f"{TAG[method]}-CL", Score(method, node, CHILDREN),
                      Score(method, node, LEAVES), symbol, symbol)
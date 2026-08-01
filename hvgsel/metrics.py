"""
Ranking-agreement statistics between the two score vectors of a comparison.

changed% and Jaccard read the top-n selection, Spearman, Kendall and RBO the full ranking.
"""

# IMPORTS
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from hvgsel.comparisons import Comparison
from hvgsel.store import digest


# ---------------------------------------------------------------------


STAT_NAMES = ("changed%", "Jaccard", "Spearman", "Kendall", "RBO")
RBO_P = 0.98


def rbo(order_a, order_b, k: int, p: float = RBO_P) -> float:
    """
    Rank-biased overlap truncated at k: top-weighted similarity of two orderings, in [0, 1].
    """
    seen_a, seen_b, total = set(), set(), 0.0
    for depth in range(k):
        seen_a.add(order_a[depth])
        seen_b.add(order_b[depth])
        total += (p ** depth) * (len(seen_a & seen_b) / (depth + 1))
    return (1 - p) * total


def ranking_stats(a, b, n_top: int) -> dict:
    """
    Selection overlap at the top-n_top, plus global rank agreement.

    Note: ranks handle the +inf of a perfect separator, so no gene is dropped.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    order_a = np.argsort(-a, kind="stable")
    order_b = np.argsort(-b, kind="stable")
    top_a, top_b = set(order_a[:n_top].tolist()), set(order_b[:n_top].tolist())
    kept = len(top_a & top_b)

    rank_a, rank_b = rankdata(-a), rankdata(-b)
    return {
        "kept": kept,
        "changed": n_top - kept,
        "changed%": (n_top - kept) / n_top * 100,
        "Jaccard": kept / len(top_a | top_b),
        "Spearman": float(np.corrcoef(rank_a, rank_b)[0, 1]),
        "Kendall": float(kendalltau(rank_a, rank_b).correlation),
        "RBO": rbo(order_a.tolist(), order_b.tolist(), k=n_top),
    }


@dataclass
class Result:
    """
    One comparison's two score vectors, its statistics, and the per-gene table behind them.
    """

    comparison: Comparison
    a: np.ndarray
    b: np.ndarray
    genes: np.ndarray
    symbols: np.ndarray
    n_top: int
    stats: dict = field(init=False)

    def __post_init__(self):
        self.a = np.asarray(self.a, float)
        self.b = np.asarray(self.b, float)
        self.stats = ranking_stats(self.a, self.b, self.n_top)

    @property
    def key(self) -> str:
        return self.comparison.key

    @property
    def table(self) -> pd.DataFrame:
        """Per-gene scores, ranks and which side selected it, rank_shift > 0 means better at a."""
        name_a, name_b = self.comparison.a.name, self.comparison.b.name
        rank_a, rank_b = rankdata(-self.a, method="min"), rankdata(-self.b, method="min")
        in_a = rank_a <= self.n_top
        in_b = rank_b <= self.n_top
        return pd.DataFrame({
            "gene": self.genes,
            "symbol": self.symbols,
            f"score[{name_a}]": self.a,
            f"score[{name_b}]": self.b,
            f"rank[{name_a}]": rank_a.astype(int),
            f"rank[{name_b}]": rank_b.astype(int),
            "rank_shift": (rank_b - rank_a).astype(int),
            "selected_in": np.where(in_a & in_b, "both",
                           np.where(in_a, "a only", np.where(in_b, "b only", "neither"))),
        })

    def movers(self, n: int = 12) -> pd.DataFrame:
        """Selected genes whose rank moves most between the two selections."""
        table = self.table[lambda d: d["selected_in"] != "neither"]
        order = table["rank_shift"].abs().sort_values(ascending=False).index
        columns = [c for c in table.columns if c.startswith("rank[")]
        return table.loc[order, ["symbol", *columns, "rank_shift", "selected_in"]].head(n).reset_index(drop=True)

    def summary(self) -> str:
        s = self.stats
        return (f"[{self.key}] top-{self.n_top}: kept {s['kept']}  changed {s['changed']} "
                f"({s['changed%']:.0f}%)  Jaccard {s['Jaccard']:.3f}  "
                f"Spearman {s['Spearman']:.3f}  Kendall {s['Kendall']:.3f}  RBO {s['RBO']:.3f}")

    def summary_short(self) -> str:
        s = self.stats
        return (f"changed {s['changed%']:.0f}%   J {s['Jaccard']:.2f}   "
                f"$\\rho$ {s['Spearman']:.2f}   RBO {s['RBO']:.2f}")

    def row(self) -> dict:
        """Flat record of this comparison, for the shared metrics table."""
        spec = self.comparison
        return {"id": digest({**spec.spec, "n_top": self.n_top}),
                "key": spec.key, "n_top": self.n_top,
                **{f"a_{k}": v for k, v in spec.a.spec.items()},
                **{f"b_{k}": v for k, v in spec.b.spec.items()},
                **{k: self.stats[k] for k in ("kept", "changed", *STAT_NAMES)}}

"""
Scorer: the data, its taxonomy and the results store, behind one interface.

scores(spec) computes a score vector once and reads it back from disk ever after.
compare(comparison) returns the statistics and logs them to the shared metrics table.
"""

# IMPORTS
from __future__ import annotations

import numpy as np
import pandas as pd

from hvgsel import scores as score_fns
from hvgsel.comparisons import ALL_NR, Comparison, Score, node_vs_root
from hvgsel.datasets import stratified_draw
from hvgsel.metrics import STAT_NAMES, Result
from hvgsel.scores import SUPERVISED
from hvgsel.store import Store, dataset_id, score_id
from hvgsel.taxonomy import CHILDREN, Taxonomy


# ---------------------------------------------------------------------


DEFAULT_N_TOP = 2000


def select_genes(adata, taxonomy: Taxonomy, spec: Score, n_top: int, params: dict | None = None):
    """Gene names catnap's own selector returns for spec: the HVG set, not a ranking."""
    params = dict(params or {})
    if spec.method in SUPERVISED:
        mask, classes = taxonomy.partition(spec.node, spec.cut)
        # f_statistic and kruskal_wallis only read X and var_names, so no copy is needed.
        # At the root that avoids materializing the whole matrix a second time.
        subset = adata if mask.all() else adata[mask]
        return score_fns.catnap_genes(subset, spec.method, labels=classes,
                                      n_top=int(n_top), **params)
    # seurat_v3 writes its results into .var, so it gets its own copy
    return score_fns.catnap_genes(adata[taxonomy.mask(spec.node)].copy(), "vst",
                                  n_top=int(n_top), **params)


class Scorer:
    """
    Parameters
    ----------
    adata      : AnnData with raw counts in .X, every method requires counts.
    label_cols : taxonomy columns of adata.obs, coarse -> fine.
    results    : directory holding the score cache and the metrics table.
    n_top      : selection size the comparisons are evaluated at.
    params     : per-method overrides of catnap's HVG defaults,
                 e.g. {'vst': {'span_loess': 0.3}, 'f_stat': {'min_cells_per_child': 2}}.
    """

    def __init__(self, adata, label_cols, results="results", dataset="dataset",
                 n_top: int = DEFAULT_N_TOP, params: dict | None = None):
        self.adata = adata
        self.taxonomy = Taxonomy(adata.obs, label_cols)
        self.n_top = int(n_top)
        self.params = {method: dict(params.get(method, {})) for method in score_fns.METHODS} \
            if params else {method: {} for method in score_fns.METHODS}
        self.dataset = dataset_id(dataset, adata)
        self.store = Store(results, self.dataset, info={
            "name": dataset, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
            "label_cols": list(label_cols), "n_top": self.n_top, "params": self.params,
        })
        self._memo: dict[str, dict] = {}

    # --- Data views --------------------------------------------------
    @property
    def genes(self) -> np.ndarray:
        return np.asarray(self.adata.var_names, dtype=str)

    @property
    def symbols(self) -> np.ndarray:
        var = self.adata.var
        column = var["feature_name"] if "feature_name" in var.columns else self.adata.var_names
        return np.asarray(column, dtype=str)

    # --- Scores ------------------------------------------------------
    def scores(self, spec: Score) -> np.ndarray:
        """Per-gene score vector for spec, from memory, then disk, then computed."""
        return self._record(spec)["scores"]

    def trend(self, node: str):
        """The seurat_v3 mean-variance LOESS fitted on node's cells, catnap's shared trend."""
        record = self._record(Score("vst", node))
        return record["trend_x"], record["trend_y"]

    def cache_key(self, spec: Score) -> dict:
        """spec plus the parameters that change its scores, what the cached file is keyed on."""
        return {**spec.spec, **score_fns.result_params(spec.method, self.params[spec.method])}

    def _compute(self, spec: Score, X, classes=None, trend=None) -> tuple[np.ndarray, dict, tuple]:
        """spec's score vector over an explicit matrix -> (scores, metadata, LOESS knots)."""
        params = self.params[spec.method]
        if spec.method == "f_stat":
            values, info = score_fns.f_stat_scores(X, classes, **params)
        elif spec.method == "kw":
            values, info = score_fns.kw_scores(X, classes, **params)
        else:
            values, fitted, info = score_fns.vst_scores(X, trend=trend, **params)
            return values, info, fitted
        return values, info, ()

    def _record(self, spec: Score) -> dict:
        sid = score_id(self.cache_key(spec))
        if sid in self._memo:
            return self._memo[sid]

        stored = self.store.load(sid)
        if stored is None:
            if spec.method in SUPERVISED:
                mask, classes = self.taxonomy.partition(spec.node, spec.cut)
                trend = None
            else:
                mask, classes = self.taxonomy.mask(spec.node), None
                trend = None if spec.loess_from is None else self.trend(spec.loess_from)

            values, meta, fitted = self._compute(spec, self.adata.X[mask], classes, trend)
            arrays = {"scores": values}
            if spec.method == "vst" and spec.loess_from is None:  # descendants reuse this trend
                arrays |= {"trend_x": np.asarray(fitted[0]), "trend_y": np.asarray(fitted[1])}

            self.store.save(sid, {"genes": self.genes, **arrays},
                            {**spec.spec, "dataset": self.dataset, **meta})
            stored = self.store.load(sid)
        elif not np.array_equal(stored["genes"], self.genes):
            raise ValueError(f"stored score '{sid}' was computed on a different gene space")

        self._memo[sid] = stored
        return stored

    # --- Comparisons -------------------------------------------------
    def compare(self, comparison: Comparison, n_top: int | None = None) -> Result:
        """Run one comparison and append its statistics to the store's metrics table."""
        result = Result(comparison, self.scores(comparison.a), self.scores(comparison.b),
                        self.genes, self.symbols, int(n_top or self.n_top))
        self.store.log(result.row())
        return result

    def summary(self, nodes, keys=ALL_NR, cut: str = CHILDREN,
                n_top: int | None = None) -> pd.DataFrame:
        """
        Tidy table of every keys comparison for every node in nodes.

        Rows are node x key, columns the ranking statistics.
        """
        rows = []
        for node in nodes:
            for comparison in node_vs_root(node, keys, cut=cut):
                stats = self.compare(comparison, n_top).stats
                rows.append({"node": node, "key": comparison.key,
                             **{name: stats[name] for name in STAT_NAMES}})
        return pd.DataFrame(rows)

    # --- Checks ------------------------------------------------------
    def parity(self, spec: Score, n_top: int | None = None, max_cells: int | None = None,
               seed: int = 0) -> dict:
        """
        Check that ranking genes by our score vector and keeping the top-n reproduces exactly
        what the shipped select_hvgs returns on the same cells.

        max_cells runs the check on a random subset of the node's cells. Both sides see the same
        subset, so it still validates the assembly, and it stays affordable on large nodes.
        """
        n_top = int(n_top or self.n_top)
        supervised = spec.method in SUPERVISED

        if supervised:
            mask, classes = self.taxonomy.partition(spec.node, spec.cut)
        else:
            mask, classes = self.taxonomy.mask(spec.node), None

        if max_cells is not None and mask.sum() > max_cells:
            chosen = np.random.default_rng(seed).choice(np.flatnonzero(mask), max_cells,
                                                        replace=False)
            smaller = np.zeros_like(mask)
            smaller[np.sort(chosen)] = True
            if classes is not None:
                classes = classes[smaller[mask]]
            mask = smaller
            values = self._compute(spec, self.adata.X[mask], classes)[0]
        else:
            values = self.scores(spec)          # full node, reuse the cached vector

        subset = self.adata[mask] if supervised else self.adata[mask].copy()
        theirs = set(score_fns.catnap_genes(subset, spec.method, labels=classes,
                                            n_top=n_top, **self.params[spec.method]))
        ours = set(self.genes[np.argsort(-values, kind="stable")[:n_top]])
        return {"spec": spec.name, "n_cells": int(mask.sum()), "n_top": n_top,
                "identical": len(ours & theirs)}

    # --- Virtual nodes -----------------------------------------------
    def random_node(self, like_nodes, seed: int = 0, name: str | None = None) -> str:
        """
        Negative control: a pseudo-node the size of the median node in like_nodes.

        Drawn at random but stratified on the finest label, so only the subsampling differs
        from the whole dataset, with no cell-type structure at all.
        """
        target = int(np.median([self.taxonomy.size(node) for node in like_nodes]))
        fraction = target / self.taxonomy.n_cells
        drawn = stratified_draw(self.taxonomy.finest, fraction, np.random.default_rng(seed),
                                min_one=False)

        mask = np.zeros(self.taxonomy.n_cells, bool)
        mask[drawn] = True
        return self.taxonomy.add_virtual(name or f"random strat. (n={int(mask.sum())})", mask)

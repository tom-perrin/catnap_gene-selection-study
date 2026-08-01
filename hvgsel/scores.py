"""
Per-gene HVG scores, computed with catnap_core's own selection code.

Each function returns the full-length vector the shipped selector ranks genes by, so ranking
all genes by it and keeping the top-n reproduces catnap_core.hvg.<method>.select_hvgs exactly.
Scorer.parity checks that on real data.

Note: the per-gene maths is imported from catnap_core (_child_moments, _kruskal_H,
_normalized_variance), only the loop assembling a whole-genome vector lives here, because the
shipped selectors return gene names rather than scores.
"""

# IMPORTS
from __future__ import annotations

import numpy as np

from catnap_core.hvg import f_statistic as _f_mod
from catnap_core.hvg import kruskal_wallis as _kw_mod
from catnap_core.hvg import seurat_v3 as _vst_mod
from catnap_core.hvg import select_genes


# ---------------------------------------------------------------------


METHODS = ("f_stat", "kw", "vst")
SUPERVISED = ("f_stat", "kw")

CATNAP_NAME = {"f_stat": "f_statistic", "kw": "kruskal_wallis", "vst": "seurat_v3"}
DEFAULTS = {"f_stat": _f_mod.DEFAULTS, "kw": _kw_mod.DEFAULTS, "vst": _vst_mod.DEFAULTS}

# Parameters that change the scores, so part of a cached score's identity.
# Note: gene_block is absent on purpose, it sets how many genes are densified at once, not the result.
RESULT_PARAMS = {"f_stat": ("min_cells_per_child",), "kw": ("min_cells_per_child",),
                 "vst": ("span_loess",)}


def result_params(method: str, params: dict | None = None) -> dict:
    """Effective value of every result-affecting parameter, defaults filled in."""
    params = params or {}
    return {key: params.get(key, DEFAULTS[method][key]) for key in RESULT_PARAMS[method]}


Trend = tuple[np.ndarray, np.ndarray]  # (log10-mean, fitted log10-var), sorted by mean

# Widening ladder, used only when the seurat_v3 LOESS solver fails on a degenerate subtree
# (too few or too collinear genes -> 'svddc failed'). catnap_core would raise there.
_SPAN_LADDER = (0.5, 0.8, 1.0)


def _param(method: str, key: str, value):
    return DEFAULTS[method][key] if value is None else value


def f_stat_scores(X, classes, min_cells_per_child=None) -> tuple[np.ndarray, dict]:
    """One-way ANOVA F per gene, assembled exactly as hvg.f_statistic.select_hvgs does."""
    min_cells = max(2, int(_param("f_stat", "min_cells_per_child", min_cells_per_child)))
    classes = np.asarray(classes)
    n_genes = X.shape[1]

    n_cells = n_kept = 0
    totals = np.zeros(n_genes)          # V_g, running sum of counts over the kept classes
    within = np.zeros(n_genes)          # W_g, within-class variance
    per_class = []
    for label in np.unique(classes):
        size, sums, sumsq = _f_mod._child_moments(X, classes == label)
        if size < min_cells:
            continue
        mean = sums / size
        variance = np.maximum(sumsq / size - mean * mean, 0.0)
        n_cells += size
        n_kept += 1
        totals += sums
        within += size * variance
        per_class.append((size, mean))

    info = {"n_cells": int(n_cells), "n_classes": int(n_kept), "min_cells_per_child": min_cells}
    if n_kept < 2 or n_cells <= n_kept:
        return np.zeros(n_genes), info

    grand_mean = totals / n_cells
    between = np.zeros(n_genes)
    for size, mean in per_class:
        delta = mean - grand_mean
        between += size * delta * delta

    ms_between = between / (n_kept - 1)
    ms_within = within / (n_cells - n_kept)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = ms_between / ms_within
    silent = ms_within == 0
    scores[silent & (ms_between == 0)] = 0.0      # no signal, no noise
    scores[silent & (ms_between > 0)] = np.inf    # perfect separator
    return np.nan_to_num(scores, nan=0.0, posinf=np.inf), info


def kw_scores(X, classes, min_cells_per_child=None, gene_block=None) -> tuple[np.ndarray, dict]:
    """Kruskal-Wallis H per gene, assembled exactly as hvg.kruskal_wallis.select_hvgs does."""
    import scipy.sparse as sp

    min_cells = max(2, int(_param("kw", "min_cells_per_child", min_cells_per_child)))
    gene_block = max(1, int(_param("kw", "gene_block", gene_block)))
    classes = np.asarray(classes)
    n_genes = X.shape[1]

    kept = [c for c in np.unique(classes) if int(np.count_nonzero(classes == c)) >= min_cells]
    info = {"n_classes": len(kept), "min_cells_per_child": min_cells, "gene_block": gene_block}
    if len(kept) < 2:
        return np.zeros(n_genes), {**info, "n_cells": 0}

    keep = np.isin(classes, kept)
    X_kept = X[keep]
    if sp.issparse(X_kept):
        X_kept = X_kept.tocsc()
    n_cells = int(X_kept.shape[0])
    onehot = (classes[keep][:, None] == np.asarray(kept)[None, :]).astype(np.float64)
    sizes = onehot.sum(axis=0)

    scores = np.empty(n_genes, dtype=np.float64)
    for start in range(0, n_genes, gene_block):
        block = X_kept[:, start:start + gene_block]
        dense = block.toarray() if sp.issparse(block) else np.asarray(block, dtype=np.float64)
        scores[start:start + dense.shape[1]] = _kw_mod._kruskal_H(dense, onehot, sizes, n_cells)
    return scores, {**info, "n_cells": n_cells}


def vst_scores(X, trend: Trend | None = None, span_loess=None) -> tuple[np.ndarray, Trend, dict]:
    """
    Seurat v3 normalized variance per gene, straight from hvg.seurat_v3._normalized_variance:
    VST_{g,v} = 1/(N_v - 1) * sum_c z_{cg,v}^2, z the count centered on the gene's mean at v,
    divided by the LOESS-expected standard deviation sqrt(V~_{g,v}) and clipped at sqrt(N_v).

    trend reuses an already fitted LOESS (catnap's share_loess) instead of fitting on X.
    The trend actually used is returned, so descendants can reuse it.
    """
    span = float(_param("vst", "span_loess", span_loess))
    spans = [span] if trend is not None else [span, *(s for s in _SPAN_LADDER if s > span)]

    for attempt, current in enumerate(spans):
        try:
            scores, fitted = _vst_mod._normalized_variance(X, X.shape[0], trend, current)
        except ValueError:  # degenerate subtree: the LOESS solver gave up
            continue
        return scores, fitted, {"n_cells": int(X.shape[0]), "span_loess": current,
                                "loess_widened": attempt > 0, "loess_fallback": False}

    # still failing at the widest span: fit the trend here, dropping to a linear local fit then
    # to a global polynomial, and hand it back as a frozen trend
    fallback = _fallback_trend(X, span)
    scores, fitted = _vst_mod._normalized_variance(X, X.shape[0], fallback, span)
    return scores, fitted, {"n_cells": int(X.shape[0]), "span_loess": span,
                            "loess_widened": True, "loess_fallback": True}


def _fallback_trend(X, span: float) -> Trend:
    """Mean-variance trend for a subtree degree-2 LOESS cannot fit, as sorted np.interp knots."""
    from scanpy.preprocessing._utils import _get_mean_var
    from skmisc.loess import loess

    mean, var = _get_mean_var(X)
    usable = var > 0
    x, y = np.log10(mean[usable]), np.log10(var[usable])

    fitted = None
    for current in (s for s in (span, *_SPAN_LADDER) if s >= span):
        model = loess(x, y, span=current, degree=1)
        try:
            model.fit()
        except ValueError:
            continue
        fitted = np.asarray(model.outputs.fitted_values, dtype=np.float64)
        break
    if fitted is None:  # no local fit converges, a global least-squares trend always does
        fitted = (np.poly1d(np.polyfit(x, y, deg=min(2, x.size - 1)))(x) if x.size >= 2
                  else np.zeros_like(x))

    order = np.argsort(x)
    return x[order], fitted[order]


def catnap_genes(adata, method: str, labels=None, n_top: int = 2000, **params) -> list[str]:
    """The shipped selector's own output: the top-n_top gene names it would train on."""
    return select_genes(adata, CATNAP_NAME[method], params={"n_top_genes": int(n_top), **params},
                        labels=labels)

"""
The datasets the study runs on, and the loader that puts raw counts into adata.X.

Every HVG method needs raw counts, and each dataset keeps them elsewhere, in a layer or in .raw.
load_dataset normalizes that: counts in .X, gene symbols in var['feature_name'], deepest known
label per cell in obs['_finest'].

Note: Dataset.path is host-specific, see README's Data table for where the matrices come from.
"""

# IMPORTS
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from catnap_core.utils import finest_labels


# ---------------------------------------------------------------------


HERE = Path(__file__).resolve().parent.parent

ALL_METHODS = ("f_stat", "kw", "vst")


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    label_cols: list[str]
    raw_source: tuple[str, str | None]  # ("layer", key) or ("raw", None) or ("X", None)
    symbol_col: str | None              # var column holding gene symbols (None: var_names are symbols)
    config_file: str                    # catnap config used by the downstream accuracy experiment
    methods: tuple[str, ...] = ALL_METHODS   # score methods affordable on this dataset
    examples: dict = field(default_factory=dict)
    # examples: every node and pair the notebook draws a figure for, in one pass.
    #   nodes     node-vs-root, one figure each
    #   pairs     node-vs-node, one figure each
    #   cl_nodes  children-vs-leaf cut, needs two levels below the node

    @property
    def config_path(self) -> Path:
        return HERE / self.config_file


DATASETS = {
    "aifi": Dataset(
        name="aifi",
        path="/home/baia/data/SIRA-CT/2_gong/gong.h5ad",
        label_cols=["AIFI_L1", "AIFI_L2", "AIFI_L3"],
        raw_source=("raw", None),   # .X is z-scaled and unusable; counts live in .raw.X (uint16)
        symbol_col=None,
        config_file="config_aifi.yml",
        methods=("f_stat", "vst"),  # kw densifies the matrix and is too slow on AIFI
        examples=dict(nodes=["T cell", "Treg"],
                      pairs=[("Monocyte", "NK cell"), ("Naive CD4 T cell", "Naive CD8 T cell")],
                      cl_nodes=["root", "T cell"]),
    ),
    "hao": Dataset(
        name="hao",
        path="/home/baia/data/SIRA-CT/1_hao/hao.h5ad",
        label_cols=["celltype.l1", "celltype.l2", "celltype.l3"],
        raw_source=("layer", "corrected_counts"),  # on-disk .X is log-normalized
        symbol_col="feature_name",
        config_file="config_hao.yml",
        examples=dict(nodes=["CD4 T", "Treg"],
                      pairs=[("Mono", "NK"), ("CD4 T", "CD8 T")],
                      cl_nodes=["root", "CD4 T"]),
    ),
}


def stratified_draw(labels, fraction: float, rng, min_one: bool = True) -> np.ndarray:
    """
    Positions of a draw holding fraction of every label's cells, preserving the composition.

    min_one keeps a label that would round to nothing, what a cap on the cells wants.
    The random control wants the plain proportion instead, and lets a tiny label drop out.
    """
    parts = []
    for label in np.unique(labels[labels != ""]):
        positions = np.flatnonzero(labels == label)
        share = round(positions.size * fraction)
        take = min(positions.size, max(1, share) if min_one else int(share))
        if take:
            parts.append(rng.choice(positions, size=take, replace=False))
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def load_dataset(name: str, max_cells: int | None = None, seed: int = 0, verbose: bool = True):
    """
    Load name with raw counts in .X.

    max_cells caps the cells with a draw stratified on the finest label, None uses every cell.
    obs and var are read backed first, so only the retained cells are ever materialized.
    """
    dataset = DATASETS[name]
    backed = sc.read_h5ad(dataset.path, backed="r")
    finest = finest_labels(backed, dataset.label_cols)

    if max_cells is None or max_cells >= backed.n_obs:
        keep = np.arange(backed.n_obs)
    else:
        keep = np.sort(stratified_draw(finest, max_cells / backed.n_obs,
                                       np.random.default_rng(seed)))
    if verbose:
        print(f"{name}: {keep.size:,}/{backed.n_obs:,} cells from {dataset.path}")

    subset = backed[keep].to_memory()
    del backed

    kind, key = dataset.raw_source
    if kind == "layer":
        counts, var = subset.layers[key], subset.var.copy()
    elif kind == "raw":
        raw = subset.raw.to_adata()
        counts, var = raw.X, raw.var.copy()
    elif kind == "X":
        counts, var = subset.X, subset.var.copy()
    else:
        raise ValueError(f"unknown raw_source kind: {kind}")

    # float32, uint16 counts overflow as soon as they are squared
    adata = ad.AnnData(X=counts.astype(np.float32), obs=subset.obs.copy(), var=var)
    del subset

    symbols = adata.var[dataset.symbol_col] if dataset.symbol_col else adata.var_names
    adata.var["feature_name"] = np.asarray(symbols).astype(str)
    adata.obs["_finest"] = finest_labels(adata, dataset.label_cols)

    if verbose:
        print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes | "
              f"X counts in [{adata.X.min():.0f}, {adata.X.max():.0f}]")
    return adata

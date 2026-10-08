"""
Which atlas marker genes each run's selections kept, read back from the saved node models.

A node's markers are what marker.node_markers gives for its children, the set the marker method
would train that node on. Each arm is measured against them where its model decides: reselect
at the node, on the node's own selection, fixed_global on the root's selection, which it reuses
everywhere.

Also the per-class reading of one node from the saved predictions (class_recall), and the split
of a finer level's accuracy into routing and the node's own split (routing_split): the per-node
breakdowns score a node's split on cells routed through every node above it, so a loss there can
come from upstream.

Note: the selection depends on the HVG method, the budget and the split, not on the backend, so
one backend's runs are enough for the coverage. Nothing here trains.

Note: a node with a single child is left out of the coverage. Its model is constant and never
reads a gene, and a supervised method, which cannot score one class, keeps every gene there. On
AIFI that is the Level 1 Progenitor cell, whose only child is the Level 2 one.
"""

# IMPORTS
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from catnap_core import hierarchy

from hvgsel import marker
from hvgsel.runs import ARMS, TAG_PATTERN, Run


# ---------------------------------------------------------------------


def node_children(config_path) -> dict[str, list[str]]:
    """Every internal node's model path, root/B cell say, and its children's names."""
    out = {}

    def walk(path, node):
        children = hierarchy.children_of(node)
        if children:
            out[path] = [hierarchy.norm(child) for child in children]
        for child, below in children.items():
            walk(f"{path}/{hierarchy.norm(child)}", below)

    walk("root", hierarchy.root_item(hierarchy.load_config(config_path))[1])
    return out


def split_nodes(config_path) -> dict[str, list[str]]:
    """node_children without the single-child nodes, whose constant model reads no gene."""
    return {path: names for path, names in node_children(config_path).items() if len(names) > 1}


def node_genes(run: Run, arm: str) -> dict[str, list[str]]:
    """Model path -> the genes that node's model was trained on."""
    models = run.arm_dir(arm) / "models"
    return {str(meta.parent.relative_to(models)): json.loads(meta.read_text())["gene_names"]
            for meta in models.rglob("model_metadata.json")}


def _describe(run: Run) -> dict:
    parts = TAG_PATTERN.match(run.tag)
    return dict(tag=run.tag, hvg=parts["hvg"].replace("_shareloess", ""), n_hvg=int(parts["n_hvg"]),
                backend=parts["backend"], fold=run.meta.get("fold"))


def coverage_frame(runs, config_path) -> pd.DataFrame:
    """
    One row per run and internal node: how many of the node's markers each arm's genes hold.

    reselect and fixed are those fractions, delta their difference. lost names the markers the
    root selection holds and the node's own drops, gained the reverse.
    """
    rows = []
    for run in runs:
        own, root = node_genes(run, ARMS[0]), set(node_genes(run, ARMS[1])["root"])
        for path, children in split_nodes(config_path).items():
            wanted, kept = marker.node_markers(children), set(own[path])
            rows.append(dict(**_describe(run), path=path, node=path.rsplit("/", 1)[-1],
                             depth=path.count("/"), n_markers=len(wanted),
                             reselect=len(wanted & kept) / len(wanted),
                             fixed=len(wanted & root) / len(wanted),
                             lost=sorted(wanted & root - kept), gained=sorted(wanted & kept - root)))
    out = pd.DataFrame(rows)
    out["delta"] = out["reselect"] - out["fixed"]
    return out


def gene_frame(runs, config_path) -> pd.DataFrame:
    """
    One row per run and marker gene: whether the root selection holds it (fixed), and in what
    fraction of the nodes it marks the node's own selection does (reselect). The root marks every
    gene, so a gene naming a Level 1 population only is read at the root alone.
    """
    marks = {path: marker.node_markers(names) for path, names in split_nodes(config_path).items()}
    table = marker.marker_table()
    rows = []
    for run in runs:
        own, root = node_genes(run, ARMS[0]), set(node_genes(run, ARMS[1])["root"])
        for gene in sorted(set(table["gene"])):
            nodes = [path for path, genes in marks.items() if gene in genes]
            rows.append(dict(**_describe(run), gene=gene, n_nodes=len(nodes), fixed=float(gene in root),
                             reselect=np.mean([gene in own[path] for path in nodes]) if nodes else np.nan))
    return pd.DataFrame(rows)


def coverage_accuracy(coverage: pd.DataFrame, runs, levels) -> pd.DataFrame:
    """
    The per-node accuracy gain of every run beside its node's coverage change.

    A node at depth d is the one breakdown_L{d+1}_by_L{d} scores. The coverage is joined on the
    split, not the backend, so a logit and an lgbm run of one fold share it.
    """
    rows = []
    for run in runs:
        for by in range(len(levels) - 1):
            table = run.read(f"breakdown_{levels[by + 1]}_by_{levels[by]}")
            rows += [dict(**_describe(run), depth=by + 1, node=row.node, gain=row.delta,
                          n_test=row.n_test) for row in table.itertuples()]
    gains = pd.DataFrame(rows)
    keys = ["hvg", "n_hvg", "fold", "depth", "node"]
    one = coverage.drop_duplicates(keys)[[*keys, "path", "n_markers", "reselect", "fixed", "delta"]]
    return gains.merge(one, on=keys, how="left", validate="many_to_one")


def coverage_correlation(joined: pd.DataFrame) -> pd.DataFrame:
    """Spearman of the coverage change against the accuracy gain across nodes, per configuration
    and pooled per budget, both averaged over folds first."""
    nodes = joined.groupby(["n_hvg", "hvg", "backend", "node", "depth"])[["delta", "gain"]].mean()
    nodes = nodes.reset_index()
    rows = []
    for (n_hvg, hvg, backend), block in [*nodes.groupby(["n_hvg", "hvg", "backend"]),
                                         *(((n, "all", "all"), b) for n, b in nodes.groupby("n_hvg"))]:
        rho, p = spearmanr(block["delta"], block["gain"])
        rows.append(dict(n_hvg=n_hvg, hvg=hvg, backend=backend, nodes=len(block), spearman=rho, p=p))
    return pd.DataFrame(rows)


# --- Reading one node from the predictions ---------------------------

def _predictions(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.dir / "predictions.csv.gz", index_col="cell")


def class_recall(runs, level: str, node: str, child_level: str) -> pd.DataFrame:
    """
    At node, the recall of each child class under each arm, on the true cells of node that both
    arms routed into it: what the node's own model does, free of the levels above.
    """
    rows = []
    for run in runs:
        table = _predictions(run)
        inside = (table[f"true_{level}"] == node) & np.logical_and.reduce(
            [table[f"{arm}_{level}_pred"] == node for arm in ARMS])
        for child, cells in table[inside].groupby(f"true_{child_level}"):
            rows.append(dict(**_describe(run), node=node, child=child, n=len(cells),
                             **{arm: (cells[f"{arm}_{child_level}_pred"] == child).mean() for arm in ARMS}))
    out = pd.DataFrame(rows)
    out["delta"] = out[ARMS[0]] - out[ARMS[1]]
    return out


def routing_split(runs, level: str, node: str, child_level: str) -> pd.DataFrame:
    """
    The true cells of node, a level-`level` population, under each arm: the share routed to it,
    and the share of those its own model then splits right at child_level. Their product is the
    breakdown's accuracy for the node, up to the both-arms-routed filter.
    """
    rows = []
    for run in runs:
        table = _predictions(run)
        cells = table[table[f"true_{level}"] == node]
        row = dict(**_describe(run), node=node, n=len(cells))
        for arm in ARMS:
            routed = cells[f"{arm}_{level}_pred"] == node
            row[f"routed_{arm}"] = routed.mean()
            row[f"split_{arm}"] = (cells.loc[routed, f"{arm}_{child_level}_pred"]
                                   == cells.loc[routed, f"true_{child_level}"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)

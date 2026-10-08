"""
How many genes a node should select: accuracy against the budget, the overlap between a node's
selection and the global one, and each node's own accuracy against the budget beside its shape, for
f_statistic re-selected at every node with logit node models.

A budget run trains only the reselect arm, every node selecting its own top-n, under
runs_quantity/<dataset>/f_statistic_<n>hvg_logit_fold<k>/, on the same donor folds as runs_cv/:

    reselect/             catnap project dir (config.yml, models/)
    predictions.csv.gz    per held-out cell: donor, true label per level, the predicted ones
    metrics.json          hierarchical and per-level metrics, training time, sizes

Hierarchical metrics compare each cell's set of labels along its true path with the set along its
predicted path (Kiritchenko et al. 2006): precision and recall over the labels, pooled over cells
(micro) or averaged over leaf classes (macro), which weighs a population of tens of cells like one
of a hundred thousand.

Every node's f_statistic F vector is scored once per fold, on the whole genome, and cached under
runs_quantity/<dataset>/_scores/. A budget run takes each node's top-n from it with f_statistic's
own procedure (the RANKED selector), the exact selection f_statistic makes, so no budget scores the
genome again, and trains on only the genes some node takes.

The overlap curves need no training: the same F vectors ranked, against the root's leaf-cut
ranking, already cached per fold under runs_cv/<dataset>/_baselines.
"""

# IMPORTS
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from catnap_core import hierarchy
from catnap_core.hvg.base import register_hvg
from catnap_core.predict import predict_labels
from catnap_core.train import train_config
from catnap_core.utils import finest_labels

from hvgsel import scores as score_fns
from hvgsel.runs import BASELINES, Run, make_split, score_levels, truth_table, variant_config


# ---------------------------------------------------------------------


ARM = "reselect"
HVG = "f_statistic"
BACKEND = "logit"
RANKED = "f_statistic_ranked"   # the selector reading a node's cached F vector, see below
FIXED = "fixed_global"           # the arm reusing the root's leaf-cut top n at every node

# The fold's node scores: the genome's gene names, and every split node's f_statistic F vector by
# the set of its children, what RANKED reads. train_budget sets them before training, so a budget
# run takes its genes from them instead of scoring the genome at every node, and one scoring serves
# every budget.
_GENES: list = []
_SCORES: dict[frozenset, np.ndarray] = {}


def top_genes(scores: np.ndarray, genes, n_top: int) -> list[str]:
    """
    The n_top best genes of a F vector, ties broken exactly as hvg.f_statistic.select_hvgs breaks
    them: its argpartition then descending sort, whose order among equal F depends on n_top.
    """
    total = len(scores)
    if n_top <= 0 or total <= n_top:
        return [str(gene) for gene in genes]
    top = np.argpartition(scores, total - n_top)[total - n_top:]
    top = top[np.argsort(scores[top])[::-1]]
    return [str(genes[i]) for i in top]


@register_hvg(RANKED)
def select_ranked(adata, labels=None, **hvg_args) -> list[str]:
    """
    f_statistic's own selection at this node, from its cached F vector: the same genes in the same
    order, without scoring the genome again. The node is read off its children's labels.

    Note: the F vector is the whole genome's, scored on the node's training cells, so the
    selection does not depend on which genes adata was cut down to, as long as it holds them.
    """
    if labels is None:
        raise ValueError(f"hvg:{RANKED} reads the node off its children's labels, none were passed")
    children = frozenset(str(label) for label in np.unique(np.asarray(labels)))
    if len(children) < 2:   # as f_statistic does: no split to score, every gene, a constant model
        return adata.var_names.astype(str).tolist()
    if children not in _SCORES:
        raise KeyError(f"hvg:{RANKED}: no cached scores for a node with children {sorted(children)}")
    genes = top_genes(_SCORES[children], _GENES, int(hvg_args.get("n_top_genes", 2000)))
    missing = set(genes) - set(adata.var_names.astype(str))
    if missing:
        raise ValueError(f"hvg:{RANKED}: {len(missing)} selected gene(s) not in the data, cut too far")
    return genes


def _children_by_path(config_path) -> dict:
    """Every split node's model path and the set of its children's names."""
    out = {}

    def walk(path, node):
        below = hierarchy.children_of(node)
        if len(below) >= 2:
            out[path] = frozenset(hierarchy.norm(child) for child in below)
        for child, sub in below.items():
            walk(f"{path}/{hierarchy.norm(child)}", sub)

    walk("root", hierarchy.root_item(hierarchy.load_config(config_path))[1])
    return out


def use_scores(genes, scores: dict, config_path) -> None:
    """Make scores, F vectors by model path over genes, the ones RANKED reads."""
    children = _children_by_path(config_path)
    _GENES[:] = [str(gene) for gene in genes]
    _SCORES.clear()
    _SCORES.update({children[path]: np.asarray(vector) for path, vector in scores.items()})


# --- Hierarchical metrics --------------------------------------------

def _paths(table: pd.DataFrame, columns) -> list[frozenset]:
    """Per cell, the distinct labels along a path, the levels it reaches."""
    columns = table[list(columns)].to_numpy(dtype=object)
    return [frozenset(str(v) for v in row if isinstance(v, str) and v != "") for row in columns]


def _leaves(table: pd.DataFrame, columns) -> np.ndarray:
    """
    Per cell, the deepest label of a path: a cell predicted as a leaf of Level 2 carries no Level 3
    prediction, where its truth repeats the Level 2 label.
    """
    out = np.full(len(table), "", dtype=object)
    for column in columns:
        values = table[column].to_numpy(dtype=object)
        known = np.array([isinstance(v, str) and v != "" for v in values])
        out[known] = values[known]
    return out


def hierarchical_metrics(table: pd.DataFrame, levels, arm=ARM) -> dict:
    """
    Hierarchical precision, recall and F1 of arm's predictions in table, a predictions.csv.gz frame.

    micro pools the label counts over cells. macro averages a per-leaf-class F1, recall over the
    cells truly of the class and precision over the cells predicted as it, so a rare leaf weighs as
    much as an abundant one. exact is the share of cells whose whole path is right, their deepest
    label then.

    Note: a label spanning levels (Transitional B cell, L2 and L3) counts once on its path.
    """
    truth = _paths(table, [f"true_{level}" for level in levels])
    pred = _paths(table, [f"{arm}_{level}_pred" for level in levels])
    hit = np.array([len(t & p) for t, p in zip(truth, pred)], dtype=float)
    n_true = np.array([len(t) for t in truth], dtype=float)
    n_pred = np.array([len(p) for p in pred], dtype=float)
    precision, recall = hit.sum() / n_pred.sum(), hit.sum() / n_true.sum()

    true_leaf = _leaves(table, [f"true_{level}" for level in levels])
    pred_leaf = _leaves(table, [f"{arm}_{level}_pred" for level in levels])
    per_class = []
    for leaf in np.unique(true_leaf[true_leaf != ""]):
        mine, called = true_leaf == leaf, pred_leaf == leaf
        r = hit[mine].sum() / n_true[mine].sum()
        p = hit[called].sum() / n_pred[called].sum() if called.any() else 0.0
        per_class.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"hP": precision, "hR": recall, "hF": 2 * precision * recall / (precision + recall),
            "macro_hF": float(np.mean(per_class)), "exact": float((true_leaf == pred_leaf).mean())}


def arm_file(run: Run, stem: str, arm=ARM, suffix="") -> Path:
    """
    run's predictions.csv.gz or metrics.json for arm: so named for ARM, the arm's name appended
    for any other, predictions_fixed_global.csv.gz say, then suffix.
    """
    name = stem + ("" if arm == ARM else f"_{arm}") + suffix
    return run.dir / (f"{name}.csv.gz" if stem == "predictions" else f"{name}.json")


def _predictions_path(run: Run, arm=ARM) -> Path:
    """arm's own predictions file, or the shared one of a runs_cv/ run holding both arms."""
    path = arm_file(run, "predictions", arm)
    return path if path.exists() else run.dir / "predictions.csv.gz"


def run_metrics(run: Run, levels, arm=ARM) -> dict:
    """Every metric of run's saved predictions: hierarchical, then accuracy and macro-F1 per level."""
    table = pd.read_csv(_predictions_path(run, arm), index_col="cell")
    out = hierarchical_metrics(table, levels, arm)
    truth = table[[f"true_{level}" for level in levels]].rename(columns=lambda c: c[5:]).fillna("")
    pred = table[[f"{arm}_{level}_pred" for level in levels]].rename(columns=lambda c: c[len(arm) + 1:])
    for row in score_levels(truth, {arm: pred}, levels).itertuples():
        out[f"accuracy_{row.level}"], out[f"macro_f1_{row.level}"] = row.accuracy, row.macro_f1
    return out


def _split_cells(table: pd.DataFrame, levels, config_path):
    """
    (model path, depth, the column of its children's labels, mask of the cells truly below it) of
    every split node, walking the config's tree. The root holds every cell.
    """
    out = []

    def walk(path, name, node, depth):
        children = hierarchy.children_of(node)
        if len(children) >= 2:
            below = (np.ones(len(table), bool) if depth == 0
                     else (table[f"true_{levels[depth - 1]}"] == name).to_numpy())
            out.append((path, depth, levels[depth], below))
        for child_name, sub in children.items():
            walk(f"{path}/{hierarchy.norm(child_name)}", hierarchy.norm(child_name), sub, depth + 1)

    walk("root", "root", hierarchy.root_item(hierarchy.load_config(config_path))[1], 0)
    return out


def own_decisions(cells, run: Run, label_cols, arm=ARM, root="runs_quantity") -> pd.DataFrame:
    """
    Every split node's own decision in run, free of the nodes above: its saved model predicting
    every held-out cell truly below it, whichever node the tree routed the cell to. Accuracy, the
    share whose next label is right, macro-F1 over its children, which weighs a rare child like an
    abundant one, n_routed the cells the tree did send there, and
    agree, the share of those the replay predicts as the tree did: 1 when it scores the tree's own
    model on the tree's own input.

    Every model reads its genes through catnap's own alignment, as predict_labels does: a gene
    whose symbol resolves to the Ensembl id of an earlier column reads that column (CAST reads
    MIR583HG in AIFI), where model.predict would read the gene by name.

    cells holds run's held-out cells, more being left out: one fold's copy serves every budget.
    Written to root/<dataset>/_decisions/, read from there when present.

    Note: the tree's own predictions score a node only on the cells its parent routed right, at
    small budgets few and the easiest: 18 Treg cells reach the Treg node at 10 genes, 6,035 at 300.
    """
    from catnap_core.backends import load_model_dir
    from catnap_core.predict import _align, _build_canon_index
    from sklearn.metrics import f1_score
    path = _decisions_path(run, root)
    if path.exists():
        return pd.read_csv(path)
    levels = [f"L{k}" for k in range(1, len(label_cols) + 1)]
    table = _predictions(run, levels, arm)
    cells = cells[cells.obs_names.isin(table.index)]
    if cells.n_obs != len(table):
        raise ValueError(f"{run.tag}: {cells.n_obs:,} of its {len(table):,} held-out cells in cells")
    table = table.loc[cells.obs_names]
    canon_index = _build_canon_index(cells)
    rows = []
    for node_path, depth, child, below in _split_cells(table, levels, run.arm_dir(arm) / "config.yml"):
        model = load_model_dir(run.arm_dir(arm) / "models" / node_path)
        predicted, _ = model.predict_top(_align(cells[below], model.gene_names, canon_index))
        truth = table.loc[below, f"true_{child}"].astype(str).str.strip().to_numpy()
        routed = (np.ones(below.sum(), bool) if depth == 0 else
                  (table.loc[below, f"{arm}_{levels[depth - 1]}_pred"] == node_path.rsplit("/", 1)[-1]).to_numpy())
        tree = table.loc[below, f"{arm}_{child}_pred"].fillna("").astype(str).str.strip().to_numpy()
        rows.append({"path": node_path, "node": node_path.rsplit("/", 1)[-1], "depth": depth,
                     "n_cells": int(below.sum()), "n_routed": int(routed.sum()),
                     "accuracy": float((predicted == truth).mean()),
                     "macro_f1": float(f1_score(truth, predicted, labels=sorted(set(truth)),
                                                average="macro", zero_division=0)),
                     "agree": float((predicted[routed] == tree[routed]).mean()) if routed.any() else np.nan})
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def node_properties(table: pd.DataFrame, levels, config_path) -> pd.DataFrame:
    """
    Every split node's shape, from the true labels of table, the five folds' held-out cells
    together being the whole dataset: depth, arity (children holding cells), size (cells below
    it), smallest and largest child, and balance, the entropy of its children's shares over its
    maximum, log arity: 1 when even, near 0 when one child holds nearly every cell.
    """
    rows = []
    for path, depth, child, below in _split_cells(table, levels, config_path):
        counts = table.loc[below, f"true_{child}"].value_counts()
        share = counts.to_numpy() / counts.sum()
        rows.append({"path": path, "node": path.rsplit("/", 1)[-1], "depth": depth,
                     "branch": path.split("/")[1] if "/" in path else "root",
                     "arity": len(counts), "size": int(counts.sum()),
                     "smallest_child": int(counts.min()), "largest_child": int(counts.max()),
                     "balance": float(-(share * np.log(share)).sum() / np.log(len(counts)))})
    return pd.DataFrame(rows)


# --- Budget runs ------------------------------------------------------

def budget_tag(n_hvg: int, fold: int) -> str:
    return f"{HVG}_{n_hvg:04d}hvg_{BACKEND}_fold{fold}"


def fold_scores(adata, label_cols, config_path, split, root="runs_quantity", dataset="aifi"):
    """
    (genes, F vectors by model path) of split's training cells, read from root/dataset/_scores
    when there, computed and written there otherwise. Named by the split's key, so a file never
    serves another split.
    """
    path = Path(root) / dataset / "_scores" / f"fold{split.fold}_{split.key}.npz"
    if path.exists():
        return load_scores(path)
    genes, scores = node_scores(adata[split.train].copy(), label_cols, config_path)
    save_scores(genes, scores, path)
    return genes, scores


def train_budget(adata, dataset: str, label_cols, source_config, n_hvg: int, fold: int,
                 donor_col: str, root="runs_quantity", n_folds=5, seed=0, node_n=None,
                 variant="pernode", max_iter=None) -> Run:
    """
    Train the reselect arm alone at n_hvg genes per node, predict the fold's held-out donors and
    write predictions.csv.gz and metrics.json. Skipped when metrics.json is already there.

    Every node takes its genes from the fold's cached F vectors (RANKED), exactly the selection
    f_statistic itself makes, and trains on only the genes some node takes at this budget: no node
    scores the genome, and no node's cells are copied with every gene.

    node_n maps a node name to its own budget, every other node keeping n_hvg: the per-node count
    the overlap curves suggest. Its run is tagged by variant, one tag per set of budgets.

    max_iter, when given, caps every node's lbfgs iterations instead of catnap's default 1000.
    """
    tag = budget_tag(n_hvg, fold) if node_n is None else f"{HVG}_{variant}_{BACKEND}_fold{fold}"
    run = Run(root, dataset, tag)
    if (run.dir / "metrics.json").exists():
        print(f"{tag}: already scored, skipping")
        return run
    config = variant_config(source_config, HVG, n_hvg, backend=BACKEND)
    _set_selector(config["root"], RANKED)
    if max_iter:
        _set_backend_params(config["root"], max_iter=int(max_iter))
    if node_n:
        _set_node_budgets(config["root"], "root", node_n)
    project = run.arm_dir(ARM)
    if (project / "models").exists():
        shutil.rmtree(project / "models")   # a crashed run's partial tree
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))

    split = make_split(adata, label_cols, project / "config.yml", donor_col=donor_col,
                       n_folds=n_folds, fold=fold, seed=seed)
    all_genes, scores = fold_scores(adata, label_cols, project / "config.yml", split, root, dataset)
    use_scores(all_genes, scores, project / "config.yml")
    budget = {path: int((node_n or {}).get(path.rsplit("/", 1)[-1], n_hvg)) for path in scores}
    genes = sorted({gene for path, vector in scores.items()
                    for gene in top_genes(vector, all_genes, budget[path])})
    adata_train = adata[split.train][:, genes].copy()
    start = time.time()
    train_config(adata_train, project, label_cols, reselect_hvg=True, verbose=1)
    seconds = time.time() - start
    del adata_train

    adata_test = adata[split.test].copy()
    pred = predict_labels(adata_test, project)
    truth = truth_table(adata_test, label_cols)
    table = truth.add_prefix("true_")
    table.insert(0, "donor", adata_test.obs[donor_col].astype(str).to_numpy())
    table = table.join(pred.add_prefix(f"{ARM}_")).rename_axis("cell")
    table.to_csv(run.dir / "predictions.csv.gz")

    levels = list(truth.columns)
    meta = {"dataset": dataset, "tag": tag, "n_hvg": n_hvg, "node_n": node_n,
            "variant": variant if node_n else None, "fold": fold, "max_iter": max_iter,
            "selector": RANKED, "n_genes_trained": len(genes), "train_seconds": seconds,
            "label_cols": list(label_cols), **split.meta}
    run.write_meta(meta)
    (run.dir / "metrics.json").write_text(json.dumps({**meta, **run_metrics(run, levels)}, indent=1))
    return run


def _set_selector(node: dict, hvg: str) -> None:
    """Every node's HVG method set to hvg, its parameters kept."""
    if "hvg" in node:
        node["hvg"] = hvg
    for below in (node.get("children") or {}).values():
        _set_selector(below, hvg)


def _set_backend_params(node: dict, **params) -> None:
    """params merged into every node's backend parameters, walking the config's tree."""
    if "backend_params" in node:
        node["backend_params"] = {**(node["backend_params"] or {}), **params}
    for below in (node.get("children") or {}).values():
        _set_backend_params(below, **params)


def _set_node_budgets(node: dict, name: str, node_n: dict) -> None:
    """n_top_genes of every node node_n names set to its value, walking the config's tree."""
    if name in node_n and "hvg_params" in node:
        node["hvg_params"]["n_top_genes"] = int(node_n[name])
    for child, below in (node.get("children") or {}).items():
        _set_node_budgets(below, hierarchy.norm(child), node_n)


def metrics_table(roots=("runs_quantity",), extra=()) -> pd.DataFrame:
    """
    Every scored arm's metrics, one row per (run, arm): metrics.json for ARM, metrics_<arm>.json
    for another, an arm column telling them apart. extra are runs scored from elsewhere, the
    runs_cv/ and runs/ trees of the same configuration, as (Run, n_hvg, fold), their ARM only.
    """
    rows = []
    for root in roots:
        for path in sorted(Path(root).glob("*/*/metrics*.json")):
            if "maxiter" in path.name:   # the metrics a retraining replaced
                continue
            arm = path.stem[len("metrics_"):] if path.stem != "metrics" else ARM
            row = json.loads(path.read_text())
            rows.append({"selector": HVG, **row, "arm": arm,
                         "fit_seconds": _fit_seconds(Run(root, row["dataset"], row["tag"]), arm)})
    for run, n_hvg, fold in extra:
        levels = [f"L{k}" for k in range(1, len(run.meta["label_cols"]) + 1)]
        rows.append({"tag": run.tag, "n_hvg": n_hvg, "fold": fold, "node_n": None, "arm": ARM,
                     "train_seconds": _arm_seconds(run), "fit_seconds": _fit_seconds(run),
                     **run_metrics(run, levels)})
    return pd.DataFrame(rows)


def _fit_seconds(run: Run, arm=ARM) -> float:
    """
    The node models' own fitting time summed over the tree, from catnap's per-node records: what a
    budget costs whatever selected the genes, where wall time also counts how they were selected.
    """
    return float(sum(json.loads(path.read_text()).get("fit_metadata", {}).get("train_seconds", 0)
                     for path in (run.arm_dir(arm) / "models").rglob("training_metadata.json")))


def _arm_seconds(run: Run, arm=ARM) -> float:
    """Wall time of a saved run's arm, from catnap's training record."""
    record = json.loads((run.arm_dir(arm) / "training_metadata.json").read_text())
    start, end = (pd.Timestamp(record[key]) for key in ("started_at", "finished_at"))
    return (end - start).total_seconds()


# --- Convergence ------------------------------------------------------

# A retraining worker's data, set in the parent before the workers fork so they share it unread:
# the dataset, every cell's finest label, and each fold's training mask
_FORKED: dict = {}

MODEL_FILES = ("logit_model.joblib", "model_config.json", "model_metadata.json", "training_metadata.json")
CAPPED_DIR = "models_maxiter1000"   # beside models/, the capped models a retraining replaced


def import_run(source: Run, n_hvg: int, fold: int, root="runs_quantity", arm=ARM, tag=None,
               selector=HVG) -> Run:
    """
    A run trained elsewhere copied in as the budget run of (n_hvg, fold), tag naming it otherwise:
    its arm, its predictions down to that arm, run.json and the arm's metrics as train_budget
    writes them. The source is left as it is. Skipped when the arm's copy is already there.

    Note: the 300-HVG folds of runs_cv/ and the 2000-HVG fold 0 of runs/ select with f_statistic
    itself where a budget run reads its cached F vectors: the same genes, checked at every node.
    """
    run = Run(root, source.dataset, tag or budget_tag(n_hvg, fold))
    if arm_file(run, "metrics", arm).exists():
        return run
    shutil.copytree(source.arm_dir(arm), run.arm_dir(arm), dirs_exist_ok=True)
    table = pd.read_csv(source.dir / "predictions.csv.gz", index_col="cell")
    keep = ["donor"] + [c for c in table.columns if c.startswith("true_") or c.startswith(f"{arm}_")]
    table[keep].to_csv(arm_file(run, "predictions", arm))
    levels = [c[5:] for c in table.columns if c.startswith("true_")]
    origin = source.meta
    if origin.get("fold", fold) != fold:
        raise ValueError(f"{source.tag} is fold {origin['fold']}, not {fold}")
    meta = {"dataset": source.dataset, "tag": run.tag, "n_hvg": n_hvg, "node_n": None, "variant": None,
            "fold": fold, "max_iter": None, "selector": selector, "n_genes_trained": None,
            "train_seconds": _arm_seconds(source, arm), "copied_from": str(source.dir),
            **{k: origin[k] for k in ("label_cols", "seed", "test_frac", "subsample", "train_key", "n_train",
                                      "n_test", "donor_col", "n_folds", "n_train_donors", "n_test_donors")
               if k in origin}}
    if not (run.dir / "run.json").exists():
        run.write_meta(meta)
    arm_file(run, "metrics", arm).write_text(json.dumps({**meta, **run_metrics(run, levels, arm)}, indent=1))
    return run


def train_fixed(adata, dataset: str, label_cols, source_config, n_hvg: int, fold: int, donor_col: str,
                root="runs_quantity", root_cv="runs_cv", n_folds=5, seed=0, max_iter=None) -> Run:
    """
    The FIXED arm of the budget run of (n_hvg, fold): the top n_hvg of the root's leaf-cut
    f_statistic ranking, cached under root_cv/<dataset>/_baselines, reused at every node, as
    runs.Run.train trains it. Writes predictions_fixed_global.csv.gz and
    metrics_fixed_global.json beside the reselect arm's files. Skipped when they are there.
    """
    import anndata as ad
    run = Run(root, dataset, budget_tag(n_hvg, fold))
    if arm_file(run, "metrics", FIXED).exists():
        return run
    config = variant_config(source_config, HVG, n_hvg, backend=BACKEND)
    if max_iter:
        _set_backend_params(config["root"], max_iter=int(max_iter))
    project = run.arm_dir(FIXED)
    if (project / "models").exists():
        shutil.rmtree(project / "models")   # a crashed run's partial tree
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))

    split = make_split(adata, label_cols, project / "config.yml", donor_col=donor_col,
                       n_folds=n_folds, fold=fold, seed=seed)
    genes = [str(gene) for gene in global_ranking(root_cv, dataset, split.key)[:n_hvg]]
    rows = np.flatnonzero(split.train)
    # the genes taken first, so no copy holds every gene of the training cells
    adata_train = ad.AnnData(X=adata.X[:, adata.var_names.get_indexer(genes)][rows],
                             obs=adata.obs.iloc[rows][list(label_cols)].copy(), var=pd.DataFrame(index=genes))
    start = time.time()
    train_config(adata_train, project, label_cols, reselect_hvg=False, verbose=1)
    seconds = time.time() - start
    del adata_train

    adata_test = adata[split.test].copy()
    pred = predict_labels(adata_test, project)
    truth = truth_table(adata_test, label_cols)
    table = truth.add_prefix("true_")
    table.insert(0, "donor", adata_test.obs[donor_col].astype(str).to_numpy())
    table = table.join(pred.add_prefix(f"{FIXED}_")).rename_axis("cell")
    table.to_csv(arm_file(run, "predictions", FIXED))
    meta = {"dataset": dataset, "tag": run.tag, "n_hvg": n_hvg, "node_n": None, "fold": fold,
            "max_iter": max_iter, "selector": HVG, "n_genes_trained": len(genes), "train_seconds": seconds,
            "label_cols": list(label_cols), **split.meta}
    arm_file(run, "metrics", FIXED).write_text(
        json.dumps({**meta, **run_metrics(run, list(truth.columns), FIXED)}, indent=1))
    return run


def _train_fixed(job) -> dict:
    """train_fixed in a forked worker, on the parent's data; an error is returned, not raised."""
    dataset, label_cols, config_path, n_hvg, fold, donor_col, max_iter = job
    out = {"n_hvg": n_hvg, "fold": fold, "arm": FIXED}
    try:
        start = time.time()
        train_fixed(_FORKED["adata"], dataset, label_cols, config_path, n_hvg, fold, donor_col, max_iter=max_iter)
        out["seconds"] = time.time() - start
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def arm_runs(root="runs_quantity", dataset="aifi") -> list:
    """
    (Run, n_hvg, fold, arm, selector) of every scored arm under root of a run giving every node
    the same genes or the same number of them: each budget's reselect and fixed_global arms, and
    the atlas markers' runs copied in.
    """
    out = []
    for path in sorted((Path(root) / dataset).glob("*/metrics*.json")):
        if "maxiter" in path.name:
            continue
        row = json.loads(path.read_text())
        if row.get("node_n"):
            continue
        arm = path.stem[len("metrics_"):] if path.stem != "metrics" else ARM
        out.append((Run(root, dataset, row["tag"]), row["n_hvg"], row["fold"], arm, row.get("selector", HVG)))
    return out


def _logit(node_dir: Path):
    """A saved node's sklearn LogisticRegression, None for a constant model."""
    import joblib
    path = node_dir / "logit_model.joblib"
    pipeline = joblib.load(path) if path.exists() else None
    return None if pipeline is None else pipeline.named_steps["clf"]


def capped_nodes(run: Run, max_iter: int, arm=ARM) -> list[str]:
    """
    run's node models whose lbfgs stopped at its iteration cap, a cap below max_iter: the ones
    retraining at max_iter could still move. A node already trained at max_iter stays.
    """
    out = []
    models = run.arm_dir(arm) / "models"
    for record in sorted(models.rglob("training_metadata.json")):
        clf = _logit(record.parent)
        if clf is not None and clf.max_iter < max_iter and int(np.max(clf.n_iter_)) >= clf.max_iter:
            out.append(record.parent.relative_to(models).as_posix())
    return out


def _config_node(config_path, node_path: str):
    """(config name, node) at a model path, walking the config's tree by normalized names."""
    name, node = hierarchy.root_item(hierarchy.load_config(config_path))
    for step in node_path.split("/")[1:]:
        name, node = next((child, below) for child, below in hierarchy.children_of(node).items()
                          if hierarchy.norm(child) == step)
    return name, node


def _node_data(cells: np.ndarray, genes: list):
    """
    The cells' counts at the genes alone, an AnnData: a large node takes the genes first, so
    neither copy holds every gene of a million cells.
    """
    import anndata as ad
    adata = _FORKED["adata"]
    columns = adata.var_names.get_indexer(genes)
    X = adata.X[:, columns][cells] if cells.size > 200_000 else adata.X[cells][:, columns]
    return ad.AnnData(X=X, obs=pd.DataFrame(index=adata.obs_names[cells]), var=pd.DataFrame(index=list(genes)))


def _node_cells(fold: int, name: str, node: dict, mask="train"):
    """
    A node's training cells of the fold, as indices into the forked data, and their classes, as
    catnap's _train_node builds them: each child's subtree, the node's own label kept as itself.
    mask names another of the fold's forked masks, "validation" say.
    """
    finest = _FORKED["finest"]
    cells = np.flatnonzero(_FORKED[mask][fold] & np.isin(finest, list(hierarchy.subtree_labels(name, node))))
    target = np.full(cells.size, "", dtype=object)
    for child, labels in hierarchy.child_label_sets(node).items():
        target[np.isin(finest[cells], list(labels))] = hierarchy.norm(child)
    target[finest[cells] == hierarchy.norm(name)] = hierarchy.norm(name)
    return cells, target


def _retrain(job) -> dict:
    """
    One node model of a run trained again as train_config trains it, its cells, classes, genes and
    parameters the saved model's, max_iter aside. The capped model moves to CAPPED_DIR, unless an
    earlier retraining already put one there, and the node's training record is rewritten.

    check=True only compares instead: the node retrained at its own cap, in a scratch directory.
    """
    from types import SimpleNamespace
    from catnap_core.backends import load_model_dir
    from catnap_core.train import train
    from catnap_core.training_metadata import TrainingRecorder
    run_dir, fold, node_path, max_iter, check, arm = job
    out = {"run": run_dir, "arm": arm, "node": node_path, "max_iter": max_iter}
    try:
        project = Path(run_dir) / arm
        node_dir = project / "models" / node_path
        name, node = _config_node(project / "config.yml", node_path)
        cells, target = _node_cells(fold, name, node)
        record = json.loads((node_dir / "training_metadata.json").read_text())
        if cells.size != record["fit_metadata"]["n_training_cells"]:
            raise RuntimeError(f"{cells.size:,} training cells, the saved model trained on "
                               f"{record['fit_metadata']['n_training_cells']:,}")

        old = load_model_dir(node_dir)
        before = _logit(node_dir)
        if not check and before.max_iter >= max_iter:   # retrained since it was queued
            return out | {"skipped": True}
        params = {**(node.get("backend_params") or {}), "max_iter": before.max_iter if check else max_iter}
        data = _node_data(cells, old.gene_names)
        model = train(data, target, node["backend"], params, node.get("hvg"), node.get("hvg_params"),
                      genes=old.gene_names)
        clf = model._model.named_steps["clf"]
        out |= {"cells": int(cells.size), "genes": len(old.gene_names), "seconds": model.last_train_seconds,
                "n_iter_before": int(np.max(before.n_iter_)), "n_iter": int(np.max(clf.n_iter_)),
                "capped_again": bool(np.max(clf.n_iter_) >= clf.max_iter)}
        if check:
            sample = np.random.default_rng(0).choice(cells.size, min(cells.size, 20_000), replace=False)
            same = old.predict_top(data[sample])[0] == model.predict_top(data[sample])[0]
            out |= {"agree": float(same.mean()),
                    "coef_diff": float(np.abs(clf.coef_ - before.coef_).max() / np.abs(before.coef_).max())}
            return out

        # the new model and its record staged beside the node, then swapped in file by file
        staged = node_dir.parent / f".{node_dir.name}.retrained"
        model.save(staged)
        TrainingRecorder.record_node(SimpleNamespace(project_dir=project, nodes={}), node_dir=staged,
                                     node_config={**node, "backend_params": params}, model=model,
                                     n_training_cells=int(cells.size))
        new_record = json.loads((staged / "training_metadata.json").read_text())
        new_record["node_path"] = node_path
        new_record["retrained"] = {"from_max_iter": int(before.max_iter), "n_iter_before": out["n_iter_before"],
                                   "train_seconds_before": record["fit_metadata"]["train_seconds"]}
        (staged / "training_metadata.json").write_text(json.dumps(new_record, indent=2))
        backup = project / CAPPED_DIR / node_path
        if not (backup / "training_metadata.json").exists():
            backup.mkdir(parents=True, exist_ok=True)
            for file in MODEL_FILES:
                if (node_dir / file).exists():
                    shutil.copy2(node_dir / file, backup / file)
        for file in MODEL_FILES:
            if (staged / file).exists():
                (staged / file).replace(node_dir / file)
        shutil.rmtree(staged, ignore_errors=True)
    except Exception as error:   # one node failing leaves the others to finish
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def _fit_node(job) -> dict:
    """
    One node of a run's arm trained from scratch, in a forked worker, as train_config trains it:
    its cells and classes from _node_cells, the genes given, the parameters of the arm's config.
    Saves the model and its catnap training record; an error is returned, not raised.
    """
    from types import SimpleNamespace
    from catnap_core.train import train
    from catnap_core.training_metadata import TrainingRecorder
    run_dir, arm, node_path, genes, fold = job
    out = {"run": run_dir, "arm": arm, "node": node_path}
    try:
        project = Path(run_dir) / arm
        name, node = _config_node(project / "config.yml", node_path)
        cells, target = _node_cells(fold, name, node)
        model = train(_node_data(cells, genes), target, node["backend"], node.get("backend_params"),
                      node.get("hvg"), node.get("hvg_params"), genes=genes)
        model.save(project / "models" / node_path)
        TrainingRecorder.record_node(SimpleNamespace(project_dir=project, nodes={}),
                                     node_dir=project / "models" / node_path, node_config=node, model=model,
                                     n_training_cells=int(cells.size))
        clf = None if model._model is None else model._model.named_steps["clf"]
        out |= {"cells": int(cells.size), "genes": len(genes), "seconds": model.last_train_seconds,
                "n_iter": None if clf is None else int(np.max(clf.n_iter_))}
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def _plan(adata, dataset: str, label_cols, source_config, n_hvg, fold: int, donor_col: str, arms, max_iter,
          root, root_cv, scores_root, node_n=None, tag=None):
    """
    A budget run's arms set up for train_parallel: configs written, split and gene sets taken, one
    job per node of every arm not yet scored. None when every arm is.
    """
    run = Run(root, dataset, tag or budget_tag(n_hvg, fold))
    arms = [arm for arm in arms if not arm_file(run, "metrics", arm).exists()]
    if not arms:
        return None
    for arm in arms:   # each arm's config as train_budget and train_fixed write it
        config = variant_config(source_config, HVG, n_hvg or 1, backend=BACKEND)
        if arm == ARM:
            _set_selector(config["root"], RANKED)
        if max_iter:
            _set_backend_params(config["root"], max_iter=int(max_iter))
        project = run.arm_dir(arm)
        if (project / "models").exists():
            shutil.rmtree(project / "models")
        project.mkdir(parents=True, exist_ok=True)
        (project / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    config_path = run.arm_dir(arms[0]) / "config.yml"
    split = make_split(adata, label_cols, config_path, donor_col=donor_col, n_folds=5, fold=fold, seed=0)
    all_genes, scores = fold_scores(adata, label_cols, config_path, split, scores_root, dataset)
    top = {path: top_genes(vector, all_genes, int((node_n or {}).get(path, n_hvg))) for path, vector in scores.items()}
    union = sorted({gene for genes in top.values() for gene in genes})
    ranked = [str(gene) for gene in global_ranking(root_cv, dataset, split.key)[:n_hvg]] if FIXED in arms else []

    nodes = []   # every node with children, as train_config trains them, constant models included

    def walk(path, node):
        below = hierarchy.children_of(node)
        if below:
            nodes.append(path)
        for child, sub in below.items():
            walk(f"{path}/{hierarchy.norm(child)}", sub)

    walk("root", hierarchy.root_item(hierarchy.load_config(config_path))[1])
    jobs = [(str(run.dir), arm, path, (top.get(path, union) if arm == ARM else ranked), fold)
            for arm in arms for path in nodes]
    return {"run": run, "arms": arms, "split": split, "jobs": jobs, "union": union, "ranked": ranked,
            "config_path": config_path, "n_hvg": n_hvg, "node_n": node_n, "max_iter": max_iter}


def _score_plan(adata, plan, label_cols, donor_col, seconds, log=print, variant=None) -> None:
    """A planned run's arms, every node trained, predicting the held-out donors: predictions and metrics."""
    run, split = plan["run"], plan["split"]
    adata_test = adata[split.test].copy()
    truth = truth_table(adata_test, label_cols)
    for arm in plan["arms"]:
        project = run.arm_dir(arm)
        (project / "training_metadata.json").write_text(json.dumps(
            {"status": "completed", "trained_by": "quantity.train_parallel", "wall_seconds": seconds}, indent=2))
        pred = predict_labels(adata_test, project)
        table = truth.add_prefix("true_")
        table.insert(0, "donor", adata_test.obs[donor_col].astype(str).to_numpy())
        table = table.join(pred.add_prefix(f"{arm}_")).rename_axis("cell")
        table.to_csv(arm_file(run, "predictions", arm))
        meta = {"dataset": run.dataset, "tag": run.tag, "n_hvg": plan["n_hvg"], "node_n": plan["node_n"],
                "variant": variant, "fold": split.fold, "max_iter": plan["max_iter"],
                "selector": RANKED if arm == ARM else HVG,
                "n_genes_trained": len(plan["union"] if arm == ARM else plan["ranked"]), "train_seconds": seconds,
                "trained_by": "quantity.train_parallel", "label_cols": list(label_cols), **split.meta}
        if arm == ARM:
            run.write_meta(meta)
        arm_file(run, "metrics", arm).write_text(
            json.dumps({**meta, **run_metrics(run, list(truth.columns), arm)}, indent=1))
        log(f"train_parallel: {run.tag} {arm} scored")


def _run_plans(adata, plans, label_cols, donor_col, workers, log, variant=None) -> None:
    """Every node of every plan in one pool of forked workers, largest first, then each plan scored."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    plans = [plan for plan in plans if plan]
    if not plans:
        return
    _FORKED.update(adata=adata, finest=finest_labels(adata, label_cols))
    _FORKED.setdefault("train", {}).update({plan["split"].fold: plan["split"].train for plan in plans})
    sized = []
    for plan in plans:
        for job in plan["jobs"]:
            cells = _node_cells(job[4], *_config_node(plan["config_path"], job[2]))[0].size
            sized.append((len(job[3]) * cells, job))
    start, results = time.time(), []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(_fit_node, job) for _, job in sorted(sized, key=lambda item: -item[0])]
        for future in as_completed(futures):
            results.append(future.result())
            log(f"train_parallel: {json.dumps(results[-1])}")
    errors = [r for r in results if "error" in r]
    if errors:
        raise RuntimeError(f"train_parallel: {len(errors)} node(s) failed, first {errors[0]}")
    for plan in plans:
        _score_plan(adata, plan, label_cols, donor_col, time.time() - start, log, variant)


def train_parallel(adata, dataset: str, label_cols, source_config, n_hvg: int, fold: int, donor_col: str,
                   arms=(ARM, FIXED), max_iter=None, root="runs_quantity", root_cv="runs_cv", workers=12,
                   scores_root="runs_quantity", log=print) -> Run:
    """
    The budget run of (n_hvg, fold), arms of it at once, every node of every arm trained in its
    own forked worker instead of one after another: what train_budget and train_fixed train, the
    same cells, classes, genes and parameters, in the time of the slowest node. Each arm then
    predicts the held-out donors and writes its predictions and metrics as they do. An arm whose
    metrics are there is skipped.

    Reselect's nodes take their top n_hvg from the fold's cached F vectors, a node with a single
    child, a constant model, every gene some node takes; fixed_global's take the top n_hvg of the
    root's leaf-cut ranking at every node.
    """
    plan = _plan(adata, dataset, label_cols, source_config, n_hvg, fold, donor_col, arms, max_iter, root,
                 root_cv, scores_root)
    _run_plans(adata, [plan], label_cols, donor_col, workers, log)
    return Run(root, dataset, budget_tag(n_hvg, fold))


# --- Per-node budgets tuned on held-out donors -----------------------

TUNE_GRID = (10, 25, 50, 100, 200, 300, 500, 1000)
TUNED = "tuned"   # the variant, and the tag f_statistic_tuned_logit_fold<k>


def inner_split(adata, split, label_cols, donor_col, n_splits=5, seed=0):
    """
    (inner_train, validation) masks over adata: split's training donors dealt into n_splits groups
    balancing the finest labels, as make_split deals every donor, the first group held out.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    train = np.flatnonzero(split.train)
    finest = finest_labels(adata, label_cols)[train]
    donors = adata.obs[donor_col].to_numpy()[train]
    _, held = next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                   .split(train, finest, groups=donors))
    validation = np.zeros(adata.n_obs, bool)
    validation[train[held]] = True
    return split.train & ~validation, validation


def _inner_scores(job):
    """A fold's F vectors on its inner training cells, in a forked worker, cached beside fold_scores'."""
    fold, key, label_cols, config_path, path = job
    out = {"fold": fold, "path": str(path)}
    try:
        if not Path(path).exists():
            genes, scores = node_scores(_FORKED["adata"][_FORKED["train"][fold]].copy(), label_cols, config_path)
            save_scores(genes, scores, path)
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def _sweep_node(job) -> dict:
    """
    One node trained on a fold's inner training cells at one budget, its own decision scored on
    every validation cell truly below it: accuracy and macro-F1 over its children, its genes read
    by name. Nothing is saved.
    """
    from catnap_core.train import train
    from sklearn.metrics import f1_score
    config_path, fold, path, n_hvg, genes = job
    out = {"fold": fold, "path": path, "n_hvg": n_hvg}
    try:
        name, node = _config_node(config_path, path)
        cells, target = _node_cells(fold, name, node)
        model = train(_node_data(cells, genes), target, node["backend"], node.get("backend_params"),
                      node.get("hvg"), node.get("hvg_params"), genes=genes)
        held, truth = _node_cells(fold, name, node, "validation")
        pred = model.predict_top(_node_data(held, genes))[0]
        truth = truth.astype(str)
        clf = model._model.named_steps["clf"]
        out |= {"cells": int(cells.size), "validation_cells": int(held.size), "seconds": model.last_train_seconds,
                "n_iter": int(np.max(clf.n_iter_)), "accuracy": float((pred == truth).mean()),
                "macro_f1": float(f1_score(truth, pred, labels=sorted(set(truth)), average="macro", zero_division=0))}
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def choose_counts(curves: pd.DataFrame, metric="macro_f1") -> dict:
    """Per fold, every node's budget of best validation metric, the smaller on a tie: {fold: {path: n}}."""
    best = (curves.sort_values(["fold", "path", metric, "n_hvg"], ascending=[True, True, False, True])
                  .groupby(["fold", "path"]).head(1))
    return {int(fold): dict(zip(block["path"], block["n_hvg"].astype(int))) for fold, block in best.groupby("fold")}


def train_tuned(adata, dataset: str, label_cols, source_config, donor_col: str, folds=range(5), grid=TUNE_GRID,
                metric="macro_f1", max_iter=2000, root="runs_quantity", workers=16, log=print,
                variant=TUNED) -> pd.DataFrame:
    """
    A tree per fold whose every node selects its own number of genes, chosen on donors the fold's
    training holds out: the training donors split again (inner_split), every node's F vectors
    scored on the inner training cells alone, every node trained there at each budget of grid and
    its own decision scored on the validation cells (_sweep_node), each node given its budget of
    best validation metric (choose_counts), then the tree trained on the whole training split with
    those budgets, selected from the fold's own F vectors, and scored on the held-out donors as
    every budget run is. Tagged f_statistic_<variant>_logit_fold<k>, its node_n in metrics.json.

    The validation curves are kept under root/<dataset>/_tuning/, shared by every grid: another
    rule of choice needs no new sweep, a wider grid sweeps only its new budgets. Returns them.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tuning = Path(root) / dataset / "_tuning"
    tuning.mkdir(parents=True, exist_ok=True)
    config = variant_config(source_config, HVG, max(grid), backend=BACKEND)
    _set_selector(config["root"], RANKED)
    _set_backend_params(config["root"], max_iter=int(max_iter))
    config_path = tuning / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    splits = {fold: make_split(adata, label_cols, config_path, donor_col=donor_col, n_folds=5, fold=fold, seed=0)
              for fold in folds}
    inner = {fold: inner_split(adata, split, label_cols, donor_col) for fold, split in splits.items()}
    _FORKED.update(adata=adata, finest=finest_labels(adata, label_cols),
                   train={fold: masks[0] for fold, masks in inner.items()},
                   validation={fold: masks[1] for fold, masks in inner.items()})
    for fold, (train, held) in inner.items():
        donors = adata.obs[donor_col].to_numpy()
        log(f"train_tuned: fold {fold}: {int(train.sum()):,} inner training cells, {int(held.sum()):,} validation "
            f"cells of {len(set(donors[held]))} donors")

    score_paths = {fold: Path(root) / dataset / "_scores" / f"fold{fold}_{splits[fold].key}_inner.npz" for fold in folds}
    with ProcessPoolExecutor(min(3, len(score_paths)), mp_context=mp.get_context("fork")) as pool:   # ~100 GB each
        for future in as_completed([pool.submit(_inner_scores, (fold, splits[fold].key, label_cols, config_path, path))
                                    for fold, path in score_paths.items()]):
            result = future.result()
            log(f"train_tuned: inner scores {json.dumps(result)}")
            if "error" in result:
                raise RuntimeError(f"train_tuned: inner scores failed {result}")

    curves_path = tuning / f"validation_{metric}.csv"
    done = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame(columns=["fold", "path", "n_hvg"])
    seen = set(zip(done["fold"], done["path"], done["n_hvg"]))
    jobs = []
    for fold in folds:
        genes, scores = load_scores(score_paths[fold])
        for path, vector in scores.items():
            cells = int(_node_cells(fold, *_config_node(config_path, path))[0].size)
            for n_hvg in grid:
                if (fold, path, n_hvg) not in seen:
                    jobs.append((n_hvg * cells, (str(config_path), fold, path, n_hvg, top_genes(vector, genes, n_hvg))))
    log(f"train_tuned: {len(jobs)} node trainings to sweep")
    rows = done.to_dict("records")
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(_sweep_node, job) for _, job in sorted(jobs, key=lambda item: -item[0])]
        for future in as_completed(futures):
            result = future.result()
            log(f"train_tuned: sweep {json.dumps(result)}")
            if "error" not in result:
                rows.append(result)
                pd.DataFrame(rows).to_csv(curves_path, index=False)
    curves = pd.DataFrame(rows)
    choices = choose_counts(curves[curves["fold"].isin(list(folds)) & curves["n_hvg"].isin(grid)], metric)
    name = f"choices_{metric}.json" if variant == TUNED else f"choices_{metric}_{variant}.json"
    (tuning / name).write_text(json.dumps(choices, indent=1))
    log(f"train_tuned: budgets chosen {json.dumps(choices)}")

    _FORKED.pop("validation", None)
    _FORKED["train"] = {}
    plans = [_plan(adata, dataset, label_cols, source_config, None, fold, donor_col, (ARM,), max_iter, root, "runs_cv",
                   root, node_n=choices[fold], tag=f"{HVG}_{variant}_{BACKEND}_fold{fold}") for fold in folds]
    _run_plans(adata, plans, label_cols, donor_col, workers, log, variant=variant)
    return curves


def convergence_table(targets) -> pd.DataFrame:
    """
    Per selection, arm and budget, over the node models of targets as arm_runs gives them: how
    many there are, how many a retraining took past catnap's cap of 1000 iterations, and how many
    stop at their cap still.
    """
    rows = []
    for run, n_hvg, fold, arm, selector in targets:
        for record in (run.arm_dir(arm) / "models").rglob("training_metadata.json"):
            clf = _logit(record.parent)
            if clf is None:
                continue
            rows.append({"selector": selector, "arm": arm, "n_hvg": n_hvg, "fold": fold, "max_iter": clf.max_iter,
                         "retrained": "retrained" in json.loads(record.read_text()),
                         "capped": int(np.max(clf.n_iter_)) >= clf.max_iter})
    frame = pd.DataFrame(rows)
    return frame.groupby(["selector", "arm", "n_hvg"]).agg(
        folds=("fold", "nunique"), models=("capped", "size"), retrained=("retrained", "sum"),
        capped=("capped", "sum"), max_iter=("max_iter", "max"))


def _stale(run: Run, arm=ARM) -> bool:
    """True when a node model of run's arm is newer than the arm's predictions."""
    newest = max(path.stat().st_mtime for path in (run.arm_dir(arm) / "models").rglob("model_metadata.json"))
    return newest > arm_file(run, "predictions", arm).stat().st_mtime


def retrain_capped(adata, targets, label_cols, config_path, donor_col, max_iter=2000, heavy=1e8,
                   heavy_workers=3, light_workers=6, log_path=None) -> pd.DataFrame:
    """
    Every node of targets, (Run, n_hvg, fold, arm, selector) as arm_runs gives them, whose lbfgs
    stopped at a cap below max_iter, trained again at max_iter, then every arm with a node newer
    than its predictions predicted again: its predictions and metrics rewritten, the replaced ones
    kept with a _maxiter1000 suffix, a reselect arm's own_decisions dropped.

    A node still at the cap after max_iter stays there. Nodes train in forked workers sharing
    adata, those of more than heavy cells x genes in a pool of their own so that few of the
    largest copies are ever held at once. The first, smallest job is first trained again at its
    own cap and compared, the check that a retrained node is the one train_config trained.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    log_path = Path(log_path) if log_path else None

    def note(text):
        print(text, flush=True)
        if log_path:
            with open(log_path, "a") as handle:
                handle.write(text + "\n")

    levels = [f"L{k}" for k in range(1, len(label_cols) + 1)]
    splits = {fold: make_split(adata, label_cols, config_path, donor_col=donor_col, n_folds=5, fold=fold, seed=0)
              for fold in sorted({target[2] for target in targets})}
    for run, _, fold, _, _ in targets:
        if run.meta.get("train_key") != splits[fold].key:
            raise ValueError(f"{run.tag}: trained on another split than fold {fold}")
    _FORKED.update(adata=adata, finest=finest_labels(adata, label_cols),
                   train={fold: split.train for fold, split in splits.items()})

    jobs = []
    for run, _, fold, arm, _ in targets:
        for node_path in capped_nodes(run, max_iter, arm):
            record = json.loads((run.arm_dir(arm) / "models" / node_path / "training_metadata.json").read_text())
            size = record["fit_metadata"]["n_training_cells"] * record["fit_metadata"]["n_features"]
            jobs.append((size, (str(run.dir), fold, node_path, max_iter, False, arm)))
    note(f"retrain_capped: {len(jobs)} capped node(s) over {len(targets)} arm(s), max_iter {max_iter}")
    results = []
    if jobs:
        jobs.sort(key=lambda job: -job[0])
        smallest = min(jobs, key=lambda job: job[0])[1]
        check = _retrain((*smallest[:4], True, smallest[5]))
        note(f"retrain_capped: check {json.dumps(check)}")
        if "error" in check or check["agree"] < 0.99:
            raise RuntimeError(f"retraining does not reproduce the saved model: {check}")
        context = mp.get_context("fork")
        with ProcessPoolExecutor(heavy_workers, mp_context=context) as big, \
             ProcessPoolExecutor(light_workers, mp_context=context) as small:
            futures = [(big if size > heavy else small).submit(_retrain, job) for size, job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                note(f"retrain_capped: {json.dumps(result)}")

    stale = [(run, fold, arm) for run, _, fold, arm, _ in targets if _stale(run, arm)]
    note(f"retrain_capped: predicting {len(stale)} arm(s) again")
    for fold in sorted({fold for _, fold, _ in stale}):
        test = adata[splits[fold].test].copy()
        for run, _, arm in [target for target in stale if target[1] == fold]:
            for stem in ("predictions", "metrics"):
                if not arm_file(run, stem, arm, "_maxiter1000").exists():
                    shutil.copy2(arm_file(run, stem, arm), arm_file(run, stem, arm, "_maxiter1000"))
            pred = predict_labels(test, run.arm_dir(arm))
            table = pd.read_csv(arm_file(run, "predictions", arm), index_col="cell",
                                usecols=["cell", "donor"] + [f"true_{level}" for level in levels])
            table = table.join(pred.add_prefix(f"{arm}_").rename_axis("cell"))
            table.to_csv(arm_file(run, "predictions", arm))
            models = run.arm_dir(arm) / "models"
            retrained = sorted(path.parent.relative_to(models).as_posix()
                               for path in models.rglob("training_metadata.json")
                               if "retrained" in json.loads(path.read_text()))
            metrics = {**json.loads(arm_file(run, "metrics", arm).read_text()), "max_iter_retrained": max_iter,
                       "retrained_nodes": retrained, **run_metrics(run, levels, arm)}
            arm_file(run, "metrics", arm).write_text(json.dumps(metrics, indent=1))
            if arm == ARM:
                _decisions_path(run).unlink(missing_ok=True)
            note(f"retrain_capped: {run.tag} {arm} predicted again, hF {metrics['hF']:.4f}, "
                 f"{len(retrained)} node(s) retrained")
        del test
    return pd.DataFrame(results)


# --- Accuracy per node -----------------------------------------------

def budget_runs(root="runs_quantity", dataset="aifi", extra=()) -> list:
    """(Run, n_hvg, fold) of every scored run under root giving every node the same n, then extra."""
    out = []
    for path in sorted((Path(root) / dataset).glob("*/metrics.json")):
        row = json.loads(path.read_text())
        if not row.get("node_n") and row.get("selector", HVG) in (HVG, RANKED):
            out.append((Run(root, dataset, row["tag"]), row["n_hvg"], row["fold"]))
    return out + list(extra)


def _predictions(run: Run, levels, arm=ARM) -> pd.DataFrame:
    columns = ["cell"] + [f"true_{level}" for level in levels] + [f"{arm}_{level}_pred" for level in levels]
    return pd.read_csv(_predictions_path(run, arm), index_col="cell", usecols=columns)


def _decisions_path(run: Run, root="runs_quantity") -> Path:
    return Path(root) / run.dataset / "_decisions" / f"{run.dir.parent.parent.name}_{run.tag}.csv"


def decision_curves(adata, runs, label_cols, arm=ARM) -> pd.DataFrame:
    """
    Every split node's own_decisions in every (Run, n_hvg, fold) of runs, one row per (path, n,
    fold). A fold's held-out cells are copied out of adata once, for the runs not yet replayed.
    """
    frames = []
    for fold in sorted({fold for _, _, fold in runs}):
        mine = [(run, n_hvg) for run, n_hvg, f in runs if f == fold]
        cells = None
        if not all(_decisions_path(run).exists() for run, _ in mine):
            held = set().union(*(pd.read_csv(run.dir / "predictions.csv.gz", usecols=["cell"])["cell"]
                                 for run, _ in mine))
            cells = adata[adata.obs_names.isin(held)].copy()
        frames += [own_decisions(cells, run, label_cols, arm).assign(n_hvg=n_hvg, fold=fold)
                   for run, n_hvg in mine]
        del cells
    return pd.concat(frames, ignore_index=True)


def held_out_truth(runs, levels, arm=ARM) -> pd.DataFrame:
    """The true labels of every cell, the held-out cells of one n's folds, which together hold them all."""
    n_hvg = min(n for _, n, _ in runs)
    folds = {fold: run for run, n, fold in runs if n == n_hvg}
    return pd.concat([_predictions(run, levels, arm).filter(like="true_") for run in folds.values()])


def complete_budgets(curves: pd.DataFrame, folds=None) -> pd.DataFrame:
    """The rows of curves at the budgets every fold holds, of folds only when given."""
    if folds is not None:
        curves = curves[curves["fold"].isin(folds)]
    complete = curves.groupby("n_hvg")["fold"].nunique()
    return curves[curves["n_hvg"].isin(complete.index[complete == complete.max()])]


def saturation(curves: pd.DataFrame, metric: str, share=0.9, folds=None) -> pd.DataFrame:
    """
    Per node, the genes its decision needs: the n at which the mean over folds of metric first
    reaches share of its gain from the smallest n to its best, linear in log n between budgets.
    Only complete_budgets count.

    Note: a budget at which no cell reached the node is left out of its curve: at 10 genes the
    root routes no cell to Progenitor cell.
    """
    rows = []
    for path, block in complete_budgets(curves, folds).groupby("path"):
        mean = block.groupby("n_hvg")[metric].mean().dropna()
        n, value = np.log(mean.index.to_numpy(float)), mean.to_numpy()
        target = value[0] + share * (value.max() - value[0])
        k = int(np.argmax(value >= target))
        at = n[0] if k == 0 else n[k - 1] + (target - value[k - 1]) / (value[k] - value[k - 1]) * (n[k] - n[k - 1])
        rows.append({"path": path, f"n{int(share * 100)}": float(np.exp(at)), "first": value[0],
                     "best": value.max(), "n_best": int(mean.idxmax()), "gain": value.max() - value[0]})
    return pd.DataFrame(rows)


# --- Overlap with the global selection -------------------------------

def node_scores(adata_train, label_cols, config_path, min_cells_per_child=2):
    """
    (genes, F vectors by model path): every split node's f_statistic F over the whole genome, on
    its children, as the reselect arm's selector computes it on its training cells, NaN to 0 and
    perfect separations infinite as it ranks them. Keys are model paths, root/B cell.
    """
    finest = finest_labels(adata_train, label_cols)
    genes = adata_train.var_names.astype(str).to_numpy()
    out = {}

    def walk(path, node):
        children = hierarchy.children_of(node)
        if len(children) >= 2:   # a single child has no split to score, its model is constant
            target = np.full(finest.shape, "", dtype=object)
            for child, labels in hierarchy.child_label_sets(node).items():
                target[np.isin(finest, list(labels))] = hierarchy.norm(child)
            cells = target != ""
            f, _ = score_fns.f_stat_scores(adata_train.X[cells], target[cells],
                                           min_cells_per_child=min_cells_per_child)
            out[path] = np.nan_to_num(np.asarray(f, dtype=float), nan=0.0, posinf=np.inf)
            print(f"  {path}: {int(cells.sum()):,} cells", flush=True)
        for child, below in children.items():
            walk(f"{path}/{hierarchy.norm(child)}", below)

    _, root = hierarchy.root_item(hierarchy.load_config(config_path))
    walk("root", root)
    return genes, out


def save_scores(genes, scores: dict, path) -> None:
    """node_scores to a compressed npz: the genes, the model paths, a F vector per path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    paths = list(scores)
    staged = Path(path).with_suffix(".tmp.npz")   # written aside then renamed, never read half-written
    # fixed-width strings: an object array would need pickle to load back
    np.savez_compressed(staged, genes=np.asarray(genes).astype(str),
                        paths=np.asarray(paths).astype(str),
                        scores=np.stack([scores[p] for p in paths]))
    staged.replace(path)


def load_scores(path):
    with np.load(path, allow_pickle=False) as data:
        return data["genes"], dict(zip(data["paths"].tolist(), data["scores"]))


def node_rankings(genes, scores: dict) -> dict:
    """Every node's whole genome best first, from its F vector: the order its top-n follow."""
    return {path: np.asarray(genes)[np.argsort(-vector, kind="stable")]
            for path, vector in scores.items()}


def global_ranking(root_cv: str, dataset: str, train_key: str) -> np.ndarray:
    """The root's leaf-cut f_statistic ranking cached for the split whose training cells hash to
    train_key, the fixed_global arm's selection at any budget."""
    path = Path(root_cv) / dataset / BASELINES / f"f_stat-root-leaves-{train_key}.json"
    return np.asarray(json.loads(path.read_text())["ranking"])


def overlap_curves(rankings: dict, reference: np.ndarray, ns) -> pd.DataFrame:
    """Per node and n, the share of the node's top-n genes also in reference's top-n."""
    rows = []
    position = {gene: i for i, gene in enumerate(reference)}
    for path, ranking in rankings.items():
        # rank of each of the node's genes in the reference, the overlap at n counts those < n
        ref_rank = np.array([position.get(gene, len(reference)) for gene in ranking])
        for n in ns:
            rows.append({"path": path, "node": path.rsplit("/", 1)[-1], "depth": path.count("/"),
                         "n": int(n), "overlap": float((ref_rank[:n] < n).mean())})
    return pd.DataFrame(rows)


def overlap_features(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Per fold and node, from overlap_curves frames carrying a fold column: n_first, the first n at
    which the node's top-n shares a gene with the global top-n, and n50, the first n at which half
    the genes are shared.

    Note: the share has no single minimum to read a budget from. It is k/n for the k shared genes,
    so it falls by 1/n at every n bringing no new shared gene and jumps at each that does, a
    sawtooth whose lowest point after n_first sits, at 19 of the 21 AIFI nodes, just before the
    second shared gene: a count of one, not a feature of the node.
    """
    rows = []
    for (fold, path), d in curves.groupby(["fold", "path"]):
        d = d.sort_values("n")
        rows.append({"fold": fold, "path": path, "node": path.rsplit("/", 1)[-1],
                     "depth": path.count("/"), "branch": path.split("/")[1] if "/" in path else "root",
                     "n_first": d.loc[d["overlap"] > 0, "n"].min(),
                     "n50": d.loc[d["overlap"] >= 0.5, "n"].min()})
    return pd.DataFrame(rows)


# --- Figures ----------------------------------------------------------

# Level 1 compartments in the validated categorical order, the root in ink
BRANCH_COLOUR = {"root": "#3a3a3a", "B cell": "#2a78d6", "T cell": "#eb6834", "NK cell": "#1baf7a",
                 "Monocyte": "#eda100", "DC": "#e87ba4", "Progenitor cell": "#008300"}


def overlap_figure(curves: pd.DataFrame, features: pd.DataFrame, n_genes: int, path=None,
                   width=None, height=2.6):
    """
    Per depth, every node's share of its top-n in the global top-n against n, the mean over folds,
    coloured by Level 1 branch, and chance (n over the genome) dotted.
    """
    import matplotlib.pyplot as plt
    from hvgsel import figures
    from hvgsel.report import _finish
    width = width or figures.FULL
    means = curves.groupby(["path", "n"], as_index=False)["overlap"].mean()
    depths = sorted(features["depth"].unique())
    figure, axes = plt.subplots(1, len(depths), figsize=(width, height), sharey=True, squeeze=False)
    grid = np.unique(means["n"])
    for ax, depth in zip(axes[0], depths):
        for node_path, block in means[means["path"].isin(features.loc[features["depth"] == depth, "path"])].groupby("path"):
            branch = node_path.split("/")[1] if "/" in node_path else "root"
            colour = BRANCH_COLOUR.get(branch, "#8c8c8c")
            ax.plot(block["n"], block["overlap"], color=colour, lw=0.8, alpha=0.85)
        ax.plot(grid, grid / n_genes, color="#8c8c8c", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title({0: "root", 1: "Level 1 nodes", 2: "Level 2 nodes"}.get(depth, f"depth {depth}"),
                     loc="left")
        ax.set_xlabel("genes per node, n")
        ax.grid(color="#ececec")
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("share of the node's top-n\nin the global top-n")
    handles = [plt.Line2D([], [], color=c, lw=1.2, label=b) for b, c in BRANCH_COLOUR.items()
               if b in set(features["branch"])]
    handles += [plt.Line2D([], [], color="#8c8c8c", ls=":", lw=0.8, label="chance")]
    return _finish(figure, handles, -(-len(handles) // 2), path)


LEVEL_COLOUR = ("#86b6ef", "#2a78d6", "#104281")   # L1 to L3, one hue light to dark: ordered


ARM_STYLE = {ARM: ("-", "genes selected at each node"), FIXED: ("--", "genes selected once at root")}
MARKER_STYLE = (":", "all atlas markers")


def _arm_stats(metrics: pd.DataFrame):
    """
    Per arm of metrics' budget rows, the mean and s.d. over folds per budget, and which budgets
    hold fewer folds than the arm's most; then the marker rows' mean, s.d. and number of folds,
    or None without any.
    """
    budgets, markers = metrics[metrics["selector"] != "marker"], metrics[metrics["selector"] == "marker"]
    numeric = [c for c in budgets.select_dtypes("number").columns if c != "n_hvg"]
    out, short = {}, []
    for arm in ARM_STYLE:
        rows = budgets[budgets["arm"] == arm]
        if rows.empty:
            continue
        stats = rows.groupby("n_hvg")[numeric].agg(["mean", "std"])
        folds = rows.groupby("n_hvg")["fold"].unique().reindex(stats.index)
        partial = (folds.map(len) < folds.map(len).max()).to_numpy()
        short += [tuple(sorted(f)) for f in folds[partial]]
        out[arm] = (stats, partial, None)
    # one fold, the same everywhere, says which; runs still coming in, that they are
    label = (f"fold {short[0][0]} only" if short and all(len(f) == 1 and f == short[0] for f in short)
             else "fewer folds so far" if short else None)
    out = {arm: (stats, partial, label) for arm, (stats, partial, _) in out.items()}
    return out, ((markers[numeric].mean(), markers[numeric].std().fillna(0), markers["fold"].nunique())
                 if len(markers) else None)


def _style_handles(arms, markers, partial_label):
    """Legend entries for the line styles: the arms drawn, the markers, a budget on fewer folds."""
    import matplotlib.pyplot as plt
    handles = [plt.Line2D([], [], color="#3a3a3a", ls=ARM_STYLE[arm][0], lw=1.0, label=ARM_STYLE[arm][1])
               for arm in arms]
    if markers is not None:
        handles.append(plt.Line2D([], [], color="#3a3a3a", ls=MARKER_STYLE[0], lw=1.0, label=MARKER_STYLE[1]))
    if partial_label:
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec="#3a3a3a", mew=0.9,
                                  label=partial_label))
    return handles


def _two_rows(top: list, bottom: list):
    """Legend handles and columns putting top on the first row and bottom on the second, as the
    legend fills its columns first."""
    import matplotlib.pyplot as plt
    width = max(len(top), len(bottom))
    blank = lambda: plt.Line2D([], [], ls="", label=" ")
    top, bottom = top + [blank() for _ in range(width - len(top))], bottom + [blank() for _ in range(width - len(bottom))]
    return [handle for pair in zip(top, bottom) for handle in pair], width


PER_LEVEL = {"accuracy": "accuracy", "macro_f1": "macro-F1"}   # per-level metric: its name


def budget_figure(metrics: pd.DataFrame, path=None, width=None, height=2.7, per_level="accuracy", cells_only=False):
    """
    Against the genes per node, on a log axis: hierarchical F1 pooled over cells and averaged over
    leaf classes, per_level (accuracy or macro_f1) per level, and the time the tree's node models
    take to fit, summed over nodes. Genes selected at each node solid, selected once at the root and reused (fixed_global)
    dashed, in the metric's colour; every atlas marker at every node dotted across, the level to
    beat, a band of one s.d. over its folds. Mean over folds, a bar of one s.d.; a budget run on
    fewer folds shows what it has, hollow.

    metrics is metrics_table's: budget rows of either arm, marker rows told by selector "marker".
    cells_only shows hierarchical precision alone on the left: hP, the share of the predicted path
    labels that are correct, pooled over cells.

    Note: fitting time leaves out the gene selection, scored once per fold for every budget here,
    and was measured with several runs sharing the machine: it compares budgets more than it times
    catnap.
    """
    import matplotlib.pyplot as plt
    from hvgsel import figures
    from hvgsel.report import _finish
    width = width or figures.FULL
    arms, markers = _arm_stats(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(width, height), squeeze=False)
    ax_f, ax_a, ax_t = axes[0]
    series = ([(ax_f, "hP", "#1a1a1a", "hP", "o", 1.0)] if cells_only else
              [(ax_f, "hF", "#1a1a1a", "hF over cells", "o", 1.0),
               (ax_f, "macro_hF", "#eb6834", "hF over leaf classes", "s", 1.0)])
    series += [(ax_a, f"{per_level}_{level}", colour, f"{PER_LEVEL[per_level]} {level}", "o", 1.0)
               for colour, level in zip(LEVEL_COLOUR, ("L1", "L2", "L3"))]
    series += [(ax_t, "fit_seconds", "#1a1a1a", None, "o", 1 / 60)]
    partial_fold = None
    for ax, column, colour, label, marker, scale in series:
        for arm, (stats, partial, label) in arms.items():
            n = stats.index.to_numpy()
            mean, std = stats[(column, "mean")] * scale, stats[(column, "std")].fillna(0) * scale
            ax.errorbar(n, mean, yerr=std, color=colour, marker=marker, ms=3.2 if arm == ARM else 2.6,
                        ls=ARM_STYLE[arm][0], lw=1.0, elinewidth=0.7, capsize=0)
            ax.plot(n[partial], mean[partial], ls="", marker=marker, ms=3.2 if arm == ARM else 2.6,
                    mfc="white", mec=colour, mew=0.9, zorder=3)
            if partial.any():
                partial_fold = label
        if markers is not None:
            mean, sd = markers[0][column] * scale, markers[1][column] * scale
            ax.axhspan(mean - sd, mean + sd, color=colour, alpha=0.12, lw=0, zorder=0)
            ax.axhline(mean, color=colour, ls=MARKER_STYLE[0], lw=1.0, zorder=1)
    budgets = sorted({n for stats, _, _ in arms.values() for n in stats.index})
    for ax, title, ylabel in ((ax_f, *(("hierarchical precision", "hP") if cells_only else ("hierarchical F1", "hF"))),
                              (ax_a, f"{PER_LEVEL[per_level]} per level", PER_LEVEL[per_level]),
                              (ax_t, "model fitting time", "model fitting time (min)")):
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(int(v)) for v in budgets], rotation=90)
        ax.minorticks_off()
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("genes per node")
        ax.grid(color="#ececec")
        ax.set_axisbelow(True)
    ax_t.set_yscale("log")
    handles, ncols = _two_rows([plt.Line2D([], [], color=colour, marker=marker, ms=3.2, lw=1.0, label=label)
                                for _, _, colour, label, marker, _ in series if label],
                               _style_handles(arms, markers, partial_fold))
    return _finish(figure, handles, ncols, path)


def time_figure(metrics: pd.DataFrame, path=None, width=None, height=2.9, per_level="accuracy", cells_only=False):
    """
    hF and per_level (accuracy or macro_f1) per level against the time the tree's node models take to fit, summed over
    nodes, on a log axis: one point per budget, mean over folds, a bar of one s.d. each way, the
    genes per node beside the points selected at each node. Selected once at the root dashed,
    every atlas marker dotted across with a diamond at its own time, as budget_figure. A budget
    run on fewer folds is hollow.

    cells_only shows hierarchical precision alone on the left: hP, the share of the predicted path
    labels that are correct, pooled over cells.

    Note: the times were measured with several runs sharing the machine, see budget_figure.
    """
    import matplotlib.pyplot as plt
    from hvgsel import figures
    from hvgsel.report import _finish
    width = width or figures.FULL
    arms, markers = _arm_stats(metrics)
    figure, axes = plt.subplots(1, 2, figsize=(width, height), squeeze=False)
    ax_f, ax_a = axes[0]
    series = ([(ax_f, "hP", "#1a1a1a", "hP", "o", "above")] if cells_only else
              [(ax_f, "hF", "#1a1a1a", "hF over cells", "o", "above"),
               (ax_f, "macro_hF", "#eb6834", "hF over leaf classes", "s", None)])
    # the genes per node where the panel leaves room: under L3 for accuracy, L1 sitting at the top,
    # above L1 for macro-F1, whose L3 runs between the root selection's and the markers' lines
    at_level, side = ("L1", "above") if per_level == "macro_f1" else ("L3", "below")
    series += [(ax_a, f"{per_level}_{level}", colour, f"{PER_LEVEL[per_level]} {level}", "o",
                side if level == at_level else None)
               for colour, level in zip(LEVEL_COLOUR, ("L1", "L2", "L3"))]
    partial_fold = None
    for ax, column, colour, label, marker, labels_at in series:
        for arm, (stats, partial, label) in arms.items():
            minutes, minutes_sd = stats[("fit_seconds", "mean")] / 60, stats[("fit_seconds", "std")].fillna(0) / 60
            mean, std = stats[(column, "mean")], stats[(column, "std")].fillna(0)
            ax.errorbar(minutes, mean, xerr=minutes_sd, yerr=std, color=colour, marker=marker,
                        ms=3.2 if arm == ARM else 2.6, ls=ARM_STYLE[arm][0], lw=1.0, elinewidth=0.7, capsize=0)
            ax.plot(minutes[partial], mean[partial], ls="", marker=marker, ms=3.2 if arm == ARM else 2.6,
                    mfc="white", mec=colour, mew=0.9, zorder=3)
            if partial.any():
                partial_fold = label
            if labels_at and arm == ARM:   # genes per node, at two heights in turn so neighbours never meet
                sign = 1 if labels_at == "above" else -1
                for i, (n, x, y) in enumerate(zip(stats.index, minutes, mean)):
                    ax.annotate(str(int(n)), (x, y), xytext=(0, sign * (4 if i % 2 == 0 else 11)),
                                textcoords="offset points", fontsize=5.5, color="#6a6a6a", ha="center",
                                va="bottom" if sign > 0 else "top")
        if markers is not None:
            mean, sd = markers[0][column], markers[1][column]
            ax.axhspan(mean - sd, mean + sd, color=colour, alpha=0.12, lw=0, zorder=0)
            ax.axhline(mean, color=colour, ls=MARKER_STYLE[0], lw=1.0, zorder=1)
            ax.plot([markers[0]["fit_seconds"] / 60], [mean], ls="", marker="D", ms=3.0,
                    mfc="white", mec=colour, mew=0.9, zorder=3)
    for ax, title, ylabel in ((ax_f, *(("hierarchical precision", "hP") if cells_only else ("hierarchical F1", "hF"))),
                              (ax_a, f"{PER_LEVEL[per_level]} per level", PER_LEVEL[per_level])):
        ax.set_xscale("log")
        ax.margins(y=0.08)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("model fitting time (min)")
        ax.grid(color="#ececec")
        ax.set_axisbelow(True)
    handles, ncols = _two_rows([plt.Line2D([], [], color=colour, marker=marker, ms=3.2, lw=1.0, label=label)
                                for _, _, colour, label, marker, _ in series],
                               _style_handles(arms, markers, partial_fold))
    return _finish(figure, handles, ncols, path)


TUNED_VARIANTS = {500: "tuned500", 1000: TUNED, 2000: "tuned2000"}   # gene cap: the tuned trees capped there
CAP_COLOUR = {"uniform": ("#86b6ef", "#2a78d6", "#104281"),   # one hue each, light to dark as the cap grows
              "tuned": ("#f5b38f", "#eb6834", "#a8401a")}


def tuned_figure(metrics: pd.DataFrame, path=None, width=None, height=1.9, caps=(500, 1000, 2000), step=0.66,
                 bar_in=0.15, gap_in=0.015, timings="runs_quantity/aifi/_timing/fold0_tree_minutes.json",
                 compact=False):
    """
    At each gene cap G, G genes at every node against trees whose nodes choose their own number of
    genes up to G (train_tuned, TUNED_VARIANTS): hierarchical macro-F1, then model fitting time.
    Bars from zero, darker as G grows, mean over folds, a bar of one s.d., every value written
    above its bar; step spaces the caps along x. A cap without its runs is left out.

    The macro-F1 axis is broken just below its first labelled value: zero up to there squeezed into
    a strip, and a thin slanted cut through the axis and every bar, the same slope on each. Bars are
    bar_in inches wide on both panels, a pair gap_in apart, set once the layout is known.

    timings, when the file is there, gives the fitting times instead: every tree of fold 0 timed
    under the same conditions, its node models fitted one after another (retime.py), one bar each
    without spread. Otherwise the times recorded when the trees were trained, which shared the
    machine in different ways.

    compact draws a shorter version for a poster: smaller y and value labels.
    """
    if compact and height == 1.9:
        height = 1.35
    value_size = 4.8 if compact else 5.5
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation
    from hvgsel import figures
    width = width or figures.FULL * 0.68
    budgets = metrics[metrics["node_n"].isna() & (metrics["arm"] == ARM) & (metrics["selector"] != "marker")]
    sets = {kind: {cap: rows for cap in caps
                   if len(rows := (budgets[budgets["n_hvg"] == cap] if kind == "uniform"
                                   else metrics[metrics["variant"] == TUNED_VARIANTS[cap]]))}
            for kind in ("uniform", "tuned")}
    stats = {(kind, cap, column): (rows[column].mean() * scale, rows[column].std() * scale)
             for kind, by_cap in sets.items() for cap, rows in by_cap.items()
             for column, scale in (("macro_hF", 1.0), ("fit_seconds", 1 / 60))}
    clean = timings and Path(timings).exists()
    if clean:   # the same-conditions times replace the recorded ones, fold 0, no spread
        minutes = json.loads(Path(timings).read_text())
        for kind in sets:
            for cap in caps:
                key = f"{cap} genes at every node" if kind == "uniform" else f"tuned up to {cap}"
                if key in minutes:
                    stats[(kind, cap, "fit_seconds")] = (minutes[key], 0.0)
                    sets[kind].setdefault(cap, None)
    figure, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(width, height))

    # macro-F1: [0, low] squeezed into a strip of `strip` of the zoomed range's height
    f1 = [value for (kind, cap, column), value in stats.items() if column == "macro_hF"]
    first = np.floor((min(m - sd for m, sd in f1) - 0.003) * 100) / 100   # first labelled value
    low, top = first - 0.004, max(m + sd for m, sd in f1)
    high, strip = top + 0.5 * (top - low), 0.12

    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= low, y / low * strip, strip + (y - low) / (high - low))

    def inverse(z):
        z = np.asarray(z, dtype=float)
        return np.where(z <= strip, z / strip * low, low + (z - strip) * (high - low))

    ax_f.set_yscale("function", functions=(forward, inverse))
    ax_f.set_ylim(0, high)
    ticks = np.arange(first, high + 1e-9, 0.01)
    ax_f.set_yticks([0, *ticks], ["0", *[f"{tick:.2f}" for tick in ticks]])
    # in the compact version the labels hang from the top of their axis, past its foot if they must
    ax_f.set_ylabel("hierarchical macro-F1", **({"fontsize": 6.5, "loc": "top"} if compact else {}))
    times = [value for (kind, cap, column), value in stats.items() if column == "fit_seconds"]
    ax_t.set_ylim(0, max(m + sd for m, sd in times) * 1.12)
    ax_t.set_ylabel("fitting time (min)" if compact else "model fitting time (min)",
                    **({"fontsize": 6.5, "loc": "top"} if compact else {}))
    for ax in (ax_f, ax_t):
        ax.set_xticks([i * step for i in range(len(caps))], [str(cap) for cap in caps])
        ax.set_xlim(-0.4, (len(caps) - 1) * step + 0.4)
        ax.text(1.02, 0, "$G$", transform=ax.transAxes, ha="left", va="center")   # at the end of the axis
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#ececec")
        ax.set_axisbelow(True)
    handles = [Patch(color=CAP_COLOUR["uniform"][1], label="$G$ genes at every node"),
               Patch(color=CAP_COLOUR["tuned"][1], label="tuned per node, up to $G$")]
    figure.legend(handles=handles, loc="outside lower center", ncols=2, columnspacing=1.2)
    figure.canvas.draw()   # the layout fixed, inches convert to data along x

    slope, cut = 0.45, 0.035   # the break: rise over run, and its thickness, in inches
    for ax, column, digits in ((ax_f, "macro_hF", 3), (ax_t, "fit_seconds", 0)):
        per_inch = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_window_extent().width / figure.dpi)
        for side, kind in ((-1, "uniform"), (1, "tuned")):
            for i, cap in enumerate(caps):
                if (kind, cap, column) not in stats:
                    continue
                mean, sd = stats[(kind, cap, column)]
                x = i * step + side * (bar_in + gap_in) / 2 * per_inch
                ax.bar(x, mean, bar_in * per_inch, yerr=sd or None, color=CAP_COLOUR[kind][i],
                       error_kw={"elinewidth": 0.7, "ecolor": "#3a3a3a"})
                ax.text(x, mean + sd, f" {mean:.{digits}f}" if digits else f"{mean:.0f}", ha="center", va="bottom",
                        fontsize=value_size, color="#3a3a3a", rotation=90 if digits else 0)
                if ax is ax_f:   # the cut through the bar, a parallelogram in inches about (x, low)
                    run = bar_in / 2 + 0.01
                    ax.add_patch(Polygon([(-run, -cut / 2 - slope * run), (run, -cut / 2 + slope * run),
                                          (run, cut / 2 + slope * run), (-run, cut / 2 - slope * run)],
                                         transform=figure.dpi_scale_trans + ScaledTranslation(x, low, ax.transData),
                                         facecolor="white", edgecolor="none", zorder=3))
    # the cut through the y axis: white across the spine, framed by two strokes of the same slope
    anchor = figure.dpi_scale_trans + ScaledTranslation(ax_f.get_xlim()[0], low, ax_f.transData)
    run = 0.045
    ax_f.add_patch(Polygon([(-run, -cut / 2 - slope * run), (run, -cut / 2 + slope * run),
                            (run, cut / 2 + slope * run), (-run, cut / 2 - slope * run)],
                           transform=anchor, facecolor="white", edgecolor="none", zorder=4, clip_on=False))
    for edge in (-cut / 2, cut / 2):
        ax_f.add_line(Line2D([-run, run], [edge - slope * run, edge + slope * run], transform=anchor,
                             color="#1a1a1a", lw=0.8, zorder=5, clip_on=False))
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


def _cells(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n / 1e3:.1f}k"


NODE_METRIC = {"accuracy": ("accuracy", "-", "o"), "macro_f1": ("macro-F1 over children", (0, (3, 1.5)), "s")}


def node_curves_figure(curves: pd.DataFrame, properties: pd.DataFrame, saturations: dict, path=None,
                       width=None, ncols=6, row_height=1.3):
    """
    Every split node's own accuracy and macro-F1 over its children against the genes per node, in
    boxes by layer: the root alone, the Level 1 nodes beside it (going down when they outgrow the
    row), then the Level 2 nodes, each in tree order. Mean over folds at the budgets every fold
    holds, a band of one s.d., the node's arity and size in the corner, and dotted, the budget of
    best mean macro-F1 among those measured, the smaller on a tie. A budget on fewer folds shows
    what it has, hollow, joined faintly to the last complete budget. saturations is unused, kept for
    the notebook's call.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, NullFormatter
    from hvgsel import figures
    from hvgsel.report import _finish
    width = width or figures.FULL
    full = complete_budgets(curves)
    rest = curves[~curves["n_hvg"].isin(full["n_hvg"].unique())]
    order = properties.sort_values("depth", kind="stable")
    root, level1, level2 = (order[order["depth"] == depth] for depth in (0, 1, 2))
    rows1, rows2 = -(-len(level1) // (ncols - 1)), -(-len(level2) // ncols)

    def draw(ax, node, title=True):
        colour = BRANCH_COLOUR.get(node.branch, "#8c8c8c")
        block = full[full["path"] == node.path]
        for metric, (_, style, mark) in NODE_METRIC.items():
            stats = block.groupby("n_hvg")[metric].agg(["mean", "std"]).dropna(subset=["mean"])
            n, mean, std = stats.index.to_numpy(), stats["mean"].to_numpy(), stats["std"].fillna(0).to_numpy()
            ax.fill_between(n, mean - std, mean + std, color=colour, alpha=0.18, lw=0)
            ax.plot(n, mean, color=colour, ls=style, lw=0.9)
            if metric == "macro_f1":   # the best budget measured, read off the mean curve
                best = int(stats.index[stats["mean"] == stats["mean"].max()].min())
                ax.axvline(best, color=colour, ls=":", lw=0.9, zorder=1)
                ax.text(best, 1.0, f"{best}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=5, color=colour)
            more = rest[rest["path"] == node.path].groupby("n_hvg")[metric].mean().dropna()
            if len(more):
                ax.plot([n[-1], *more.index], [mean[-1], *more.to_numpy()], color=colour, ls="-", lw=0.6, alpha=0.5)
                ax.scatter(more.index, more.to_numpy(), s=7, marker=mark, facecolor="white",
                           edgecolor=colour, lw=0.6, zorder=3)
        # the root's box names it, an empty title keeps its panel level with the others
        ax.set_title(node.node if title else " ", loc="left", fontsize=6.5, pad=6)
        ax.text(0.97, 0.05, f"{node.arity} ch. · {_cells(node.size)}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color="#6a6a6a",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6})
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000, 10000], ["10", "100", "1k", "10k"])
        ax.set_xticks(sorted(set(curves["n_hvg"]) - {10, 100, 1000, 10000}), minor=True)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.tick_params(which="minor", length=1.2)
        ax.grid(color="#ececec")
        ax.set_axisbelow(True)

    figure = plt.figure(figsize=(width, row_height * (rows1 + rows2) + 1.0))
    top, bottom = figure.subfigures(2, 1, height_ratios=[rows1, rows2], hspace=0.04)
    left, right = top.subfigures(1, 2, width_ratios=[1, ncols - 1], wspace=0.03)
    for box, nodes, cols, heading in ((left, root, 1, "root"), (right, level1, ncols - 1, "L1 nodes"),
                                      (bottom, level2, ncols, "L2 nodes")):
        # the frame drawn just inside the box, so the edges of the figure do not clip it
        box.add_artist(Rectangle((0.004, 0.004), 0.992, 0.992, transform=box.transSubfigure, fill=False,
                                 edgecolor="#c8c8c8", linewidth=0.8))
        box.suptitle(heading, x=0.03, ha="left", fontsize=7, fontweight="bold", color="#4a4a4a")
        axes = box.subplots(rows1 if box is left else -(-len(nodes) // cols), cols, squeeze=False)
        for ax, node in zip(axes.flat, nodes.itertuples()):
            draw(ax, node, title=box is not left)
        for ax in axes.flat[len(nodes):]:
            ax.set_visible(False)
        for column in axes.T:   # the lowest panel drawn in each column names the axis
            drawn = [a for a in column if a.get_visible()]
            if drawn:
                drawn[-1].set_xlabel("genes per node", fontsize=6.5)
    handles = [plt.Line2D([], [], color="#3a3a3a", ls=style, lw=0.9, label=label)
               for label, style, _ in NODE_METRIC.values()]
    handles.append(plt.Line2D([], [], color="#3a3a3a", ls=":", lw=0.9, label="best macro-F1 budget"))
    if len(rest):
        handles.append(plt.Line2D([], [], color="#3a3a3a", ls="-", lw=0.6, alpha=0.5, marker="o", ms=2.5,
                                  mfc="white", mew=0.6, label=f"fold {int(rest['fold'].min())} only"))
    return _finish(figure, handles, -(-len(handles) // 2) if len(handles) > 4 else len(handles), path)


NODE_PROPERTY = {"depth": ("depth", False), "arity": ("children", False), "size": ("cells below", True),
                 "smallest_child": ("smallest child, cells", True), "balance": ("children's balance", False),
                 "best": ("best value reached", False)}   # the node's difficulty, not its shape


def node_needs_figure(saturations: dict, properties: pd.DataFrame, path=None, width=None, height=1.5):
    """
    The genes each node needs, the n reaching 90 % of its gain, against its shape: one row per
    metric, one panel per property, coloured by Level 1 branch, Spearman's rho over the nodes in
    the corner.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from hvgsel import figures
    from hvgsel.report import _finish
    width = width or figures.FULL
    figure, axes = plt.subplots(len(saturations), len(NODE_PROPERTY), sharey=True, squeeze=False,
                                figsize=(width, height * len(saturations) + 0.4))
    for row, (metric, sat) in zip(axes, saturations.items()):
        frame = sat.drop(columns=[c for c in ("node", "depth") if c in sat]).merge(properties, on="path")
        need = frame.filter(regex=r"^n\d+$").columns[0]
        for ax, (prop, (label, log)) in zip(row, NODE_PROPERTY.items()):
            x = frame[prop].to_numpy(float)
            if prop in ("depth", "arity"):   # integers: spread the ties
                x = x + np.random.default_rng(0).uniform(-0.12, 0.12, len(x))
            ax.scatter(x, frame[need], s=11, linewidths=0, alpha=0.9,
                       color=[BRANCH_COLOUR.get(b, "#8c8c8c") for b in frame["branch"]])
            rho, p = spearmanr(frame[prop], frame[need])
            ax.text(0.04, 0.96, rf"$\rho$ = {rho:+.2f}" + ("*" if p < 0.05 else ""), transform=ax.transAxes,
                    ha="left", va="top", fontsize=6)
            if log:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.grid(color="#ececec")
            ax.set_axisbelow(True)
            if ax in axes[-1]:
                ax.set_xlabel(label, fontsize=6.5)
        row[0].set_ylabel(f"genes to 90 % of the\n{NODE_METRIC[metric][0]} gain", fontsize=6.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=c, label=b)
               for b, c in BRANCH_COLOUR.items() if b in set(properties["branch"])]
    return _finish(figure, handles, len(handles), path)

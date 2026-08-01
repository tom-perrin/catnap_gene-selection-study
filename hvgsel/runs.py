"""
The downstream experiment: per-node re-selection against one global selection.

A Run owns runs/<dataset>/<tag>/ and holds the two arms of one comparison:

    run.json              dataset, config, HVG method, n_hvg, baseline spec, split seed, sizes
    reselect/             catnap project dir (config.yml, models/), HVGs reselected per node
    fixed_global/         the same, gene space restricted to the root selection
    scores.csv            accuracy and macro-F1 per level, per arm
    breakdown_L*.csv      per-node split accuracy of both arms

Training is skipped when a completed models/ is already there, so re-running the notebook reuses
the saved models. Predictions are not stored, they follow from the models and the seeded split.

Note: everything here writes to disk, nothing draws. Figures and tables are report.py.
"""

# IMPORTS
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

from catnap_core import hierarchy
from catnap_core.hierarchy import norm
from catnap_core.predict import predict_labels
from catnap_core.train import train_config
from catnap_core.utils import finest_labels

from hvgsel.comparisons import Score
from hvgsel.datasets import stratified_draw
from hvgsel.scorer import select_genes
from hvgsel.scores import SUPERVISED
from hvgsel.taxonomy import Taxonomy


# ---------------------------------------------------------------------


ARMS = ("reselect", "fixed_global")         # (treatment, baseline) -- drives every delta
TRAIN_ORDER = ("fixed_global", "reselect")  # cheapest first, so a crash costs less

BASELINES = "_baselines"   # shared cache of fixed_global gene rankings, per dataset
TAG_PATTERN = re.compile(r"^(?P<hvg>.+)_(?P<n_hvg>\d+)hvg_(?P<backend>[^_]+)$")

# HVG blocks a configuration variant can be built with, and the root selection defining its
# fixed_global arm. The baseline always matches the method under test.
HVG_PARAMS = {
    "f_statistic": {"min_cells_per_child": 2},
    "kruskal_wallis": {"min_cells_per_child": 2, "gene_block": 4096},
    "seurat_v3": {"batch_key": None, "share_loess": True, "span_loess": 0.3},
}
BASELINE = {
    "f_statistic": Score("f_stat", "root", cut="leaves"),
    "kruskal_wallis": Score("kw", "root", cut="leaves"),
    "seurat_v3": Score("vst", "root"),
}

# catnap's own LGBM defaults, device pinned. Deliberately untuned: the logit arm runs on its
# defaults too, so tuning one side only would confound the model class with the tuning effort.
# If a node overfits or training drags, the levers are min_child_samples (up), colsample_bytree
# (down, matters most at 2000 genes) and max_bin (down to 127).
LGBM_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.1,
    "max_bin": 255,
    "min_child_samples": 20,
    "colsample_bytree": 1.0,
    "n_jobs": -1,
    "device": "cpu",                # the CUDA driver/NVML versions disagree on this host
    "early_stopping_patience": 20,  # most nodes stop far short of n_estimators
    "validation_size": 0.1,
    "seed": 0,
}

# Node models a run can be built with. None means the config's own block for that backend,
# which is what the logit runs were trained under.
BACKEND_PARAMS = {"logit": None, "lgbm": LGBM_PARAMS}


# --- Split -----------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """A seeded, finest-label-stratified train/test split, restricted to the config's tree."""

    train: np.ndarray
    test: np.ndarray
    seed: int
    test_frac: float
    subsample: int | None

    @property
    def key(self) -> str:
        """Digest of exactly which cells train, what a cached selection depends on."""
        return hashlib.md5(np.packbits(self.train).tobytes()).hexdigest()[:10]

    @property
    def meta(self) -> dict:
        return {"seed": self.seed, "test_frac": self.test_frac, "subsample": self.subsample,
                "train_key": self.key,
                "n_train": int(self.train.sum()), "n_test": int(self.test.sum())}


def make_split(adata, label_cols, config_path, test_frac=0.2, seed=0, subsample=None) -> Split:
    finest = finest_labels(adata, label_cols)
    root_name, root_node = hierarchy.root_item(hierarchy.load_config(config_path))
    pool = np.flatnonzero(np.isin(finest, list(hierarchy.subtree_labels(root_name, root_node))))

    rng = np.random.default_rng(seed)
    keep = pool
    if subsample is not None and subsample < pool.size:
        keep = pool[stratified_draw(finest[pool], subsample / pool.size, rng)]

    test = np.zeros(adata.n_obs, bool)
    for label in np.unique(finest[keep]):
        idx = keep[finest[keep] == label]
        test[rng.choice(idx, size=int(round(idx.size * test_frac)), replace=False)] = True
    train = np.zeros(adata.n_obs, bool)
    train[keep] = True
    return Split(train & ~test, test, seed, test_frac, subsample)


# --- Run on disk -----------------------------------------------------

def config_tag(config_path) -> str:
    """<hvg>_<n>hvg_<backend>, read off the config's root node: the run's directory name."""
    _, root = hierarchy.root_item(hierarchy.load_config(config_path))
    hvg = str(root.get("hvg", "none"))
    if root.get("hvg_params", {}).get("share_loess"):
        hvg += "_shareloess"
    n_hvg = int(root.get("hvg_params", {}).get("n_top_genes", 0))
    return f"{hvg}_{n_hvg:04d}hvg_{root.get('backend', 'none')}"


class Run:
    """One reselect-vs-fixed_global comparison, kept under root/dataset/tag."""

    def __init__(self, root, dataset: str, tag: str):
        # absolute, catnap's training loop mixes project_dir with resolved paths and a relative
        # one makes its Path.relative_to raise part-way through the tree
        self.dir = (Path(root) / dataset / tag).resolve()
        self.dataset, self.tag = dataset, tag
        self.dir.mkdir(parents=True, exist_ok=True)

    def arm_dir(self, arm: str) -> Path:
        return self.dir / arm

    def arm_trained(self, arm: str) -> bool:
        """
        True only if the arm holds a model tree catnap finished writing.

        A crashed run leaves a partial models/ behind, so the recorded status is what decides.
        Otherwise the arm would be skipped on the next pass and silently keep an incomplete tree.
        """
        path = self.arm_dir(arm)
        if not (path / "models").exists():
            return False
        record = path / "training_metadata.json"
        if not record.exists():
            return True                       # pre-dates the status field, assume complete
        return json.loads(record.read_text()).get("status") == "completed"

    @property
    def trained(self) -> bool:
        return all(self.arm_trained(arm) for arm in ARMS)

    def train(self, adata_train, label_cols, config_path, genes, force=False) -> None:
        """
        Train both arms, skipping any arm that already holds a completed model tree unless force.

        fixed_global goes first: it trains in the restricted gene space and takes minutes, where
        reselect reselects over the whole genome at every node. If the long arm dies, the short
        one is already on disk and complete.
        """
        for arm in TRAIN_ORDER:
            path = self.arm_dir(arm)
            if self.arm_trained(arm) and not force:
                print(f"{arm}: reusing {path}")
                continue
            if (path / "models").exists():
                print(f"{arm}: discarding an incomplete model tree and retraining")
                shutil.rmtree(path / "models")
            path.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path, path / "config.yml")
            data = adata_train if arm == "reselect" else adata_train[:, genes].copy()
            train_config(data, path, label_cols, reselect_hvg=(arm == "reselect"), verbose=1)

    def predict(self, adata_test) -> dict:
        return {arm: predict_labels(adata_test, self.arm_dir(arm)) for arm in ARMS}

    def write(self, name: str, table: pd.DataFrame) -> None:
        table.to_csv(self.dir / f"{name}.csv", index=False)

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.dir / f"{name}.csv")

    def write_meta(self, meta: dict) -> None:
        (self.dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))

    @property
    def meta(self) -> dict:
        path = self.dir / "run.json"
        return json.loads(path.read_text()) if path.exists() else {}


# --- Scoring the held-out cells --------------------------------------

def truth_table(adata_test, label_cols) -> pd.DataFrame:
    """Ground truth per level, normed exactly like the predicted node names."""
    return pd.DataFrame(
        {f"L{k}": adata_test.obs[col].map(lambda v: "" if pd.isna(v) else norm(v)).values
         for k, col in enumerate(label_cols, start=1)},
        index=adata_test.obs_names,
    )


def score_levels(truth, pred, levels) -> pd.DataFrame:
    """Accuracy and macro-F1 per level and arm, where the truth is known and the arm routed."""
    rows = []
    for arm, table in pred.items():
        for level in levels:
            actual = truth[level].astype(str)
            predicted = table[f"{level}_pred"]
            keep = (actual != "") & predicted.notna()
            rows.append({
                "arm": arm, "level": level, "n": int(keep.sum()),
                "accuracy": accuracy_score(actual[keep], predicted[keep].astype(str)),
                "macro_f1": f1_score(actual[keep], predicted[keep].astype(str),
                                     average="macro", zero_division=0),
            })
    return pd.DataFrame(rows)


def breakdown(truth, pred, levels, by: int, min_n: int = 20) -> pd.DataFrame:
    """
    Accuracy of the level-by nodes' own split, at level by+1, grouped by the true level-by node.
    Restricted to cells both arms routed that deep, in groups of min_n or more.

    Note: nodes that are leaves of the taxonomy at this level are skipped. Their cells carry the
    node's own name as the finer label, so the only ones surviving the both-arms-routed filter
    are the misrouted ones, and the group scores 0 by construction instead of measuring a split.
    """
    child = levels[by + 1]
    group = truth[levels[by]].astype(str)
    routed = np.logical_and.reduce([pred[arm][f"{child}_pred"].notna().to_numpy() for arm in ARMS])

    rows = []
    for name in sorted(group.unique()):
        labelled = (group == name) & (truth[child] != "")
        if name == "" or truth[child][labelled].nunique() < 2:
            continue
        keep = labelled & routed
        if keep.sum() < min_n:
            continue
        actual = truth[child][keep].astype(str)
        accuracy = {arm: accuracy_score(actual, pred[arm][f"{child}_pred"][keep].astype(str))
                    for arm in ARMS}
        rows.append({"node": name, "n_test": int(keep.sum()),
                     **{f"acc_{arm}": accuracy[arm] for arm in ARMS},
                     "delta": accuracy[ARMS[0]] - accuracy[ARMS[1]]})
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def evaluate(run: Run, adata_test, label_cols, min_n: int = 20) -> pd.DataFrame:
    """Predict the held-out cells with both arms, write scores.csv and breakdown_*.csv."""
    pred = run.predict(adata_test)
    truth = truth_table(adata_test, label_cols)
    levels = list(truth.columns)

    scores = score_levels(truth, pred, levels)
    run.write("scores", scores)
    for by in range(len(levels) - 1):
        run.write(f"breakdown_{levels[by + 1]}_by_{levels[by]}",
                  breakdown(truth, pred, levels, by, min_n))
    return scores


def baseline_selection(adata_train, label_cols, spec: Score, n_top: int, cache_dir, split) -> list:
    """
    Gene names for the fixed_global arm, cached on disk.

    The most expensive single step of a run, a supervised selection over every training cell and
    the whole genome, and it depends only on the method, the cut and the split, not the budget.
    The supervised selectors return their genes in descending score order, so one full ranking is
    computed, written out before any training starts, and sliced for any n_top.

    Note: seurat_v3 returns an unordered mask, so it is selected per budget instead, in seconds.
    """
    taxonomy = Taxonomy(adata_train.obs, label_cols)
    supervised = spec.method in SUPERVISED
    if not supervised:
        return select_genes(adata_train, taxonomy, spec, n_top)

    key = "-".join([spec.method, str(spec.node), str(spec.cut), split.key])
    path = Path(cache_dir) / f"{key.replace(' ', '_')}.json"
    if path.exists():
        ranking = json.loads(path.read_text())["ranking"]
        print(f"baseline {spec.name}: reusing the cached ranking ({len(ranking):,} genes)")
    else:
        print(f"baseline {spec.name}: ranking the whole genome over "
              f"{int(split.train.sum()):,} training cells", flush=True)
        # n_top must stay below n_vars, at or above it the selector short-circuits and returns
        # every gene in var order, which would not be a ranking
        ranking = select_genes(adata_train, taxonomy, spec, adata_train.n_vars - 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spec": spec.spec, "split": split.meta,
                                    "n_genes": adata_train.n_vars, "ranking": ranking}, indent=1))
        print(f"baseline {spec.name}: cached to {path}")
    return ranking[:n_top]


def compare_arms(adata, dataset: str, label_cols, config_path, baseline: Score, n_hvg: int,
                 root="runs", tag: str | None = None, test_frac=0.2, seed=0, subsample=None,
                 min_n=20, force=False, split: Split | None = None) -> Run:
    """
    Full experiment: split, train or reuse both arms, predict, score, write under root/dataset/tag.

    baseline is the root selection that defines the fixed_global gene set.
    """
    config_path = Path(config_path)
    run = Run(root, dataset, tag or config_tag(config_path))
    if split is None:
        split = make_split(adata, label_cols, config_path, test_frac, seed, subsample)
    adata_test = adata[split.test].copy()

    if run.trained and not force:
        print(f"both arms already trained under {run.dir}")
    else:  # training cells and baseline selection are only needed to train
        adata_train = adata[split.train].copy()
        genes = baseline_selection(adata_train, label_cols, baseline, n_hvg,
                                   Path(root) / dataset / BASELINES, split)
        run.train(adata_train, label_cols, config_path, genes, force=force)
        del adata_train

    evaluate(run, adata_test, label_cols, min_n)
    run.write_meta({**run.meta, "dataset": dataset, "tag": run.tag, "label_cols": list(label_cols),
                    "config": str(config_path), "n_hvg": int(n_hvg),
                    "baseline": baseline.spec, "n_genes": int(adata.n_vars), **split.meta})
    return run


# --- Configuration variants ------------------------------------------

def variant_config(source, hvg: str, n_top: int, hvg_params: dict | None = None,
                   backend: str | None = None, backend_params: dict | None = None) -> dict:
    """
    source with every node's HVG block replaced by hvg at n_top genes, backend too if given.

    The config applies both blocks to every internal node through a YAML anchor, so all of them
    have to be overridden.

    Note: backend_params=None falls back to the config's own <backend>_default block, so naming
    a backend the config already uses reproduces its runs instead of clearing their parameters.
    """
    config = yaml.safe_load(Path(source).read_text())   # safe_load resolves the anchors
    if backend and backend_params is None:
        backend_params = config.get(f"{backend}_default", {}).get("backend_params", {})

    def override(node):
        if not isinstance(node, dict):
            return
        if "hvg" in node:
            node["hvg"] = hvg
            node["hvg_params"] = {"n_top_genes": int(n_top), **HVG_PARAMS[hvg], **(hvg_params or {})}
        if backend and "backend" in node:
            node["backend"] = backend
            node["backend_params"] = dict(backend_params or {})
        for child in (node.get("children") or {}).values():
            override(child)

    override(config["root"])
    return config


def _dump(config: dict) -> Path:
    """The config on a temporary file, so config_tag can read back the tag it implies."""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return Path(handle.name)


def run_variants(adata, dataset: str, label_cols, source_config, configs, root="runs",
                 hvg_params: dict | None = None, backend: str | None = None,
                 backend_params: dict | None = None, **kwargs) -> list[Run]:
    """
    Train and score one run per (hvg, n_top) of configs, each derived from source_config.

    The variant is written to its own run directory, so the tag it implies and the directory it
    lives in always agree, and compare_arms skips whatever is already trained.
    """
    runs = []
    for hvg, n_top in configs:
        staged = _dump(variant_config(source_config, hvg, n_top, hvg_params, backend, backend_params))
        tag = config_tag(staged)
        destination = Path(root) / dataset / tag
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(staged, destination / "config.yml")

        print(f"\n===== {tag} =====", flush=True)
        runs.append(compare_arms(adata, dataset, label_cols, destination / "config.yml",
                                 baseline=BASELINE[hvg], n_hvg=n_top, root=root, tag=tag, **kwargs))
    return runs


def run_repeats(adata, dataset: str, label_cols, source_config, hvg: str, n_top: int,
                seeds=range(10), root="runs_repeats", hvg_params: dict | None = None,
                backend: str | None = None, backend_params: dict | None = None,
                **kwargs) -> list[Run]:
    """
    One configuration re-run over seeds, to size the noise on each per-node gain.

    A single run shows a gain per node but not whether it reproduces. Each repeat draws its own
    stratified split, which re-draws the HVG selection, both arms' models and the held-out cells
    together. The logit backend is deterministic given its data, so the split is the only thing
    worth varying.

    Monte Carlo rather than K-fold, because the test fraction is then independent of the repeat
    count: 20% held out estimates a node's accuracy far more tightly than the 10% a 10-fold
    design would force, and that is where the noise question is sharpest.

    Note: repeats share training cells, so their spread is a noise diagnostic, not a standard
    error. Seed 0 draws the split of the matching runs/ run, so it doubles as a consistency check.
    root is separate to keep the cross-run tables one row per configuration.
    """
    staged = _dump(variant_config(source_config, hvg, n_top, hvg_params, backend, backend_params))
    base = config_tag(staged)

    runs = []
    for seed in seeds:
        tag = f"{base}_seed{seed}"
        destination = Path(root) / dataset / tag
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(staged, destination / "config.yml")

        print(f"\n===== {tag} =====", flush=True)
        runs.append(compare_arms(adata, dataset, label_cols, destination / "config.yml",
                                 baseline=BASELINE[hvg], n_hvg=n_top, root=root, tag=tag,
                                 seed=seed, **kwargs))
    staged.unlink()
    return runs


# --- Reading back what is on disk ------------------------------------

def refresh_runs(adata, dataset: str, label_cols, root="runs", min_n=20, tags=None) -> list[Run]:
    """
    Re-score every trained run of dataset from its saved models, data loaded once.

    No training and no baseline selection: this is how to produce scores.csv and breakdown_*.csv
    for runs whose models exist but whose numbers predate them.

    Note: the split is rebuilt from each run.json, runs written before those fields existed fall
    back to the study's defaults (20% test, seed 0, no subsample).
    """
    runs = []
    for run_dir in sorted((Path(root) / dataset).iterdir()):
        if run_dir.name.startswith(("_", ".")) or (tags and run_dir.name not in tags):
            continue
        run = Run(root, dataset, run_dir.name)
        if not run.trained:
            print(f"{run.tag}: no models, skipping")
            continue
        meta = run.meta
        split = make_split(adata, label_cols, run.arm_dir("reselect") / "config.yml",
                           meta.get("test_frac", 0.2), meta.get("seed", 0), meta.get("subsample"))
        print(f"{run.tag}: predicting {int(split.test.sum()):,} held-out cells", flush=True)
        evaluate(run, adata[split.test].copy(), label_cols, min_n)
        run.write_meta({**meta, "dataset": dataset, "tag": run.tag, "label_cols": list(label_cols),
                        "n_genes": int(adata.n_vars), **split.meta})
        runs.append(run)
    return runs


def configurations(root, dataset: str, n_hvg: int) -> list[Run]:
    """
    Every run of dataset at n_hvg that already has its per-node breakdowns on disk.

    These are the configurations report.config_breakdown_figure reads together.
    """
    runs = []
    for run_dir in sorted((Path(root) / dataset).iterdir()):
        parts = TAG_PATTERN.match(run_dir.name)
        if parts is None or int(parts["n_hvg"]) != n_hvg:
            continue
        if not list(run_dir.glob("breakdown_*.csv")):
            print(f"  {run_dir.name}: no breakdown on disk, skipping")
            continue
        runs.append(Run(root, dataset, run_dir.name))
    return runs

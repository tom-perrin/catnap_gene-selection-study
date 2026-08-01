"""
On-disk results, so every figure redraws without recomputing anything.

Layout, under <root>/<dataset_id>/:

    dataset.json          the matrix the scores were computed on
    scores/<id>.npz       gene names, per-gene score vector, LOESS knots for a local vst fit
    scores/index.json     <id> -> the score's spec and metadata
    metrics.csv           one row per comparison, its two specs and its statistics

Note: the directory is self-describing, sharing it shares the results.
"""

# IMPORTS
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------


INDEX = "index.json"
METRICS = "metrics.csv"


def digest(payload) -> str:
    """Short stable hash of any JSON-able payload, what every id here is built from."""
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _slug(text: str, limit: int = 28) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in str(text))
    return "-".join(filter(None, cleaned.split("-")))[:limit] or "x"


def dataset_id(name: str, adata) -> str:
    """Id of the matrix in use, so a different subsample gets its own results directory."""
    step = max(1, adata.n_obs // 500)
    fingerprint = digest({
        "shape": [adata.n_obs, adata.n_vars],
        "genes": adata.var_names[::max(1, adata.n_vars // 200)].astype(str).tolist(),
        "cells": adata.obs_names[::step].astype(str).tolist(),
    })
    return f"{name}-{fingerprint}"


def score_id(spec: dict) -> str:
    """Readable, deterministic file name for a score spec."""
    qualifier = spec.get("cut") or spec.get("loess_from") or "local"
    return f"{spec['method']}-{_slug(spec['node'])}-{_slug(qualifier, 12)}-{digest(spec)}"


class Store:
    """Reads and writes one dataset's score vectors, their metadata and its metrics table."""

    def __init__(self, root, dataset: str, info: dict | None = None):
        self.dir = Path(root) / dataset
        self.scores_dir = self.dir / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        if info is not None and not (self.dir / "dataset.json").exists():
            _write_json(self.dir / "dataset.json", {"dataset": dataset, **info})

    # --- Scores ------------------------------------------------------
    def load(self, sid: str) -> dict | None:
        path = self.scores_dir / f"{sid}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def save(self, sid: str, arrays: dict, meta: dict) -> None:
        np.savez_compressed(self.scores_dir / f"{sid}.npz", **arrays)
        index = self.index()
        index[sid] = {**meta, "saved": datetime.now().isoformat(timespec="seconds")}
        _write_json(self.scores_dir / INDEX, index)

    def index(self) -> dict:
        path = self.scores_dir / INDEX
        return json.loads(path.read_text()) if path.exists() else {}

    def catalog(self) -> pd.DataFrame:
        """The stored score vectors as a table: what, on how many cells, when."""
        index = self.index()
        return pd.DataFrame(index.values(), index=pd.Index(index, name="score_id"))

    # --- Metrics -----------------------------------------------------
    def log(self, row: dict) -> None:
        """Upsert one comparison's statistics into metrics.csv, keyed on id."""
        table = self.metrics()
        if not table.empty and "id" in table.columns:
            table = table[table["id"] != row["id"]]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(self.dir / METRICS, index=False)

    def metrics(self) -> pd.DataFrame:
        path = self.dir / METRICS
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))

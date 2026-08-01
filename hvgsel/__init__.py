"""
HVG-selection comparisons for catnap_core.

A comparison is two Score specs (method, node, cut / LOESS source).
Scores are computed with catnap_core's own HVG code, cached on disk with their metadata,
and compared with ranking-agreement statistics.
runs.py then asks whether the difference moves accuracy.

See README.md for the notation and the experiment list.

Note: importing this sets matplotlib's rcParams to the paper style.
"""

# IMPORTS
from hvgsel import figures, plots, report, runs
from hvgsel.comparisons import ALL_NN, ALL_NR, Comparison, Score, cl, keys_for, node_vs_node, node_vs_root
from hvgsel.datasets import DATASETS, Dataset, load_dataset
from hvgsel.scorer import Scorer

__all__ = [
    "DATASETS", "Dataset", "load_dataset",
    "Score", "Comparison", "ALL_NR", "ALL_NN", "keys_for",
    "node_vs_root", "node_vs_node", "cl",
    "Scorer", "plots", "figures", "runs", "report",
]

figures.use_paper_style()

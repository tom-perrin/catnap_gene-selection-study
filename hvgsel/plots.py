"""
Selection-comparison figures: two panels per comparison, and the multi-node summary.

Reads Result objects, writes PDFs and one LaTeX digest.
The downstream experiment's own figures are report.py.
"""

# IMPORTS
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvgsel import figures
from hvgsel.metrics import STAT_NAMES, Result


# ---------------------------------------------------------------------


BOTH, NEITHER = "#2a9d8f", "#cfcfcf"
ONLY_A, ONLY_B = "#e76f51", "#4c72b0"
KEY_COLORS = {"F-NR": "#c2410c", "KW-NR": "#7c3aed", "VST(v)-NR": "#1d4ed8", "VST(v_0)-NR": "#60a5fa",
              "F-NN": "#c2410c", "KW-NN": "#7c3aed", "VST(v_0)-NN": "#1d4ed8",
              "F-CL": "#c2410c", "KW-CL": "#7c3aed"}


def _log_scale(values: np.ndarray) -> np.ndarray:
    """
    log10 for display, perfect separators (F = +inf) capped just above the finite max.
    """
    values = np.asarray(values, float).copy()
    finite = np.isfinite(values) & (values > 0)
    cap = values[finite].max() * 2 if finite.any() else 1.0
    values[~np.isfinite(values)] = cap
    return np.log10(np.clip(values, cap * 1e-9, None))


def _selection_class(result: Result) -> np.ndarray:
    top_a = np.argsort(-result.a, kind="stable")[:result.n_top]
    top_b = np.argsort(-result.b, kind="stable")[:result.n_top]
    labels = np.full(result.a.size, "neither", dtype=object)
    labels[top_b] = "b only"
    labels[top_a] = "a only"
    labels[np.intersect1d(top_a, top_b)] = "both"
    return labels


def panels(result: Result, axes, legend: bool = True) -> None:
    """
    [0] gene scores, coloured by which selection keeps the gene.
    [1] top-k overlap vs k.
    """
    comparison = result.comparison
    x, y = _log_scale(result.b), _log_scale(result.a)
    classes = _selection_class(result)

    style = {"neither": (NEITHER, 1.5, 0.3), "both": (BOTH, 3, 0.6),
             "a only": (ONLY_A, 3.5, 0.7), "b only": (ONLY_B, 3.5, 0.7)}
    names = {"a only": f"${comparison.latex_a}$ only", "b only": f"${comparison.latex_b}$ only",
             "both": "both", "neither": "neither"}
    for label, (color, size, alpha) in style.items():
        keep = classes == label
        axes[0].scatter(x[keep], y[keep], s=size, c=color, alpha=alpha, linewidths=0,
                        label=names[label], rasterized=True)
    span = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[0].plot(span, span, "--", color="k", lw=0.6, alpha=0.5)
    axes[0].set_xlabel(rf"$\log_{{10}}\ {comparison.latex_b}$")
    axes[0].set_ylabel(rf"$\log_{{10}}\ {comparison.latex_a}$")
    if legend:
        axes[0].legend(markerscale=2.5, loc="lower right")

    order_a = np.argsort(-result.a, kind="stable").tolist()
    order_b = np.argsort(-result.b, kind="stable").tolist()
    ks = np.unique(np.round(np.logspace(1, np.log10(result.a.size), 40)).astype(int))
    overlap = [len(set(order_a[:k]) & set(order_b[:k])) / k for k in ks]
    axes[1].plot(ks, overlap, color=BOTH)
    axes[1].axvline(result.n_top, color="k", ls="--", lw=0.6, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(f"top-$k$ (dashed: $n={result.n_top}$)")
    axes[1].set_ylabel(r"$|A_k \cap B_k|\,/\,k$")


def show(results, title: str = "", path=None, width=figures.FULL, row_height: float = 1.9):
    """
    One row of two panels per comparison.

    Pass one Result or several, e.g. the four methods of a node-vs-root experiment, to stack them.
    """
    results = [results] if isinstance(results, Result) else list(results)
    figure, axes = plt.subplots(len(results), 2, figsize=(width, row_height * len(results)),
                               squeeze=False)
    for row, result in enumerate(results):
        panels(result, axes[row], legend=(row == 0))
        axes[row][0].set_title(f"[{result.key}]", loc="left", fontweight="bold")
        axes[row][1].set_title(result.summary_short(), loc="left")
        print(result.summary())
    if title:
        figure.suptitle(title)
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


# --- Multi-node summary ----------------------------------------------

def _pivot(summary: pd.DataFrame, rows, keys):
    keys = list(keys or summary["key"].unique())
    rows = list(rows or summary["node"].unique())
    return rows, keys, summary.set_index(["node", "key"])


def summary_figure(summary: pd.DataFrame, path=None, rows=None, keys=None, control: str | None = None,
                   stats=("changed%", "Jaccard", "Spearman", "RBO"), width=figures.FULL):
    """
    One panel per statistic, one line per node, one marker per method.

    The wide node x statistic table's numbers, readable at \\textwidth.
    """
    rows, keys, lookup = _pivot(summary, rows, keys)
    y = np.arange(len(rows))[::-1]

    height = max(1.6, 0.17 * len(rows) + 0.9)
    figure, axes = plt.subplots(1, len(stats), figsize=(width, height), sharey=True, squeeze=False)
    for column, (ax, stat) in enumerate(zip(axes[0], stats)):
        for key in keys:
            ax.scatter([lookup.loc[(node, key), stat] for node in rows], y, s=9,
                       color=KEY_COLORS.get(key, "#333"), zorder=3, linewidths=0,
                       label=key if column == len(stats) - 1 else None)
        ax.set_title(stat)
        ax.grid(axis="x", color="#e8e8e8", zorder=0)
        ax.set_axisbelow(True)
        if stat == "changed%":
            ax.set_xlim(0, 100)
    axes[0][0].set_yticks(y, rows)
    axes[0][0].tick_params(axis="y", length=0)
    for label in axes[0][0].get_yticklabels():
        if label.get_text() == control:
            label.set_fontstyle("italic")
    figure.legend(*axes[0][-1].get_legend_handles_labels(), loc="outside lower center",
                  ncols=len(keys))
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


def summary_html(summary: pd.DataFrame, rows=None, keys=None) -> str:
    """
    All the numbers, for browsing in the notebook. Wrap in IPython.display.HTML.
    """
    rows, keys, lookup = _pivot(summary, rows, keys)

    def cell(node, stat):
        lines = "<br>".join(
            f"<span style='color:{KEY_COLORS.get(key, '#333')}'>{key} "
            f"{_format(stat, lookup.loc[(node, key), stat])}</span>" for key in keys)
        return f"<div style='line-height:1.4'>{lines}</div>"

    table = pd.DataFrame({node: {stat: cell(node, stat) for stat in STAT_NAMES} for node in rows}).T
    table = table[list(STAT_NAMES)].reindex(rows)
    table.index.name = "node"
    css = ("<style>table.hvgsum{border-collapse:collapse}"
           "table.hvgsum td,table.hvgsum th{text-align:center;padding:4px 10px;"
           "font-family:monospace;border:1px solid #d0d0d0}"
           "table.hvgsum th{background:rgba(128,128,128,.12)}</style>")
    return css + table.to_html(escape=False, classes="hvgsum", border=0)


def _format(stat: str, value: float) -> str:
    return f"{value:.0f}%" if stat == "changed%" else f"{value:.3f}"


def summary_latex(summary: pd.DataFrame, path=None, keys=None, control: str | None = None,
                  stats=STAT_NAMES, caption: str = "", label: str = "") -> str:
    """
    Digest table for the paper: per method, the median over nodes and the [min, max] range.

    The random control gets its own row.
    """
    keys = list(keys or summary["key"].unique())
    nodes = summary[summary["node"] != control]
    control_rows = summary[summary["node"] == control]

    def line(frame, key, name):
        cells = []
        for stat in stats:
            values = frame.loc[frame["key"] == key, stat]
            fmt = "{:.0f}" if stat == "changed%" else "{:.2f}"
            cells.append(f"{fmt.format(values.median())} "
                         f"[{fmt.format(values.min())}, {fmt.format(values.max())}]")
        return " & ".join([name, *cells]) + r" \\"

    tex = lambda text: text.replace("_", r"\_").replace("%", r"\%")
    header = " & ".join(["method", *(tex(s) for s in stats)]) + r" \\"
    body = [line(nodes, key, tex(key)) for key in keys]
    if not control_rows.empty:
        body.append(r"\midrule")
        body += [line(control_rows, key, rf"\textit{{control}} {tex(key)}") for key in keys]

    return figures.latex_table(header, body, "l" + "c" * len(stats), path, caption, label)

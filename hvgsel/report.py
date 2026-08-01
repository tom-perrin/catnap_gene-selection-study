"""
Figures and tables of the downstream experiment, read back from what runs.py wrote.

Nothing here trains or predicts. Every function takes Run objects, or the tidy frame runs_table
builds out of every scores.csv on disk, and returns a figure or a LaTeX table.

Four figures share one reading: one horizontal row per node, one panel per level pair, coloured
by whether re-selection helps that node. They are built on the same skeleton (_grid, _axis,
_range_row, _xpad, _finish), and differ in the anchor each row is measured from and in the
verdict its colour states.
"""

# IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from hvgsel import figures
from hvgsel.runs import ARMS, TAG_PATTERN, Run


# ---------------------------------------------------------------------


C_RESELECT, C_FIXED = "#2a78d6", "#e34948"
C_FLIP = "#9a9a9a"     # a node whose gain changes sign across repeats, within split noise

BACKEND_MARK = {"logit": "o", "lgbm": "s"}   # fixed, a backend keeps its marker across figures
# Where a figure shows both at once, the marker's fill is the HVG method: solid for the supervised
# scores, hollow for the unsupervised one, so the two families read apart at a glance.
HVG_FILL = {"f_statistic": 1.0, "kruskal_wallis": 0.4, "seurat_v3": 0.0}

SUPERVISED_HVG = ("f_statistic", "kruskal_wallis")

GAIN_LABEL = "accuracy gain (reselect $-$ fixed_global)"
ACC_LABEL = "accuracy within node"


# --- Reading the runs ------------------------------------------------

def runs_table(root="runs") -> pd.DataFrame:
    """Every saved run's per-level scores in one tidy frame, the cross-configuration table."""
    rows = []
    for scores_path in sorted(Path(root).glob("*/*/scores.csv")):
        run = Run(scores_path.parents[2], scores_path.parents[1].name, scores_path.parent.name)
        parts = TAG_PATTERN.match(run.tag)
        if parts is None:
            raise ValueError(f"run directory '{run.tag}' is not <hvg>_<n>hvg_<backend>")
        rows.append(run.read("scores").assign(
            dataset=run.dataset, tag=run.tag, hvg=parts["hvg"], backend=parts["backend"],
            n_hvg=int(parts["n_hvg"]), models_saved=bool(run.meta.get("models_saved", run.trained))))
    if not rows:
        raise FileNotFoundError(f"no runs under {root}/")
    return pd.concat(rows, ignore_index=True)


def repeat_column(runs, levels, by: int, column: str) -> pd.DataFrame:
    """One breakdown column per node, one table column per repeat, NaN where a node is absent."""
    name = f"breakdown_{levels[by + 1]}_by_{levels[by]}"
    columns = [run.read(name).set_index("node")[column].rename(run.tag) for run in runs]
    return pd.concat(columns, axis=1)


def repeat_deltas(runs, levels, by: int) -> pd.DataFrame:
    """Per-node reselect-minus-fixed_global gain, one column per run, NaN where a node absent."""
    return repeat_column(runs, levels, by, "delta")


def repeat_levels(runs, metric="accuracy") -> pd.DataFrame:
    """Per level and arm, the metric's mean and spread over the repeats, plus the gain's."""
    scores = pd.concat([run.read("scores").assign(run=run.tag) for run in runs], ignore_index=True)
    wide = scores.pivot_table(index=["run", "level"], columns="arm", values=metric)
    wide["delta"] = wide[ARMS[0]] - wide[ARMS[1]]
    return wide.groupby("level").agg(["mean", "std", "min", "max", "count"])


def config_label(tag: str) -> str:
    """f_statistic_0300hvg_lgbm -> 'f_statistic x lgbm', naming a cross-configuration column."""
    parts = TAG_PATTERN.match(tag)
    return f"{parts['hvg'].replace('_shareloess', '')} x {parts['backend']}"


def config_title(dataset: str, runs, n_hvg: int) -> str:
    """The configurations named as the grid they cover when they cover one, else listed."""
    parts = [TAG_PATTERN.match(run.tag) for run in runs]
    methods = list(dict.fromkeys(p["hvg"].replace("_shareloess", "") for p in parts))
    backends = list(dict.fromkeys(p["backend"] for p in parts))
    grid = f"{', '.join(methods)}  $\\times$  {', '.join(backends)}"
    if len(methods) * len(backends) != len(runs):
        grid = ", ".join(config_label(run.tag) for run in runs)
    return f"{dataset}, {n_hvg} HVGs: {len(runs)} configurations   ({grid})"


# --- Shared per-node panel -------------------------------------------

def _grid(n_panels: int, tallest: int, width, row_height, min_height, extra=1.1, pad_y=0.9,
          title=None):
    """
    Figure with n_panels panels side by side, all on the same y grid.

    A node then occupies the same height in every one of them. -> (figure, axes).
    """
    height = max(min_height, row_height * tallest + extra + (0.25 if title else 0))
    figure, axes = plt.subplots(1, n_panels, figsize=(width, height), squeeze=False)
    for ax in axes[0]:
        ax.set_ylim(-pad_y, tallest - (1 - pad_y))
    return figure, axes[0]


def _rows(tallest: int, n: int) -> np.ndarray:
    """Row positions of n nodes, top-aligned on a grid tallest rows deep."""
    return tallest - 1 - np.arange(n)


def _axis(ax, y, labels, xlabel: str, title: str) -> None:
    """The frame every per-node panel shares: node names down the side, a grid behind."""
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color="#ececec", zorder=0)
    ax.set_axisbelow(True)


def _xpad(ax, low: float, high: float, fraction=0.08) -> None:
    """annotate() arrows do not autoscale, so a panel drawing them sets its range by hand."""
    pad = max(high - low, 1e-3) * fraction
    ax.set_xlim(low - pad, high + pad)


def _range_row(ax, y, values, anchor: float, colour: str, forward: bool, mean: float,
               dots=True) -> None:
    """
    One node's row: an arrow from anchor to the near end of values, the range as a line, mean
    as a bar. The arrow stops at the near end so it spans the gain without crossing the dots.

    Note: dots draws the values themselves, which the cross-configuration figure styles per
    configuration and draws itself.
    """
    low, high = np.nanmin(values), np.nanmax(values)
    ax.annotate("", xy=(low if forward else high, y), xytext=(anchor, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.1,head_length=0.26",
                                color=colour, lw=0.7, shrinkA=1.5, shrinkB=0), zorder=2)
    ax.plot([low, high], [y, y], color=colour, lw=0.8, alpha=0.5, solid_capstyle="round", zorder=3)
    if dots:
        ax.scatter(values, np.full(np.size(values), y), s=3, color=colour, alpha=0.7,
                   linewidths=0, zorder=4)
    ax.scatter([mean], [y], s=22, marker="|", color=colour, lw=0.9, zorder=5)


def _finish(figure, handles, ncols: int, path, title=None):
    """Legend outside the panels, optional suptitle, then write and show."""
    figure.legend(handles=handles, loc="outside lower center", ncols=ncols)
    if title:
        figure.suptitle(title, fontsize=plt.rcParams["axes.titlesize"])
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


def _ranked(deltas: pd.DataFrame, minimum: int, label: str) -> tuple[pd.Index, str]:
    """Nodes present in at least minimum columns, ordered by mean gain, plus a note."""
    enough = deltas.notna().sum(axis=1) >= minimum
    missing = int((~enough).sum())
    order = deltas[enough].mean(axis=1).sort_values(ascending=False).index
    return order, (f"   [{missing} node(s) not in {label}]" if missing else "")


# --- Per-node figures ------------------------------------------------

def breakdown_figure(breakdowns, path=None, width=figures.FULL, row_height=0.16, min_height=1.7):
    """
    One run, one panel per breakdown.

    A dumbbell per node from fixed_global to reselect, coloured by the sign of the gain.
    """
    breakdowns = [(label, table) for label, table in breakdowns if len(table)]
    treat, base = ARMS
    tallest = max(len(table) for _, table in breakdowns)
    figure, axes = _grid(len(breakdowns), tallest, width, row_height, min_height,
                         extra=0.9, pad_y=0.8)

    for ax, (label, table) in zip(axes, breakdowns):
        y = _rows(tallest, len(table))
        for yy, (_, row) in zip(y, table.iterrows()):
            gain = row["delta"] >= 0
            ax.annotate("", xy=(row[f"acc_{treat}"], yy), xytext=(row[f"acc_{base}"], yy),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.3",
                                        color=C_RESELECT if gain else C_FIXED, lw=1.1,
                                        shrinkA=0, shrinkB=0))
        ax.scatter(table[f"acc_{base}"], y, s=7, color="#8c8c8c", zorder=3, linewidths=0)
        _axis(ax, y, table["node"], ACC_LABEL, label)
        _xpad(ax, min(table[f"acc_{base}"].min(), table[f"acc_{treat}"].min()),
              max(table[f"acc_{base}"].max(), table[f"acc_{treat}"].max()))

    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color="#8c8c8c", label="fixed_global"),
               plt.Line2D([], [], color=C_RESELECT, lw=1.2, label="reselect better"),
               plt.Line2D([], [], color=C_FIXED, lw=1.2, label="fixed_global better")]
    return _finish(figure, handles, 3, path)


def repeat_breakdown_figure(runs, levels, path=None, width=figures.FULL, row_height=0.20,
                            min_height=2.0, min_repeats=None):
    """
    The single-run breakdown read across repeats, on the accuracy axis.

    fixed_global is one black point, its mean over the repeats. reselect is one dot per repeat,
    their range a line and their mean a bar, with an arrow from fixed_global to the near end.

    Note: colour is the reproducibility verdict and comes from the paired per-repeat gain, not
    from the plotted spread. Blue or red when every repeat agrees on the sign, grey when it flips.
    The two arms share a split so their accuracies move together, and reading the reselect spread
    against the black point would count that shared movement twice.
    """
    min_repeats = min_repeats or len(runs)
    treat, base = ARMS
    blocks = []
    for by in range(len(levels) - 1):
        deltas = repeat_deltas(runs, levels, by)
        order, note = _ranked(deltas, min_repeats, "all repeats")
        blocks.append((f"{levels[by + 1]} accuracy by true {levels[by]}" + note, deltas.loc[order],
                       repeat_column(runs, levels, by, f"acc_{treat}").loc[order],
                       repeat_column(runs, levels, by, f"acc_{base}").loc[order].mean(axis=1)))

    tallest = max(len(deltas) for _, deltas, _, _ in blocks)
    figure, axes = _grid(len(blocks), tallest, width, row_height, min_height)

    for ax, (label, deltas, reselect, fixed) in zip(axes, blocks):
        y = _rows(tallest, len(deltas))
        gains, values = deltas.to_numpy(), reselect.to_numpy()
        consistent = np.all(np.nan_to_num(gains, nan=0) > 0, axis=1) | \
                     np.all(np.nan_to_num(gains, nan=0) < 0, axis=1)
        for yy, row, gain, anchor, sure in zip(y, values, gains.mean(axis=1), fixed, consistent):
            colour = (C_RESELECT if gain > 0 else C_FIXED) if sure else C_FLIP
            _range_row(ax, yy, row, anchor, colour, forward=gain > 0, mean=np.nanmean(row))
        ax.scatter(fixed, y, s=6, color="#1a1a1a", linewidths=0, zorder=6)
        _axis(ax, y, deltas.index, ACC_LABEL, label)
        _xpad(ax, min(fixed.min(), np.nanmin(values)), max(fixed.max(), np.nanmax(values)))

    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color="#1a1a1a", label="fixed_global"),
               plt.Line2D([], [], color=C_RESELECT, lw=1.2, label="gain in every repeat"),
               plt.Line2D([], [], color=C_FIXED, lw=1.2, label="loss in every repeat"),
               plt.Line2D([], [], color=C_FLIP, lw=1.2, label="sign flips")]
    return _finish(figure, handles, 4, path)


def config_breakdown_figure(runs, levels, labels=None, path=None, width=figures.FULL,
                            row_height=0.20, min_height=2.0, min_configs=None, title=None):
    """
    repeat_breakdown_figure's reading with configurations in place of repeats.

    Every (HVG method x backend) at one budget on one dataset, so a node's colour answers whether
    re-selection helps it whatever picks the genes and whatever model does the splitting.

    On the gain axis, where the repeats use the accuracy axis: one fixed_global holds across the
    repeats and can be the anchor point, but here every configuration brings its own and only the
    paired difference is comparable. Zero takes that anchor's place.

    Shape is the backend (BACKEND_MARK) and fill is the HVG method (HVG_FILL), so fill separates
    the supervised scores from the unsupervised one, the split that usually decides a node.

    Note: colour is the verdict on those gains, blue when every configuration gains on the node,
    red when every one loses, grey when the sign flips. A configuration leaving a node exactly
    unchanged does not make it a flip, ties break neither way.
    """
    min_configs = min_configs or len(runs)
    parts = [TAG_PATTERN.match(run.tag) for run in runs]
    labels = list(labels or [config_label(run.tag) for run in runs])
    hvgs = [p["hvg"].replace("_shareloess", "") for p in parts]
    spare = iter(np.linspace(0.75, 0.25, len(set(hvgs))))   # a method HVG_FILL does not name
    fills = {h: HVG_FILL[h] if h in HVG_FILL else next(spare) for h in dict.fromkeys(hvgs)}
    styles = [(BACKEND_MARK.get(p["backend"], "D"), fills[h]) for p, h in zip(parts, hvgs)]

    blocks = []
    for by in range(len(levels) - 1):
        deltas = repeat_deltas(runs, levels, by)
        order, note = _ranked(deltas, min_configs, "every configuration")
        blocks.append((f"{levels[by + 1]} gain by true {levels[by]}" + note, deltas.loc[order]))

    tallest = max(len(deltas) for _, deltas in blocks)
    figure, axes = _grid(len(blocks), tallest, width, row_height, min_height, title=title)

    for ax, (label, deltas) in zip(axes, blocks):
        y = _rows(tallest, len(deltas))
        gains = deltas.to_numpy()
        wins, losses = (gains > 0).sum(axis=1), (gains < 0).sum(axis=1)   # NaN counts as neither
        means = np.nanmean(gains, axis=1)
        colours = np.where(losses == 0, C_RESELECT, np.where(wins == 0, C_FIXED, C_FLIP))
        colours[(wins == 0) & (losses == 0)] = C_FLIP                    # unchanged everywhere

        ax.axvline(0, color="#1a1a1a", lw=0.8, zorder=1)
        for yy, row, gain, colour in zip(y, gains, means, colours):
            _range_row(ax, yy, row, 0, colour, forward=gain > 0, mean=gain, dots=False)
        for style in dict.fromkeys(styles):
            mark, fill = style
            columns = [i for i, s in enumerate(styles) if s == style]
            points = gains[:, columns].ravel()
            rows, colour = np.repeat(y, len(columns)), np.repeat(colours, len(columns))
            # an edge whatever the fill, so a pale marker keeps its outline -- and the alpha rides
            # in the colours, since scatter's own `alpha` would overwrite it and fill every marker
            ax.scatter(points, rows, s=11, marker=mark, linewidths=0.7,
                       edgecolors=[to_rgba(c, 0.9) for c in colour],
                       facecolors=[to_rgba(c, 0.9 * fill) for c in colour], zorder=4)

        _axis(ax, y, deltas.index, GAIN_LABEL, label)
        _xpad(ax, min(0, np.nanmin(gains)), max(0, np.nanmax(gains)))

    handles = [plt.Line2D([], [], color=C_RESELECT, lw=1.2, label="gain in every configuration"),
               plt.Line2D([], [], color=C_FIXED, lw=1.2, label="loss in every configuration"),
               plt.Line2D([], [], color=C_FLIP, lw=1.2, label="sign flips")]
    handles += [plt.Line2D([], [], marker=mark, ls="", ms=3.6, color="#6a6a6a", label=name,
                           markerfacecolor=to_rgba("#6a6a6a", fill), markeredgewidth=0.7,
                           markeredgecolor="#6a6a6a") for (mark, fill), name in zip(styles, labels)]
    # three rows: the verdicts fill the first column, the configurations the rest
    return _finish(figure, handles, -(-len(handles) // 3), path, title)


def gain_figure(runs, levels, labels=None, path=None, width=figures.FULL, row_height=0.24,
                min_height=2.2):
    """
    Per-node gain over fixed_global, one marker per run.

    For runs differing in something other than the split, a different node model say. Ordered by
    the mean gain, so a node one run never routed to does not decide the ordering.
    """
    labels = list(labels or [run.tag for run in runs])
    colour = dict(zip(labels, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    blocks = []
    for by in range(len(levels) - 1):
        table = repeat_deltas(runs, levels, by)
        table.columns = labels
        table = table.dropna(how="all")
        order = table.mean(axis=1).sort_values(ascending=False, na_position="last").index
        blocks.append((f"{levels[by + 1]} gain by true {levels[by]}", table.loc[order]))

    tallest = max(len(table) for _, table in blocks)
    figure, axes = _grid(len(blocks), tallest, width, row_height, min_height)

    for ax, (title, table) in zip(axes, blocks):
        y = _rows(tallest, len(table))
        for yy, (_, row) in zip(y, table.iterrows()):
            values = row.dropna()
            if len(values) > 1:
                ax.plot([values.min(), values.max()], [yy, yy], color="#c9c9c9", lw=1.2,
                        zorder=2, solid_capstyle="round")
            for name, value in values.items():
                ax.scatter([value], [yy], s=18, color=colour[name], zorder=3, linewidths=0,
                           label=name if yy == y[0] else None)
        ax.axvline(0, color="#4a4a4a", lw=0.8, zorder=1)
        _axis(ax, y, table.index, GAIN_LABEL, title)

    return _finish(figure, axes[0].get_legend_handles_labels()[0], len(labels), path)


# --- Cross-run figures -----------------------------------------------

def _wide(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(dataset, hvg, n_hvg, backend) x level: reselect value and its gain over fixed_global."""
    treat, base = ARMS
    pivot = table.pivot_table(index=["dataset", "hvg", "n_hvg", "backend"],
                              columns=["level", "arm"], values=metric)
    levels = table["level"].unique()
    out = pd.DataFrame(index=pivot.index)
    for level in levels:
        out[(level, treat)] = pivot[(level, treat)]
        out[(level, "delta")] = pivot[(level, treat)] - pivot[(level, base)]
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def runs_figure(table: pd.DataFrame, metric="accuracy", path=None, width=figures.FULL, height=2.6):
    """Gain over fixed_global per level: one panel per dataset, one line per configuration."""
    wide = _wide(table, metric)
    levels = list(dict.fromkeys(level for level, _ in wide.columns))
    datasets = list(dict.fromkeys(entry[0] for entry in wide.index))

    # one colour per HVG method, one linestyle per budget, one marker per backend
    methods = sorted({entry[1] for entry in wide.index})
    budgets = sorted({entry[2] for entry in wide.index})
    # named backends first then by name, a set iterates in hash order and without the second
    # key the legend comes out in a different order from one process to the next
    backends = sorted({entry[3] for entry in wide.index}, key=lambda b: (b not in BACKEND_MARK, b))
    color = dict(zip(methods, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    dash = dict(zip(budgets, ["-", "--", ":", "-."]))
    spare = (m for m in ("^", "D", "v", "P"))
    mark = {b: BACKEND_MARK.get(b) or next(spare) for b in backends}

    figure, axes = plt.subplots(1, len(datasets), figsize=(width, height), sharey=True,
                               squeeze=False)
    x = np.arange(len(levels))
    for ax, dataset in zip(axes[0], datasets):
        for (hvg, n_hvg, backend), row in wide.loc[dataset].iterrows():
            ax.plot(x, [row[(level, "delta")] for level in levels], marker=mark[backend], ms=3.5,
                    color=color[hvg], ls=dash[n_hvg])
        ax.axhline(0, color="#8c8c8c", lw=0.7)
        ax.set_xticks(x, levels)
        ax.set_title(dataset, loc="left")
        ax.grid(axis="y", color="#ececec")
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel(f"{metric} gain (reselect $-$ fixed)")
    handles = ([plt.Line2D([], [], color=color[m], label=m.replace("_", " ")) for m in methods]
               + [plt.Line2D([], [], color="#555", ls=dash[n], label=f"{n} HVGs") for n in budgets]
               + [plt.Line2D([], [], color="#555", ls="", marker=mark[b], ms=3.5, label=b)
                  for b in backends])
    axes[0][-1].legend(handles=handles, loc="best", fontsize=5.5)
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


def backend_frame(table: pd.DataFrame, dataset: str, metric="accuracy",
                  arm="reselect") -> pd.DataFrame:
    """(HVG method, budget) x (level, backend) for one arm, the logit-vs-lgbm view."""
    block = table[(table["dataset"] == dataset) & (table["arm"] == arm)]
    if block.empty:
        raise KeyError(f"no {arm} rows for dataset {dataset!r}")
    wide = block.pivot_table(index=["hvg", "n_hvg"], columns=["level", "backend"], values=metric)
    order = sorted(wide.index, key=lambda r: (not r[0].startswith(SUPERVISED_HVG), r[0], r[1]))
    return wide.reindex(order)


def backend_figure(table: pd.DataFrame, dataset: str, metric="accuracy", arm="reselect",
                   path=None, width=figures.FULL):
    """One panel per level, one row per configuration, one marker per backend."""
    wide = backend_frame(table, dataset, metric, arm)
    levels = list(dict.fromkeys(level for level, _ in wide.columns))
    backends = sorted({b for _, b in wide.columns})
    rows = [f"{hvg.replace('_', ' ')} @{n}" for hvg, n in wide.index]
    colour = dict(zip(backends, ("#2a78d6", "#e07a1f", "#7c3aed")))
    y = np.arange(len(rows))[::-1]

    height = max(1.7, 0.26 * len(rows) + 1.0)
    figure, axes = plt.subplots(1, len(levels), figsize=(width, height), sharey=True, squeeze=False)
    for column, (ax, level) in enumerate(zip(axes[0], levels)):
        for backend in backends:
            if (level, backend) not in wide.columns:
                continue
            ax.scatter(wide[(level, backend)], y, s=16, color=colour[backend], zorder=3,
                       linewidths=0, label=backend if column == len(levels) - 1 else None)
        ax.set_title(level)
        ax.grid(axis="x", color="#ececec", zorder=0)
        ax.set_axisbelow(True)
    axes[0][0].set_yticks(y, rows)
    axes[0][0].tick_params(axis="y", length=0)
    axes[0][0].set_xlabel(f"{metric} ({arm} arm)")
    figure.legend(*axes[0][-1].get_legend_handles_labels(), loc="outside lower center",
                  ncols=len(backends))
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


# --- Verdict table ---------------------------------------------------

def consistency_frame(runs, levels, labels=None) -> pd.DataFrame:
    """
    Per node, every configuration's gain plus the verdict the figure colours by: gain when none
    of them loses, loss when none gains, flip when both happen. One frame per level pair.

    Also the gain's mean per HVG method, and the two spreads saying which half of a configuration
    moved it: between_method, the range of those per-method means, against within_method, the mean
    range the backends span inside a method. between_method the larger means the genes decided the
    node rather than the model, and it usually is.
    """
    labels = list(labels or [run.tag for run in runs])
    methods = [TAG_PATTERN.match(run.tag)["hvg"].replace("_shareloess", "") for run in runs]
    frames = []
    for by in range(len(levels) - 1):
        deltas = repeat_deltas(runs, levels, by)
        deltas.columns = labels
        wins, losses = (deltas > 0).sum(axis=1), (deltas < 0).sum(axis=1)
        per_method = deltas.T.groupby(methods).mean().T
        backend_range = deltas.T.groupby(methods).agg(lambda gains: gains.max() - gains.min()).T
        frames.append(deltas.assign(
            split=f"{levels[by + 1]} by {levels[by]}", n_config=deltas.notna().sum(axis=1),
            n_gain=wins, n_loss=losses, mean=deltas.mean(axis=1),
            verdict=np.where((losses == 0) & (wins > 0), "gain",
                             np.where((wins == 0) & (losses > 0), "loss", "flip")),
            between_method=per_method.max(axis=1) - per_method.min(axis=1),
            within_method=backend_range.mean(axis=1), **per_method))
    out = pd.concat(frames).rename_axis("node").reset_index()
    return out[["split", "node", "verdict", "n_config", "n_gain", "n_loss", "mean",
                "between_method", "within_method", *per_method.columns, *labels]]


# --- LaTeX tables ----------------------------------------------------

def runs_latex(table: pd.DataFrame, metric="accuracy", path=None, caption="", label="") -> str:
    """
    Compact cross-run comparison.

    One row per configuration, one column pair per level: reselect, and its gain over fixed_global.
    """
    wide = _wide(table, metric)
    levels = list(dict.fromkeys(level for level, _ in wide.columns))

    header = (" & ".join(["dataset", "HVG", "$n$", "model",
                          *(rf"\multicolumn{{2}}{{c}}{{{level}}}" for level in levels)]) + r" \\"
              + "\n" + " & & & & " + " & ".join(["reselect", r"$\Delta$"] * len(levels)) + r" \\")
    body, previous = [], None
    for (dataset, hvg, n_hvg, backend), row in wide.iterrows():
        head = [dataset, hvg.replace("_", r"\_")] if (dataset, hvg) != previous else ["", ""]
        if (dataset, hvg) != previous and previous is not None:
            body.append(r"\addlinespace")
        previous = (dataset, hvg)
        cells = []
        for level in levels:
            cells += [f"{row[(level, ARMS[0])]:.4f}", f"{row[(level, 'delta')]:+.4f}"]
        body.append(" & ".join([*head, str(n_hvg), backend, *cells]) + r" \\")
    return figures.latex_table(header, body, "llrl" + "rr" * len(levels), path, caption, label)


def comparison_frame(table: pd.DataFrame, dataset: str, metric="accuracy") -> pd.DataFrame:
    """
    One dataset's models as rows, HVG method x budget with supervised first, both arms per level
    as columns. The per-dataset comparison table, where runs_latex is the cross-dataset digest.
    """
    block = table[table["dataset"] == dataset]
    if block.empty:
        raise KeyError(f"no runs for dataset {dataset!r}")
    wide = block.pivot_table(index=["hvg", "n_hvg", "backend"], columns=["level", "arm"],
                             values=metric)
    wide = wide.reindex(columns=pd.MultiIndex.from_product(
        [sorted({level for level, _ in wide.columns}), list(ARMS)]))
    order = sorted(wide.index, key=lambda row: (not row[0].startswith(SUPERVISED_HVG), *row))
    return wide.reindex(order)


def comparison_latex(table: pd.DataFrame, dataset: str, metric="accuracy", path=None,
                     caption="", label="") -> str:
    """
    comparison_frame as LaTeX.

    Bold marks the better arm of each pair, underline the best value in the level overall, and a
    rule separates the supervised methods from the unsupervised ones.
    """
    wide = comparison_frame(table, dataset, metric)
    levels = list(dict.fromkeys(level for level, _ in wide.columns))
    best = {level: wide[level].to_numpy().max() for level in levels}
    tex = lambda text: str(text).replace("_", r"\_")

    header = (" & ".join(["", "", "", *(rf"\multicolumn{{2}}{{c}}{{{level}}}" for level in levels)])
              + r" \\" + "\n"
              + "".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(levels)))
              + "\n" + " & ".join(["HVG method", "$n$", "model",
                                   *sum(([r"reselect", r"fixed"] for _ in levels), [])]) + r" \\")

    body, previous_supervised = [], None
    for (hvg, n_hvg, backend), row in wide.iterrows():
        supervised = hvg.startswith(SUPERVISED_HVG)
        if previous_supervised is not None and supervised != previous_supervised:
            body.append(r"\midrule")
        previous_supervised = supervised
        cells = []
        for level in levels:
            pair = [row[(level, arm)] for arm in ARMS]
            for value in pair:
                text = f"{value:.4f}"
                if value > min(pair):          # a tie bolds neither arm
                    text = rf"\textbf{{{text}}}"
                if value == best[level]:
                    text = rf"\underline{{{text}}}"
                cells.append(text)
        body.append(" & ".join([tex(hvg), str(n_hvg), tex(backend), *cells]) + r" \\")
    return figures.latex_table(header, body, "lrl" + "rr" * len(levels), path, caption, label)


def backend_latex(table: pd.DataFrame, dataset: str, metric="accuracy", arm="reselect",
                  path=None, caption="", label="", reference="logit") -> str:
    """Per level, each backend's value and its gain over reference."""
    wide = backend_frame(table, dataset, metric, arm)
    levels = list(dict.fromkeys(level for level, _ in wide.columns))
    others = [b for b in sorted({b for _, b in wide.columns}) if b != reference]
    tex = lambda t: str(t).replace("_", r"\_")

    header = (" & ".join(["", "", *(rf"\multicolumn{{{1 + 2 * len(others)}}}{{c}}{{{lv}}}"
                                    for lv in levels)]) + r" \\" + "\n"
              + "".join(rf"\cmidrule(lr){{{3 + (1 + 2 * len(others)) * i}-"
                        rf"{2 + (1 + 2 * len(others)) * (i + 1)}}}" for i in range(len(levels)))
              + "\n" + " & ".join(["HVG method", "$n$", *sum(
                  ([reference, *sum(([b, r"$\Delta$"] for b in others), [])] for _ in levels), [])])
              + r" \\")

    body, previous = [], None
    for (hvg, n_hvg), row in wide.iterrows():
        head = [tex(hvg)] if hvg != previous else [""]
        if hvg != previous and previous is not None:
            body.append(r"\addlinespace")
        previous = hvg
        cells = []
        for level in levels:
            base = row.get((level, reference), float("nan"))
            cells.append(f"{base:.4f}")
            for other in others:
                value = row.get((level, other), float("nan"))
                cells += [f"{value:.4f}", f"{value - base:+.4f}"]
        body.append(" & ".join([*head, str(n_hvg), *cells]) + r" \\")
    spec = "lr" + ("r" + "rr" * len(others)) * len(levels)
    return figures.latex_table(header, body, spec, path, caption, label)


def repeat_levels_latex(runs, metric="accuracy", path=None, caption="", label="") -> str:
    """Per level, each arm's mean over repeats with its spread, and the gain's."""
    stats = repeat_levels(runs, metric)
    treat, base = ARMS
    header = r"level & $r$ & reselect & fixed & $\Delta$ & $\Delta$ range \\"
    body = []
    for level, row in stats.iterrows():
        body.append(" & ".join([
            level, f"{int(row[(treat, 'count')])}",
            f"{row[(treat, 'mean')]:.4f} $\\pm$ {row[(treat, 'std')]:.4f}",
            f"{row[(base, 'mean')]:.4f} $\\pm$ {row[(base, 'std')]:.4f}",
            f"{row[('delta', 'mean')]:+.4f} $\\pm$ {row[('delta', 'std')]:.4f}",
            f"[{row[('delta', 'min')]:+.4f}, {row[('delta', 'max')]:+.4f}]",
        ]) + r" \\")
    return figures.latex_table(header, body, "lrrrrr", path, caption, label)

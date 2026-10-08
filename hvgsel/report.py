"""
Figures and tables of the downstream experiment, read back from what runs.py wrote.

Nothing here trains or predicts. Every function takes Run objects, or the tidy frame runs_table
builds out of every scores.csv on disk, and returns a figure or a LaTeX table.

Five figures share one reading: one horizontal row per node, one panel per level pair, coloured
by whether re-selection helps that node. They are built on the same skeleton (_grid, _axis,
_range_row, _xpad, _finish), and differ in the anchor each row is measured from and in the
verdict its colour states.

Cross-validated configurations (runs_cv/) sit in the same tidy frame as the single-split ones,
one row set per fold. fold_frame folds them back into one row per configuration, which is what
the comparison table and level_gain_figure read.
"""

# IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

from hvgsel import figures, marker
from hvgsel.coverage import split_nodes
from hvgsel.runs import ARMS, TAG_PATTERN, Run


# ---------------------------------------------------------------------


C_RESELECT, C_FIXED = "#2a78d6", "#e34948"
C_FLIP = "#9a9a9a"     # a node whose gain changes sign across repeats, within split noise
C_BAND = "#f4f4f4"     # every other node, where several sub-rows share one node label

# One colour per gene budget, fixed so a figure missing a budget does not repaint the others.
# Clear of the blue/red verdict pair so a colour keeps one meaning across the figures. Apart under
# protan/deutan/tritan simulation (OKLab dE >= 9, 300 and 2000 >= 29) and in lightness, so the
# 300/2000 pair survives a greyscale print. 0 is a method without budget, the atlas markers.
BUDGET_COLOUR = {300: "#eb6834", 2000: "#4a3aa7", 0: "#1baf7a"}
SPARE_COLOURS = ("#eda100", "#e87ba4")


def budget_name(n_hvg: int) -> str:
    """300 -> '300 HVGs', and a method without budget (n_hvg 0) -> 'atlas markers'."""
    return f"{n_hvg} HVGs" if n_hvg else "atlas markers"


def _budget_key(n_hvg: int) -> tuple:
    """Budgets smallest first, a method without one last."""
    return (n_hvg == 0, n_hvg)

BACKEND_MARK = {"logit": "o", "lgbm": "s"}   # fixed, a backend keeps its marker across figures
# Where a figure shows both at once, the marker's fill is the HVG method: solid for the supervised
# scores, hollow for the unsupervised one, so the two families read apart at a glance.
HVG_FILL = {"f_statistic": 1.0, "kruskal_wallis": 0.4, "seurat_v3": 0.0, "marker": 0.5,
            "f_statistic_markers": 1.0}

SUPERVISED_HVG = ("f_statistic", "kruskal_wallis")

GAIN_LABEL = "accuracy gain (reselect $-$ fixed_global)"
ACC_LABEL = "accuracy within node"


# --- Reading the runs ------------------------------------------------

def runs_table(*roots) -> pd.DataFrame:
    """
    Every saved run's per-level scores in one tidy frame, the cross-configuration table.

    Several roots read into one frame, runs/ and runs_cv/ say, a fold's rows told apart by fold.
    No root reads runs/.
    """
    roots = roots or ("runs",)
    rows = []
    for root in roots:
        for scores_path in sorted(Path(root).glob("*/*/scores.csv")):
            run = Run(scores_path.parents[2], scores_path.parents[1].name, scores_path.parent.name)
            parts = TAG_PATTERN.match(run.tag)
            if parts is None:
                raise ValueError(f"run directory '{run.tag}' is not <hvg>_<n>hvg_<backend>[_fold<k>]")
            rows.append(run.read("scores").assign(
                dataset=run.dataset, tag=run.tag, hvg=parts["hvg"], backend=parts["backend"],
                n_hvg=int(parts["n_hvg"]), fold=run.meta.get("fold"),
                models_saved=bool(run.meta.get("models_saved", run.trained))))
    if not rows:
        raise FileNotFoundError(f"no runs under {', '.join(f'{root}/' for root in roots)}")
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


def fold_frame(table: pd.DataFrame, dataset: str, metric="accuracy") -> pd.DataFrame:
    """
    (budget, HVG method, backend, level) x (reselect, fixed_global, delta) x statistic, over each
    configuration's runs: its donor folds when cross-validated, its one split otherwise.

    Rows run budget by budget, supervised methods first. Each arm and the gain get a mean and an
    s.d., NaN for a single split, and the gain also n, gain and loss: how many folds, and how many
    on either side of zero.

    Note: the gain is paired, taken within a fold before it is averaged. The arms share the fold's
    donors and move together, so its spread is far tighter than either arm's, and the arm spreads
    say nothing on whether one arm beats the other.
    """
    block = table[table["dataset"] == dataset]
    if block.empty:
        raise KeyError(f"no runs for dataset {dataset!r}")
    config, keys = ["n_hvg", "hvg", "backend"], ["n_hvg", "hvg", "backend", "fold", "level"]
    if block.duplicated([*keys, "arm"]).any():
        raise ValueError(f"{dataset}: a configuration holds one fold twice, "
                         "read from both runs/ and runs_cv/?")
    per_fold = block.set_index([*keys, "arm"])[metric].unstack("arm")[list(ARMS)]
    per_fold["delta"] = per_fold[ARMS[0]] - per_fold[ARMS[1]]

    grouped = per_fold.groupby([*config, "level"])
    out = grouped.agg(["mean", "std"])
    out[("delta", "n")] = grouped["delta"].count()
    out[("delta", "gain")] = grouped["delta"].agg(lambda gains: int((gains > 0).sum()))
    out[("delta", "loss")] = grouped["delta"].agg(lambda gains: int((gains < 0).sum()))
    order = sorted(out.index, key=lambda row: (*_budget_key(row[0]), *_config_key(row[1], row[2]),
                                               row[3]))
    return out.reindex(order)


def fold_note(counts) -> str:
    """
    How a budget's rows were measured, from its per-configuration fold counts.

    Note: $\\pm$ is read by LaTeX and by matplotlib's mathtext alike, the table and the figure
    legend share the phrase.
    """
    n = sorted({int(count) for count in counts})
    if n == [1]:
        return "one donor split"
    span = str(n[0]) if len(n) == 1 else f"{n[0]}-{n[-1]}"
    return rf"mean $\pm$ s.d. over {span} donor folds"


def config_label(tag: str) -> str:
    """f_statistic_0300hvg_lgbm -> 'f_statistic x lgbm', naming a cross-configuration column."""
    parts = TAG_PATTERN.match(tag)
    return f"{parts['hvg'].replace('_shareloess', '')} x {parts['backend']}"


def config_title(dataset: str, runs, n_hvg: int) -> str:
    """
    The configurations named as the grid they cover when they cover one, else listed.

    Folds of one configuration count once, and their number is named after it.
    """
    parts = [TAG_PATTERN.match(run.tag) for run in runs]
    configs = list(dict.fromkeys(config_label(run.tag) for run in runs))
    methods = list(dict.fromkeys(p["hvg"].replace("_shareloess", "") for p in parts))
    backends = list(dict.fromkeys(p["backend"] for p in parts))
    grid = f"{', '.join(methods)}  $\\times$  {', '.join(backends)}"
    if len(methods) * len(backends) != len(configs):
        grid = ", ".join(configs)
    folds = {p["fold"] for p in parts} - {None}
    reruns = f" $\\times$ {len(folds)} donor folds" if folds else ""
    return f"{dataset}, {n_hvg} HVGs: {len(configs)} configurations{reruns}   ({grid})"


def _config_key(hvg: str, backend: str) -> tuple:
    """Supervised HVG methods first, then by name, and the backends in BACKEND_MARK's order."""
    backends = list(BACKEND_MARK)
    return (not hvg.startswith(SUPERVISED_HVG), hvg,
            backends.index(backend) if backend in backends else len(backends), backend)


def _verdict(gains: np.ndarray) -> np.ndarray:
    """
    Per row of gains, blue when none of its columns loses, red when none gains, grey when both
    happen. NaN counts as neither, and a row unchanged everywhere is grey, ties break neither way.
    """
    wins, losses = (gains > 0).sum(axis=1), (gains < 0).sum(axis=1)
    colours = np.where(losses == 0, C_RESELECT, np.where(wins == 0, C_FIXED, C_FLIP))
    colours[(wins == 0) & (losses == 0)] = C_FLIP
    return colours


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


def _finish(figure, handles, ncols: int, path, title=None, **legend):
    """Legend outside the panels, optional suptitle, then write and show. legend goes to the
    legend itself, its label spacing say."""
    figure.legend(handles=handles, loc="outside lower center", ncols=ncols, **legend)
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
        means = np.nanmean(gains, axis=1)
        colours = _verdict(gains)

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


def fold_breakdown_figure(runs, levels, path=None, width=figures.FULL, row_height=0.30,
                          min_height=2.0, spread=0.66, title=None):
    """
    config_breakdown_figure with every fold of every configuration drawn, the folds of runs_cv/.

    A node's row splits into one sub-row per configuration, spread over `spread` of the row, in
    the same styles: shape is the backend, fill the HVG method. A sub-row holds one marker per
    fold, their range as a line and their mean as a bar, on the gain axis.

    Note: colour is the verdict per configuration, over its folds, where config_breakdown_figure
    states one across configurations: blue when every fold of that configuration gains on the
    node, red when every one loses, grey when the sign flips. A node gaining whatever the
    configuration is then a row of blue sub-rows. Nodes are ordered by their mean gain over every
    run, and a node a run did not score leaves that run's marker out.
    """
    folds = {}
    for run in runs:
        parts = TAG_PATTERN.match(run.tag)
        folds.setdefault((parts["hvg"], parts["backend"]), []).append(run.tag)
    configs = sorted(folds, key=lambda config: _config_key(*config))
    offsets = np.linspace(spread / 2, -spread / 2, len(configs)) if len(configs) > 1 else [0.0]

    blocks = []
    for by in range(len(levels) - 1):
        deltas = repeat_deltas(runs, levels, by).dropna(how="all")
        order = deltas.mean(axis=1).sort_values(ascending=False).index
        blocks.append((f"{levels[by + 1]} gain by true {levels[by]}", deltas.loc[order]))

    tallest = max(len(deltas) for _, deltas in blocks)
    figure, axes = _grid(len(blocks), tallest, width, row_height, min_height, pad_y=0.6,
                         title=title)

    handles = [plt.Line2D([], [], color=C_RESELECT, lw=1.2, label="gain in every fold"),
               plt.Line2D([], [], color=C_FIXED, lw=1.2, label="loss in every fold"),
               plt.Line2D([], [], color=C_FLIP, lw=1.2, label="sign flips across folds")]
    for ax, (label, deltas) in zip(axes, blocks):
        y = _rows(tallest, len(deltas))
        for yy in y[::2]:
            ax.axhspan(yy - 0.5, yy + 0.5, color=C_BAND, lw=0, zorder=0)
        ax.axvline(0, color="#1a1a1a", lw=0.8, zorder=1)

        for offset, (hvg, backend) in zip(offsets, configs):
            mark = BACKEND_MARK.get(backend, "D")
            fill = HVG_FILL.get(hvg.replace("_shareloess", ""), 0.5)
            gains = deltas[folds[(hvg, backend)]].to_numpy()
            colours, rows = _verdict(gains), y + offset
            for yy, row, colour in zip(rows, gains, colours):
                row = row[~np.isnan(row)]
                if row.size:
                    ax.plot([row.min(), row.max()], [yy, yy], color=colour, lw=0.8, alpha=0.5,
                            solid_capstyle="round", zorder=2)
                    ax.scatter([row.mean()], [yy], s=16, marker="|", color=colour, lw=0.9,
                               zorder=5)
            colour = np.repeat(colours, gains.shape[1])
            ax.scatter(gains.ravel(), np.repeat(rows, gains.shape[1]), s=9, marker=mark,
                       linewidths=0.6, edgecolors=[to_rgba(c, 0.9) for c in colour],
                       facecolors=[to_rgba(c, 0.8 * fill) for c in colour], zorder=4)

        _axis(ax, y, deltas.index, GAIN_LABEL, label)
        _xpad(ax, min(0, np.nanmin(deltas)), max(0, np.nanmax(deltas)))

    for hvg, backend in configs:
        method = hvg.replace("_shareloess", "")
        handles.append(plt.Line2D([], [], marker=BACKEND_MARK.get(backend, "D"), ls="", ms=3.6,
                                  color="#6a6a6a", markeredgewidth=0.7, markeredgecolor="#6a6a6a",
                                  markerfacecolor=to_rgba("#6a6a6a", HVG_FILL.get(method, 0.5)),
                                  label=f"{method} x {backend}"))
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


METRIC_NAME = {"accuracy": "accuracy", "macro_f1": "macro-F1"}


def _tint(colour, fill: float):
    """colour laid over white at opacity fill: a hollow marker that still hides the line under it."""
    return tuple(fill * c + (1 - fill) for c in to_rgba(colour)[:3])


def level_gain_figure(table: pd.DataFrame, dataset: str, metrics=("accuracy", "macro_f1"),
                      path=None, width=figures.FULL, height=3.4, spread=0.56):
    """
    Gain over fixed_global per level, one panel per metric, every configuration at every budget.

    Colour is the budget (BUDGET_COLOUR), or the method's own in METHOD_COLOUR, which then gets a
    colour entry of its own below the budgets (METHOD_LEGEND) and no column: its shapes are those of
    the method it extends. Shape is the backend and fill the HVG method, as in the per-node figures,
    and the line repeats the fill, solid for the supervised scores. A budget run
    over donor folds shows the mean gain and a bar of one s.d. of the paired per-fold gain, a single
    split its one value. The configurations are spread over `spread` of a level, so no bar hides
    another.

    Note: the folds and the single split do not hold out the same donors, the single split is fold
    0 of the same five. Their gains compare, but not to the last digit.
    """
    first = fold_frame(table, dataset, metrics[0])
    configs = list(dict.fromkeys(row[:3] for row in first.index))   # fold_frame's order
    # sorted for the lookups below, an index in fold_frame's order is not lexsorted
    stats = {metric: fold_frame(table, dataset, metric).sort_index() for metric in metrics}
    levels = list(dict.fromkeys(first.index.get_level_values("level")))
    budgets = list(dict.fromkeys(n_hvg for n_hvg, _, _ in configs))
    spare = iter(SPARE_COLOURS)
    colour = {n_hvg: BUDGET_COLOUR.get(n_hvg) or next(spare) for n_hvg in budgets}
    offsets = np.linspace(-spread / 2, spread / 2, len(configs)) if len(configs) > 1 else [0.0]

    def style(n_hvg, hvg, backend, grey=None):
        fill = HVG_FILL.get(hvg.replace("_shareloess", ""), 0.5)
        ink = grey or METHOD_COLOUR.get(hvg) or colour[n_hvg]
        return dict(color=ink, marker=BACKEND_MARK.get(backend, "D"), ms=3.8, mew=0.7, mec=ink,
                    mfc=_tint(ink, fill), ls="-" if fill else (0, (2.4, 1.4)), lw=0.8)

    figure, axes = plt.subplots(1, len(metrics), figsize=(width, height), squeeze=False)
    x = np.arange(len(levels))
    for ax, metric in zip(axes[0], metrics):
        ax.axhline(0, color="#8c8c8c", lw=0.7, zorder=1)
        for offset, config in zip(offsets, configs):
            rows = stats[metric].loc[config].reindex(levels)
            ax.errorbar(x + offset, rows[("delta", "mean")], yerr=rows[("delta", "std")].fillna(0),
                        elinewidth=0.8, capsize=0, zorder=3, **style(*config))
        ax.set_xticks(x, levels)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_ylabel("reselect $-$ fixed_global")
        ax.set_title(f"{METRIC_NAME.get(metric, metric)} gain", loc="left")
        ax.grid(axis="y", color="#ececec")
        ax.set_axisbelow(True)

    # a budget gets an entry when some method draws in its colour, a method of METHOD_COLOUR
    # carries its own colour on its entries instead
    shown = [n_hvg for n_hvg in budgets
             if any(n == n_hvg and h not in METHOD_COLOUR for n, h, _ in configs)]
    handles = [plt.Line2D([], [], color=colour[n_hvg], lw=1.2,
                          label=f"{budget_name(n_hvg)}, "
                                + fold_note(first.loc[n_hvg][("delta", "n")]))
               for n_hvg in shown]
    for n_hvg, hvg in dict.fromkeys((n, h) for n, h, _ in configs if h in METHOD_COLOUR):
        folds = first.xs((n_hvg, hvg), level=("n_hvg", "hvg"))[("delta", "n")]
        handles.append(plt.Line2D([], [], color=METHOD_COLOUR[hvg], lw=1.2,
                                  label=f"{budget_name(n_hvg)} {METHOD_LEGEND.get(hvg, METHOD_NAME.get(hvg, hvg))}, "
                                        + fold_note(folds)))
    # one column per method, its name heading its backends, after a column of budgets padded to
    # the same height: a legend fills its columns top to bottom, so every column holds as many
    # entries and the method names line up as a header row
    pairs = [pair for pair in dict.fromkeys(config[1:] for config in configs)   # logit first
             if pair[0] not in METHOD_COLOUR]
    methods = list(dict.fromkeys(hvg for hvg, _ in pairs))
    backends = list(dict.fromkeys(backend for _, backend in pairs))
    blank = lambda label="": plt.Line2D([], [], ls="", label=label)
    handles += [blank() for _ in range(1 + len(backends) - len(handles))]
    headers = []
    for hvg in methods:
        headers.append(METHOD_NAME.get(hvg, hvg.replace("_shareloess", "")))
        handles.append(blank(headers[-1]))
        handles += [plt.Line2D([], [], label=backend, **style(None, hvg, backend, grey="#6a6a6a"))
                    for backend in backends if (hvg, backend) in pairs]
    legend = figure.legend(handles=handles, loc="outside lower center", ncols=1 + len(methods),
                           columnspacing=1.4)
    for text in legend.get_texts():
        if text.get_text() in headers:
            text.set_fontweight("bold")
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


# How a gene set's arm reads where both are shown: the atlas markers, the node's own or all of them
MARKER_ARM = {"reselect": "node", "fixed_global": "all"}
# A method coloured apart from its budget, and how it reads: the crossover of markers and F
METHOD_COLOUR = {"f_statistic_markers": "#e87ba4"}
METHOD_NAME = {"f_statistic_markers": "f_statistic + markers", "marker": "markers"}
# How a method of METHOD_COLOUR reads beside its budget where the colour stands for it in a legend
METHOD_LEGEND = {"f_statistic_markers": "including markers"}


def gene_set_figure(table: pd.DataFrame, dataset: str, metrics=("accuracy", "macro_f1"),
                    both_arms=("marker",), path=None, width=figures.FULL, row_height=0.15):
    """
    Every gene set's score, not its gain: one panel per metric and level, one block of rows per
    backend, one row per gene set within it, so the gene sets line up for one backend at a time.

    What the gain figures cannot say: whether one gene set predicts as well as another, the atlas
    markers against the HVG selections, say. An HVG selection is shown re-selected at every node,
    the arm a budget is meant to be used with. Methods of both_arms show both: the atlas markers'
    node's own against all of them at every node. Colour is the budget, or the method's own in
    METHOD_COLOUR, shape the backend, which also heads each block, so the legend names the colours
    only. A
    budget run over donor folds shows the mean and a bar of one s.d. across folds, a single split
    its value.

    Note: the single split is fold 0 of the folds, so a budget's rows compare to within the fold
    spread only.
    """
    stats = {metric: fold_frame(table, dataset, metric) for metric in metrics}
    first = stats[metrics[0]]
    levels = list(dict.fromkeys(first.index.get_level_values("level")))
    backends = sorted(set(first.index.get_level_values("backend")),
                      key=lambda b: _config_key("", b))
    sets = [(n_hvg, hvg, arm)
            for n_hvg, hvg in dict.fromkeys((n, h) for n, h, _, _ in first.index)
            for arm in (ARMS if hvg in both_arms else ARMS[:1])]
    method = lambda hvg: METHOD_NAME.get(hvg, hvg.replace("_shareloess", ""))
    name = lambda n_hvg, hvg, arm: (f"{budget_name(n_hvg)} · {MARKER_ARM[arm]}" if hvg in both_arms
                                    else f"{budget_name(n_hvg)} · {method(hvg)}")
    spare = iter(SPARE_COLOURS)
    budget = {n: BUDGET_COLOUR.get(n) or next(spare) for n in dict.fromkeys(s[0] for s in sets)}
    colour = lambda n_hvg, hvg: METHOD_COLOUR.get(hvg, budget[n_hvg])

    # rows top to bottom: a header per backend, then its gene sets
    rows, labels, y = [], [], 0
    for backend in backends:
        labels.append((y, backend, True))
        y += 1
        for entry in sets:
            rows.append((y, backend, entry))
            labels.append((y, name(*entry), False))
            y += 1
    depth = y
    flip = lambda position: depth - 1 - position   # matplotlib counts rows from the bottom

    figure, axes = plt.subplots(len(metrics), len(levels), sharey=True, squeeze=False,
                                figsize=(width, len(metrics) * (row_height * depth + 0.5) + 0.5))
    for r, metric in enumerate(metrics):
        lookup = stats[metric].sort_index()
        for ax, level in zip(axes[r], levels):
            for position, backend, (n_hvg, hvg, arm) in rows:
                key = (n_hvg, hvg, backend, level)
                if key not in lookup.index:
                    continue
                mean, std = lookup.loc[key, (arm, "mean")], lookup.loc[key, (arm, "std")]
                ax.errorbar([mean], [flip(position)], xerr=[0 if np.isnan(std) else std],
                            marker=BACKEND_MARK.get(backend, "D"), ms=3.4, mew=0.6, ls="",
                            color=colour(n_hvg, hvg), elinewidth=0.8, capsize=0, zorder=3)
            for position, _, header in labels:
                if header:   # a thin rule through the header row, the block's divider
                    ax.axhline(flip(position), color="#4a4a4a", lw=plt.rcParams["axes.linewidth"],
                               zorder=1)
            ax.set_title(f"{METRIC_NAME.get(metric, metric)}, {level}", loc="left")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="x", color="#ececec", zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks([flip(p) for p, _, _ in labels], [text for _, text, _ in labels])
    for row in axes:
        for tick, (_, _, header) in zip(row[0].get_yticklabels(), labels):
            if header:
                tick.set_fontweight("bold")
    axes[0][0].set_ylim(-0.6, depth - 0.4)

    # one entry per colour, the row blocks already name the backend: a budget, or a method with
    # a colour of its own. In columns of two, those run over donor folds first, then those on a
    # single split, so the gene sets measured alike sit together
    entries = {}
    for n_hvg, hvg, _ in sets:
        own = hvg in METHOD_COLOUR
        folds = first.xs((n_hvg, hvg), level=("n_hvg", "hvg"))[("delta", "n")]
        what = budget_name(n_hvg) + (f", {method(hvg)}" if own else "")
        entries.setdefault(hvg if own else n_hvg, (
            (folds.max() == 1, *_budget_key(n_hvg), own), colour(n_hvg, hvg),
            f"{what}{'' if n_hvg == 0 else ' re-selected'}, {fold_note(folds)}"))
    handles = [plt.Line2D([], [], color=paint, lw=1.2, label=text)
               for _, paint, text in sorted(entries.values())]
    return _finish(figure, handles, -(-len(handles) // 2), path)


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


# --- Marker genes ----------------------------------------------------

COVERAGE_LABEL = "share of the node's atlas markers selected"


def _selections(frame: pd.DataFrame) -> list[tuple[str, int]]:
    """Every (HVG method, budget) in frame, budget first, then supervised first."""
    pairs = {(row.hvg, row.n_hvg) for row in frame[["hvg", "n_hvg"]].itertuples()}
    return sorted(pairs, key=lambda pair: (pair[1], not pair[0].startswith(SUPERVISED_HVG), pair[0]))


def _bands(ax, y, groups) -> None:
    """A light band behind every other run of equal groups, one row high."""
    shade, previous = False, None
    for yy, group in zip(y, groups):
        shade = shade ^ (group != previous) if previous is not None else False
        previous = group
        if shade:
            ax.axhspan(yy - 0.5, yy + 0.5, color=C_BAND, lw=0, zorder=0)


def coverage_figure(coverage: pd.DataFrame, config_path, path=None, width=figures.FULL,
                    row_height=0.15):
    """
    Per internal node, the share of its atlas markers each arm's genes hold.

    fixed_global's, the root selection, is the black point, reselect's, the node's own, the arrow
    head: blue when re-selection holds more of them, red when fewer. One panel per HVG method and
    budget, nodes in tree order with their marker count, a band per Level 1 compartment. Means
    over folds, the reselect range across them as a line.
    """
    order = list(split_nodes(config_path))
    stats = coverage.groupby(["hvg", "n_hvg", "path"]).agg(
        fixed=("fixed", "mean"), reselect=("reselect", "mean"), low=("reselect", "min"),
        high=("reselect", "max"), n=("n_markers", "first"))
    panels = _selections(coverage)
    y = np.arange(len(order))[::-1]
    compartments = [path.split("/")[1] if "/" in path else "root" for path in order]

    figure, axes = plt.subplots(1, len(panels), figsize=(width, row_height * len(order) + 1.0),
                                sharey=True, squeeze=False)
    for ax, (hvg, n_hvg) in zip(axes[0], panels):
        block = stats.loc[(hvg, n_hvg)].reindex(order)
        _bands(ax, y, compartments)
        for yy, row in zip(y, block.itertuples()):
            colour = (C_RESELECT if row.reselect > row.fixed else
                      C_FIXED if row.reselect < row.fixed else C_FLIP)
            if row.high > row.low:
                ax.plot([row.low, row.high], [yy, yy], color=colour, lw=0.8, alpha=0.45,
                        solid_capstyle="round", zorder=2)
            if row.reselect != row.fixed:
                ax.annotate("", xy=(row.reselect, yy), xytext=(row.fixed, yy), zorder=3,
                            arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.3",
                                            color=colour, lw=1.0, shrinkA=0, shrinkB=0))
        ax.scatter(block["fixed"], y, s=7, color="#1a1a1a", linewidths=0, zorder=4)
        ax.set_xlim(-0.04, 1.04)
        ax.set_title(f"{hvg}, {n_hvg} HVGs", loc="left")
        ax.set_xlabel("share of markers")
        ax.grid(axis="x", color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    labels = [f"{path.rsplit('/', 1)[-1]} ({int(n)})"
              for path, n in stats.loc[panels[0]].reindex(order)["n"].items()]
    axes[0][0].set_yticks(y, labels)
    axes[0][0].set_ylim(-0.7, len(order) - 0.3)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color="#1a1a1a",
                          label="fixed_global (root selection)"),
               plt.Line2D([], [], color=C_RESELECT, lw=1.2, label="reselect holds more"),
               plt.Line2D([], [], color=C_FIXED, lw=1.2, label="reselect holds fewer")]
    return _finish(figure, handles, 3, path)


def marker_gene_figure(genes: pd.DataFrame, path=None, width=figures.FULL, blocks=3,
                       row_height=0.072):
    """
    Every atlas marker against every selection, a heatmap split into `blocks` side by side.

    Per (HVG method, budget), two columns: root, the share of folds whose root selection
    (fixed_global) holds the gene, and nodes, the share of the nodes it marks whose own selection
    (reselect) holds it, averaged over folds. Genes by the compartment first naming them, then by
    name, a rule between compartments.
    """
    table = marker.marker_table()
    rank = {name: i for i, name in enumerate(dict.fromkeys(map(marker.compartment, marker.PARENT)))}
    home = table.assign(rank=table["population"].map(lambda p: rank[marker.compartment(p)]),
                        home=table["population"].map(marker.compartment))
    home = home.sort_values("rank").drop_duplicates("gene").set_index("gene")
    order = sorted(home.index, key=lambda gene: (home.loc[gene, "rank"], gene))

    panels = _selections(genes)
    means = genes.groupby(["hvg", "n_hvg", "gene"])[["fixed", "reselect"]].mean()
    columns = [(pair, arm) for pair in panels for arm in ("fixed", "reselect")]
    grid = np.array([[means.loc[(*pair, gene), arm] for pair, arm in columns] for gene in order])
    names = [f"{hvg} {n_hvg} · {'root' if arm == 'fixed' else 'nodes'}"
             for (hvg, n_hvg), arm in columns]

    per = -(-len(order) // blocks)
    figure, axes = plt.subplots(1, blocks, figsize=(width, row_height * per + 1.3), squeeze=False)
    for index, ax in enumerate(axes[0]):
        rows = slice(index * per, min((index + 1) * per, len(order)))
        part, part_genes = grid[rows], order[rows]
        image = ax.imshow(part, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                          interpolation="nearest")
        ax.set_yticks(range(len(part_genes)), part_genes, fontsize=4.6)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
        ax.tick_params(length=0, pad=1.5)
        ax.xaxis.tick_top()
        for x in np.arange(1.5, len(columns) - 1, 2):
            ax.axvline(x, color="white", lw=1.2)
        homes = [home.loc[gene, "home"] for gene in part_genes]
        for row in range(1, len(homes)):
            if homes[row] != homes[row - 1]:
                ax.axhline(row - 0.5, color="#1a1a1a", lw=0.6)
        for row, name in enumerate(homes):
            if row == 0 or name != homes[row - 1]:
                ax.text(len(columns) - 0.4, row - 0.4, name, fontsize=4.6, style="italic",
                        va="top", ha="left", clip_on=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    figure.colorbar(image, ax=axes[0].tolist(), orientation="horizontal", fraction=0.025,
                    pad=0.02, aspect=50, label="share holding the gene")
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


PALE = 0.35   # the strength a figure keeps of a colour for the nodes it is not about
VERDICT_COLOUR = {"gain": C_RESELECT, "loss": C_FIXED, "flip": C_FLIP}

# where a point's name may go, tried in order: (dx, dy) in points, horizontal, vertical alignment.
# First the four corners hugging the point, then a ring further out, every 45 degrees, which gets
# a leader line back to the point.
def _ring(radius: float) -> list:
    spots = []
    for angle in np.radians([45, -45, 135, -135, 0, 180, 90, -90]):
        dx, dy = radius * np.cos(angle), radius * np.sin(angle)
        ha = "left" if dx > 1 else "right" if dx < -1 else "center"
        va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
        spots.append((round(dx, 1), round(dy, 1), ha, va))
    return spots


LABEL_SPOTS = ([(4, 2, "left", "bottom"), (4, -2, "left", "top"), (-4, 2, "right", "bottom"),
                (-4, -2, "right", "top")] + _ring(14))
LEADER_AT = 10   # points: a spot this far out draws its leader line


def _place_labels(figure, ax, items, obstacles, merge=14, **text) -> None:
    """
    Name each (x, y, name, group) of items at the first of LABEL_SPOTS overlapping neither a name
    already placed nor a point of obstacles, [(x, y)], and staying inside ax, or, none being free,
    at the spot overlapping least. A name already placed for the same group within merge points is not repeated: points
    of one group read alike, one name covers them.

    Note: reads extents off a drawn figure, so the layout around ax must be final: call it after
    every legend is in.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    points = figure.dpi / 72
    taken = []
    for x, y in obstacles:
        cx, cy = ax.transData.transform((x, y))
        taken.append((cx - 3 * points, cy - 3 * points, cx + 3 * points, cy + 3 * points))

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    frame = ax.get_window_extent(renderer).extents
    outside = lambda e: (e[2] - e[0]) * (e[3] - e[1]) - overlap(e, frame)   # area past ax's edges

    def annotate(x, y, name, spot):
        dx, dy, ha, va = spot
        leader = ({"arrowprops": dict(arrowstyle="-", lw=0.5, color="#6a6a6a", shrinkA=1, shrinkB=2.5)}
                  if np.hypot(dx, dy) >= LEADER_AT else {})
        return ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                           **leader, **text)

    anchors = []
    for x, y, name, group in items:
        here = ax.transData.transform((x, y))
        if any((name, group) == other and np.hypot(*(here - there)) < merge * points
               for other, there in anchors):
            continue
        anchors.append(((name, group), here))
        best = None
        for spot in LABEL_SPOTS:
            label = annotate(x, y, name, spot)
            extent = label.get_window_extent(renderer).extents
            clash = sum(overlap(extent, other) for other in taken) + 4 * outside(extent)
            if best is None or clash < best[0]:
                best = (clash, spot, tuple(extent))
            label.remove()
            if clash == 0:
                break
        annotate(x, y, name, best[1])
        taken.append(best[2])


def _short(node: str) -> str:
    """'Naive B cell' -> 'Naive B', but 'B cell' stays whole, a lone letter reads as a fragment."""
    short = node.removesuffix(" cell")
    return short if len(short) > 1 else node


def coverage_gain_figure(joined: pd.DataFrame, correlation: pd.DataFrame, focus=None, zoom=None,
                         path=None, width=figures.FULL, height=2.9, named=5):
    """
    Per node, the marker coverage re-selection adds (x) against the accuracy it gains (y), means
    over folds, one panel per budget.

    Shape is the backend and fill the HVG method, as elsewhere. Each configuration's Spearman per
    budget sits under its name in the legend, rho_300 for the 300-HVG panel and so on.

    Colour is each configuration's verdict over its folds: blue when every fold gains, red when
    every one loses, grey when the sign flips. focus maps a budget to the nodes it is about,
    {300: ["Treg", ...]}: those keep the full colour and are named once per HVG method, without
    their trailing "cell" (a method's two backends share the x, its selection, and two methods
    landing on one spot with the same colours share a name), every other node is drawn behind
    them in a pale version of its colour. A budget focus does not name has every node in full
    colour and its `named` largest gains or losses named.

    zoom maps a budget to a crowded region, {300: ((x0, x1, y0, y1), (left, bottom, w, h))}: the
    region in data units, magnified in an inset at the given axes fractions, an empty corner of
    the panel. Names of points inside it go in the inset.

    Note: a single-split budget has one fold, its verdict is the sign of its one gain.
    """
    grouped = joined.groupby(["n_hvg", "hvg", "backend", "node", "depth"])
    nodes = grouped[["delta", "gain"]].mean()
    nodes["verdict"] = grouped["gain"].agg(
        lambda gains: "gain" if (gains > 0).all() else "loss" if (gains < 0).all() else "flip")
    nodes = nodes.reset_index()
    budgets = sorted(nodes["n_hvg"].unique())
    figure, axes = plt.subplots(1, len(budgets), figsize=(width, height), squeeze=False)
    ink = "#3a3a3a"

    def draw(ax, block, chosen, size):
        ax.axhline(0, color="#8c8c8c", lw=0.7, zorder=1)
        ax.axvline(0, color="#8c8c8c", lw=0.7, zorder=1)
        for (hvg, backend), points in block.groupby(["hvg", "backend"]):
            fill, mark = HVG_FILL.get(hvg, 0.5), BACKEND_MARK.get(backend, "D")
            front = points["node"].isin(chosen) if chosen else points["node"].notna()
            colours = [VERDICT_COLOUR[v] for v in points.loc[front, "verdict"]]
            back = [_tint(VERDICT_COLOUR[v], PALE) for v in points.loc[~front, "verdict"]]
            for part, colour, z in ((points[~front], back, 2), (points[front], colours, 4)):
                ax.scatter(part["delta"], part["gain"], s=size, marker=mark, linewidths=0.6,
                           facecolors=[_tint(c, fill) for c in colour], edgecolors=colour, zorder=z)
        ax.grid(color="#ececec", zorder=0)
        ax.set_axisbelow(True)

    names = []   # (axes, items, obstacles)
    for ax, n_hvg in zip(axes[0], budgets):
        block = nodes[nodes["n_hvg"] == n_hvg]
        chosen = set((focus or {}).get(n_hvg, ()))
        unknown = chosen - set(block["node"])
        if unknown:
            raise KeyError(f"{n_hvg} HVGs: no node {sorted(unknown)} in the breakdowns")
        draw(ax, block, chosen, 14)
        ax.set_title(f"{n_hvg} HVGs", loc="left")
        ax.set_xlabel("marker coverage diff. (reselect $-$ fixed_global)")

        if chosen:
            front = block[block["node"].isin(chosen)]
            # one name per node and method, at whichever backend's point lies further from zero,
            # grouped by the verdicts its points show: methods merge only when they read alike
            look = front.groupby(["node", "hvg"])["verdict"].agg(lambda v: tuple(sorted(v)))
            far = front.assign(r=front["gain"].abs()).sort_values("r", ascending=False)
            far = far.drop_duplicates(["node", "hvg"])
            items = [(r.delta, r.gain, _short(r.node), look[(r.node, r.hvg)]) for r in far.itertuples()]
            obstacles = list(zip(front["delta"], front["gain"]))
        else:
            # by the gain alone: the coverage extremes crowd one edge, the gain extremes spread out
            far = block.assign(r=block["gain"].abs()).sort_values("r", ascending=False)
            items = [(r.delta, r.gain, r.node, None)
                     for r in far.drop_duplicates("node").head(named).itertuples()]
            obstacles = []

        if n_hvg in (zoom or {}):
            (x0, x1, y0, y1), rect = zoom[n_hvg]
            inset = ax.inset_axes(rect)
            inset.set_facecolor("white")   # nothing of the panel shows through
            draw(inset, block, chosen, 11)
            inset.set_xlim(x0, x1)
            inset.set_ylim(y0, y1)
            # no tick labels, the frame and its connectors tie it to the panel's scale
            inset.tick_params(labelleft=False, labelbottom=False, length=0)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#8c8c8c")
                spine.set_linewidth(0.6)
            ax.indicate_inset_zoom(inset, edgecolor="#8c8c8c", alpha=1, linewidth=0.6)
            inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
            names.append((inset, [i for i in items if inside(*i[:2])],
                          [o for o in obstacles if inside(*o)]))
            items = [i for i in items if not inside(*i[:2])]
        names.append((ax, items, obstacles))
    axes[0][0].set_ylabel("acc. gain (reselect $-$ fixed_global)")   # GAIN_LABEL outgrows the height
    key = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=VERDICT_COLOUR[v], label=text)
           for v, text in (("gain", "gain in every fold"), ("loss", "loss in every fold"),
                           ("flip", "sign flips across folds"))]
    axes[0][0].legend(handles=key, loc="upper left", fontsize=5.5, handletextpad=0.2,
                      borderaxespad=0.4, labelspacing=0.3)

    # two entries per configuration, filling a column: the marker beside the name, then the
    # Spearmans behind a blank handle, so they start where the name starts
    rho = correlation.set_index(["hvg", "backend", "n_hvg"])["spearman"]
    handles = []
    for hvg, backend in sorted({(r.hvg, r.backend) for r in nodes.itertuples()},
                               key=lambda c: _config_key(*c)):
        stats = "   ".join(f"$\\rho_{{{n_hvg}}}$ = {rho[(hvg, backend, n_hvg)]:+.2f}"
                           for n_hvg in budgets if (hvg, backend, n_hvg) in rho.index)
        handles += [plt.Line2D([], [], marker=BACKEND_MARK.get(backend, "D"), ls="", ms=3.6,
                               color=ink, markerfacecolor=_tint(ink, HVG_FILL.get(hvg, 0.5)),
                               markeredgewidth=0.7, label=f"{hvg} x {backend}"),
                    plt.Line2D([], [], ls="", label=stats)]
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles) // 2,
                  labelspacing=0.25)
    # names last, once every legend has settled the layout they are placed against
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    for ax, items, obstacles in names:
        _place_labels(figure, ax, items, obstacles, fontsize=5.5, zorder=5, path_effects=halo,
                      color="#1a1a1a" if obstacles else ink)
    if path:
        figures.save(figure, path)
    plt.show()
    return figure


def retrain_figure(frame: pd.DataFrame, colours: dict | None = None, path=None,
                   width=figures.FULL, row_height=0.2, ncols=3):
    """
    runs.retrain_node's recalls: one panel per child class and one for every held-out cell, one
    row per gene set in frame's order, the mean over folds and a bar of one s.d. Panels fill
    ncols columns, the worst-recalled class first.

    colours maps a gene set to its colour, grey where it names none.
    """
    sets = list(dict.fromkeys(frame["genes"]))
    children = [c for c in dict.fromkeys(frame["child"]) if c != "all"]
    # the classes the node loses most on first, the whole node last
    worst = frame[frame["child"] != "all"].groupby("child")["recall"].mean()
    children = sorted(children, key=lambda c: worst[c]) + ["all"]
    stats = frame.groupby(["genes", "child"])["recall"].agg(["mean", "std"])
    sizes = frame.groupby("genes")["n_genes"].mean()
    y = np.arange(len(sets))[::-1]
    colours = colours or {}

    nrows = -(-len(children) // ncols)
    figure, axes = plt.subplots(nrows, ncols, sharey=True, squeeze=False,
                                figsize=(width, nrows * (row_height * len(sets) + 0.55) + 0.35))
    for ax in axes.flat[len(children):]:
        ax.set_visible(False)
    for ax, child in zip(axes.flat, children):
        for yy, name in zip(y, sets):
            mean, std = stats.loc[(name, child)]
            ax.errorbar([mean], [yy], xerr=[0 if np.isnan(std) else std], marker="o", ms=3.6,
                        color=colours.get(name, "#8c8c8c"), elinewidth=0.8, capsize=0, ls="")
        ax.set_title(child if child != "all" else "every cell", loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="x", color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    for row in axes:
        row[0].set_yticks(y, [f"{name} ({sizes[name]:.0f})" for name in sets])
        row[0].set_ylim(-0.6, len(sets) - 0.4)
    figure.supxlabel("recall on held-out cells", fontsize=plt.rcParams["axes.labelsize"])
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


def comparison_latex(table: pd.DataFrame, dataset: str, metric="accuracy", path=None,
                     caption="", label="") -> str:
    """
    fold_frame as LaTeX: one row per configuration, both arms per level, a block per budget.

    A cross-validated budget shows each arm's mean over its folds with the s.d. in small type on
    the line below, a single split its value. Bold marks the arm better in every fold, so on the paired gain and not
    on the two means. A single split bolds the better arm, a tie neither. Underline marks the best
    value of the level within the budget, whose rows share their test donors, where two budgets
    need not. A rule separates the budgets, a gap the HVG methods.
    """
    stats = fold_frame(table, dataset, metric)
    levels = list(dict.fromkeys(stats.index.get_level_values("level")))
    tex = lambda text: str(text).replace("_", r"\_")

    header = (" & ".join(["", "", *(rf"\multicolumn{{2}}{{c}}{{{level}}}" for level in levels)])
              + r" \\" + "\n"
              + "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(levels)))
              + "\n" + " & ".join(["HVG method", "model", *["reselect", "fixed"] * len(levels)])
              + r" \\")

    body = []
    for n_hvg, budget in stats.groupby(level="n_hvg", sort=False):
        if body:
            body.append(r"\midrule")
        note = fold_note(budget[("delta", "n")])
        body.append(rf"\multicolumn{{{2 + 2 * len(levels)}}}{{l}}"
                    rf"{{\emph{{{budget_name(n_hvg)}, {note}}}}} \\")
        best = budget[[(arm, "mean") for arm in ARMS]].max(axis=1).groupby(level="level").max()

        previous = None
        for (hvg, backend), rows in budget.groupby(level=["hvg", "backend"], sort=False):
            if previous is not None and hvg != previous:
                body.append(r"\addlinespace")
            head = tex(hvg.replace("_shareloess", "")) if hvg != previous else ""
            previous = hvg
            rows = rows.droplevel(["n_hvg", "hvg", "backend"])
            cells, spreads = [], []
            for level in levels:
                row = rows.loc[level]
                n = row[("delta", "n")]
                for arm, wins in zip(ARMS, (row[("delta", "gain")], row[("delta", "loss")])):
                    mean = row[(arm, "mean")]
                    text = f"{mean:.4f}"
                    if wins == n:                  # a tie bolds neither arm
                        text = rf"\textbf{{{text}}}"
                    if mean == best[level]:
                        text = rf"\underline{{{text}}}"
                    cells.append(text)
                    spreads.append(rf"{{\scriptsize$\pm${row[(arm, 'std')]:.4f}}}" if n > 1 else "")
            body.append(" & ".join([head, tex(backend), *cells]) + r" \\")
            # the s.d. on a line of its own under the mean, beside it the table outgrows \textwidth
            if any(spreads):
                body.append(" & ".join(["", "", *spreads]) + r" \\[1pt]")
    return figures.latex_table(header, body, "ll" + "rr" * len(levels), path, caption, label)


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

"""
Vector output sized for an A4 LaTeX document.

use_paper_style sets the rcParams.
save writes a PDF, text vector and dense scatter layers rasterized.
latex_table wraps rows in the skeleton every table of the study shares.
"""

# IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


# ---------------------------------------------------------------------


TEXT_WIDTH = 6.3          # inches, \textwidth of A4 with 2.5 cm margins
FULL = TEXT_WIDTH

SCATTER_DPI = 450         # resolution of the rasterized point layers

PAPER_STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3,
    "figure.dpi": 110, "savefig.dpi": SCATTER_DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
}


def use_paper_style() -> None:
    mpl.rcParams.update(PAPER_STYLE)


def save(figure, path, formats=("pdf",)) -> list[Path]:
    """Write figure to path once per format. A suffix on path overrides formats."""
    path = Path(path)
    suffixes = [path.suffix.lstrip(".")] if path.suffix else list(formats)
    written = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        out = path.with_suffix(f".{suffix}")
        figure.savefig(out, facecolor="white")
        written.append(out)
    print("saved -> " + ", ".join(str(p) for p in written))
    return written


def latex_table(header, body, spec, path=None, caption: str = "", label: str = "") -> str:
    """
    Header and body rows in a booktabs table, \\input-able as written.

    spec is the tabular column spec, a body row may be a rule instead of cells.
    """
    latex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}",
        *([rf"\caption{{{caption}}}"] if caption else []),
        *([rf"\label{{{label}}}"] if label else []),
        r"\end{table}",
    ])
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(latex)
        print(f"saved -> {path}")
    return latex

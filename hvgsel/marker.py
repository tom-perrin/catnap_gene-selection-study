"""
The AIFI Human Immune Health Atlas marker genes, as a catnap HVG method taking no parameter.

Transcribed from the atlas' seven cell type description pages, read on 2026-09-30:
    https://apps.allenimmunology.org/aifi/resources/imm-health-atlas/cell-type-descriptions/

Three kinds of entry, each as the pages state it:

    DEFINED    per population, the genes its written definition names (italicised on the page),
               expressed or absent alike: "the lack of ITGB1" separates naive from memory as much
               as ITGB1 does. A leading '-' marks a gene the page calls absent or low, '~' one it
               calls intermediate or variable, none a gene it calls expressed or high.
    DIVISION   genes a level's introduction names as the basis of splitting a node into its
               children ("Memory CD8 T cells were divided ... based of expression of CD27, GZMK,
               GZMB, and KLRF1"), attached to the node that is split.
    PANEL      per page, the genes of its two 'Key markers' figures (violins per Level 3 type,
               then UMAPs), attached to every Level 3 type the violin figure has a row for.

A node's gene set is every marker of every population below it: each child, the child's own
descendants, and the division markers of the node the child splits from. The root therefore
gets the whole panel, the natural fixed_global set, and a node whose children the pages leave
undescribed (NK cell, ILC, the three progenitors) still gets its page's figure panel.

Where the pages name a gene by anything but its HGNC symbol, the symbol is used and the page's
spelling kept in a comment:
    CD21 -> CR2              (B cell)
    HLA-DAP1 -> HLA-DPA1     (Intermediate monocyte, a typo: no such gene, the DC page names
                              the same class II trio as HLA-DPA1, HLA-DRA, CD74)
    IGHG, IGHA -> IGHG1-4, IGHA1-2   (the B cell Level 2 introduction names the families)

Left out on purpose: surface proteins named in prose rather than as genes (CD14/CD16/HLA-D/CCR2
protein, CD56/CD16 protein, CD161, "HLA-DR and HLA-DQ" loci), and CD45RB, which the Early memory
B cell text names as the surface marker its transcriptional labelling could NOT use.

Note: the B cell page's figure caption says "Key markers used to define NK and ILC cell types",
a copy of the NK page's caption. Its figure lists B cell genes, and is taken as the B panel.
"""

# IMPORTS
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd

from catnap_core.hvg import f_statistic
from catnap_core.hvg.base import register_hvg
from catnap_core.utils import resolve_params, warn


# ---------------------------------------------------------------------


SOURCE = "https://apps.allenimmunology.org/aifi/resources/imm-health-atlas/cell-type-descriptions/"

# The atlas hierarchy, as the table heading every page gives it. None marks a Level 1 population.
# A population spanning several levels ("Levels 2, 3") is one entry, at the first of them.
PARENT = {
    "B cell": None, "T cell": None, "NK cell": None, "ILC": None, "Monocyte": None, "DC": None,
    "Progenitor cell": None, "Erythrocyte": None, "Platelet": None,

    "Transitional B cell": "B cell", "Naive B cell": "B cell", "Memory B cell": "B cell",
    "Effector B cell": "B cell", "Plasma cell": "B cell",
    "Core naive B cell": "Naive B cell", "ISG+ naive B cell": "Naive B cell",
    "Early memory B cell": "Memory B cell", "Core memory B cell": "Memory B cell",
    "Type 2 polarized memory B cell": "Memory B cell", "CD95 memory B cell": "Memory B cell",
    "Activated memory B cell": "Memory B cell",
    "CD27+ effector B cell": "Effector B cell", "CD27- effector B cell": "Effector B cell",

    "Naive CD4 T cell": "T cell", "Memory CD4 T cell": "T cell", "Treg": "T cell",
    "DN T cell": "T cell", "Proliferating T cell": "T cell", "Naive CD8 T cell": "T cell",
    "Memory CD8 T cell": "T cell", "CD8aa": "T cell", "gdT": "T cell", "MAIT": "T cell",
    "Core naive CD4 T cell": "Naive CD4 T cell", "SOX4+ naive CD4 T cell": "Naive CD4 T cell",
    "ISG+ naive CD4 T cell": "Naive CD4 T cell",
    "CM CD4 T cell": "Memory CD4 T cell", "GZMB- CD27- EM CD4 T cell": "Memory CD4 T cell",
    "GZMB- CD27+ EM CD4 T cell": "Memory CD4 T cell",
    "KLRF1- GZMB+ CD27- memory CD4 T cell": "Memory CD4 T cell",
    "ISG+ memory CD4 T cell": "Memory CD4 T cell",
    "Naive CD4 Treg": "Treg", "Memory CD4 Treg": "Treg", "KLRB1+ memory CD4 Treg": "Treg",
    "GZMK+ memory CD4 Treg": "Treg", "Memory CD8 Treg": "Treg", "KLRB1+ memory CD8 Treg": "Treg",
    "Core naive CD8 T cell": "Naive CD8 T cell", "SOX4+ naive CD8 T cell": "Naive CD8 T cell",
    "ISG+ naive CD8 T cell": "Naive CD8 T cell",
    "CM CD8 T cell": "Memory CD8 T cell", "GZMK- CD27+ EM CD8 T cell": "Memory CD8 T cell",
    "GZMK+ CD27+ EM CD8 T cell": "Memory CD8 T cell",
    "KLRF1- GZMB+ CD27- EM CD8 T cell": "Memory CD8 T cell",
    "KLRF1+ GZMB+ CD27- EM CD8 T cell": "Memory CD8 T cell",
    "ISG+ memory CD8 T cell": "Memory CD8 T cell",
    "Naive Vd1 gdT": "gdT", "SOX4+ Vd1 gdT": "gdT", "KLRF1- effector Vd1 gdT": "gdT",
    "KLRF1+ effector Vd1 gdT": "gdT", "GZMB+ Vd2 gdT": "gdT", "GZMK+ Vd2 gdT": "gdT",
    "CD8 MAIT": "MAIT", "CD4 MAIT": "MAIT", "ISG+ MAIT": "MAIT",

    "CD56dim NK cell": "NK cell", "CD56bright NK cell": "NK cell",
    "Proliferating NK cell": "NK cell",
    "GZMK+ CD56dim NK cell": "CD56dim NK cell", "Adaptive NK cell": "CD56dim NK cell",
    "ISG+ CD56dim NK cell": "CD56dim NK cell", "GZMK- CD56dim NK cell": "CD56dim NK cell",

    "CD14 monocyte": "Monocyte", "CD16 monocyte": "Monocyte", "Intermediate monocyte": "Monocyte",
    "IL1B+ CD14 monocyte": "CD14 monocyte", "ISG+ CD14 monocyte": "CD14 monocyte",
    "Core CD14 monocyte": "CD14 monocyte",
    "C1Q+ CD16 monocyte": "CD16 monocyte", "ISG+ CD16 monocyte": "CD16 monocyte",
    "Core CD16 monocyte": "CD16 monocyte",

    "ASDC": "DC", "cDC1": "DC", "cDC2": "DC", "pDC": "DC",
    "CD14+ cDC2": "cDC2", "HLA-DRhi cDC2": "cDC2", "ISG+ cDC2": "cDC2",

    "CLP cell": "Progenitor cell", "CMP cell": "Progenitor cell", "BaEoMaP cell": "Progenitor cell",
}

DEFINED = {
    # --- b-cells-and-plasma-cells ---
    "B cell": "CD19 CD79A CD79B CR2 IGHM IGHD IGHA1 IGHA2 IGHG1 IGHG2 IGHG3 IGHG4 IGHE "
              "IGKC IGLC2 IGLC3 MS4A1 PAX5",                                 # CR2: page's "CD21"
    "Transitional B cell": "MME CD9 CD38 PAX5 FCER2",
    "Naive B cell": "IL4R FCER2 -MME -CD24 -CD9",
    "Memory B cell": "CD27 AIM2 IGHG1 IGHG2 IGHG3 IGHG4 IGHA1 IGHA2 IGHE",
    "Effector B cell": "ITGAX FCRL4 FCRL5 TBX21 ZEB2 PDCD1 ~CD27 ~AIM2",
    "Plasma cell": "-MS4A1 -PAX5 CD19 PRDM1 XBP1 MZB1 SLAMF7 CD27 CD38 -IGHD",
    "ISG+ naive B cell": "STAT3 STAT1 IFI44L ISG15",
    "Early memory B cell": "~CD27 ~AIM2 ~IGHA1 ~IGHA2 ~IGHG1 ~IGHG2 ~IGHG3 ~IGHG4",
    "Type 2 polarized memory B cell": "IL4R FCER2 COCH IGHG1 IGHG4 IGHE",
    "CD95 memory B cell": "FAS AIM2 IGHA1 IGHA2 IGHG1 IGHG2 IGHG3 IGHG4",
    "Activated memory B cell": "FOS CD69 JUN MCL1 MYC",
    "CD27+ effector B cell": "CD27",
    "CD27- effector B cell": "-CD27 -AIM2 ITGAX TBX21 ZEB2",

    # --- cd4-t-cells-dn-t-cells-and-tregs, cd8-t-cells-gdt-cells-and-mait-cells ---
    "T cell": "TRAC TRDC CD3D CD3E CD3G",
    "Naive CD4 T cell": "CD27 CCR7 SELL TCF7 LEF1 -ITGB1",
    "Memory CD4 T cell": "ITGB1",
    "Treg": "FOXP3 IL2RA IKZF2 RTKN2",
    "DN T cell": "TRAC -CD4 -CD8A -CD8B",
    "Proliferating T cell": "MKI67",
    "Core naive CD4 T cell": "CD27 CCR7 SELL TCF7 LEF1 -ITGB1",
    "SOX4+ naive CD4 T cell": "SOX4",
    "ISG+ naive CD4 T cell": "MX1 IFI44",
    "CM CD4 T cell": "CCR7 SELL LEF1",
    "GZMB- CD27- EM CD4 T cell": "-CCR7 -SELL -LEF1 -GZMA -GZMK -GZMB",
    "GZMB- CD27+ EM CD4 T cell": "CD27 GZMK -GZMB -CCR7 -SELL -LEF1",
    "KLRF1- GZMB+ CD27- memory CD4 T cell": "GZMB CCL5 -KLRF1 -CD27 -GZMK",
    "ISG+ memory CD4 T cell": "MX1 IFI44",
    "Naive CD4 Treg": "CD27 CCR7 SELL TCF7 LEF1 -ITGB1",
    "Memory CD4 Treg": "ITGB1",
    "KLRB1+ memory CD4 Treg": "KLRB1 -GZMK",
    "GZMK+ memory CD4 Treg": "GZMK -KLRB1",
    "Memory CD8 Treg": "CD8A -CD4",
    "KLRB1+ memory CD8 Treg": "KLRB1",
    "Naive CD8 T cell": "CD27 CCR7 SELL TCF7 LEF1 -GZMA -GZMK -GZMB -ITGB1",
    "Memory CD8 T cell": "ITGB1 GZMA GZMB GZMK -CCR7 TRAC",
    "CD8aa": "CD8A KLRC2 IKZF2 IL21R",
    "MAIT": "SLC4A10 KLRB1",
    "gdT": "TRDC TRGC1 TRGC2 -TRAC",
    "Core naive CD8 T cell": "CD27 CCR7 SELL TCF7 LEF1 -ITGB1",
    "SOX4+ naive CD8 T cell": "SOX4",
    "ISG+ naive CD8 T cell": "MX1 IFI44",
    "CM CD8 T cell": "CCR7 SELL LEF1",
    "GZMK- CD27+ EM CD8 T cell": "CD27 -CCR7 -SELL -LEF1 -GZMK -GZMB",
    "GZMK+ CD27+ EM CD8 T cell": "CD27 GZMK -GZMB -CCR7 -SELL -LEF1",
    "KLRF1- GZMB+ CD27- EM CD8 T cell": "GZMB CCL5 -KLRF1 -CD27 -GZMK",
    "KLRF1+ GZMB+ CD27- EM CD8 T cell": "KLRF1 GZMB CCL5 -CD27 -GZMK",
    "ISG+ memory CD8 T cell": "MX1 IFI44",
    "Naive Vd1 gdT": "CCR7 SELL LEF1 -ITGB1",
    "SOX4+ Vd1 gdT": "SOX4",
    "KLRF1- effector Vd1 gdT": "TRDC TRDV1 -KLRF1",
    "KLRF1+ effector Vd1 gdT": "TRDC TRDV1 KLRF1",
    "GZMB+ Vd2 gdT": "TRDC TRDV2 GZMB",
    "GZMK+ Vd2 gdT": "TRDC TRDV2 GZMK",
    "CD8 MAIT": "CD8A",
    "CD4 MAIT": "CD4",
    "ISG+ MAIT": "MX1 IFI44",

    # --- dendritic-cells ---
    "DC": "CST3 FLT3 HLA-DPA1 HLA-DRA CD74",
    "ASDC": "AXL SIGLEC6 HAMP",
    "cDC1": "CLEC9A XCR1 IDO1 C1orf54",
    "cDC2": "CD1C FCN1 PILRA",
    "pDC": "PTCRA SMIM5 LAMP5 IL3RA JCHAIN",
    "CD14+ cDC2": "CST3 CD74 HLA-DRA HLA-DPA1 CD14 S100A8 S100A9 VCAN CD163",
    "HLA-DRhi cDC2": "HLA-DPA1 HLA-DRA CD1C",
    "ISG+ cDC2": "MX1 IFI44L IFI6",

    # --- monocytes ---
    "Monocyte": "FCN1 CTSS -CD1C",
    "CD14 monocyte": "CD14 VCAN S100A8 S100A9 -FCGR3A -HLA-DRA",
    "CD16 monocyte": "FCGR3A CDKN1C LST1",
    "Intermediate monocyte": "CD14 FCGR3A HLA-DPA1 HLA-DOA HLA-DRA CD74",   # page: "HLA-DAP1"
    "IL1B+ CD14 monocyte": "IL1B CCL3 CXCL8",
    "ISG+ CD14 monocyte": "MX1 IFI44L IFI6",
    "C1Q+ CD16 monocyte": "C1QA C1QB",
    "ISG+ CD16 monocyte": "MX1 IFI44L IFI6",

    # --- nk-cells-and-ilcs ---
    "CD56dim NK cell": "-NCAM1 FCGR3A",
    "CD56bright NK cell": "NCAM1 -FCGR3A",
    "Proliferating NK cell": "MKI67",
    "GZMK+ CD56dim NK cell": "GZMK",
    "Adaptive NK cell": "KLRC2",
    "ISG+ CD56dim NK cell": "ISG15 MX1 MX2",

    # --- other-cell-types ---
    "Erythrocyte": "HBA1 HBA2 HBB",
    "Progenitor cell": "SMIM24 CD34",
    "Platelet": "PPBP TUBB1",
}

# "The same gene expression profile described for level 2 Naive B cells applies for this level 3
# label", and likewise Core memory for Memory. The atlas defines these by reference.
SAME_AS = {"Core naive B cell": "Naive B cell", "Core memory B cell": "Memory B cell"}

# Defined only as "the remaining" cells of their parent, without a gene of their own: Core CD14
# and Core CD16 monocyte, GZMK- CD56dim NK cell. The pages give no marker at all for NK cell,
# ILC, CLP, CMP and BaEoMaP cell. Their figure panel is what they get.

DIVISION = {
    "B cell": "IGHM IGHD IGHG1 IGHG2 IGHG3 IGHG4 IGHA1 IGHA2 IGHE",   # page: "IGHG, IGHA, or IGHE"
    "T cell": "TRAC TRDC CD4 CD8A CD8B",
    "Memory CD4 T cell": "CD27 GZMK GZMB KLRF1",   # named on the CD4 page, but not italicised
    "Memory CD8 T cell": "CD27 GZMK GZMB KLRF1",
    "gdT": "TRDV1 TRDV2",
}

# page -> (the figure's genes, the Level 3 rows of its violin figure)
PANEL = {
    "b-cells-and-plasma-cells": (
        "MS4A1 CD19 MME CD24 CD27 AIM2 ZEB2 FAS IL4R FCER2 CD69 FOS MKI67 SLAMF7 PRDM1 IGHD IGHG1 "
        "IGHA2 IGHE",
        ["Transitional B cell", "Core naive B cell", "ISG+ naive B cell", "Early memory B cell",
         "Core memory B cell", "Activated memory B cell", "Type 2 polarized memory B cell",
         "CD95 memory B cell", "CD27+ effector B cell", "CD27- effector B cell", "Plasma cell"]),
    "cd4-t-cells-dn-t-cells-and-tregs": (
        "CD8A SOX4 SELL CCR7 ITGB1 ITGA4 FAS GZMK GZMA GZMB IL2RA KLRF1 KLRB1 KLRD1 ISG15 MX1",
        ["SOX4+ naive CD4 T cell", "Core naive CD4 T cell", "ISG+ naive CD4 T cell",
         "CM CD4 T cell", "GZMB- CD27+ EM CD4 T cell", "GZMB- CD27- EM CD4 T cell",
         "ISG+ memory CD4 T cell", "KLRF1- GZMB+ CD27- memory CD4 T cell", "Naive CD4 Treg",
         "Memory CD4 Treg", "KLRB1+ memory CD4 Treg", "GZMK+ memory CD4 Treg", "Memory CD8 Treg",
         "KLRB1+ memory CD8 Treg", "DN T cell", "Proliferating T cell"]),
    "cd8-t-cells-gdt-cells-and-mait-cells": (
        "CD8A SOX4 LEF1 SELL CCR7 ITGB1 ITGA4 FAS GZMK GZMA GZMB KLRF1 KLRB1 KLRD1 KLRC2 TRAC TRDC "
        "TRGC1 TRDV1 TRDV2 ISG15 MX1 CD27",                     # CD27: UMAP figure only
        ["SOX4+ naive CD8 T cell", "Core naive CD8 T cell", "ISG+ naive CD8 T cell",
         "CM CD8 T cell", "GZMK+ CD27+ EM CD8 T cell", "KLRF1- GZMB+ CD27- EM CD8 T cell",
         "GZMK- CD27+ EM CD8 T cell", "KLRF1+ GZMB+ CD27- EM CD8 T cell", "ISG+ memory CD8 T cell",
         "CD8aa", "SOX4+ Vd1 gdT", "Naive Vd1 gdT", "KLRF1- effector Vd1 gdT",
         "KLRF1+ effector Vd1 gdT", "GZMK+ Vd2 gdT", "GZMB+ Vd2 gdT", "CD8 MAIT", "CD4 MAIT",
         "ISG+ MAIT"]),
    "dendritic-cells": (
        "CST3 CD163 VCAN CD14 CD1C HLA-DRA ISG15 MX1 IFI44L CLEC9A C1orf54 AXL SIGLEC6 S100A10 SPI1 "
        "ITM2C IL3RA IRF8 PLAC8",
        ["CD14+ cDC2", "HLA-DRhi cDC2", "ISG+ cDC2", "cDC1", "pDC", "ASDC"]),
    "monocytes": (
        "S100A8 S100A9 CD14 VCAN MX1 ISG15 IFI44L NFKBIA IL1B CCL3 SOD2 CD74 FCGR3A CDKN1C LST1 "
        "C1QA C1QB CD68",
        ["Core CD14 monocyte", "ISG+ CD14 monocyte", "IL1B+ CD14 monocyte",
         "Intermediate monocyte", "Core CD16 monocyte", "ISG+ CD16 monocyte",
         "C1Q+ CD16 monocyte"]),
    "nk-cells-and-ilcs": (
        "CD3E CD3D NCAM1 FCGR3A KLRC1 KLRC2 FCER1G GZMB GZMK MX1 MKI67",
        ["CD56bright NK cell", "GZMK+ CD56dim NK cell", "GZMK- CD56dim NK cell",
         "Adaptive NK cell", "ISG+ CD56dim NK cell", "Proliferating NK cell", "ILC"]),
    "other-cell-types": (
        "HBB CD34 CD38 PTPRC DNTT",
        ["CLP cell", "CMP cell", "BaEoMaP cell", "Platelet", "Erythrocyte"]),
}


# --- Reading the tables ----------------------------------------------

def _entries(text: str) -> list[tuple[str, str]]:
    """'A -B ~C' -> [(A, expressed), (B, absent), (C, intermediate)]."""
    kind = {"-": "absent", "~": "intermediate"}
    return [(token.lstrip("-~"), kind.get(token[0], "expressed")) for token in text.split()]


def level(population: str) -> int:
    """1 for a population without parent, one more per ancestor."""
    return 1 if PARENT[population] is None else 1 + level(PARENT[population])


def compartment(population: str) -> str:
    """The Level 1 population population sits under, itself for a Level 1 one."""
    return population if PARENT[population] is None else compartment(PARENT[population])


def children(population: str) -> list[str]:
    return [name for name, parent in PARENT.items() if parent == population]


def subtree(population: str) -> list[str]:
    """population, then every population below it."""
    return [population, *(below for child in children(population) for below in subtree(child))]


@lru_cache(maxsize=None)
def marker_table() -> pd.DataFrame:
    """
    Every (population, gene) the atlas names, one row each: source (defined, same_as, division or
    panel), the direction the page states, and the page or node it comes from.
    """
    page_of = {row: page for page, (_, rows) in PANEL.items() for row in rows}
    rows = []
    for population in PARENT:
        own = DEFINED.get(population, "")
        for gene, direction in _entries(own):
            rows.append((population, gene, "defined", direction, population))
        if population in SAME_AS:
            for gene, direction in _entries(DEFINED[SAME_AS[population]]):
                rows.append((population, gene, "same_as", direction, SAME_AS[population]))
        if PARENT[population] in DIVISION:
            for gene, _ in _entries(DIVISION[PARENT[population]]):
                rows.append((population, gene, "division", "split", PARENT[population]))
        if population in page_of:
            for gene, _ in _entries(PANEL[page_of[population]][0]):
                rows.append((population, gene, "panel", "figure", page_of[population]))
    table = pd.DataFrame(rows, columns=["population", "gene", "source", "direction", "from"])
    table.insert(1, "level", table["population"].map(level))
    return table


def population_markers(population: str) -> set[str]:
    """Every marker of population and of each population below it."""
    table = marker_table()
    return set(table.loc[table["population"].isin(subtree(population)), "gene"])


def node_markers(child_names) -> set[str]:
    """A node's gene set, from the names of its children: everything any of them is marked by."""
    unknown = sorted(set(child_names) - set(PARENT))
    if unknown:
        raise KeyError(f"not an AIFI atlas population: {unknown}")
    return set().union(*(population_markers(child) for child in child_names))


def fill(markers, ranked, budget: int) -> list[str]:
    """
    A crossover selection: every gene of markers, then ranked's other genes in its order, until
    budget genes. ranked is a selection best first, f_statistic's say.
    """
    markers = list(dict.fromkeys(markers))
    rest = [gene for gene in ranked if gene not in set(markers)]
    return markers + rest[:max(0, budget - len(markers))]


# --- The HVG method --------------------------------------------------

@register_hvg("marker")
def select_hvgs(adata, labels=None) -> List[str]:
    """
    The atlas markers of every population below the node, in adata's gene order.

    The node is read off labels, the name of the child each cell belongs to, so the method takes
    no parameter and makes no use of the counts. A marker absent from adata is left out, with a
    warning.
    """
    if labels is None:
        raise ValueError("hvg:marker reads the node off its children's labels, none were passed")
    names = sorted({str(label) for label in np.unique(np.asarray(labels))})
    wanted = node_markers(names)
    genes = [gene for gene in adata.var_names.astype(str) if gene in wanted]
    missing = sorted(wanted - set(genes))
    if missing:
        warn(f"hvg:marker: {len(missing)} marker(s) not in the data, left out: {missing}")
    return genes


@register_hvg("f_statistic_markers")
def select_f_statistic_markers(adata, labels=None, **hvg_args) -> List[str]:
    """
    A crossover: the node's atlas markers, then f_statistic's best other genes, n_top_genes in all.

    The markers are marker's at this node, the fill f_statistic's ranking on the node's children,
    so a node short of markers on its own gets them at the cost of its lowest-scoring genes.
    hvg_args are f_statistic's.
    """
    if labels is None:
        raise ValueError("hvg:f_statistic_markers reads the node off its children's labels, "
                         "none were passed")
    params = resolve_params(hvg_args, f_statistic.DEFAULTS, context="hvg:f_statistic_markers")
    names = sorted({str(label) for label in np.unique(np.asarray(labels))})
    wanted = node_markers(names)
    markers = [gene for gene in adata.var_names.astype(str) if gene in wanted]
    ranked = f_statistic.select_hvgs(adata, labels=labels, **params)
    return fill(markers, ranked, int(params["n_top_genes"]))

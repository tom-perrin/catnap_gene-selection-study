# HVG selection study

Per-node HVG re-selection vs a single global selection, in catnap_core.

To reproduce results, set parameters in `hvg_selection.ipynb`'s cells.  
The code used is in `hvgsel/`.

Note: `results/`, `figures/` and `tables/` ship with the repo, `runs/` and `runs_repeats` do not. The selection cells will only read the cached scores but the downstream cells will train all models.


## Install & run

```bash
conda env create -f environment.yml
conda activate hvg_selection
pip install -e <path to the catnap_core repo>
jupyter lab hvg_selection.ipynb
```

## Data

| dataset | cells × genes | source |
|---|---|---|
| Hao PBMC | 161,764 × 20,264 | [CELLxGENE](https://cellxgene.cziscience.com/collections/b0cf0afa-ec40-4d65-b570-ed4ceacc6813) |
| Human Immune Health Atlas (AIFI) | 1,821,725 × 33,538 | [Allen Institute for Immunology](https://apps.allenimmunology.org/aifi/resources/imm-health-atlas/downloads/scrna/) |

Paths, label columns and where the raw counts live are in `hvgsel/datasets.py`.

Note: the paths are the ones the study ran on, point `Dataset.path` at your own copies.


## Notations

| symbol | |
|---|---|
| $v$, $v'$, $v_0$ | a node, another node outside $\mathcal{T}_v$, the root |
| $\mathcal{C}_v$, $N_v$ | the cells of $v$'s subtree, and how many |
| $\bar x_{g,v}$, $V_{g,v}$ | mean and variance of gene $g$ over $\mathcal{C}_v$ |
| **cut** | the partition of $\mathcal{C}_v$ a supervised score is computed against: `children` or `leaves` |
| $F_{g,v}$, $F_{g,v}(\mathrm{leaves})$ | ANOVA $F$ at $v$, on the children / leaf cut |
| $H_{g,v}$ | Kruskal–Wallis $H$, same convention |
| $\widetilde f_w$ | LOESS of $\log_{10} V_{g,w}$ on $\log_{10}\bar x_{g,w}$, fitted over $\mathcal{C}_w$ |
| $\widetilde V_{g,v}$ | its fitted response at $v$'s mean, $10^{\widetilde f_w(\log_{10}\bar x_{g,v})}$ |

$$\mathrm{VST}^{(\widetilde f_w)}_{g,v} = \frac{1}{N_v-1}\sum_{c \in \mathcal{C}_v} z_{cg,v}^2,
\qquad z_{cg,v} = \min\!\left( \frac{x_{cg}-\bar x_{g,v}}{\sqrt{\widetilde V_{g,v}}},\ \sqrt{N_v} \right).$$


## Experiments

| key | $a$ | $b$ |
|---|---|---|
| `F-NR` / `KW-NR` | $F_{g,v}$ | $F_{g,v_0}(\mathrm{leaves})$ |
| `F-NN` / `KW-NN` | $F_{g,v}$ | $F_{g,v'}$ |
| `F-CL` / `KW-CL` | $F_{g,v}$ | $F_{g,v}(\mathrm{leaves})$ |
| `VST(v)-NR` | $\mathrm{VST}^{(\widetilde f_v)}_{g,v}$ | $\mathrm{VST}^{(\widetilde f_{v_0})}_{g,v_0}$ |
| `VST(v_0)-NR` | $\mathrm{VST}^{(\widetilde f_{v_0})}_{g,v}$ | $\mathrm{VST}^{(\widetilde f_{v_0})}_{g,v_0}$ |
| `VST(v_0)-NN` | $\mathrm{VST}^{(\widetilde f_{v_0})}_{g,v}$ | $\mathrm{VST}^{(\widetilde f_{v_0})}_{g,v'}$ |

Panels: gene scores $a$ vs $b$, top-$k$ overlap vs $k$.

Selection statistics: `changed%`, `Jaccard` at the top-$n$; `Spearman`, `Kendall`, `RBO` over the full ranking.

Prediction statistics: accuracy and macro-F1 per level, accuracy per node.


## Faithfulness to catnap_core

Since the selectors in catnap_core.hvg return gene names and not scores, only the per-gene maths is imported from catnap_core. The array containing scores for all genes is built here.

Note: When the seurat_v3 LOESS solver fails on a degenerate subtree, catnap_core raises but `vst_scores` instead widens the span, recording `loess_widened` in the metadata.
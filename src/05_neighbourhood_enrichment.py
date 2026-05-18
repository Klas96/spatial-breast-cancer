"""
Spatial neighbourhood enrichment — which cell types co-localise?
Input:  results/03_deconvolved.h5ad
Output: results/05_nhood_enrichment.png
"""

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import RESULTS_DIR

sc.settings.figdir = RESULTS_DIR


def assign_dominant_cell_type(adata: sc.AnnData) -> sc.AnnData:
    abundances = adata.obsm["means_cell_abundance_w_sf"]
    dominant = abundances.idxmax(axis=1)
    # Strip common cell2location prefix so labels are bare cell type names
    from os.path import commonprefix
    prefix = commonprefix(abundances.columns.tolist())
    if prefix:
        dominant = dominant.str[len(prefix):]
    adata.obs["dominant_cell_type"] = dominant.astype("category")
    return adata


def run_nhood_enrichment(adata: sc.AnnData) -> sc.AnnData:
    sq.gr.spatial_neighbors(adata, coord_type="grid")
    sq.gr.nhood_enrichment(adata, cluster_key="dominant_cell_type")
    return adata


def plot(adata: sc.AnnData) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap of enrichment z-scores
    zscore = pd.DataFrame(
        adata.uns["dominant_cell_type_nhood_enrichment"]["zscore"],
        index=adata.obs["dominant_cell_type"].cat.categories,
        columns=adata.obs["dominant_cell_type"].cat.categories,
    )
    sns.heatmap(
        zscore, center=0, cmap="RdBu_r", ax=axes[0],
        square=True, xticklabels=True, yticklabels=True
    )
    axes[0].set_title("Neighbourhood enrichment (z-score)")
    axes[0].tick_params(axis="x", rotation=45)

    # Spatial scatter coloured by dominant cell type
    sq.pl.spatial_scatter(adata, color="dominant_cell_type", ax=axes[1], img=True)
    axes[1].set_title("Dominant cell type per spot")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_nhood_enrichment.png", dpi=150)
    plt.close()
    print("Saved: results/05_nhood_enrichment.png")


def main():
    adata = sc.read(RESULTS_DIR / "03_deconvolved.h5ad")
    adata = assign_dominant_cell_type(adata)
    adata = run_nhood_enrichment(adata)
    plot(adata)
    adata.write(RESULTS_DIR / "05_nhood_enrichment.h5ad")
    print("Saved: results/05_nhood_enrichment.h5ad")


if __name__ == "__main__":
    main()

"""
Dimensionality reduction, spatial clustering, and visualisation.
Input:  results/01_qc_preprocessed.h5ad
Output: results/02_clustered.h5ad
"""

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
sc.settings.figdir = RESULTS_DIR
sc.settings.verbosity = 2


def compute_neighbours(adata: sc.AnnData) -> sc.AnnData:
    # Transcriptional neighbours
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    # Spatial neighbours (for spatially-aware clustering)
    sq.gr.spatial_neighbors(adata, coord_type="grid")
    return adata


def cluster(adata: sc.AnnData, resolution: float = 0.5) -> sc.AnnData:
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden", flavor="igraph", directed=False, n_iterations=2)
    return adata


def plot(adata: sc.AnnData) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc.pl.umap(adata, color="leiden", ax=axes[0], show=False)
    sq.pl.spatial_scatter(adata, color="leiden", ax=axes[1], img=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_clusters.png", dpi=150)
    plt.close()
    print("Saved: results/02_clusters.png")


def main():
    adata = sc.read(RESULTS_DIR / "01_qc_preprocessed.h5ad")
    adata = compute_neighbours(adata)
    adata = cluster(adata)
    plot(adata)
    adata.write(RESULTS_DIR / "02_clustered.h5ad")
    print("Saved: results/02_clustered.h5ad")


if __name__ == "__main__":
    main()

"""
Spatially variable gene (SVG) detection using Moran's I.
Input:  results/03_deconvolved.h5ad
Output: results/04_svg_results.csv
"""

import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
sc.settings.figdir = RESULTS_DIR
N_TOP_SVGS = 12


def compute_morans_i(adata: sc.AnnData) -> sc.AnnData:
    sq.gr.spatial_neighbors(adata, coord_type="visium")
    sq.gr.spatial_autocorr(adata, mode="moran", n_jobs=4)
    return adata


def save_results(adata: sc.AnnData) -> pd.DataFrame:
    svg_df = (
        adata.uns["moranI"]
        .sort_values("I", ascending=False)
        .reset_index()
        .rename(columns={"index": "gene"})
    )
    svg_df.to_csv(RESULTS_DIR / "04_svg_results.csv", index=False)
    print(f"Top SVG: {svg_df['gene'].iloc[0]}  (Moran's I = {svg_df['I'].iloc[0]:.3f})")
    return svg_df


def plot_top_svgs(adata: sc.AnnData, svg_df: pd.DataFrame) -> None:
    top_genes = svg_df["gene"].head(N_TOP_SVGS).tolist()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, gene in enumerate(top_genes):
        sq.pl.spatial_scatter(adata, color=gene, ax=axes[i], img=False, colormap="viridis")
        axes[i].set_title(gene, fontsize=9)
    fig.suptitle("Top spatially variable genes (Moran's I)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_top_svgs.png", dpi=150)
    plt.close()
    print("Saved: results/04_top_svgs.png")


def main():
    adata = sc.read(RESULTS_DIR / "03_deconvolved.h5ad")
    adata = compute_morans_i(adata)
    svg_df = save_results(adata)
    plot_top_svgs(adata, svg_df)


if __name__ == "__main__":
    main()

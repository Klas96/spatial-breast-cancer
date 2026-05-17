"""
Cell type deconvolution with Cell2location.
Requires a matched scRNA-seq reference (Wu et al. 2021, GEO: GSE176078).

Input:  results/02_clustered.h5ad
        data/scrna_reference.h5ad  (preprocessed scRNA-seq atlas)
Output: results/03_deconvolved.h5ad
"""

import scanpy as sc
import squidpy as sq
import cell2location
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
sc.settings.figdir = RESULTS_DIR
sc.settings.verbosity = 2

N_CELLS_PER_SPOT = 8   # approximate for breast tissue Visium


def fit_reference_model(adata_ref: sc.AnnData) -> cell2location.models.RegressionModel:
    cell2location.models.RegressionModel.setup_anndata(
        adata_ref,
        labels_key="cell_type",
    )
    model = cell2location.models.RegressionModel(adata_ref)
    model.train(max_epochs=200, use_gpu=False)
    return model


def get_signature(model: cell2location.models.RegressionModel, adata_ref: sc.AnnData):
    adata_ref = model.export_posterior(adata_ref, sample_kwargs={"num_samples": 1000})
    return adata_ref.uns["means_per_cluster_mu_fg"]


def fit_spatial_model(adata: sc.AnnData, cell_type_signatures) -> sc.AnnData:
    cell2location.models.Cell2location.setup_anndata(adata)
    model = cell2location.models.Cell2location(
        adata,
        cell_state_df=cell_type_signatures,
        N_cells_per_location=N_CELLS_PER_SPOT,
        detection_alpha=20,
    )
    model.train(max_epochs=10000, batch_size=None, use_gpu=False)
    adata = model.export_posterior(adata, sample_kwargs={"num_samples": 1000})
    return adata


def plot_cell_types(adata: sc.AnnData) -> None:
    cell_types = adata.obsm["means_cell_abundance_w_sf"].columns.tolist()
    n = len(cell_types)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for i, ct in enumerate(cell_types):
        sq.pl.spatial_scatter(adata, color=ct, ax=axes[i], img=False, colormap="magma")
        axes[i].set_title(ct, fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_cell_type_maps.png", dpi=150)
    plt.close()
    print("Saved: results/03_cell_type_maps.png")


def main():
    adata = sc.read(RESULTS_DIR / "02_clustered.h5ad")
    adata_ref = sc.read(Path("data") / "scrna_reference.h5ad")

    ref_model = fit_reference_model(adata_ref)
    signatures = get_signature(ref_model, adata_ref)
    adata = fit_spatial_model(adata, signatures)
    plot_cell_types(adata)
    adata.write(RESULTS_DIR / "03_deconvolved.h5ad")
    print("Saved: results/03_deconvolved.h5ad")


if __name__ == "__main__":
    main()

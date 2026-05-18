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
    model.train(max_epochs=200, accelerator="gpu")
    return model


def get_signature(model: cell2location.models.RegressionModel, adata_ref: sc.AnnData):
    adata_ref = model.export_posterior(adata_ref, sample_kwargs={"num_samples": 1000})
    return adata_ref.varm["means_per_cluster_mu_fg"]


def fit_spatial_model(adata: sc.AnnData, cell_type_signatures) -> sc.AnnData:
    cell2location.models.Cell2location.setup_anndata(adata)
    model = cell2location.models.Cell2location(
        adata,
        cell_state_df=cell_type_signatures,
        N_cells_per_location=N_CELLS_PER_SPOT,
        detection_alpha=20,
    )
    model.train(max_epochs=10000, batch_size=None, accelerator="gpu")
    adata = model.export_posterior(adata, sample_kwargs={"num_samples": 1000})
    return adata


def plot_cell_types(adata: sc.AnnData) -> None:
    abundances = adata.obsm["means_cell_abundance_w_sf"]
    cell_types = abundances.columns.tolist()
    # Copy to obs so squidpy can access them for plotting
    for ct in cell_types:
        adata.obs[ct] = abundances[ct].values
    n = len(cell_types)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for i, ct in enumerate(cell_types):
        sq.pl.spatial_scatter(adata, color=ct, ax=axes[i], img=False, cmap="magma")
        axes[i].set_title(ct, fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_cell_type_maps.png", dpi=150)
    plt.close()
    print("Saved: results/03_cell_type_maps.png")


def main():
    # Load clustered data to get QC-filtered spot barcodes and spatial info
    adata_clustered = sc.read(RESULTS_DIR / "02_clustered.h5ad")

    # Reload raw Visium counts — cell2location requires raw integer counts
    adata_raw = sc.read_visium(Path("data/visium"))
    adata_raw.var_names_make_unique()
    adata_raw = adata_raw[adata_clustered.obs_names].copy()  # same spots as post-QC
    # Transfer cluster labels and spatial embedding for downstream plots
    adata_raw.obs = adata_raw.obs.join(adata_clustered.obs[["leiden"]])
    adata_raw.obsm["spatial"] = adata_clustered.obsm["spatial"]

    # Wu et al. 2021 atlas — use raw counts and remap Ensembl IDs to gene symbols
    adata_ref_full = sc.read(Path("data") / "wu2021_scrna_atlas.h5ad")
    adata_ref = adata_ref_full.raw.to_adata()
    adata_ref.var_names = adata_ref.var["feature_name"].values
    adata_ref.var_names_make_unique()
    adata_ref.obs["cell_type"] = adata_ref_full.obs["celltype_major"].values.astype(str)
    adata_ref.obs["cell_type"] = adata_ref.obs["cell_type"].astype("category")
    del adata_ref_full

    ref_model = fit_reference_model(adata_ref)
    signatures = get_signature(ref_model, adata_ref)

    # Restrict to genes present in both datasets
    shared_genes = adata_raw.var_names.intersection(signatures.index)
    print(f"Shared genes: {len(shared_genes)}")
    adata_raw = adata_raw[:, shared_genes].copy()
    signatures = signatures.loc[shared_genes]

    adata_raw = fit_spatial_model(adata_raw, signatures)
    plot_cell_types(adata_raw)
    adata_raw.write(RESULTS_DIR / "03_deconvolved.h5ad")
    print("Saved: results/03_deconvolved.h5ad")


if __name__ == "__main__":
    main()

"""
QC and preprocessing of 10X Visium breast cancer data.
Input:  data/visium/  (raw SpaceRanger output)
Output: results/01_qc_preprocessed.h5ad
"""

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from pathlib import Path
from config import VISIUM_DIR, RESULTS_DIR

RESULTS_DIR.mkdir(exist_ok=True)
sc.settings.figdir = RESULTS_DIR
sc.settings.verbosity = 2


def load_data(data_dir: Path) -> sc.AnnData:
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()
    return adata


def run_qc(adata: sc.AnnData) -> sc.AnnData:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(adata.obs["total_counts"], bins=50)
    axes[0].set_xlabel("Total counts")
    axes[1].hist(adata.obs["n_genes_by_counts"], bins=50)
    axes[1].set_xlabel("Genes detected")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=50)
    axes[2].set_xlabel("% mitochondrial counts")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_qc_distributions.png", dpi=150)
    plt.close()

    adata = adata[
        (adata.obs["total_counts"] > 1000) &
        (adata.obs["n_genes_by_counts"] > 500) &
        (adata.obs["pct_counts_mt"] < 20)
    ].copy()
    print(f"Spots after QC filtering: {adata.n_obs}")
    return adata


def preprocess(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", subset=False)
    adata.raw = adata
    sc.pp.scale(adata)
    sc.pp.pca(adata, mask_var="highly_variable")
    return adata


def main():
    adata = load_data(VISIUM_DIR)
    adata = run_qc(adata)
    adata = preprocess(adata)
    adata.write(RESULTS_DIR / "01_qc_preprocessed.h5ad")
    print(f"Saved: {RESULTS_DIR}/01_qc_preprocessed.h5ad")


if __name__ == "__main__":
    main()

"""
Build scrna_reference.h5ad from Wu et al. 2021 Zenodo download.
Reads per-sample 10X matrices + metadata CSVs, concatenates, saves.
"""

import anndata
import pandas as pd
import scipy.io
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "scrna_raw"
OUT = RAW.parent / "scrna_reference.h5ad"


def read_10x_plain(sample_dir: Path) -> anndata.AnnData:
    """Read 10X mtx directory where files lack .gz extension."""
    X = scipy.io.mmread(sample_dir / "matrix.mtx").T.tocsr()
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv", header=None)[0].values
    features = pd.read_csv(sample_dir / "features.tsv", header=None, sep="\t")
    gene_names = features[0].values
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )


samples = sorted(RAW.glob("filtered_count_matrices/*_filtered_count_matrix"))

adatas = []
for sample_dir in samples:
    sample_id = sample_dir.name.replace("_filtered_count_matrix", "")
    adata = read_10x_plain(sample_dir)
    adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

    meta_path = RAW / "metadata" / f"{sample_id}_metadata.csv"
    meta = pd.read_csv(meta_path, index_col=0)
    meta.index = [f"{sample_id}_{bc}" for bc in meta.index]

    shared = adata.obs_names.intersection(meta.index)
    adata = adata[shared]
    adata.obs = adata.obs.join(meta[["Classification", "subtype", "patientid"]])
    adatas.append(adata)
    print(f"  {sample_id}: {adata.n_obs} cells")

combined = adatas[0].concatenate(adatas[1:], batch_key="sample", index_unique=None)
combined.obs.rename(columns={"Classification": "cell_type"}, inplace=True)
combined.var_names_make_unique()

print(f"\nTotal cells: {combined.n_obs}, genes: {combined.n_vars}")
print(f"Cell types: {combined.obs['cell_type'].unique().tolist()}")

combined.write(OUT)
print(f"\nSaved: {OUT}")

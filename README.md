# Spatial Transcriptomics — Breast Cancer Tumour Microenvironment

End-to-end analysis of 10X Visium spatial transcriptomics data from human breast cancer tissue. The pipeline combines standard bioinformatics (QC, clustering, deconvolution, spatially variable genes, neighbourhood enrichment) with a graph convolutional network that learns tissue domain labels directly from the spatial gene expression graph.

## Methods overview

```
10X Visium FFPE data
        │
        ▼
01  QC & preprocessing     — filtering, normalisation, HVG selection, PCA
        │
        ▼
02  Clustering             — UMAP, Leiden clustering, spatial overlay
        │
        ▼
03  Cell type deconvolution — Cell2location (Bayesian NMF, scRNA-seq reference)
        │
        ▼
04  Spatially variable genes — Moran's I autocorrelation
        │
        ▼
05  Neighbourhood enrichment — cell type co-localisation (z-score heatmap)
        │
        ▼
06  GNN domain prediction   — GCN on spatial neighbour graph (PyTorch Geometric)
```

## Key technologies

| Area | Tools |
|---|---|
| Spatial analysis | [Squidpy](https://squidpy.readthedocs.io), [Scanpy](https://scanpy.readthedocs.io) |
| Cell type deconvolution | [Cell2location](https://cell2location.readthedocs.io) |
| Graph neural network | [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) (2-layer GCN) |
| Data format | [AnnData](https://anndata.readthedocs.io) / HDF5 |
| Environment | [uv](https://docs.astral.sh/uv/) |

## Dataset

| Data | Source |
|---|---|
| Spatial (Visium FFPE) | [10X Genomics Human Breast Cancer](https://www.10xgenomics.com/datasets) — SpaceRanger output into `data/visium/` |
| scRNA-seq reference | Wu et al. 2021 breast cancer atlas ([Zenodo 4739739](https://zenodo.org/records/4739739)) — built into `data/scrna_reference.h5ad` via `scripts/build_scrna_reference.py` |

## Pipeline scripts

| Script | Input | Output |
|---|---|---|
| `src/01_qc_preprocessing.py` | `data/visium/` | `results/01_qc_preprocessed.h5ad` |
| `src/02_clustering.py` | `01_qc_preprocessed.h5ad` | `results/02_clustered.h5ad` |
| `src/03_deconvolution.py` | `02_clustered.h5ad` + `scrna_reference.h5ad` | `results/03_deconvolved.h5ad` |
| `src/04_spatially_variable_genes.py` | `03_deconvolved.h5ad` | `results/04_svg_results.csv` |
| `src/05_neighbourhood_enrichment.py` | `03_deconvolved.h5ad` | `results/05_nhood_enrichment.h5ad` |
| `src/06_gnn_domain_prediction.py` | `05_nhood_enrichment.h5ad` | `results/06_gnn_metrics.txt` |

Scripts chain via `.h5ad` files in `results/`. Run them in order.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync
source .venv/bin/activate
```

### Download data

```bash
# 10X Visium FFPE breast cancer (SpaceRanger output)
mkdir -p data/visium
wget -P data/visium \
  https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
wget -O - \
  https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz \
  | tar -xz -C data/visium

# scRNA-seq reference (Wu et al. 2021)
mkdir -p data/scrna_raw
wget -P data/scrna_raw \
  "https://zenodo.org/records/4739739/files/filtered_count_matrices.tar.gz?download=1" \
  "https://zenodo.org/records/4739739/files/metadata.tar.gz?download=1"
tar -xzf data/scrna_raw/filtered_count_matrices.tar.gz -C data/scrna_raw
tar -xzf data/scrna_raw/metadata.tar.gz -C data/scrna_raw
python scripts/build_scrna_reference.py
```

## Run

```bash
python src/01_qc_preprocessing.py
python src/02_clustering.py
python src/03_deconvolution.py   # CPU-only: ~1–2 h
python src/04_spatially_variable_genes.py
python src/05_neighbourhood_enrichment.py
python src/06_gnn_domain_prediction.py
```

## GNN architecture

The graph convolutional network in `src/06_gnn_domain_prediction.py` treats each Visium spot as a node (features = PCA embedding of gene expression) and spatial adjacency as edges. A two-layer GCN with dropout predicts tissue domain labels, evaluated against cell-type assignments from Cell2location.

```
Input: X_pca (n_spots × 50)  +  spatial adjacency edges
  → GCNConv(50 → 64) + ReLU + Dropout(0.3)
  → GCNConv(64 → 64) + ReLU
  → Linear(64 → n_classes)
  → CrossEntropyLoss
```

## References

- Ståhl et al. 2016 *Science* — spatial transcriptomics
- Kleshchevnikov et al. 2022 *Nature Biotechnology* — Cell2location
- Palla et al. 2022 *Nature Methods* — Squidpy
- Wu et al. 2021 *Nature Genetics* — breast cancer scRNA-seq atlas
- Kipf & Welling 2017 *ICLR* — graph convolutional networks

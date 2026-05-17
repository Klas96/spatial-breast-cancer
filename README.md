# Spatial Transcriptomics — Breast Cancer Tumour Microenvironment

End-to-end analysis of 10X Visium spatial transcriptomics data from human breast cancer tissue, combining standard bioinformatics methods with a graph neural network for spatial domain prediction.

## Dataset

- **Spatial:** [10X Genomics Human Breast Cancer (FFPE)](https://www.10xgenomics.com/datasets) — download SpaceRanger output into `data/visium/`
- **scRNA-seq reference:** Wu et al. 2021 breast cancer atlas ([GEO: GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078)) — preprocess and save as `data/scrna_reference.h5ad`

## Analysis pipeline

| Script | What it does |
|---|---|
| `01_qc_preprocessing.py` | QC filtering, normalisation, HVG selection, PCA |
| `02_clustering.py` | UMAP, Leiden clustering, spatial overlay on H&E |
| `03_deconvolution.py` | Cell type deconvolution with Cell2location |
| `04_spatially_variable_genes.py` | Spatially variable genes via Moran's I |
| `05_neighbourhood_enrichment.py` | Cell type co-localisation analysis |
| `06_gnn_domain_prediction.py` | GCN trained on spatial gene expression graph to predict tissue domains |

Scripts chain via `.h5ad` files in `results/`. Run them in order.

## Setup

```bash
conda env create -f environment.yml
conda activate spatial-bc
```

## Run

```bash
python src/01_qc_preprocessing.py
python src/02_clustering.py
python src/03_deconvolution.py   # slow — CPU-only ~1–2h
python src/04_spatially_variable_genes.py
python src/05_neighbourhood_enrichment.py
python src/06_gnn_domain_prediction.py
```

## Key references

- Ståhl et al. 2016 Science — spatial transcriptomics
- Kleshchevnikov et al. 2022 Nature Biotechnology — Cell2location
- Palla et al. 2022 Nature Methods — Squidpy
- Wu et al. 2021 Nature Genetics — breast cancer scRNA-seq atlas

#!/usr/bin/env bash
# Run the full spatial transcriptomics pipeline on any Visium dataset.
#
# Usage:
#   ./scripts/run_pipeline.sh [VISIUM_DIR] [RESULTS_DIR]
#
# Defaults:
#   VISIUM_DIR  = data/visium
#   RESULTS_DIR = results
#
# Example (second dataset):
#   ./scripts/run_pipeline.sh data/visium2 results2

set -euo pipefail

VISIUM_DIR="${1:-data/visium}"
RESULTS_DIR="${2:-results}"

export VISIUM_DIR RESULTS_DIR

PYTHON=".venv/bin/python"

echo "======================================"
echo "Spatial breast cancer pipeline"
echo "  VISIUM_DIR  : $VISIUM_DIR"
echo "  RESULTS_DIR : $RESULTS_DIR"
echo "======================================"

mkdir -p "$RESULTS_DIR"

PYTHONPATH=src $PYTHON src/01_qc_preprocessing.py
PYTHONPATH=src $PYTHON src/02_clustering.py
PYTHONPATH=src $PYTHON src/03_deconvolution.py
PYTHONPATH=src $PYTHON src/04_spatially_variable_genes.py
PYTHONPATH=src $PYTHON src/05_neighbourhood_enrichment.py
PYTHONPATH=src $PYTHON src/06_gnn_domain_prediction.py

echo ""
echo "Pipeline complete. Results in: $RESULTS_DIR"

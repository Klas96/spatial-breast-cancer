"""
Central config: reads dataset paths from env vars so the same scripts
can be run on multiple datasets without code changes.

Usage:
    VISIUM_DIR=data/visium2 RESULTS_DIR=results2 python src/01_qc_preprocessing.py
"""

import os
from pathlib import Path

VISIUM_DIR = Path(os.environ.get("VISIUM_DIR", "data/visium"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
SCRNA_ATLAS = Path(os.environ.get("SCRNA_ATLAS", "data/wu2021_scrna_atlas.h5ad"))

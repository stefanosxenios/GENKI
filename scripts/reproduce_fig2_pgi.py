#!/usr/bin/env python3
"""
reproduce_fig2_pgi.py
=====================
Reproduces Figure 2 from Xenios et al. (2025):
  (a) KPI distribution — ORACLE vs GENKI
  (b) Threshold bar plot (20th, 10th, 1st percentile, best model)
  (c) Predicted-vs-observed flux scatter for the PGI perturbation

Usage
-----
    python scripts/reproduce_fig2_pgi.py --km-file outputs/ecoli_pgi_km.csv

The --km-file is the Km parameter CSV exported from Section 1 of
notebooks/01_ecoli_cvae_training.ipynb.

Outputs land in results/fig2/images/.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow imports from repo root (preprocessing/, skimpy_tools/, src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paper-specific constants (Section 3.1.1, Table 2) ─────────────────────────
ENZYME          = "PGI"
PERTURBATION    = 0.06          # proteomics-derived scaling factor (Table 2)
SCORING         = "zscore"      # zeta-score used throughout the paper
KPI_THRESHOLD   = "0.2,0.1,0.01"            # Fig 2b x-axis thresholds (Best Model added automatically)
HIST_XLIM       = (0.0, 20.0)   # Fig 2a x-axis range
OUTPUT_DIR      = "./results/fig2"

# Fixed model and data paths (relative to repo root)
KMODEL          = "./models/kmodel3_last.yml"
TMODEL          = "./models/tmodel_last.json"
SAMPLES         = "./data/tfa_samples/sample1.csv"
ORACLE_KM_FILE  = "./data/data_for_training/ORACLE_Kms.csv"
FLUX_EXP_FILE   = "./data/FCCs/fluxes_exp.csv"
ORACLE_ENZ_FILE = "./data/enzyme_perturbations/PGI.csv"
# ──────────────────────────────────────────────────────────────────────────────

import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Figure 2 — PGI downregulation.")
    p.add_argument(
        "--km-file", required=True,
        help="Km CSV exported from notebook Section 1 (ecoli_pgi_km.csv).",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help=(
            "Skip stability screening and ODE simulations; reuse the enzyme "
            "perturbation CSV from a previous run and regenerate figures only. "
            "The CSV must already exist at "
            f"{OUTPUT_DIR}/enzyme_perturbations/."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    argv = [
        "--km-file",          args.km_file,
        "--kmodel",           KMODEL,
        "--tmodel",           TMODEL,
        "--samples",          SAMPLES,
        "--enzymes",          ENZYME,
        "--changes",          str(PERTURBATION),
        "--scoring",          SCORING,
        "--oracle-km-file",   ORACLE_KM_FILE,
        "--flux-exp-file",    FLUX_EXP_FILE,
        "--oracle-enzyme-file", ORACLE_ENZ_FILE,
        "--cut-reaction",     ENZYME,
        "--thresholds",       KPI_THRESHOLD,
        "--hist-xmin",        str(HIST_XLIM[0]),
        "--hist-xmax",        str(HIST_XLIM[1]),
        "--output-dir",       OUTPUT_DIR,
    ]
    if args.plot_only:
        argv.append("--skip-perturbation")

    from src.ecoli_single_pipeline import main as _run
    _run(argv)


if __name__ == "__main__":
    main()

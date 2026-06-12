#!/usr/bin/env python3
"""
reproduce_fig7_8_generalization.py
====================================
Reproduces Figures 7 and 8 from Xenios et al. (2025):
  Fig 7 — KPI distributions for training conditions (PGI, TALA) and
           held-out conditions (PGL, RPI, PYK, G6PDH2r)
  Fig 8 — Predicted-vs-observed flux scatterplots for held-out perturbations

The cVAE was trained on PGI and TALA only; all other perturbations are
held-out (not seen during training). This script evaluates the generated
ensemble on all six conditions.

Usage
-----
    python scripts/reproduce_fig7_8_generalization.py \
        --km-file outputs/ecoli_generalization_km.csv

The --km-file is the Km CSV exported from Section 3 of
notebooks/01_ecoli_cvae_training.ipynb (generalization scenario).

Outputs land in results/fig7_8/images/.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paper-specific constants (Section 3.1.5, Table 4) ─────────────────────────
# All six perturbations evaluated — training and held-out (Table 4)
PERTURBATIONS = [
    "PGI:0.06",       # training condition
    "TALA:0.33",      # training condition
    "PGL:0.10",       # held-out
    "RPI:0.50",       # held-out
    "PYK:0.34",       # held-out
    "G6PDH2r:0.013",  # held-out (minimum detectable abundance, Section 2.2)
]

SCORING           = "zscore"
THRESHOLD_PCT     = 40.0
KPI_THRESHOLDS    = "0.2,0.1,0.01,0.00001"
HIST_XLIM         = (0.0, 20.0)
OUTPUT_DIR        = "./results/fig7_8"

# Fixed model and data paths
KMODEL            = "./models/kmodel3_last.yml"
TMODEL            = "./models/tmodel_last.json"
SAMPLES           = "./data/tfa_samples/sample1.csv"
ORACLE_KM_FILE    = "./data/data_for_training/ORACLE_Kms.csv"
FLUX_EXP_FILE     = "./data/FCCs/fluxes_exp.csv"
ORACLE_ENZYME_DIR = "./data/enzyme_perturbations"
# ──────────────────────────────────────────────────────────────────────────────

import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Figures 7, 8 — generalization to unseen perturbations."
    )
    p.add_argument(
        "--km-file", required=True,
        help="Km CSV exported from notebook Section 3 (ecoli_generalization_km.csv).",
    )
    p.add_argument(
        "--n-jobs", type=int, default=6,
        help="Parallel workers for perturbation simulations (one per mutant).",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help=(
            "Skip stability screening and ODE simulations; reuse the enzyme "
            "perturbation CSVs from a previous run and regenerate figures only. "
            f"CSVs must already exist under {OUTPUT_DIR}/enzyme_perturbations/."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    argv = [
        "--km-file",           args.km_file,
        "--kmodel",            KMODEL,
        "--tmodel",            TMODEL,
        "--samples",           SAMPLES,
        "--perturbations",     *PERTURBATIONS,
        "--scoring",           SCORING,
        "--threshold-pct",     str(THRESHOLD_PCT),
        "--thresholds",        KPI_THRESHOLDS,
        "--hist-xmin",         str(HIST_XLIM[0]),
        "--hist-xmax",         str(HIST_XLIM[1]),
        "--oracle-km-file",    ORACLE_KM_FILE,
        "--flux-exp-file",     FLUX_EXP_FILE,
        "--oracle-enzyme-dir", ORACLE_ENZYME_DIR,
        "--output-dir",        OUTPUT_DIR,
        "--n-jobs",            str(args.n_jobs),
    ]
    # Figures in the paper: histograms, Km comparison, G6PDH2r scatter, UpSet (p40).
    argv += [
        "--skip-threshold-plots",
        "--skip-per-model-r2",
        "--pred-vs-obs-reactions", "G6PDH2r,PGL,PGI,TALA",
    ]

    if args.plot_only:
        argv.append("--skip-perturbation")

    from src.ecoli_multi_pipeline import main as _run
    _run(argv)


if __name__ == "__main__":
    main()

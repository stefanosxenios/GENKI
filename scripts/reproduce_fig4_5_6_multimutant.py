#!/usr/bin/env python3
"""
reproduce_fig4_5_6_multimutant.py
===================================
Reproduces Figures 4, 5, and 6 from Xenios et al. (2025):
  Fig 4 — Top shifted Km parameters (GENKI vs ORACLE, joint robustness)
  Fig 5 — KPI distributions across all five perturbations
  Fig 6 — UpSet analysis: multi-perturbation constraint satisfaction

Usage
-----
    python scripts/reproduce_fig4_5_6_multimutant.py \
        --km-file outputs/ecoli_multimutant_km.csv

The --km-file is the Km CSV exported from Section 2 of
notebooks/01_ecoli_cvae_training.ipynb (5-mutant joint robustness).

Outputs land in results/fig4_5_6/images/.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paper-specific constants (Section 3.1.4) ──────────────────────────────────
# Proteomics-derived enzyme scaling factors (Methods, Section 2.2)
PERTURBATIONS = [
    "PGI:0.06",
    "PGL:0.10",
    "RPI:0.50",
    "TALA:0.33",
    "PYK:0.34",
]

SCORING           = "zscore"
THRESHOLD_PCT     = 40.0        # per-mutant oracle percentile for UpSet (Section 3.1.4)
KPI_THRESHOLDS    = "0.2,0.1,0.01,0.00001"
HIST_XLIM         = (0.0, 20.0)
OUTPUT_DIR        = "./results/fig4_5_6"

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
        description="Reproduce Figures 4, 5, 6 — 5-mutant joint robustness."
    )
    p.add_argument(
        "--km-file", required=True,
        help="Km CSV exported from notebook Section 2 (ecoli_multimutant_km.csv).",
    )
    p.add_argument(
        "--n-jobs", type=int, default=5,
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
    # Only produce the figures in the paper: histograms, Km comparison, UpSet.
    argv += ["--skip-threshold-plots", "--skip-pred-vs-obs", "--skip-per-model-r2"]

    if args.plot_only:
        argv.append("--skip-perturbation")

    from src.ecoli_multi_pipeline import main as _run
    _run(argv)


if __name__ == "__main__":
    main()

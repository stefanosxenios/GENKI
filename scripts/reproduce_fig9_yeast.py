#!/usr/bin/env python3
"""
reproduce_fig9_yeast.py
========================
Reproduces Figure 9 from Xenios et al. (2025):
  (a) Percentage of GENKI models below ORACLE KPI percentile thresholds
      for the nanoaerobic → microaerobic transition
  (b) Top 15 Km parameter distribution shifts (GENKI vs ORACLE)
  (c) Global KPI distributions for ORACLE and GENKI ensembles

The S. cerevisiae model was constrained under nanoaerobic steady-state
conditions. The microaerobic transition is simulated by applying
proteomics-derived fold-changes to 8 reactions simultaneously (Methods,
Section 2.2).

Usage
-----
    python scripts/reproduce_fig9_yeast.py --km-file outputs/yeast_cvae_km.csv

The --km-file is the Km CSV exported from
notebooks/02_yeast_cvae_training.ipynb.

Outputs land in results/fig9/images/.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paper-specific constants (Section 2.2, Section 3.2) ───────────────────────
# Proteomics-derived fold-changes for nanoaerobic→microaerobic transition
# Gene-level changes mapped to reactions; only reactions with fold-change
# < 0.85 or > 1.20 were selected (Methods, Section 2.2)
ENZYMES = "ALCD2x,ICL,ACS,ACSm,CSm,TKT1,TKT2,O2t"
CHANGES = (
    "1.5474868277824088,"   # ALCD2x
    "1.3791628669281644,"   # ICL
    "1.482889052206379,"    # ACS
    "1.482889052206379,"    # ACSm
    "0.7910305944103543,"   # CSm
    "0.832499825755204,"    # TKT1
    "0.832499825755204,"    # TKT2
    "1.365"                 # O2t (includes change in oxygen availability)
)

SCORING          = "zscore"
CONDITION        = "microaerobic"          # column in flux_exp_file
THRESHOLDS       = "0.10,0.05,0.01"        # Fig 9a x-axis thresholds
IMPORTANT_RXNS   = "GLCt1,LMPD_s_0450_c_1_256,ETOHt,CO2t"
OUTPUT_DIR       = "./results/fig9"

# Fixed model and data paths
KMODEL           = "./models/nanoaerobic1_kinetic.yml"
TMODEL           = "./models/nanoaerobic1_fdp1_test.json"
SAMPLES          = "./data/tfa_samples/curated_fdp1_nano.csv"
CONSERVATION     = "./data/tfa_samples/mini_redYeast8_26Oct2020_151326_cons_relations_annotated.csv"
ORACLE_KM_FILE   = "./data/ORACLE_sample_9.csv"
ORACLE_ENZ_FILE  = "./data/enzyme_perturbations_yeast/oracle_single_tfa_sample.csv"
FLUX_EXP_FILE    = "./data/enzyme_perturbations_yeast/fluxomics_for_fdps.csv"
KM_MAPPING       = "./data/FCCs/km_mapping_yeast.csv"
# ──────────────────────────────────────────────────────────────────────────────

import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 9 — yeast nanoaerobic → microaerobic transition."
    )
    p.add_argument(
        "--km-file", required=True,
        help="Km CSV exported from notebooks/02_yeast_cvae_training.ipynb.",
    )
    p.add_argument(
        "--n-jobs-stability", type=int, default=12,
        help="Parallel workers for the stability stage.",
    )
    p.add_argument(
        "--n-jobs-simulation", type=int, default=4,
        help="Parallel workers for the ODE simulation stage.",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help=(
            "Skip stability screening and ODE simulations; reuse the enzyme "
            "perturbation CSV from a previous run and regenerate figures only. "
            f"The CSV must already exist under {OUTPUT_DIR}/enzyme_perturbations/."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    argv = [
        "--km-file",            args.km_file,
        "--kmodel",             KMODEL,
        "--tmodel",             TMODEL,
        "--samples",            SAMPLES,
        "--conservation-file",  CONSERVATION,
        "--enzymes",            ENZYMES,
        "--changes",            CHANGES,
        "--scoring",            SCORING,
        "--condition",          CONDITION,
        "--thresholds",         THRESHOLDS,
        "--important-rxns",     IMPORTANT_RXNS,
        "--oracle-km-file",     ORACLE_KM_FILE,
        "--oracle-enzyme-file", ORACLE_ENZ_FILE,
        "--flux-exp-file",      FLUX_EXP_FILE,
        "--km-mapping",         KM_MAPPING,
        "--output-dir",         OUTPUT_DIR,
        "--n-jobs-stability",   str(args.n_jobs_stability),
        "--n-jobs-simulation",  str(args.n_jobs_simulation),
        "--plot-label",         "GENKI",
    ]
    if args.plot_only:
        argv.append("--skip-perturbation")

    from src.yeast_pipeline import main as _run
    _run(argv)


if __name__ == "__main__":
    main()

# GENKI — Generative ENsemble KPI-Informed kinetic modeling

Code and data to reproduce the figures in:

> **Xenios et al. (2025).** *GENKI: A generative framework for scalable and robust metabolic kinetic modeling.*
> DOI: `TODO — fill in once published`

[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

---

## Data

Large data files are hosted on Zenodo and must be downloaded before running
the notebooks or figure scripts:

**Zenodo record:** [https://doi.org/TODO](https://doi.org/TODO) *(update once uploaded)*

After downloading, place the files so the repository looks like this:

```
data/
├── data_for_training/
│   ├── ORACLE_Kms.csv                  # ORACLE E. coli Km ensemble
│   ├── zeta_PGI_training_pert.csv      # cVAE training data — PGI
│   ├── zeta_PGL_training_pert.csv      # cVAE training data — PGL
│   ├── zeta_PYK_training_pert.csv      # cVAE training data — PYK
│   ├── zeta_RPI_training_pert.csv      # cVAE training data — RPI
│   └── zeta_TALA_training_pert.csv     # cVAE training data — TALA
├── data_for_training_yeast/
│   └── yeast_oracle_single_zeta.csv    # cVAE training data — yeast
├── ORACLE_sample_9.csv                 # ORACLE yeast Km ensemble
├── enzyme_perturbations/
│   ├── PGI.csv                         # ORACLE flux reference — PGI
│   ├── PGL.csv
│   ├── RPI.csv
│   ├── TALA.csv
│   ├── PYK.csv
│   └── G6PDH2r.csv
└── enzyme_perturbations_yeast/
    └── oracle_single_tfa_sample.csv    # ORACLE flux reference — yeast
```

A pre-generated set of Km parameter CSVs (outputs of the cVAE notebooks) is
also provided on Zenodo under `outputs/` so reviewers can run the figure
scripts directly without retraining:

```
outputs/
├── ecoli_pgi_km.csv
├── ecoli_multimutant_km.csv
├── ecoli_generalization_km.csv
└── yeast_cvae_km.csv
```

---

## Overview

GENKI generates ensembles of kinetically feasible metabolic models by
training a conditional Variational Autoencoder (cVAE) on ORACLE-sampled
parameter sets, using a zeta-score KPI as the conditioning signal.
The generated Km ensembles are then screened for dynamic stability and
evaluated against experimental flux data.

Two organisms are covered:

| Organism | Model | Km parameters | Figures |
|----------|-------|---------------|---------|
| *E. coli* core | `kmodel3_last.yml` | 259 | 2, 4–8 |
| *S. cerevisiae* (yeast) | `nanoaerobic1_kinetic.yml` | 1 055 | 9 |

---

## Repository structure

```
GENKI_repo/
├── notebooks/
│   ├── 01_ecoli_cvae_training.ipynb   # E. coli cVAE — 3 scenarios
│   └── 02_yeast_cvae_training.ipynb   # Yeast cVAE
├── scripts/
│   ├── reproduce_fig2_pgi.py           # Figure 2
│   ├── reproduce_fig4_5_6_multimutant.py  # Figures 4–6
│   ├── reproduce_fig7_8_generalization.py # Figures 7–8
│   └── reproduce_fig9_yeast.py         # Figure 9
├── src/                                # Pipeline internals (called by scripts)
├── ml4parameters/                      # cVAE architecture and utilities
├── preprocessing/                      # Flux/Km preprocessing helpers
├── skimpy_tools/                       # Stability checks and ODE utilities
├── functions/                          # KPI scoring functions
├── data/                               # ORACLE samples, training data, models
├── models/                             # Kinetic and thermodynamic model files
├── outputs/                            # Km CSVs exported by notebooks (gitignored)
├── results/                            # Publication figures (gitignored)
├── requirements_ml.txt                 # env_genki_ml dependencies
└── requirements_sim.txt                # env_genki_sim dependencies
```

---

## Setup

Two Python environments are required — one for training, one for simulation.

### env_genki_ml (notebooks)

No skimpy needed.

```bash
conda create -n env_genki_ml python=3.10
conda activate env_genki_ml
pip install -r requirements_ml.txt
```

### env_genki_sim (figure scripts)

Requires skimpy and pytfa. See the
[skimpy installation guide](https://github.com/EPFL-LCSB/skimpy) for
solver dependencies (CVXPY, CPLEX or Gurobi for pytfa).

```bash
conda create -n env_genki_sim python=3.10
conda activate env_genki_sim
pip install -r requirements_sim.txt
```

---

## Reproducing the figures

### Step 1 — Generate Km parameter sets (notebooks, env_genki_ml)

Run each notebook from the `notebooks/` directory:

```bash
conda activate env_genki_ml
cd notebooks/
jupyter notebook 01_ecoli_cvae_training.ipynb
jupyter notebook 02_yeast_cvae_training.ipynb
```

The notebooks have three clearly delimited sections. Run all cells in each
section; the final cell exports a CSV to `outputs/`:

| Notebook section | Output file |
|-----------------|-------------|
| `01` Section 1 — PGI single perturbation | `outputs/ecoli_pgi_km.csv` |
| `01` Section 2 — 5-mutant joint robustness | `outputs/ecoli_multimutant_km.csv` |
| `01` Section 3 — Generalization (PGI + TALA) | `outputs/ecoli_generalization_km.csv` |
| `02` Yeast nanoaerobic → microaerobic | `outputs/yeast_cvae_km.csv` |

### Step 2 — Reproduce figures (scripts, env_genki_sim)

Run each script from the **repository root**:

```bash
conda activate env_genki_sim

# Figure 2 — PGI single perturbation
python scripts/reproduce_fig2_pgi.py \
    --km-file outputs/ecoli_pgi_km.csv

# Figures 4, 5, 6 — 5-mutant joint robustness
python scripts/reproduce_fig4_5_6_multimutant.py \
    --km-file outputs/ecoli_multimutant_km.csv

# Figures 7, 8 — Generalization to unseen perturbations
python scripts/reproduce_fig7_8_generalization.py \
    --km-file outputs/ecoli_generalization_km.csv

# Figure 9 — Yeast nanoaerobic → microaerobic transition
python scripts/reproduce_fig9_yeast.py \
    --km-file outputs/yeast_cvae_km.csv
```

Publication-quality figures are saved under `results/<fig_name>/images/`.

#### Parallelism

The simulation scripts run perturbations in parallel using `joblib`.
The default number of workers is set to the number of perturbations (or
a sensible default for yeast). Adjust with `--n-jobs` (E. coli scripts)
or `--n-jobs-stability` / `--n-jobs-simulation` (yeast script):

```bash
python scripts/reproduce_fig9_yeast.py \
    --km-file outputs/yeast_cvae_km.csv \
    --n-jobs-stability 8 \
    --n-jobs-simulation 4
```

---

## Key paper parameters

All paper-specific constants (perturbation factors, KPI thresholds,
temperature values) are hardcoded at the top of each script/notebook
section and annotated with the corresponding paper section or table.
No arguments need to be changed to reproduce the publication figures.

### E. coli perturbation factors (Table 2)

| Reaction | Abundance factor |
|----------|-----------------|
| PGI | 0.06 |
| PGL | 0.10 |
| RPI | 0.50 |
| TALA | 0.33 |
| PYK | 0.34 |
| G6PDH2r | 0.013 |

### Yeast enzyme fold-changes (nanoaerobic → microaerobic, Section 2.2)

| Reaction | Fold-change |
|----------|------------|
| ALCD2x | 1.547 |
| ICL | 1.379 |
| ACS / ACSm | 1.483 |
| CSm | 0.791 |
| TKT1 / TKT2 | 0.832 |
| O2t | 1.365 |

---

## Runtime estimates

Runtimes depend on hardware. Approximate wall-clock times on a
16-core workstation:

| Step | Time |
|------|------|
| cVAE training (E. coli, 120 epochs) | 1–5 min |
| cVAE training (yeast, 180 epochs) | 1–5 min |
| Fig 2 (stability + 1 ODE perturbation) | 20–40 min |
| Fig 4–6 (5 perturbations, parallel) | 1–2 h  (depends on parallel processing)|
| Fig 7–8 (6 perturbations, parallel) | 1.5–3 h  depends on parallel processing)|
| Fig 9 (yeast, 8 perturbations) | 2–4 h depends on parallel processing)|

GPU acceleration is supported for the cVAE training steps (TensorFlow).

---

## Contact

Stefanos Xenios — NTUA

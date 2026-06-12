# Publication checklist — GENKI

Complete these steps in order before submitting / sharing with reviewers.

---

## PART 1 — Zenodo data deposit

### What goes on Zenodo

Upload these files exactly as they are (preserve directory structure by zipping
or using the Zenodo folder upload):

```
data/data_for_training/ORACLE_Kms.csv                  (38 MB)
data/data_for_training/zeta_PGI_training_pert.csv      (38 MB)
data/data_for_training/zeta_PGL_training_pert.csv      (38 MB)
data/data_for_training/zeta_PYK_training_pert.csv      (38 MB)
data/data_for_training/zeta_RPI_training_pert.csv      (39 MB)
data/data_for_training/zeta_TALA_training_pert.csv     (39 MB)
data/data_for_training_yeast/yeast_oracle_single_zeta.csv  (9.7 MB)
data/ORACLE_sample_9.csv                               (81 MB)
data/enzyme_perturbations/PGI.csv                      (11 MB)
data/enzyme_perturbations/PGL.csv                      (11 MB)
data/enzyme_perturbations/RPI.csv                      (11 MB)
data/enzyme_perturbations/TALA.csv                     (18 MB)
data/enzyme_perturbations/PYK.csv                      (19 MB)
data/enzyme_perturbations/G6PDH2r.csv                  (11 MB)
data/enzyme_perturbations_yeast/oracle_single_tfa_sample.csv  (3.4 MB)
outputs/ecoli_pgi_km.csv                               (~5 MB)
outputs/ecoli_multimutant_km.csv                       (~5 MB)
outputs/ecoli_generalization_km.csv                    (~5 MB)
outputs/yeast_cvae_km.csv                              (~57 MB)
```

Total: ~507 MB

### Steps

1. Go to https://zenodo.org and log in (create account if needed).
2. Click **New upload**.
3. Upload all files listed above. Keep the folder structure (zip each folder
   or use the drag-and-drop folder upload).
4. Fill in the metadata:
   - **Title:** Data for "GENKI: A generative framework for scalable and robust metabolic kinetic modeling"
   - **Authors:** Stefanos Xenios (+ co-authors)
   - **Description:** ORACLE Km ensembles, cVAE training data, ORACLE flux references, and pre-generated Km parameter sets for reproducing figures in Xenios et al. (2025).
   - **License:** Creative Commons Attribution 4.0 (CC BY 4.0)
   - **Related publication:** add the paper DOI once available
5. Click **Save** (do NOT publish yet — wait until the GitHub repo is ready).
6. Note the Zenodo **DOI** (shown as a badge on the draft record).
7. Update README.md and CITATION.cff in the GitHub repo with the DOI.
8. Publish the Zenodo record.

---

## PART 2 — GitHub repository

### One-time setup (do this once on your machine)

```bash
cd "/Users/stefanosxenios/projects/Generative AI for kinetics/GENKI_repo"

# Initialise git
git init
git branch -M main

# Stage everything that is not gitignored
git add .

# Verify what will be committed — check that no large files appear
git status

# Make the first commit
git commit -m "Initial release — GENKI paper code and figure scripts"
```

### Create the GitHub repo and push

1. Go to https://github.com/new
2. Name it **GENKI** (or **genki-kinetic-modeling** — your choice)
3. Set it to **Public**
4. Do NOT initialise with README (you already have one)
5. Click **Create repository**
6. Copy the remote URL shown on the next page, then run:

```bash
git remote add origin https://github.com/stefanosxenios/GENKI.git
git push -u origin main
```

### After pushing

- Go to the repo on GitHub → **Settings → About** → add a description and
  the paper DOI as the website link.
- Edit `CITATION.cff` in place to fill in your ORCID, the GitHub URL, and
  the paper DOI, then commit:
  ```bash
  git add CITATION.cff README.md
  git commit -m "Add DOIs and ORCID to citation metadata"
  git push
  ```
- Add a GitHub **release** (tag `v1.0.0`) once the paper is accepted —
  this links the exact code version to the paper.

---

## PART 3 — Running the paper results (reviewer instructions)

These instructions are also in HOW_TO_RUN.txt and README.md.

### 0. Clone the repo and download data

```bash
git clone https://github.com/stefanosxenios/GENKI.git
cd GENKI

# Download data from Zenodo and unpack into the repo root
# (Zenodo DOI: https://doi.org/TODO)
# Place files so that data/ and outputs/ match the structure in README.md
```

### 1. Create environments

```bash
# ML environment (cVAE training — no skimpy needed)
conda create -n env_genki_ml python=3.10
conda activate env_genki_ml
pip install -r requirements_ml.txt

# Simulation environment (figure scripts — requires skimpy)
conda create -n env_genki_sim python=3.10
conda activate env_genki_sim
conda install weilandtd::skimpy          # Option A (Linux/macOS)
pip install -r requirements_sim.txt
```

### 2. Train cVAE and generate Km parameters  (env_genki_ml, ~30–60 min)

Skip this step if you downloaded the pre-generated `outputs/` from Zenodo.

```bash
conda activate env_genki_ml
cd notebooks/
jupyter notebook 01_ecoli_cvae_training.ipynb   # run all cells — 3 sections
jupyter notebook 02_yeast_cvae_training.ipynb   # run all cells
```

Outputs:
- `outputs/ecoli_pgi_km.csv`
- `outputs/ecoli_multimutant_km.csv`
- `outputs/ecoli_generalization_km.csv`
- `outputs/yeast_cvae_km.csv`

### 3. Reproduce figures  (env_genki_sim, run from repo root)

```bash
conda activate env_genki_sim

# Figure 2 — PGI downregulation (KPI histogram, Km shifts, flux scatter)
python scripts/reproduce_fig2_pgi.py \
    --km-file outputs/ecoli_pgi_km.csv

# Figures 4, 5, 6 — 5-mutant joint robustness (histograms, Km shifts, UpSet)
python scripts/reproduce_fig4_5_6_multimutant.py \
    --km-file outputs/ecoli_multimutant_km.csv

# Figures 7, 8 — generalisation to unseen perturbations
python scripts/reproduce_fig7_8_generalization.py \
    --km-file outputs/ecoli_generalization_km.csv

# Figure 9 — yeast nanoaerobic → microaerobic transition
python scripts/reproduce_fig9_yeast.py \
    --km-file outputs/yeast_cvae_km.csv
```

Figures are saved to `results/<fig_name>/images/`.

### 3a. Plot-only mode (reuse existing simulations)

If ODE simulations have already run once, regenerate figures instantly:

```bash
python scripts/reproduce_fig2_pgi.py \
    --km-file outputs/ecoli_pgi_km.csv --plot-only

python scripts/reproduce_fig4_5_6_multimutant.py \
    --km-file outputs/ecoli_multimutant_km.csv --plot-only

python scripts/reproduce_fig7_8_generalization.py \
    --km-file outputs/ecoli_generalization_km.csv --plot-only

python scripts/reproduce_fig9_yeast.py \
    --km-file outputs/yeast_cvae_km.csv --plot-only
```

### Approximate runtimes (16-core workstation)

| Figure | Step | Time |
|--------|------|------|
| Fig 2  | Simulation | 20–40 min |
| Fig 4–6 | Simulation (5 workers) | 1–2 h |
| Fig 7–8 | Simulation (6 workers) | 1.5–3 h |
| Fig 9  | Simulation (yeast) | 2–4 h |
| Any    | Plot-only | < 1 min |

---

## PART 4 — Final checklist before going public

- [ ] Zenodo record published and DOI confirmed
- [ ] README.md updated with Zenodo DOI
- [ ] CITATION.cff updated with ORCID, GitHub URL, paper DOI
- [ ] `git status` shows no large files staged
- [ ] GitHub repo created and code pushed
- [ ] GitHub release `v1.0.0` tagged
- [ ] Paper supplementary material references the GitHub URL and Zenodo DOI

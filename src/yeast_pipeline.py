#!/usr/bin/env python3
"""
run_yeast_pipeline.py

Yeast end-to-end kinetic-model pipeline, mirroring run_pgi_pipeline.py /
run_multi_perturbation_pipeline.py for E. coli but adapted for the yeast
(nanoaerobic) model from GenAI_yeast_one_sample.ipynb.

Pipeline stages
---------------
1. Stability check — one fixed TFA sample (--sample-index, default 9);
   parallelised over parameter chunks (one loky worker per chunk).
2. Pruning by max-eigenvalue threshold.
3. Split pruned population into chunks, save each chunk to HDF5.
4. Parallel ODE simulation — one worker per chunk (simulate_batch from
   GenAI_yeast_one_sample.ipynb).
5. Merge perturbed fluxes and save the enzyme-perturbation CSV.
6. Scoring via build_dataset_yeast (MAPE) or build_dataset_yeast_zscore;
   z-score epsilon is fixed at 0.01 per project convention.
7. Figures
      a. Global KPI histogram   (candidate vs oracle)
      b. Per-metric histograms  (one per important reaction + OTHER)
      c. Oracle-threshold barplot
      d. Km-comparison boxplot  (compare_km_means)
      e. Predicted-vs-observed scatter + per-reaction score bar

Usage
-----
    python run_yeast_pipeline.py \\
        --km-file ./data/synthetic_yeast/my_params.csv

    # z-score scoring with LMPD signal amplification
    python run_yeast_pipeline.py \\
        --km-file ./data/synthetic_yeast/my_params.csv \\
        --scoring zscore --lmpd-scaling 1e6

    # Skip the heavy kinetic stage and go straight to scoring
    python run_yeast_pipeline.py \\
        --km-file ./data/synthetic_yeast/my_params.csv \\
        --skip-perturbation \\
        --enzyme-output ./data/enzyme_perturbations_yeast/existing.csv

All outputs land under ./results_yeast/<model-name>/.
"""

from __future__ import annotations

import argparse
import builtins
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Yeast-specific physical constants (GenAI_yeast.ipynb)
# ---------------------------------------------------------------------------
CONCENTRATION_SCALING = 1e6   # 1 mol → 1 mmol
DENSITY               = 1200  # g / L
GDW_GWW_RATIO         = 0.3   # 70 % water
TIME_SCALING          = 1
FLUX_SCALING          = 360_000

ZSCORE_EPS = 0.01             # preset epsilon for yeast z-score

DEFAULT_IMPORTANT_RXNS = ["GLCt1", "LMPD_s_0450_c_1_256", "ETOHt", "CO2t"]


# ---------------------------------------------------------------------------
# Generic helpers (shared with E. coli pipelines)
# ---------------------------------------------------------------------------

def slugify(value: str) -> str:
    value = Path(value).stem if value else "model"
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value).strip("_")
    return value or "model"


def save_table(obj, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path)
    elif isinstance(obj, pd.Series):
        obj.to_frame(name=name).to_csv(path)
    else:
        pd.DataFrame({name: np.asarray(obj)}).to_csv(path)
    print(f"[saved] {path}")


def save_figure(fig, path: Path, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


def close_all() -> None:
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# Stage 1 – Stability (parallel over parameter chunks, one fixed TFA sample)
# ---------------------------------------------------------------------------

def _build_and_prepare_kmodel(kmodel, conservation_file: str, ncpu: int = 2):
    """Load conservation relations and compile Jacobian — replicates
    build_and_prepare_kmodel from GenAI_yeast.ipynb."""
    from sympy import Matrix
    from scipy.sparse import csc_matrix as sparse_matrix
    from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
    from skimpy.utils.general import get_stoichiometry

    kmodel.prepare(mca=False)
    conservations = pd.read_csv(conservation_file)
    conservations.columns = [
        f"_{c}" if c[0].isdigit() else c for c in conservations.columns
    ]
    L0, _ = Matrix(conservations[kmodel.reactants].values).rref()
    kmodel.conservation_relation = sparse_matrix(L0, dtype=float)
    dep_ix, indep_ix = get_dep_indep_vars_from_basis(kmodel.conservation_relation)
    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix   = dep_ix
    kmodel.reduced_stoichiometry = get_stoichiometry(
        kmodel, kmodel.reactants
    )[indep_ix, :]
    kmodel.compile_jacobian(ncpu=ncpu)
    return kmodel


def _process_parameter_chunk(
    sample_row: pd.Series,
    df_params_chunk: pd.DataFrame,
    chunk_id: int,
    sample_idx: int,
    path_to_tmodel: str,
    path_to_kmodel: str,
    conservation_file: str,
) -> Tuple[dict, dict]:
    """Worker: check stability for one parameter chunk under a fixed TFA sample.

    Mirrors process_parameter_chunk from GenAI_yeast_one_sample.ipynb.

    Returns
    -------
    local_param_dict  : dict  key → {str(symbol): value}
    local_stable_info : dict  key → {"max_eigenvalue": float}
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    from sympy import Symbol
    from scipy.linalg import eigvals as eigenvalues

    from pytfa.io.json import load_json_model
    from skimpy.io.yaml import load_yaml_model
    from skimpy.analysis.oracle.load_pytfa_solution import (
        load_fluxes, load_concentrations, load_equilibrium_constants,
    )
    from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
    from skimpy.core.parameters import ParameterValues
    from skimpy_tools.check_stability import CheckStability

    tmodel = load_json_model(path_to_tmodel)
    kmodel = load_yaml_model(path_to_kmodel)
    kmodel = _build_and_prepare_kmodel(kmodel, conservation_file, ncpu=1)

    print(f"\n=== Processing chunk {chunk_id} (size={len(df_params_chunk)}) ===\n")

    fluxes_dict = load_fluxes(
        sample_row, tmodel, kmodel,
        density=DENSITY, ratio_gdw_gww=GDW_GWW_RATIO,
        concentration_scaling=CONCENTRATION_SCALING,
        time_scaling=TIME_SCALING,
    )
    concentrations_dict = load_concentrations(
        sample_row, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING,
    )
    k_eq = load_equilibrium_constants(
        sample_row, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING,
        in_place=True,
    )

    cs = CheckStability()
    cs.kmodel        = kmodel
    cs.flux_series   = fluxes_dict
    cs.conc_series   = concentrations_dict
    cs.k_eq          = k_eq
    cs.sym_conc_dict = {Symbol(k): v for k, v in concentrations_dict.items()}
    cs.CONCENTRATION_SCALING = CONCENTRATION_SCALING
    cs.TIME_SCALING   = TIME_SCALING
    cs.DENSITY        = DENSITY
    cs.GDW_GWW_RATIO  = GDW_GWW_RATIO

    sampling_params = SimpleParameterSampler.Parameters(n_samples=1)
    sampler = SimpleParameterSampler(sampling_params)
    sampler._compile_sampling_functions(cs.kmodel, cs.sym_conc_dict, [])

    cs.kmodel.parameters = cs.k_eq
    model_param = cs.kmodel.parameters

    local_param_dict  = {}
    local_stable_info = {}

    for ix2, row in df_params_chunk.iterrows():
        if ix2 % 100 == 0:
            print(f"[chunk {chunk_id}] processed: {ix2}")

        param_val = ParameterValues(row, cs.kmodel)
        cs.kmodel.parameters = cs.k_eq
        cs.kmodel.parameters = param_val
        cs.parameter_sample  = {
            v.symbol: v.value for _, v in cs.kmodel.parameters.items()
        }

        for rxn in cs.kmodel.reactions.values():
            cs.parameter_sample[rxn.parameters.vmax_forward.symbol] = 1

        cs.kmodel.flux_parameter_function(
            cs.kmodel, cs.parameter_sample, cs.sym_conc_dict, cs.flux_series,
        )
        for c in cs.conc_series.index:
            if c in model_param:
                cs.parameter_sample[cs.kmodel.parameters[c].symbol] = (
                    cs.conc_series[c]
                )

        jac = cs.kmodel.jacobian_fun(
            cs.flux_series[cs.kmodel.reactions],
            cs.conc_series[cs.kmodel.reactants],
            cs.parameter_sample,
        )
        eigs    = np.real(eigenvalues(jac.todense()))
        max_eig = float(np.max(eigs))

        if max_eig <= 0:
            key = f"{sample_idx},{ix2}"
            # Return raw dict with string keys to avoid cross-process Symbol issues
            local_param_dict[key]  = {
                str(sym): val for sym, val in cs.parameter_sample.items()
            }
            local_stable_info[key] = {"max_eigenvalue": max_eig}

    print(f"[chunk {chunk_id}] done — {len(local_param_dict)} stable models")
    return local_param_dict, local_stable_info


def run_stability_parallel(
    km_df: pd.DataFrame,
    samples: pd.DataFrame,
    args,
    output_dir: Path,
) -> Tuple[dict, dict]:
    """Parallel stability check — one fixed TFA sample, parallelised over
    parameter chunks (mirrors GenAI_yeast_one_sample.ipynb)."""
    from joblib import Parallel, delayed

    # Fixed TFA sample
    sample_row = samples.iloc[args.sample_index]
    sample_idx = samples.index[args.sample_index]   # original DataFrame index
    print(f"[stability] Using TFA sample at iloc={args.sample_index} "
          f"(index={sample_idx})")

    # Split parameter population into chunks
    chunks = np.array_split(km_df, args.n_chunks_stability)
    print(f"[stability] {len(km_df)} parameter rows → {len(chunks)} chunks, "
          f"{args.n_jobs_stability} parallel workers…")

    results = Parallel(n_jobs=args.n_jobs_stability, backend="loky", verbose=5)(
        delayed(_process_parameter_chunk)(
            sample_row, chunk, chunk_id, sample_idx,
            args.tmodel, args.kmodel, args.conservation_file,
        )
        for chunk_id, chunk in enumerate(chunks)
    )

    stable_params: dict = {}
    stable_info:   dict = {}
    for local_p, local_i in results:
        stable_params.update(local_p)
        stable_info.update(local_i)

    print(f"[stability] Total stable models: {len(stable_params)}")
    return stable_params, stable_info


# ---------------------------------------------------------------------------
# Stage 2 – Pruning
# ---------------------------------------------------------------------------

def run_pruning(
    stable_params: dict,
    stable_info: dict,
    kmodel,
    max_eigenvalue: float,
):
    """Filter stable models by max-eigenvalue threshold.

    Parameters in stable_params are raw {str_symbol: value} dicts;
    we reconstruct ParameterValues using kmodel from the main process.
    """
    from skimpy.core.parameters import ParameterValues, ParameterValuePopulation

    pruned_data  = []
    pruned_index = []
    for key, info in stable_info.items():
        if info["max_eigenvalue"] < max_eigenvalue:
            pruned_data.append(
                ParameterValues(stable_params[key], kmodel)
            )
            pruned_index.append(key)

    print(f"[prune] {len(pruned_index)} models kept (max_eig < {max_eigenvalue})")
    if not pruned_data:
        raise RuntimeError(
            "No models survived pruning. Try relaxing --max-eigenvalue."
        )
    return ParameterValuePopulation(pruned_data, kmodel=kmodel, index=pruned_index)


# ---------------------------------------------------------------------------
# Stage 3 – Split + save chunks
# ---------------------------------------------------------------------------

def split_and_save_chunks(
    pruned_pop,
    n_chunks: int,
    chunk_dir: Path,
) -> List[str]:
    """Split the pruned population into n_chunks and persist each to HDF5."""
    from skimpy.core.parameters import ParameterValuePopulation

    data       = pruned_pop._data
    index_keys = list(pruned_pop._index.keys())
    splits     = np.array_split(np.arange(len(data)), n_chunks)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        chunk = ParameterValuePopulation(
            [data[j] for j in idx],
            kmodel=pruned_pop.kmodel,
            index=[index_keys[j] for j in idx],
        )
        p = str(chunk_dir / f"chunk_{i:02d}.hdf5")
        chunk.save(p)
        print(f"[chunk] Saved chunk {i} ({len(idx)} models) → {p}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Stage 4 – Parallel ODE simulation
# ---------------------------------------------------------------------------

def _simulate_batch(
    sub_population_path: str,
    samples: pd.DataFrame,
    pert_params: dict,
    path_to_tmodel: str,
    path_to_kmodel: str,
) -> dict:
    """Worker: simulate every model in one population chunk.

    Replicates simulate_batch from GenAI_yeast.ipynb.
    Returns {model_ix: np.ndarray of fluxes / FLUX_SCALING}.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from pytfa.io.json import load_json_model
    from skimpy.io.yaml import load_yaml_model
    from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
    from skimpy.core.parameters import load_parameter_population
    from skimpy.analysis.ode.utils import make_flux_fun
    from skimpy.utils.namespace import QSSA
    from skimpy.utils.tabdict import TabDict

    TOUT = np.logspace(-8, 1, 1000)

    tmodel = load_json_model(path_to_tmodel)
    kmodel = load_yaml_model(path_to_kmodel)
    kmodel.prepare(mca=False)
    kmodel.compile_ode(sim_type=QSSA, ncpu=1)
    flux_fun = make_flux_fun(kmodel, sim_type=QSSA)

    sub_pop = load_parameter_population(sub_population_path)
    local_result = {}

    for ix in list(sub_pop._index.keys()):
        kmodel.parameters = sub_pop[ix]

        tfa_id = int(ix.split(",")[0])
        sample = samples.loc[tfa_id]
        concentrations = load_concentrations(
            sample, tmodel, kmodel,
            concentration_scaling=CONCENTRATION_SCALING,
        )

        for rid, fold in pert_params.items():
            param_name = str(
                kmodel.reactions[rid].parameters.vmax_forward.symbol
            )
            kmodel.parameters[param_name].value *= fold

        kmodel.initial_conditions = TabDict(
            [(k, v) for k, v in concentrations.items()]
        )

        try:
            sol = kmodel.solve_ode(
                TOUT,
                solver_type="cvode",
                bdf_stability_detection=False,
                rtol=1e-6,
                atol=1e-6,
                max_steps=int(1e9),
            )
        except Exception:
            continue

        if sol.time[-1] > 9:
            cons  = sol.concentrations.iloc[-1]
            flux  = flux_fun(cons, sub_pop[ix])
            for rid, fold in pert_params.items():
                flux[rid] *= fold
            local_result[ix] = np.array(list(flux.values())) / FLUX_SCALING
            print(f"[sim] {ix} stable")
        del sol

    return local_result


def run_simulation_parallel(
    pruned_pop,
    chunk_paths: List[str],
    samples: pd.DataFrame,
    args,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Parallel simulation over chunks; returns path to enzyme CSV."""
    from joblib import Parallel, delayed
    from skimpy.io.yaml import load_yaml_model

    pert_params = dict(zip(
        [e.strip() for e in args.enzymes.split(",")],
        [float(c.strip()) for c in args.changes.split(",")],
    ))
    print(f"[sim] Perturbations: {pert_params}")
    print(f"[sim] Launching {len(chunk_paths)} workers ({args.n_jobs_simulation} parallel)…")

    results = Parallel(
        n_jobs=args.n_jobs_simulation, backend="loky", verbose=10
    )(
        delayed(_simulate_batch)(
            path, samples, pert_params, args.tmodel, args.kmodel,
        )
        for path in chunk_paths
    )

    # Merge all per-chunk dicts
    merged = {}
    for r in results:
        merged.update(r)
    print(f"[sim] Total simulated models: {len(merged)}")

    if not merged:
        raise RuntimeError("No models produced stable ODE solutions.")

    # Get reaction names from a freshly loaded kmodel (no compilation needed)
    kmodel_tmp = load_yaml_model(args.kmodel)
    reaction_names = list(kmodel_tmp.reactions.keys())

    enzyme_df = pd.DataFrame(merged, index=reaction_names)
    enzyme_csv = output_dir / "enzyme_perturbations" / f"enzyme_{model_name}.csv"
    enzyme_csv.parent.mkdir(parents=True, exist_ok=True)
    enzyme_df.to_csv(enzyme_csv)
    print(f"[saved] {enzyme_csv}")
    return enzyme_csv


# ---------------------------------------------------------------------------
# Stage 5 – Load kinetic deps (lazy, keeps --help fast)
# ---------------------------------------------------------------------------

def load_kinetic_model_for_main(args):
    """Load kmodel in the main process (needed to create ParameterValuePopulation)."""
    from skimpy.io.yaml import load_yaml_model
    kmodel = load_yaml_model(args.kmodel)
    kmodel = _build_and_prepare_kmodel(kmodel, args.conservation_file, ncpu=args.ncpu_jacobian)
    return kmodel


# ---------------------------------------------------------------------------
# Stage 6 – Scoring
# ---------------------------------------------------------------------------

def run_scoring(
    args,
    km_df: pd.DataFrame,
    enzyme_csv: Path,
    output_dir: Path,
    model_name: str,
):
    """Build dataset + KPI for candidate and oracle; return payloads dict."""
    from preprocessing.flux_utlis import (
        build_dataset_yeast,
        build_dataset_yeast_zscore,
    )

    important_rxns = [r.strip() for r in args.important_rxns.split(",")]
    lmpd_scaling   = (
        {"LMPD_s_0450_c_1_256": args.lmpd_scaling}
        if args.lmpd_scaling and args.lmpd_scaling != 1.0
        else None
    )

    flux_exp_df      = pd.read_csv(args.flux_exp_file, index_col=0)
    enzyme_df        = pd.read_csv(enzyme_csv, index_col=0)
    oracle_enzyme_df = pd.read_csv(args.oracle_enzyme_file, index_col=0)

    # Enzyme CSVs produced by the pipeline have columns in "ix1,ix2" format
    # (e.g. "9,0", "9,42").  We only need ix2 to match against the Km index.
    def _strip_tfa_prefix(df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.astype(str).str.contains(",").any():
            df = df.copy()
            df.columns = [str(c).split(",")[-1] for c in df.columns]
        return df

    enzyme_df        = _strip_tfa_prefix(enzyme_df)
    oracle_enzyme_df = _strip_tfa_prefix(oracle_enzyme_df)

    # Convert columns to int; drop any that can't convert (e.g. phantom
    # "Unnamed: N" trailing columns written by pandas to_csv).
    def _numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = pd.to_numeric(df.columns, errors="coerce")
        valid = ~numeric_cols.isna()
        df = df.loc[:, valid].copy()
        df.columns = numeric_cols[valid].astype(int)
        return df

    enzyme_df        = _numeric_columns(enzyme_df)
    oracle_enzyme_df = _numeric_columns(oracle_enzyme_df)

    print(f"[score] enzyme_df: {enzyme_df.shape[1]} models after column clean")
    print(f"[score] oracle_enzyme_df: {oracle_enzyme_df.shape[1]} models after column clean")

    oracle_km_df = pd.read_csv(args.oracle_km_file, index_col=0)
    oracle_km_clean = oracle_km_df.loc[
        oracle_km_df.index.intersection(oracle_enzyme_df.columns)
    ]

    km_clean = km_df.loc[
        km_df.index.intersection(enzyme_df.columns)
    ]

    print(f"[score] km_clean: {len(km_clean)} models matched")
    print(f"[score] oracle_km_clean: {len(oracle_km_clean)} models matched")

    def _score(km, enzyme):
        if args.scoring == "mape":
            return build_dataset_yeast(
                Km_df=km,
                enzyme_df=enzyme,
                flux_exp_df=flux_exp_df,
                important_rxns=important_rxns,
                condition=args.condition,
                epsilon=args.mape_epsilon,
            )
        if args.scoring == "zscore":
            return build_dataset_yeast_zscore(
                Km_df=km,
                enzyme_df=enzyme,
                flux_exp_df=flux_exp_df,
                oracle_enzyme_df=oracle_enzyme_df,
                important_rxns=important_rxns,
                condition=args.condition,
                eps=ZSCORE_EPS,
                reaction_scaling=lmpd_scaling,
            )
        raise ValueError(f"Unknown --scoring: {args.scoring!r}")

    print(f"[score] Scoring candidate models with {args.scoring}…")
    ds_cand, score_cand, kpi_cand = _score(km_clean, enzyme_df)

    print(f"[score] Scoring oracle models with {args.scoring}…")
    ds_oracle, score_oracle, kpi_oracle = _score(oracle_km_clean, oracle_enzyme_df)

    tables_dir = output_dir / "tables"
    # Save full dataset (Km values + per-reaction scores + KPI)
    ds_cand.to_csv(tables_dir / f"dataset_{model_name}.csv")
    ds_oracle.to_csv(tables_dir / "dataset_oracle.csv")
    print(f"[saved] {tables_dir / f'dataset_{model_name}.csv'}")
    print(f"[saved] {tables_dir / 'dataset_oracle.csv'}")
    # Also save KPI series separately for convenience
    save_table(kpi_cand,   tables_dir / f"kpi_{model_name}.csv",  f"kpi_{model_name}")
    save_table(kpi_oracle, tables_dir / "kpi_oracle.csv",          "kpi_oracle")

    return {
        "ds_cand":    ds_cand,
        "kpi_cand":   kpi_cand,
        "ds_oracle":  ds_oracle,
        "kpi_oracle": kpi_oracle,
        "important_rxns": important_rxns,
    }


# ---------------------------------------------------------------------------
# Stage 7 – Figures
# ---------------------------------------------------------------------------

def _import_plot_deps():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    builtins.display = print  # for compare_km_means which calls display()

    from preprocessing.plots import (
        plot_error_histograms_vibrant,
        plot_thresholds_per_reaction,
        compare_km_means,
    )
    from preprocessing.predicted_vs_observed import plot_predicted_vs_observed

    return {
        "plot_error_histograms_vibrant": plot_error_histograms_vibrant,
        "plot_thresholds_per_reaction":  plot_thresholds_per_reaction,
        "compare_km_means":              compare_km_means,
        "plot_predicted_vs_observed":    plot_predicted_vs_observed,
    }


def _metric_prefix(scoring: str) -> str:
    return "MAPE" if scoring == "mape" else "ZSCORE"


def plot_kpi_histogram(
    scoring_payload: dict,
    model_name: str,
    images_dir: Path,
    args,
    plot_deps: dict,
):
    """Global KPI histogram — candidate vs oracle."""
    import matplotlib.pyplot as plt

    fn = plot_deps["plot_error_histograms_vibrant"]
    kpi_cand   = scoring_payload["kpi_cand"]
    kpi_oracle = scoring_payload["kpi_oracle"]

    # Outlier clipping: clip both distributions at the given percentile of the
    # combined data so extreme outliers don't compress the informative region.
    if args.kpi_clip_percentile < 100:
        combined = pd.concat([kpi_cand, kpi_oracle]).dropna()
        clip_val = float(np.percentile(combined, args.kpi_clip_percentile))
        kpi_cand_plot   = kpi_cand.clip(upper=clip_val)
        kpi_oracle_plot = kpi_oracle.clip(upper=clip_val)
        print(f"[plot] KPI clipped at {args.kpi_clip_percentile}th percentile "
              f"→ {clip_val:.4f}")
    else:
        kpi_cand_plot   = kpi_cand
        kpi_oracle_plot = kpi_oracle

    savepath = str(images_dir / f"histogram_kpi_{model_name}.png")
    fn(
        {model_name: kpi_cand_plot, "ORACLE": kpi_oracle_plot},
        title="Yeast – Global KPI",
        xlim=(args.hist_xmin, args.hist_xmax) if args.hist_xmin is not None else None,
        figsize=(args.hist_width, args.hist_height),
        savepath=savepath,
    )
    close_all()
    print(f"[saved] {savepath}")


def plot_per_metric_histograms(
    scoring_payload: dict,
    model_name: str,
    images_dir: Path,
    args,
    plot_deps: dict,
):
    """One histogram per important reaction + OTHER."""
    import matplotlib.pyplot as plt

    fn     = plot_deps["plot_error_histograms_vibrant"]
    ds_c   = scoring_payload["ds_cand"]
    ds_o   = scoring_payload["ds_oracle"]
    prefix = _metric_prefix(args.scoring)

    # Yeast-friendly short labels for file names
    label_map = {
        "LMPD_s_0450_c_1_256": "LMPD",
        "GLCt1":  "GLCt1",
        "ETOHt":  "ETOHt",
        "CO2t":   "CO2t",
    }
    metric_cols = [c for c in ds_c.columns if c.startswith(prefix)]

    for col in metric_cols:
        if col not in ds_o.columns:
            continue
        short = col.replace(f"{prefix}_", "")
        short = label_map.get(short, short)
        savepath = str(images_dir / f"histogram_{prefix.lower()}_{short}_{model_name}.png")

        cand_vals   = ds_c[col].dropna()
        oracle_vals = ds_o[col].dropna()

        # Clip at 99th percentile of combined distribution so outliers
        # (e.g. LMPD z-scores up to 1e8) don't compress the bulk of the plot.
        combined  = pd.concat([cand_vals, oracle_vals])
        clip_xmax = float(combined.quantile(0.99))

        fn(
            {model_name: cand_vals, "ORACLE": oracle_vals},
            title=f"Yeast – {short} ({prefix})",
            figsize=(6.81, 3.93),
            xlim=(0, clip_xmax),
            savepath=savepath,
        )
        close_all()
        print(f"[saved] {savepath}")


def plot_threshold_barplot(
    scoring_payload: dict,
    model_name: str,
    images_dir: Path,
    args,
    plot_deps: dict,
):
    """Oracle-threshold barplot."""
    fn = plot_deps["plot_thresholds_per_reaction"]

    threshold_values = [float(x.strip()) for x in args.thresholds.split(",")]
    threshold_data = {
        "km":  {model_name: scoring_payload["ds_cand"],
                "ORACLE":    scoring_payload["ds_oracle"]},
        "kpi": {"yeast": {model_name: scoring_payload["kpi_cand"],
                           "ORACLE":    scoring_payload["kpi_oracle"]}},
    }
    thresholds_config = {"plot": {"yeast": threshold_values}}
    fn(
        data=threshold_data,
        thresholds=thresholds_config,
        output_dir=str(images_dir),
    )
    close_all()


def plot_km_comparison(
    km_df: pd.DataFrame,
    model_name: str,
    images_dir: Path,
    args,
    plot_deps: dict,
):
    """Compare_km boxplot: candidate vs oracle Km distributions."""
    import matplotlib.pyplot as plt

    fn = plot_deps["compare_km_means"]
    oracle_km_df = pd.read_csv(args.oracle_km_file, index_col=0)

    # Drop error/metric columns if present
    error_cols = [c for c in km_df.columns
                  if c.startswith("MAPE") or c.startswith("ZSCORE") or c == "KPI"]
    km_clean = km_df.drop(columns=error_cols, errors="ignore")

    savepath = str(images_dir / f"compare_km_{model_name}.png")
    fn(
        km_clean, oracle_km_df,
        name1=model_name, name2="ORACLE",
        top=args.top_km,
        km_mapping_path=args.km_mapping,
        savepath=savepath,
        title=r"Top differing $\mathbf{log}(\mathbf{K_m})$" + f" — {model_name} vs ORACLE",
    )
    close_all()


def plot_scatter(
    enzyme_csv: Path,
    model_name: str,
    images_dir: Path,
    args,
    plot_deps: dict,
):
    """Predicted-vs-observed scatter + per-reaction score bar."""
    fn = plot_deps["plot_predicted_vs_observed"]

    save_path = str(images_dir / f"pred_vs_obs_{model_name}.png")
    important_rxns = [r.strip() for r in args.important_rxns.split(",")]
    yeast_compartment_map = {
        "GLCt1":               "Uptake",
        "LMPD_s_0450_c_1_256": "Growth",
        "ETOHt":               "Secretion",
        "CO2t":                "Secretion",
    }
    # Mark the perturbed reaction if it isn't already a KPI reaction
    if args.condition not in yeast_compartment_map:
        yeast_compartment_map[args.condition] = "Perturbed"

    fn(
        oracle_file=args.oracle_enzyme_file,
        genki_file=str(enzyme_csv),
        exp_file=args.flux_exp_file,
        perturbed_reaction=args.condition,
        perturbation_column=args.condition,
        scoring=args.scatter_scoring,
        exclude_from_scatter=tuple(
            r.strip() for r in args.exclude_from_scatter.split(",") if r.strip()
        ),
        compartment_of_map=yeast_compartment_map,
        label_top_n=args.scatter_label_top_n,
        bar_clip_percentile=args.scatter_bar_clip,
        save_path=save_path,
        dpi=args.dpi,
        show=False,
    )
    close_all()
    print(f"[saved] {save_path}")


def run_plots(
    scoring_payload: dict,
    km_df: pd.DataFrame,
    enzyme_csv: Path,
    model_name: str,
    output_dir: Path,
    args,
):
    """Orchestrate all figure generation."""
    plot_deps  = _import_plot_deps()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # a) Global KPI histogram
    plot_kpi_histogram(scoring_payload, model_name, images_dir, args, plot_deps)

    # b) Per-metric histograms
    plot_per_metric_histograms(scoring_payload, model_name, images_dir, args, plot_deps)

    # c) Oracle-threshold barplot
    plot_threshold_barplot(scoring_payload, model_name, images_dir, args, plot_deps)

    # d) Km comparison
    plot_km_comparison(km_df, model_name, images_dir, args, plot_deps)

    # e) Predicted-vs-observed scatter
    if not args.skip_scatter:
        plot_scatter(enzyme_csv, model_name, images_dir, args, plot_deps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(
        description="Yeast kinetic-model pipeline: stability → simulation → "
                    "scoring → figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- inputs ----
    p.add_argument("--km-file", required=True,
                   help="Candidate Km parameter CSV (rows = models, cols = Kms).")
    p.add_argument("--model-name", default=None,
                   help="Run label. Defaults to the --km-file stem.")
    p.add_argument("--plot-label", default="GENKI",
                   help="Display name used in plot titles and legends (default: GENKI).")
    p.add_argument("--results-root", default="./results_yeast",
                   help="Top-level output folder.")
    p.add_argument("--output-dir", default=None,
                   help="Exact output directory (overrides <results-root>/<model-name>).")

    # ---- model files ----
    p.add_argument("--kmodel",
                   default="./models/nanoaerobic1_kinetic.yml")
    p.add_argument("--tmodel",
                   default="./models/nanoaerobic1_fdp1_test.json")
    p.add_argument("--samples",
                   default="./data/tfa_samples/curated_fdp1_nano.csv")
    p.add_argument("--conservation-file",
                   default="./data/tfa_samples/"
                           "mini_redYeast8_26Oct2020_151326_cons_relations_annotated.csv")

    # ---- TFA sample ----
    p.add_argument("--sample-index", type=int, default=9,
                   help="iloc index of the TFA sample to use (default: 9, "
                        "matching GenAI_yeast_one_sample.ipynb).")

    # ---- stability / pruning ----
    p.add_argument("--ncpu-jacobian",  type=int,   default=2,
                   help="CPUs for kmodel.compile_jacobian inside the main process.")
    p.add_argument("--n-chunks-stability", type=int, default=12,
                   help="Number of parameter chunks for parallel stability check.")
    p.add_argument("--n-jobs-stability", type=int, default=12,
                   help="Parallel workers for the stability stage (one per chunk).")
    p.add_argument("--max-eigenvalue",   type=float, default=-0.3,
                   help="Pruning threshold: keep models with max_eig < this value.")

    # ---- simulation ----
    p.add_argument("--enzymes", default="ALCD2x,ICL,ACS,ACSm,CSm,TKT1,TKT2,O2t",
                   help="Comma-separated reaction IDs to perturb.")
    p.add_argument("--changes",
                   default="1.5474868277824088,1.3791628669281644,"
                           "1.482889052206379,1.482889052206379,"
                           "0.7910305944103543,0.832499825755204,"
                           "0.832499825755204,1.365",
                   help="Comma-separated fold-change values (same order as --enzymes).")
    p.add_argument("--n-chunks",         type=int,   default=4,
                   help="Number of population chunks for parallel simulation.")
    p.add_argument("--n-jobs-simulation", type=int,  default=4,
                   help="Parallel workers for the simulation stage.")
    p.add_argument("--keep-chunks",       action="store_true",
                   help="Keep intermediate HDF5 chunk files after simulation.")

    # ---- skip flags ----
    p.add_argument("--skip-perturbation", action="store_true",
                   help="Skip stability + simulation; reuse --enzyme-output.")
    p.add_argument("--enzyme-output", default=None,
                   help="Existing enzyme CSV to use when --skip-perturbation is set.")
    p.add_argument("--skip-scoring",  action="store_true")
    p.add_argument("--skip-scatter",  action="store_true",
                   help="Skip the predicted-vs-observed scatter plot.")

    # ---- scoring ----
    p.add_argument("--scoring", choices=["mape", "zscore"], default="zscore",
                   help="Per-reaction error function.")
    p.add_argument("--condition", default="microaerobic",
                   help="Column in --flux-exp-file to use as the experimental reference.")
    p.add_argument("--important-rxns",
                   default="GLCt1,LMPD_s_0450_c_1_256,ETOHt,CO2t",
                   help="Comma-separated key reactions tracked individually in KPI.")
    p.add_argument("--mape-epsilon", type=float, default=0.1,
                   help="Epsilon for MAPE denominator clipping.")
    p.add_argument("--lmpd-scaling", type=float, default=1e6,
                   help="Pre-scale factor for LMPD_s_0450_c_1_256 in z-score computation "
                        "(amplifies its signal; set to 1.0 to disable).")

    # ---- oracle / experimental files ----
    p.add_argument("--oracle-km-file",
                   default="./data/synthetic_yeast/ORACLE_sample_9.csv")
    p.add_argument("--oracle-enzyme-file",
                   default="./data/enzyme_perturbations_yeast/oracle_single_tfa_sample.csv")
    p.add_argument("--flux-exp-file",
                   default="./data/enzyme_perturbations_yeast/fluxomics_for_fdps.csv")

    # ---- plot knobs ----
    p.add_argument("--thresholds", default="0.10,0.05,0.01",
                   help="Oracle-percentile thresholds for the barplot.")
    p.add_argument("--kpi-clip-percentile", type=float, default=99.0,
                   help="Clip KPI values at this percentile of the combined "
                        "candidate+oracle distribution before plotting the histogram. "
                        "Set to 100 to disable clipping.")
    p.add_argument("--hist-xmin",   type=float, default=None)
    p.add_argument("--hist-xmax",   type=float, default=None)
    p.add_argument("--hist-width",  type=float, default=10.0)
    p.add_argument("--hist-height", type=float, default=6.0)
    p.add_argument("--top-km",      type=int,   default=15,
                   help="Number of top-diverging Kms shown in compare_km plot.")
    p.add_argument("--km-mapping",
                   default="./data/FCCs/km_mapping_yeast.csv",
                   help="CSV mapping raw Km column names to metabolite/reaction "
                        "pairs for display labels in the compare_km plot.")
    p.add_argument("--scatter-scoring", default="zeta",
                   choices=["zeta", "smape", "mape"],
                   help="Error metric used in the predicted-vs-observed bar panel.")
    p.add_argument("--exclude-from-scatter", default="",
                   help="Comma-separated reactions to omit from the scatter panel "
                        "(kept in the bar panel).")
    p.add_argument("--scatter-label-top-n", type=int, default=5,
                   help="Number of top outlier reactions to label in scatter plot "
                        "(KPI reactions are always labelled).")
    p.add_argument("--scatter-bar-clip", type=float, default=95.0,
                   help="Percentile at which to clip the bar chart y-axis. "
                        "Bars above this are shown with a ▲ prefix. "
                        "Set to 100 to disable.")
    p.add_argument("--dpi", type=int, default=300)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    km_file = Path(args.km_file)
    if not km_file.exists():
        raise FileNotFoundError(f"--km-file not found: {km_file}")

    model_name = slugify(args.model_name or km_file.stem)
    # plot_label is used for all figure titles/legends; model_name is kept for file paths
    plot_label = args.plot_label or model_name
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.results_root) / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] model_name={model_name}  plot_label={plot_label}  output_dir={output_dir}")

    km_df = pd.read_csv(km_file, index_col=0)
    print(f"[load] Km file: {km_file}  ({len(km_df)} rows)")

    # ------------------------------------------------------------------
    # Perturbation stage
    # ------------------------------------------------------------------
    if args.skip_perturbation:
        if args.enzyme_output:
            enzyme_csv = Path(args.enzyme_output)
        else:
            enzyme_csv = output_dir / "enzyme_perturbations" / f"enzyme_{model_name}.csv"
            print(f"[stage] --enzyme-output not set, using default: {enzyme_csv}")
        if not enzyme_csv.exists():
            raise FileNotFoundError(f"Enzyme CSV not found: {enzyme_csv}")
        print(f"[stage] Reusing existing enzyme CSV: {enzyme_csv}")

    else:
        samples = pd.read_csv(args.samples, header=0, index_col=0)
        print(f"[load] Samples: {args.samples}  ({len(samples)} rows)")

        # Stage 1 – Stability (parallel over TFA samples)
        stable_params, stable_info = run_stability_parallel(
            km_df, samples, args, output_dir
        )
        if not stable_params:
            raise RuntimeError("No stable models found. Check your model and parameters.")

        # Save stability summary
        stable_summary = pd.DataFrame(
            {k: v for k, v in stable_info.items()}
        ).T
        save_table(
            stable_summary,
            output_dir / "tables" / "stability_summary.csv",
            "stability",
        )

        # Stage 2 – Pruning
        print("[stage] Loading kmodel for main process…")
        kmodel_main = load_kinetic_model_for_main(args)

        pruned_pop = run_pruning(
            stable_params, stable_info, kmodel_main, args.max_eigenvalue
        )

        # Stage 3 – Split + save chunks
        tmp_dir = tempfile.mkdtemp(prefix="yeast_pipeline_")
        chunk_dir = Path(tmp_dir) / "chunks"
        chunk_paths = split_and_save_chunks(pruned_pop, args.n_chunks, chunk_dir)

        # Stage 4 – Parallel ODE simulation
        try:
            enzyme_csv = run_simulation_parallel(
                pruned_pop, chunk_paths, samples, args, output_dir, model_name
            )
        finally:
            if not args.keep_chunks:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Scoring + plotting
    # ------------------------------------------------------------------
    if not args.skip_scoring:
        scoring_payload = run_scoring(args, km_df, enzyme_csv, output_dir, plot_label)
        run_plots(scoring_payload, km_df, enzyme_csv, plot_label, output_dir, args)

    print("[done] Yeast pipeline completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise

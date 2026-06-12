#!/usr/bin/env python3
"""
Run the PGI kinetic-model pipeline from the terminal.

This script combines the two notebooks:
1. GenAI_PGI copy.ipynb
   - loads a generated Km/parameter dataframe (`df1`)
   - prepares the kinetic model and Jacobian
   - keeps stable models
   - applies the pruning criterion
   - runs PGI enzyme perturbations
   - writes only the enzyme perturbation CSV from this stage

2. scoring_visualisation.ipynb
   - uses the generated enzyme perturbation CSV in build_dataset()
   - writes the generated-model MAPE and KPI outputs
   - creates the histogram, threshold barplot, and Km-comparison plot

Typical usage:
    python run_pgi_pipeline.py \
        --km-file ./data/synthetic_parameters/graph_vae_film_2k.csv

By default, outputs are created under:
    ./results/<km-file-stem>/

For example, passing graph_vae_film_2k.csv creates:
    ./results/graph_vae_film_2k/
"""

from __future__ import annotations
import argparse
import re
import sys
import builtins # Added to fix NameError
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------


def slugify(value: str) -> str:
    """Make a safe model name for filenames."""
    value = Path(value).stem if value else "model"
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value).strip("_")
    return value or "model"


def as_dataframe(obj, name: str) -> pd.DataFrame:
    """Convert Series/list/array/scalar-like objects to a dataframe for saving."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=name)
    arr = np.asarray(obj)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    return pd.DataFrame(arr)


def save_table(obj, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    as_dataframe(obj, name=name).to_csv(path)
    print(f"[saved] {path}")


def save_current_figure(path: Path, dpi: int = 300) -> None:
    """Save the current Matplotlib figure and close it to prevent display."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    if fig is None or not fig.axes:
        print(f"[warning] No active figure to save for {path.name}")
        return
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


# -----------------------------
# GenAI_PGI notebook stage
# -----------------------------


def import_kinetic_dependencies():
    """Import heavy project dependencies only when the kinetic stage is run."""
    try:
        from pytfa.io.json import load_json_model
        from skimpy.io.yaml import load_yaml_model
        from skimpy.analysis.oracle.load_pytfa_solution import (
            load_concentrations,
            load_equilibrium_constants,
            load_fluxes,
        )
        from skimpy.analysis.ode.utils import make_flux_fun
        from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
        from skimpy.core.parameters import ParameterValues, ParameterValuePopulation
        from skimpy.core.solution import ODESolutionPopulation
        from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
        from skimpy.utils.general import get_stoichiometry
        from skimpy.utils.namespace import QSSA
        from skimpy.utils.tabdict import TabDict
        from scipy.linalg import eigvals as eigenvalues
        from scipy.sparse import csc_matrix as sparse_matrix  # noqa: F401  # kept for parity with notebook
        from sympy import Symbol
        from skimpy_tools.check_stability import CheckStability
    except ImportError as exc:
        raise ImportError(
            "Missing one of the kinetic-model dependencies used by the notebooks. "
            "Run this script in the same environment where the notebooks run successfully. "
            f"Original import error: {exc}"
        ) from exc

    return {
        "load_json_model": load_json_model,
        "load_yaml_model": load_yaml_model,
        "load_fluxes": load_fluxes,
        "load_concentrations": load_concentrations,
        "load_equilibrium_constants": load_equilibrium_constants,
        "make_flux_fun": make_flux_fun,
        "get_dep_indep_vars_from_basis": get_dep_indep_vars_from_basis,
        "ParameterValues": ParameterValues,
        "ParameterValuePopulation": ParameterValuePopulation,
        "ODESolutionPopulation": ODESolutionPopulation,
        "SimpleParameterSampler": SimpleParameterSampler,
        "get_stoichiometry": get_stoichiometry,
        "QSSA": QSSA,
        "TabDict": TabDict,
        "eigenvalues": eigenvalues,
        "Symbol": Symbol,
        "CheckStability": CheckStability,
    }


def load_and_prepare_models(args, deps):
    print("[stage] Loading tmodel, kmodel, and TFA samples")
    tmodel = deps["load_json_model"](args.tmodel)
    kmodel = deps["load_yaml_model"](args.kmodel)
    samples = pd.read_csv(args.samples, header=0, index_col=0)

    if args.samples_head is not None:
        samples_for_stability = samples.iloc[: args.samples_head]
    else:
        samples_for_stability = samples

    print("[stage] Preparing kinetic model and compiling Jacobian")
    kmodel.prepare(mca=False)
    dep_ix, indep_ix = deps["get_dep_indep_vars_from_basis"](kmodel.conservation_relation)
    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix = dep_ix
    kmodel.reduced_stoichiometry = deps["get_stoichiometry"](kmodel, kmodel.reactants)[indep_ix, :]
    kmodel.compile_jacobian(ncpu=args.ncpu_jacobian)

    return tmodel, kmodel, samples, samples_for_stability


def calculate_stable_population(args, deps, tmodel, kmodel, samples_for_stability, parameter_set: pd.DataFrame):
    """Replicate the CheckStability/Jacobian loop from GenAI_PGI."""
    print("[stage] Calculating Jacobian eigenvalues and keeping stable models")

    sample = samples_for_stability.iloc[0]
    load_fluxes = deps["load_fluxes"]
    load_concentrations = deps["load_concentrations"]
    load_equilibrium_constants = deps["load_equilibrium_constants"]
    ParameterValues = deps["ParameterValues"]
    ParameterValuePopulation = deps["ParameterValuePopulation"]
    SimpleParameterSampler = deps["SimpleParameterSampler"]
    CheckStability = deps["CheckStability"]
    Symbol = deps["Symbol"]
    eigenvalues = deps["eigenvalues"]
    QSSA = deps["QSSA"]

    fluxes_dict = load_fluxes(
        sample,
        tmodel,
        kmodel,
        density=args.density,
        ratio_gdw_gww=args.gdw_gww_ratio,
        concentration_scaling=args.concentration_scaling,
        time_scaling=args.time_scaling,
    )
    concentrations_dict = load_concentrations(
        sample, tmodel, kmodel, concentration_scaling=args.concentration_scaling
    )
    k_eq = load_equilibrium_constants(
        sample,
        tmodel,
        kmodel,
        concentration_scaling=args.concentration_scaling,
        in_place=True,
    )

    cs = CheckStability()
    cs.kmodel = kmodel
    cs.flux_series = fluxes_dict
    cs.conc_series = concentrations_dict
    cs.k_eq = k_eq
    cs.sym_conc_dict = {Symbol(k): v for k, v in concentrations_dict.items()}
    cs.CONCENTRATION_SCALING = args.concentration_scaling
    cs.TIME_SCALING = args.time_scaling
    cs.DENSITY = args.density
    cs.GDW_GWW_RATIO = args.gdw_gww_ratio
    cs.parameter_set = parameter_set

    cs.kmodel.prepare()
    cs.kmodel.compile_jacobian(sim_type=QSSA)

    sampling_parameters = SimpleParameterSampler.Parameters(n_samples=1)
    sampler = SimpleParameterSampler(sampling_parameters)
    sampler._compile_sampling_functions(cs.kmodel, cs.sym_conc_dict, [])

    model_param = cs.kmodel.parameters
    store_eigen: List[np.ndarray] = []
    param_pop = []
    stable_original_indices: List = []   # original Km row index for each stable model, in append order
    stable_count = 0

    for pos, idx in enumerate(cs.parameter_set.index):
        if pos % args.progress_every == 0:
            print(f"  Models processed: {pos}/{len(cs.parameter_set.index)}")

        param_val = cs.parameter_set.loc[idx]
        param_val = ParameterValues(param_val, cs.kmodel)

        cs.kmodel.parameters = cs.k_eq
        cs.kmodel.parameters = param_val
        cs.parameter_sample = {v.symbol: v.value for _, v in cs.kmodel.parameters.items()}

        for this_reaction in cs.kmodel.reactions.values():
            vmax_param = this_reaction.parameters.vmax_forward
            cs.parameter_sample[vmax_param.symbol] = 1

        cs.kmodel.flux_parameter_function(
            cs.kmodel,
            cs.parameter_sample,
            cs.sym_conc_dict,
            cs.flux_series,
        )

        for c in cs.conc_series.index:
            if c in model_param:
                c_sym = cs.kmodel.parameters[c].symbol
                cs.parameter_sample[c_sym] = cs.conc_series[c]

        this_jacobian = cs.kmodel.jacobian_fun(
            cs.flux_series[cs.kmodel.reactions],
            cs.conc_series[cs.kmodel.reactants],
            cs.parameter_sample,
        )
        real_eigs = sorted(np.real(eigenvalues(this_jacobian.todense())))
        is_stable = real_eigs[-1] <= args.stability_eigen_cutoff

        if is_stable:
            stable_count += 1
            store_eigen.append(real_eigs)
            param_pop.append(ParameterValues(cs.parameter_sample, cs.kmodel))
            stable_original_indices.append(idx)

    print(f"[info] Stable models: {stable_count}/{len(cs.parameter_set.index)}")
    if not param_pop:
        raise RuntimeError("No stable models were found. Try relaxing --stability-eigen-cutoff or check df1 input.")

    param_pop = ParameterValuePopulation(param_pop, kmodel)
    maximal_eigen = pd.DataFrame(np.asarray(store_eigen)[:, -1])
    return param_pop, maximal_eigen, kmodel, stable_original_indices


def prune_parameters(params_population, lambda_max_all: pd.DataFrame, kmodel, deps,
                     max_eigenvalues: float, stable_original_indices: Optional[List] = None):
    """Replicate the pruning() function from the notebook."""
    print(f"[stage] Pruning models with lambda_max < {max_eigenvalues}")
    ParameterValuePopulation = deps["ParameterValuePopulation"]

    is_selected = lambda_max_all < max_eigenvalues
    is_selected.columns = range(len(lambda_max_all.T))

    fast_parameters = []
    fast_index = []
    fast_original_indices: List = []
    for i, row in is_selected.T.iterrows():
        if any(row):
            fast_models = np.where(np.array(row))[0]
            fast_parameters.extend([params_population._data[k] for k in fast_models])
            fast_index.extend([f"{i},{k}" for k in fast_models])
            if stable_original_indices is not None:
                fast_original_indices.extend([stable_original_indices[k] for k in fast_models])

    print(f"[info] Pruned models: {len(fast_index)}")
    if not fast_parameters:
        raise RuntimeError(
            "No models passed the pruning criterion. Try relaxing --pruning-max-eigenvalues."
        )

    return (
        ParameterValuePopulation(fast_parameters, kmodel=kmodel, index=fast_index),
        kmodel,
        fast_index,
        fast_original_indices,
    )


def run_enzyme_perturbations(args, deps, tmodel, kmodel, samples, pruned_parameters, enzyme_csv: Path):
    """Run the PGI enzyme perturbation simulation and write the perturbation CSV."""
    print("[stage] Running enzyme perturbations")

    QSSA = deps["QSSA"]
    TabDict = deps["TabDict"]
    ODESolutionPopulation = deps["ODESolutionPopulation"]
    make_flux_fun = deps["make_flux_fun"]
    load_fluxes = deps["load_fluxes"]
    load_concentrations = deps["load_concentrations"]

    enzymes_to_perturb = [x.strip() for x in args.enzymes.split(",") if x.strip()]
    changes = [float(x.strip()) for x in args.changes.split(",") if x.strip()]
    if len(enzymes_to_perturb) != len(changes):
        raise ValueError("--enzymes and --changes must contain the same number of comma-separated values.")

    enzyme_perturbations = [
        (str(kmodel.reactions[r_id].parameters.vmax_forward.symbol), change)
        for r_id, change in zip(enzymes_to_perturb, changes)
    ]

    parameter_population = pruned_parameters
    kmodel.prepare(mca=False)
    kmodel.compile_ode(sim_type=QSSA, ncpu=args.ncpu_ode)

    sample = samples.loc[args.steady_sample_index]
    fluxes_steady = load_fluxes(
        sample,
        tmodel,
        kmodel,
        density=args.density,
        ratio_gdw_gww=args.gdw_gww_ratio,
        concentration_scaling=args.concentration_scaling,
        time_scaling=args.time_scaling,
    )

    f = pd.DataFrame(data=fluxes_steady / args.flux_normalization, columns=["Steady"])
    flux_fun = make_flux_fun(kmodel, sim_type=QSSA)
    solutions = []

    samples_to_simulate = list(parameter_population._index.keys())
    tout = np.logspace(args.tout_log_min, args.tout_log_max, args.tout_points)

    for count, ix in enumerate(samples_to_simulate, start=1):
        kmodel.parameters = parameter_population[ix]
        sample = samples.loc[args.steady_sample_index]
        concentrations = load_concentrations(
            sample, tmodel, kmodel, concentration_scaling=args.concentration_scaling
        )

        for parameter, change in enzyme_perturbations:
            kmodel.initial_conditions = TabDict([(k, v) for k, v in concentrations.items()])
            kmodel.parameters[parameter].value = kmodel.parameters[parameter].value * change

            this_sol_qssa = kmodel.solve_ode(
                tout,
                solver_type="cvode",
                bdf_stability_detection=False,
                rtol=args.ode_rtol,
                atol=args.ode_atol,
                max_steps=args.ode_max_steps,
            )
            cons = this_sol_qssa.concentrations.iloc[-1]
            flux = flux_fun.__call__(cons, parameter_population[ix])

            # Preserve notebook behavior: scale the perturbed enzyme flux by the perturbation factor.
            flux[enzymes_to_perturb[0]] = flux[enzymes_to_perturb[0]] * changes[0]
            f[ix] = flux.values() 
            f[ix]=f[ix] / args.flux_normalization
            solutions.append(this_sol_qssa)

        if count % args.progress_every == 0 or count == len(samples_to_simulate):
            print(f"  Perturbations simulated: {count}/{len(samples_to_simulate)}")

    # Created for parity with the notebook, although only f is saved as requested.
    _ = ODESolutionPopulation(solutions, [str(x) for x in enzyme_perturbations])

    enzyme_csv.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(enzyme_csv)
    print(f"[saved] {enzyme_csv}")
    return enzyme_csv


# -----------------------------
# scoring_visualisation stage
# -----------------------------

def import_scoring_dependencies():
    try:
        import matplotlib
        matplotlib.use('Agg') # Force non-interactive backend to avoid terminal print
        import matplotlib.pyplot as plt
        
        # FIX: Define 'display' globally so imported libraries don't crash
        builtins.display = print 
        
        from preprocessing.flux_utlis import (
            build_dataset,
            build_dataset_smape,
            build_dataset_zscore,
        )
        from preprocessing.plots import (
            compare_km_means,
            plot_error_histograms_vibrant,
            plot_thresholds_per_reaction,
        )
        from preprocessing.predicted_vs_observed import plot_predicted_vs_observed
    except ImportError as exc:
        raise ImportError(
            f"Missing dependencies: {exc}. Ensure you are in the correct environment."
        ) from exc

    return {
        "build_dataset": build_dataset,
        "build_dataset_smape": build_dataset_smape,
        "build_dataset_zscore": build_dataset_zscore,
        "compare_km_means": compare_km_means,
        "plot_error_histograms_vibrant": plot_error_histograms_vibrant,
        "plot_thresholds_per_reaction": plot_thresholds_per_reaction,
        "plot_predicted_vs_observed": plot_predicted_vs_observed,
    }

def run_scoring_and_plots(args, scoring_deps, model_name: str, km_file: Path, enzyme_csv: Path, output_dir: Path):
    print(f"[stage] Generating datasets and plots (scoring={args.scoring}, Oracle baseline)...")
    import matplotlib.pyplot as plt

    build_dataset = scoring_deps["build_dataset"]
    build_dataset_smape = scoring_deps["build_dataset_smape"]
    build_dataset_zscore = scoring_deps["build_dataset_zscore"]
    plot_error_histograms_vibrant = scoring_deps["plot_error_histograms_vibrant"]
    plot_thresholds_per_reaction = scoring_deps["plot_thresholds_per_reaction"]
    compare_km_means = scoring_deps["compare_km_means"]
    plot_predicted_vs_observed = scoring_deps["plot_predicted_vs_observed"]

    tables_dir = output_dir / "tables"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build the (model, oracle-baseline) pair under the chosen scoring scheme.
    # For zscore, both datasets use the SAME oracle_enzyme_file as the source
    # of sigma_r, so the resulting KPI distributions are directly comparable.
    def _score(enzyme_file: str):
        if args.scoring == "mape":
            return build_dataset(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                cut_reaction=args.cut_reaction,
            )
        if args.scoring == "smape":
            return build_dataset_smape(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                cut_reaction=args.cut_reaction,
            )
        if args.scoring == "zscore":
            return build_dataset_zscore(
                km_file=args.oracle_km_file,
                flux_exp_file=args.flux_exp_file,
                enzyme_file=enzyme_file,
                oracle_enzyme_file=args.oracle_enzyme_file,
                cut_reaction=args.cut_reaction,
            )
        raise ValueError(f"Unknown --scoring value: {args.scoring}")

    # 1. Build dataset for the Generated Model
    df_model_pgi, _score_model_pgi, kpi_model_pgi = _score(str(enzyme_csv))
    save_table(kpi_model_pgi, tables_dir / f"kpi_{model_name}_pgi.csv", f"kpi_{model_name}_pgi")

    # 2. Build dataset for ORACLE Baseline (re-scored with the same scheme)
    df_oracle_pgi, _, kpi_oracle_pgi = _score(args.oracle_enzyme_file)
    save_table(kpi_oracle_pgi, tables_dir / "kpi_oracle_pgi.csv", "kpi_oracle_pgi")

    # --- SAVE HISTOGRAM ---
    plt.figure(figsize=(args.hist_width, args.hist_height))
    # Comparison includes model and Oracle[cite: 1, 2]
    histogram_data = {model_name: kpi_model_pgi, "ORACLE": kpi_oracle_pgi}
    plot_error_histograms_vibrant(
        histogram_data,
        title=args.cut_reaction,
        xlim=(args.hist_xmin, args.hist_xmax)
    )
    save_current_figure(images_dir / f"histogram_{model_name}_pgi.png", dpi=args.dpi)

    # --- SAVE THRESHOLD BARPLOT ---
    threshold_values = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    # Populate structure with model and Oracle baseline
    threshold_data = {
        "km": {model_name: df_model_pgi, "ORACLE": df_oracle_pgi},
        "kpi": {args.cut_reaction: {model_name: kpi_model_pgi, "ORACLE": kpi_oracle_pgi}},
    }
    thresholds_config = {"plot": {args.cut_reaction: threshold_values}}
    
    # Save threshold barplot directly under images/ (no extra "thresholds" subfolder)
    plot_thresholds_per_reaction(
        data=threshold_data,
        thresholds=thresholds_config,
        output_dir=str(images_dir),
        title=f"{args.cut_reaction} downregulation — performance thresholds",
    )

    # --- SAVE KM COMPARISON PLOT ---
    # Keep Oracle as the default baseline comparison
    compare_to_path = Path(args.compare_km_file) if args.compare_km_file else Path(args.oracle_km_file)
    graph_kms_df = pd.read_csv(km_file, index_col=0)
    comparison_kms_df = pd.read_csv(compare_to_path, index_col=0)

    plt.figure()
    compare_km_means(graph_kms_df, comparison_kms_df, name1=model_name, name2=args.compare_name)
    save_current_figure(images_dir / f"compare_km_{model_name}_pgi.png", dpi=args.dpi)

    # --- SAVE PREDICTED VS OBSERVED SCATTER ---
    scoring_label = "zeta" if args.scoring == "zscore" else args.scoring
    scatter_path = str(images_dir / f"predicted_vs_observed_{model_name}_pgi.png")
    plot_predicted_vs_observed(
        oracle_file=args.oracle_enzyme_file,
        genki_file=str(enzyme_csv),
        exp_file=args.flux_exp_file,
        perturbed_reaction=args.cut_reaction,
        scoring=scoring_label,
        save_path=scatter_path,
        show=False,
    )
    print(f"[saved] {scatter_path}")

    return {
        "df_model_pgi": df_model_pgi,
        "kpi_model_pgi": kpi_model_pgi,
    }


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run GenAI_PGI enzyme perturbation and scoring visualisation from terminal."
    )

    parser.add_argument("--km-file", required=True, help="Changing df1 CSV file from GenAI_PGI notebook.")
    parser.add_argument("--model-name", default=None, help="Name used in output files. Defaults to the passed --km-file stem.")
    parser.add_argument(
        "--results-root",
        default="./results",
        help="Top-level folder where each run folder is created. Defaults to ./results.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional exact output directory. If omitted, uses <results-root>/<km-file-stem>.",
    )
    parser.add_argument("--enzyme-output", default=None, help="Optional exact enzyme perturbation CSV output path.")

    parser.add_argument("--skip-perturbation", action="store_true", help="Skip GenAI stage and reuse --enzyme-output.")
    parser.add_argument("--skip-scoring", action="store_true", help="Only create the enzyme perturbation CSV.")

    # Model/data paths from the notebook.
    parser.add_argument("--kmodel", default="./models/kmodel3_last.yml")
    parser.add_argument("--tmodel", default="./models/tmodel_last.json")
    parser.add_argument("--samples", default="./data/tfa_samples/sample1.csv")
    parser.add_argument("--samples-head", type=int, default=100, help="Use first N TFA samples for stability stage, matching notebook default 100.")

    # Constants from the notebook.
    parser.add_argument("--concentration-scaling", type=float, default=1e6)
    parser.add_argument("--time-scaling", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1521.0)
    parser.add_argument("--gdw-gww-ratio", type=float, default=0.3)
    parser.add_argument("--flux-normalization", type=float, default=456300.0)

    # Stability/pruning.
    parser.add_argument("--ncpu-jacobian", type=int, default=4)
    parser.add_argument("--ncpu-ode", type=int, default=6)
    parser.add_argument("--stability-eigen-cutoff", type=float, default=0.0)
    parser.add_argument("--pruning-max-eigenvalues", type=float, default=-10.0)

    # Perturbation settings.
    parser.add_argument("--enzymes", default="PGI", help="Comma-separated enzymes to perturb.")
    parser.add_argument("--changes", default="0.06", help="Comma-separated perturbation factors.")
    parser.add_argument("--steady-sample-index", type=int, default=144)
    parser.add_argument("--tout-log-min", type=float, default=-8)
    parser.add_argument("--tout-log-max", type=float, default=2)
    parser.add_argument("--tout-points", type=int, default=1000)
    parser.add_argument("--ode-rtol", type=float, default=1e-6)
    parser.add_argument("--ode-atol", type=float, default=1e-6)
    parser.add_argument("--ode-max-steps", type=float, default=1e9)
    parser.add_argument("--progress-every", type=int, default=100)

    # Scoring notebook paths/settings.
    parser.add_argument(
        "--scoring",
        choices=["mape", "smape", "zscore"],
        default="mape",
        help="Per-reaction error function used to build the training/baseline datasets. "
             "'zscore' uses |sim - exp| / sigma_r where sigma_r is the std of the "
             "ORACLE ensemble's absolute residuals (taken from --oracle-enzyme-file).",
    )
    parser.add_argument("--cut-reaction", default="PGI")
    parser.add_argument("--oracle-km-file", default="./data/data_for_training/ORACLE_Kms.csv")
    parser.add_argument("--flux-exp-file", default="./data/FCCs/fluxes_exp.csv")
    parser.add_argument("--oracle-enzyme-file", default="./data/enzyme_perturbations/PGI.csv")
    parser.add_argument("--include-oracle-baseline", action="store_true", help="Also include ORACLE in histogram and save its KPI.")
    parser.add_argument("--genki-enzyme-file", default=None, help="Optional GENKI/CVAE enzyme file to include in histogram.")
    parser.add_argument("--compare-km-file", default=None, help="Km CSV to compare against. Defaults to --oracle-km-file.")
    parser.add_argument("--compare-name", default="ORACLE")
    parser.add_argument("--thresholds", default="0.2,0.1,0.01,0.00001")
    parser.add_argument("--hist-xmin", type=float, default=0.0)
    parser.add_argument("--hist-xmax", type=float, default=20.0)
    parser.add_argument("--hist-width", type=float, default=12.9)
    parser.add_argument("--hist-height", type=float, default=7.52)
    parser.add_argument("--dpi", type=int, default=300)

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    km_file = Path(args.km_file)
    if not km_file.exists():
        raise FileNotFoundError(f"--km-file not found: {km_file}")

    # The passed df1 dataframe file determines the default model/run name.
    # Example: --km-file ./foo/my_model.csv -> model_name = my_model
    model_name = slugify(args.model_name or km_file.stem)

    # By default, every run is placed under ./results/<model_name>/
    # You can still override the full output path with --output-dir.
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.enzyme_output:
        enzyme_csv = Path(args.enzyme_output)
    else:
        enzyme_csv = output_dir / "enzyme_perturbations" / f"enzyme_perturbation_{model_name}_pgi.csv"

    if args.skip_perturbation:
        if not enzyme_csv.exists():
            raise FileNotFoundError(
                "--skip-perturbation was set, but the requested --enzyme-output does not exist: "
                f"{enzyme_csv}"
            )
        print(f"[stage] Reusing existing enzyme perturbation file: {enzyme_csv}")
    else:
        deps = import_kinetic_dependencies()
        parameter_set = pd.read_csv(km_file, index_col=0)
        tmodel, kmodel, samples, samples_for_stability = load_and_prepare_models(args, deps)
        param_pop, maximal_eigen, kmodel, stable_original_indices = calculate_stable_population(
            args, deps, tmodel, kmodel, samples_for_stability, parameter_set
        )
        save_table(maximal_eigen, output_dir / "tables" / f"maximal_eigen_{model_name}_pgi.csv", "maximal_eigen")
        pruned_parameters, kmodel, fast_index, fast_original_indices = prune_parameters(
            param_pop, maximal_eigen, kmodel, deps, args.pruning_max_eigenvalues,
            stable_original_indices=stable_original_indices,
        )

        # Save the surviving Km rows (stability + pruning), labelled with the
        # same "i,k" index used as enzyme_csv columns, so the parameter set
        # joins 1:1 with the perturbation results.
        surviving_km = parameter_set.loc[fast_original_indices].copy()
        surviving_km.index = fast_index
        save_table(
            surviving_km,
            output_dir / "tables" / f"parameter_set_{model_name}_pgi.csv",
            f"parameter_set_{model_name}_pgi",
        )

        run_enzyme_perturbations(args, deps, tmodel, kmodel, samples, pruned_parameters, enzyme_csv)

    if not args.skip_scoring:
        scoring_deps = import_scoring_dependencies()
        run_scoring_and_plots(args, scoring_deps, model_name, km_file, enzyme_csv, output_dir)

    print("[done] PGI pipeline completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise

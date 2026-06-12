#!/usr/bin/env python
"""
GenAI all-mutants pipeline.

Replicates the workflow from `GenAI_all_mutants.ipynb` as a single Python
script. Given an initial parameter-set CSV (the "dataset"), the script:

  1. Loads the thermodynamic and kinetic models.
  2. Runs the stability check: for every parameter row it back-calculates
     Vmax's, builds the Jacobian, computes eigenvalues, and keeps only
     models whose largest real eigenvalue is <= 0.
  3. Prunes the stable models by a max-eigenvalue threshold (default -10).
  4. Runs every requested enzyme perturbation in PARALLEL using joblib +
     loky (one worker per perturbation scenario). Each worker re-loads the
     models in its own process, integrates the ODEs at the perturbed Vmax
     for every pruned model, and records the post-perturbation steady-
     state fluxes.
  5. Writes ONE CSV per perturbation containing the perturbed fluxes (rows
     = reactions, columns = pruned-model index). No other outputs.

Default paths/values match the notebook exactly so the script can be run
from the project root with just `python run_pipeline.py --params <csv>`.

Example
-------
    python run_pipeline.py \
        --params ./data/synthetic_parameters/synth_cvae_all_mutants.csv \
        --output-dir ./data/enzyme_perturbations \
        --n-jobs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (match the notebook)
# ---------------------------------------------------------------------------
CONCENTRATION_SCALING = 1e6      # 1 mol -> 1 mmol
TIME_SCALING = 1
DENSITY = 1521                   # g/L
GDW_GWW_RATIO = 0.3              # assumes ~70% water
TOUT = np.logspace(-8, 2, 1000)  # ODE output time grid


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_perturbation(pert_str: str):
    """Parse a single ENZYME:CHANGE entry."""
    if ":" not in pert_str:
        raise argparse.ArgumentTypeError(
            f"Invalid perturbation '{pert_str}'. Use ENZYME:CHANGE, e.g. PGI:0.06"
        )
    enzyme, change = pert_str.split(":", 1)
    return enzyme.strip(), float(change)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stability + pruning + parallel enzyme-perturbation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--params", required=True,
        help="Path to the initial parameter-set CSV.",
    )
    p.add_argument(
        "--output-dir", default="./data/enzyme_perturbations",
        help="Directory where one CSV per perturbation will be written.",
    )
    p.add_argument(
        "--tmodel", default="./models/tmodel_last.json",
        help="Path to the pyTFA thermodynamic model (.json).",
    )
    p.add_argument(
        "--kmodel", default="./models/kmodel3_last.yml",
        help="Path to the SKiMpy kinetic model (.yml).",
    )
    p.add_argument(
        "--samples", default="./data/tfa_samples/sample1.csv",
        help="Path to the TFA samples CSV.",
    )
    p.add_argument(
        "--params-index-col", type=int, default=0,
        help="index_col arg for the params CSV. Pass -1 to disable.",
    )
    p.add_argument(
        "--stability-sample-row", type=int, default=0,
        help="iloc row in the samples CSV used as reference for the "
             "stability check (matches notebook: 0).",
    )
    p.add_argument(
        "--perturbation-sample-row", type=int, default=144,
        help="loc row label in the samples CSV used as the reference steady "
             "state for the perturbations (matches notebook: 144).",
    )
    p.add_argument(
        "--max-eigenvalue", type=float, default=-10.0,
        help="Threshold for pruning: models are kept only when their maximal "
             "eigenvalue is strictly less than this value.",
    )
    p.add_argument(
        "--ncpu-compile", type=int, default=4,
        help="ncpu passed to kmodel.compile_jacobian during the stability "
             "step (does NOT affect the parallel perturbation step).",
    )
    p.add_argument(
        "--n-jobs", type=int, default=5,
        help="Number of parallel workers for the perturbation step. "
             "Typically equal to the number of perturbations.",
    )
    p.add_argument(
        "--flux-scaling", type=float, default=456300.0,
        help="Divisor applied to every output flux (matches notebook).",
    )
    p.add_argument(
        "--perturbations",
        nargs="+",
        type=_parse_perturbation,
        default=[
            ("PGL",  0.10),
            ("PGI",  0.06),
            ("RPI",  0.50),
            ("TALA", 0.33),
            ("PYK",  0.34),
        ],
        help="Perturbation scenarios as ENZYME:CHANGE entries "
             "(e.g. PGI:0.06 RPI:0.5). Defaults match the notebook.",
    )
    p.add_argument(
        "--output-prefix", default=None,
        help="Prefix for output CSV filenames. "
             "Each file is named '<prefix>_<ENZYME>.csv'. "
             "If omitted, defaults to the stem of --params "
             "(e.g. params=./foo/test.csv -> prefix='test').",
    )
    p.add_argument(
        "--keep-pruned-hdf5", action="store_true",
        help="Keep the intermediate pruned-parameter HDF5 in the output dir. "
             "By default it is written to a temp file and deleted.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Step 1+2: stability check + pruning  (sequential, single process)
# ---------------------------------------------------------------------------
def run_stability_and_prune(
    params_df: pd.DataFrame,
    kmodel,
    tmodel,
    samples: pd.DataFrame,
    stability_sample_row: int,
    max_eigenvalue: float,
    ncpu_compile: int,
):
    """Run the notebook's stability check and prune by max eigenvalue.

    Returns
    -------
    ParameterValuePopulation
        Pruned (stable + fast) parameter population.
    """
    # Heavy SKiMpy / pyTFA imports kept local so `--help` is snappy.
    from sympy import Symbol
    from scipy.linalg import eigvals as eigenvalues

    from skimpy.utils.namespace import QSSA
    from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
    from skimpy.core.parameters import ParameterValues, ParameterValuePopulation
    from skimpy.analysis.oracle.load_pytfa_solution import (
        load_fluxes, load_concentrations, load_equilibrium_constants,
    )
    from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
    from skimpy.utils.general import get_stoichiometry
    from skimpy_tools.check_stability import CheckStability

    # Reset the parameter set index so .loc[j] is positional 0..N-1.
    params_df = params_df.reset_index(drop=True)

    sample = samples.iloc[stability_sample_row]
    fluxes_dict = load_fluxes(
        sample, tmodel, kmodel,
        density=DENSITY,
        ratio_gdw_gww=GDW_GWW_RATIO,
        concentration_scaling=CONCENTRATION_SCALING,
        time_scaling=TIME_SCALING,
    )
    concentrations_dict = load_concentrations(
        sample, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING,
    )
    k_eq = load_equilibrium_constants(
        sample, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING,
        in_place=True,
    )

    # Compile MCA pieces and the jacobian (matches the notebook).
    kmodel.prepare(mca=False)
    dep_ix, indep_ix = get_dep_indep_vars_from_basis(kmodel.conservation_relation)
    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix = dep_ix
    kmodel.reduced_stoichiometry = get_stoichiometry(kmodel, kmodel.reactants)[indep_ix, :]
    kmodel.compile_jacobian(ncpu=ncpu_compile)
    # And the QSSA jacobian variant that the notebook uses just before sampling.
    kmodel.prepare()
    kmodel.compile_jacobian(sim_type=QSSA)

    cs = CheckStability()
    cs.kmodel = kmodel
    cs.flux_series = fluxes_dict
    cs.conc_series = concentrations_dict
    cs.k_eq = k_eq
    cs.sym_conc_dict = {Symbol(k): v for k, v in concentrations_dict.items()}
    cs.CONCENTRATION_SCALING = CONCENTRATION_SCALING
    cs.TIME_SCALING = TIME_SCALING
    cs.DENSITY = DENSITY
    cs.GDW_GWW_RATIO = GDW_GWW_RATIO
    cs.parameter_set = params_df

    # Compile sampling helper functions (needed for flux_parameter_function).
    sampling_parameters = SimpleParameterSampler.Parameters(n_samples=1)
    sampler = SimpleParameterSampler(sampling_parameters)
    sampler._compile_sampling_functions(cs.kmodel, cs.sym_conc_dict, [])

    model_param = cs.kmodel.parameters
    store_eigen = []
    param_pop = []
    n = len(cs.parameter_set.index)
    print(f"[stability] Evaluating {n} parameter sets...")
    for j in range(n):
        if j % 100 == 0:
            print(f"[stability] models processed: {j}/{n}")

        param_val = ParameterValues(cs.parameter_set.loc[j], cs.kmodel)
        cs.kmodel.parameters = cs.k_eq
        cs.kmodel.parameters = param_val
        cs.parameter_sample = {v.symbol: v.value for k, v in cs.kmodel.parameters.items()}

        # Set Vmax_forward = 1 for every reaction so flux_parameter_function
        # can back-calculate the Vmax's consistent with the input fluxes.
        for rxn in cs.kmodel.reactions.values():
            cs.parameter_sample[rxn.parameters.vmax_forward.symbol] = 1

        cs.kmodel.flux_parameter_function(
            cs.kmodel,
            cs.parameter_sample,
            cs.sym_conc_dict,
            cs.flux_series,
        )
        for c in cs.conc_series.index:
            if c in model_param:
                cs.parameter_sample[cs.kmodel.parameters[c].symbol] = cs.conc_series[c]

        this_jacobian = cs.kmodel.jacobian_fun(
            cs.flux_series[cs.kmodel.reactions],
            cs.conc_series[cs.kmodel.reactants],
            cs.parameter_sample,
        )
        eigs = sorted(np.real(eigenvalues(this_jacobian.todense())))
        if eigs[-1] <= 0:
            store_eigen.append(eigs)
            param_pop.append(ParameterValues(cs.parameter_sample, cs.kmodel))

    n_stable = len(param_pop)
    print(f"[stability] {n_stable}/{n} models stable")
    if n_stable == 0:
        raise RuntimeError("No stable models found - cannot prune or perturb.")

    param_pop = ParameterValuePopulation(param_pop, kmodel)
    maximal_eigen = pd.DataFrame(np.array(store_eigen)[:, -1])

    # ------------------------------------------------------------------
    # Pruning (notebook helper inlined)
    # ------------------------------------------------------------------
    is_selected = (maximal_eigen < max_eigenvalue)
    is_selected.columns = range(len(maximal_eigen.T))
    fast_parameters = []
    fast_index = []
    for i, row in is_selected.T.iterrows():
        if any(row):
            fast_models = np.where(np.array(row))[0]
            fast_parameters.extend([param_pop._data[k] for k in fast_models])
            fast_index.extend([f"{i},{k}" for k in fast_models])

    print(f"[prune] {len(fast_index)} models kept after pruning "
          f"(max eigenvalue < {max_eigenvalue})")
    if not fast_index:
        raise RuntimeError(
            f"No models passed the pruning threshold of {max_eigenvalue}. "
            "Try a less strict --max-eigenvalue."
        )

    pruned = ParameterValuePopulation(fast_parameters, kmodel=kmodel, index=fast_index)
    return pruned


# ---------------------------------------------------------------------------
# Step 3: per-perturbation worker (runs in its own loky process)
# ---------------------------------------------------------------------------
def simulate_one_perturbation(
    perturbation,
    path_to_parameter_population,
    samples,
    path_to_tmodel,
    path_to_kmodel,
    perturbation_sample_row,
    flux_scaling,
):
    """Simulate every pruned model under one enzyme perturbation.

    Returns (enzyme_id, DataFrame[reactions x model_index]).
    """
    # Local imports: every loky worker is a fresh interpreter.
    from pytfa.io.json import load_json_model
    from skimpy.io.yaml import load_yaml_model
    from skimpy.analysis.oracle.load_pytfa_solution import (
        load_fluxes, load_concentrations,
    )
    from skimpy.core.parameters import load_parameter_population
    from skimpy.analysis.ode.utils import make_flux_fun
    from skimpy.utils.namespace import QSSA
    from skimpy.utils.tabdict import TabDict

    enzyme_id, change = perturbation

    tmodel = load_json_model(path_to_tmodel)
    kmodel = load_yaml_model(path_to_kmodel)

    sample = samples.loc[perturbation_sample_row]
    fluxes_steady = load_fluxes(
        sample, tmodel, kmodel,
        density=DENSITY,
        ratio_gdw_gww=GDW_GWW_RATIO,
        concentration_scaling=CONCENTRATION_SCALING,
        time_scaling=TIME_SCALING,
    )
    concentrations = load_concentrations(
        sample, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING,
    )

    parameter_population = load_parameter_population(path_to_parameter_population)

    kmodel.prepare(mca=False)
    kmodel.compile_ode(sim_type=QSSA, ncpu=1)
    flux_fun = make_flux_fun(kmodel, sim_type=QSSA)

    vmax_symbol = str(kmodel.reactions[enzyme_id].parameters.vmax_forward.symbol)

    samples_to_simulate = list(parameter_population._index.keys())
    results = {}
    reaction_keys = None

    for ix in samples_to_simulate:
        # Reload base parameters for this model.
        kmodel.parameters = parameter_population[ix]
        kmodel.initial_conditions = TabDict(concentrations.items())
        # Apply the perturbation.
        kmodel.parameters[vmax_symbol].value *= change

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

        cons = sol.concentrations.iloc[-1]
        if not np.all(np.isfinite(cons.values)):
            continue

        flux = flux_fun(cons, parameter_population[ix])
        if reaction_keys is None:
            reaction_keys = list(flux.keys())
        # Apply the same post-scaling the notebook does on the perturbed
        # enzyme's own flux value.
        flux[enzyme_id] = flux[enzyme_id] * change
        results[ix] = np.array([flux[k] for k in reaction_keys]) / flux_scaling
        del sol

    if reaction_keys is None:
        # Every solve failed; return an empty frame so the pipeline doesn't crash.
        return enzyme_id, pd.DataFrame()
    return enzyme_id, pd.DataFrame(results, index=reaction_keys)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the user did not pass --output-prefix, derive it from the params filename.
    if not args.output_prefix:
        args.output_prefix = Path(args.params).stem
    print(f"[output] Files will be written as "
          f"{output_dir}/{args.output_prefix}_<ENZYME>.csv")

    # Load the initial dataset.
    index_col = None if args.params_index_col < 0 else args.params_index_col
    print(f"[load] Reading initial parameter set from {args.params}")
    params_df = pd.read_csv(args.params, index_col=index_col)

    # Load samples and models.
    print(f"[load] Reading TFA samples from {args.samples}")
    samples = pd.read_csv(args.samples, header=0, index_col=0)

    from pytfa.io.json import load_json_model
    from skimpy.io.yaml import load_yaml_model
    print(f"[load] Loading tmodel/kmodel...")
    tmodel = load_json_model(args.tmodel)
    kmodel = load_yaml_model(args.kmodel)

    # Stability + pruning.
    pruned = run_stability_and_prune(
        params_df=params_df,
        kmodel=kmodel,
        tmodel=tmodel,
        samples=samples,
        stability_sample_row=args.stability_sample_row,
        max_eigenvalue=args.max_eigenvalue,
        ncpu_compile=args.ncpu_compile,
    )

    # Save pruned parameter population to disk so loky workers can reload it.
    if args.keep_pruned_hdf5:
        pruned_path = str(output_dir / "pruned_parameters.hdf5")
        cleanup_pruned = False
    else:
        tmp_dir = tempfile.mkdtemp(prefix="genai_pipeline_")
        pruned_path = os.path.join(tmp_dir, "pruned_parameters.hdf5")
        cleanup_pruned = True
    print(f"[prune] Saving pruned parameters to {pruned_path}")
    pruned.save(pruned_path)

    # Parallel perturbations.
    from joblib import Parallel, delayed
    print(f"[perturb] Launching {len(args.perturbations)} perturbations on "
          f"{args.n_jobs} workers...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(simulate_one_perturbation)(
            perturbation,
            pruned_path,
            samples,
            args.tmodel,
            args.kmodel,
            args.perturbation_sample_row,
            args.flux_scaling,
        )
        for perturbation in args.perturbations
    )

    # Save one CSV per perturbation.
    for enzyme_id, df in results:
        out_path = output_dir / f"{args.output_prefix}_{enzyme_id}.csv"
        df.to_csv(out_path)
        print(f"[save] {enzyme_id}: wrote {df.shape[0]} reactions x "
              f"{df.shape[1]} models -> {out_path}")

    # Clean up the temp HDF5 if we created one.
    if cleanup_pruned:
        try:
            shutil.rmtree(os.path.dirname(pruned_path))
        except OSError:
            pass

    print("[done] Pipeline finished.")


if __name__ == "__main__":
    sys.exit(main() or 0)

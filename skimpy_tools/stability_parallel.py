import warnings
import numpy as np
import pandas as pd

from sympy import Matrix, Symbol
from scipy.sparse import csc_matrix as sparse_matrix

# Import your domain-specific functions here
# from yourpackage import (
#     load_json_model,
#     load_yaml_model,
#     load_fluxes,
#     load_concentrations,
#     load_equilibrium_constants,
#     CheckStability,
#     ParameterValues,
#     get_dep_indep_vars_from_basis,
#     get_stoichiometry,
#     eigenvalues
# )


# ============================================================
# Model Preparation
# ============================================================

def build_and_prepare_kmodel(kmodel, ncpu=1):

    kmodel.prepare(mca=False)

    conservations = pd.read_csv(
        "./data/tfa_samples/mini_redYeast8_26Oct2020_151326_cons_relations_annotated.csv"
    )

    conservations.columns = [
        f"_{col}" if col[0].isdigit() else col
        for col in conservations.columns
    ]

    L0, pivot = Matrix(
        conservations[kmodel.reactants].values
    ).rref()

    kmodel.conservation_relation = sparse_matrix(L0, dtype=float)

    dep_ix, indep_ix = get_dep_indep_vars_from_basis(
        kmodel.conservation_relation
    )

    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix = dep_ix

    kmodel.reduced_stoichiometry = get_stoichiometry(
        kmodel, kmodel.reactants
    )[indep_ix, :]

    kmodel.compile_jacobian(ncpu=ncpu)

    return kmodel


# ============================================================
# Parallel Stability Chunk
# ============================================================

def process_parameter_chunk(sample,
                            df_params_chunk,
                            chunk_id,
                            path_to_tmodel,
                            path_to_kmodel,
                            ncpu_model=1):

    warnings.filterwarnings("ignore", category=FutureWarning)

    CONCENTRATION_SCALING = 1e6
    DENSITY = 1200
    GDW_GWW_RATIO = 0.3
    TIME_SCALING = 1

    # Load models
    tmodel = load_json_model(path_to_tmodel)
    kmodel = load_yaml_model(path_to_kmodel)
    kmodel = build_and_prepare_kmodel(kmodel, ncpu=ncpu_model)

    local_param_dict = {}
    local_stable_info = {}

    print(f"\n=== Processing chunk {chunk_id} "
          f"(size={len(df_params_chunk)}) ===\n")

    # ---------------------------------------------------------
    # Load thermodynamic data once per worker
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Stability checker
    # ---------------------------------------------------------

    cs = CheckStability()
    cs.kmodel = kmodel
    cs.flux_series = fluxes_dict
    cs.conc_series = concentrations_dict
    cs.k_eq = k_eq
    cs.sym_conc_dict = {
        Symbol(k): v for k, v in concentrations_dict.items()
    }

    cs.kmodel.parameters = cs.k_eq

    # ---------------------------------------------------------
    # Loop over chunk
    # ---------------------------------------------------------

    for ix2, (_, row) in enumerate(df_params_chunk.iterrows()):

        if ix2 % 100 == 0:
            print(f"[chunk {chunk_id}] processed: {ix2}")

        param_val = ParameterValues(row, cs.kmodel)
        cs.kmodel.parameters = param_val

        cs.parameter_sample = {
            v.symbol: v.value for k, v in cs.kmodel.parameters.items()
        }

        # Force Vmax = 1
        for rxn in cs.kmodel.reactions.values():
            vmax_param = rxn.parameters.vmax_forward
            cs.parameter_sample[vmax_param.symbol] = 1

        # Update flux
        cs.kmodel.flux_parameter_function(
            cs.kmodel,
            cs.parameter_sample,
            cs.sym_conc_dict,
            cs.flux_series,
        )

        for c in cs.conc_series.index:
            if c in cs.kmodel.parameters:
                c_sym = cs.kmodel.parameters[c].symbol
                cs.parameter_sample[c_sym] = cs.conc_series[c]

        jac = cs.kmodel.jacobian_fun(
            cs.flux_series[cs.kmodel.reactions],
            cs.conc_series[cs.kmodel.reactants],
            cs.parameter_sample,
        )

        eigs = np.real(eigenvalues(jac.todense()))
        stable = np.max(eigs) <= 0

        key = f"{chunk_id},{ix2}"

        if stable:
            local_param_dict[key] = ParameterValues(
                cs.parameter_sample, cs.kmodel
            )
            local_stable_info[key] = {
                "stable": True,
                "max_eigenvalue": float(np.max(eigs)),
            }

    return local_param_dict, local_stable_info


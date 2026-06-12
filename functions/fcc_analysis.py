import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations
from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
from skimpy.utils.general import get_stoichiometry
from skimpy.utils.namespace import QSSA
from skimpy.utils.tensor import Tensor
from skimpy.utils.tabdict import TabDict

def compute_fcc_error(tmodel, kmodel, samples, parameter_population, fccexp, 
                      NCPU=4, CONCENTRATION_SCALING=1e6, TIME_SCALING=1, 
                      DENSITY=1521, GDW_GWW_RATIO=0.3):
    """
    Compute the Flux Control Coefficients (FCC) and error function matrix.

    Parameters:
    - tmodel: The thermodynamic model.
    - kmodel: The kinetic model.
    - samples: The TFA sample data.
    - parameter_population: The kinetic parameter population.
    - fccexp: Experimental FCC data.
    - NCPU: Number of CPU cores to use.
    - CONCENTRATION_SCALING: Scaling factor for concentration.
    - TIME_SCALING: Scaling factor for time.
    - DENSITY: Yeast cell density (default: 1521 g/L).
    - GDW_GWW_RATIO: Dry-to-wet weight ratio (default: 0.3).

    Returns:
    - fcccalc: Computed flux control coefficients.
    - error_function_matrix: Computed error function matrix.
    """

    # Compile ODE-Functions
    kmodel.prepare(mca=False)

    # Transform conservation relations
    dep_ix, indep_ix = get_dep_indep_vars_from_basis(kmodel.conservation_relation)
    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix = dep_ix
    kmodel.reduced_stoichiometry = get_stoichiometry(kmodel, kmodel.reactants)[indep_ix, :]

    # Compile with parameter elasticities
    parameter_list = TabDict([(k, p.symbol) for k, p in kmodel.parameters.items()
                              if p.name.startswith('vmax_forward')])

    kmodel.compile_mca(sim_type=QSSA, ncpu=NCPU, parameter_list=parameter_list)

    # Get parameter sample keys
    this_samples = list(parameter_population._index.keys())
    samples_to_simulate = this_samples

    flux_control_data = []
    con_control_data = []

    for ix in samples_to_simulate:
        # Set parameters for this sample
        kmodel.parameters = parameter_population[ix]

        # Get reference concentrations
        tfa_id, _ = ix.split(',')
        tfa_id = int(tfa_id)
        sample = samples

        # Load fluxes and concentrations
        fluxes = load_fluxes(sample, tmodel, kmodel, density=DENSITY, ratio_gdw_gww=GDW_GWW_RATIO,
                             concentration_scaling=CONCENTRATION_SCALING, time_scaling=TIME_SCALING)

        concentrations = load_concentrations(sample, tmodel, kmodel, concentration_scaling=CONCENTRATION_SCALING)

        flux_control_coeff = kmodel.flux_control_fun(fluxes, concentrations, [parameter_population[ix]])
        con_control_coeff = kmodel.concentration_control_fun(fluxes, concentrations, [parameter_population[ix]])

        flux_control_data.append(flux_control_coeff._data)
        con_control_data.append(con_control_coeff._data)

    # Compute indices
    con_index = pd.Index([kmodel.reactants.iloc(i)[0] for i in kmodel.independent_variables_ix], name="concentration")
    
    # Convert lists to tensors
    fcc_data = np.concatenate(flux_control_data, axis=2)
    ccc_data = np.concatenate(con_control_data, axis=2)
    
    flux_index = pd.Index(kmodel.reactions.keys(), name="flux")
    parameter_index = pd.Index(kmodel.flux_control_fun.parameter_elasticity_function.respective_variables, name="parameter")
    sample_index = pd.Index(samples_to_simulate, name="sample")

    # Store FCC and CCC tensors
    flux_control_coeff = Tensor(fcc_data, [flux_index, parameter_index, sample_index])
    con_control_coeff = Tensor(ccc_data, [con_index, parameter_index, sample_index])

    # Compute mean FCC
    mean_fcc = flux_control_coeff.mean('sample')
    mean_ccc = con_control_coeff.mean('sample')

    # Remove transport reactions
    transport_reactions = ['vmax_forward_' + r.id for r in tmodel.reactions
                           if (len(r.compartments) > 1)
                           and not ('i' in r.compartments)
                           and not r.id.startswith('LMPD_')
                           and r.id in kmodel.reactions]

    mean_fcc = mean_fcc.drop(transport_reactions, axis=1)

    # Convert experimental FCC to numpy array
    fccexp = np.array(fccexp.iloc[:, 1:13])

    # Get indexes for reactions and knockouts
    reactions_indexes = {
        0: 29, 1: 52, 2: 50, 3: 19, 4: 73, 5: 28, 6: 55, 7: 61, 8: 49, 9: 54, 10: 36, 
        11: 63, 12: 64, 13: 71, 14: 69, 15: 72, 16: 14, 17: 3, 18: 37, 19: 7, 20: 67, 
        21: 25, 22: 42, 23: 57, 24: 43, 25: 38, 26: 40
    }
    knockouts_indexes = {0: 52, 1: 55, 2: 61, 3: 54, 4: 63, 5: 64, 6: 71, 7: 69}

    # Compute error function
    fcccalc = flux_control_coeff._data
    error_function = []

    for model in range(len(fcccalc[0, 0, :])):
        error = 0
        for knock1, knock2 in knockouts_indexes.items():
            for reaction1, reaction2 in reactions_indexes.items():
                error += abs(fcccalc[reaction2, knock2, model] - fccexp[reaction1, knock1]) / abs(fccexp[reaction1, knock1])
        error_function.append(error)

    # Store error function as a DataFrame
    error_function_matrix = pd.DataFrame(data=error_function, index=parameter_population._index, columns=['error_function'])

    # Plot the histogram
    plt.hist(np.log(error_function), bins=100, color='#0047AB', edgecolor='black', linewidth=1.5, alpha=0.85)
    plt.title("Histogram of log(Error Function)")
    plt.xlabel("log(error function)")
    plt.ylabel("Frequency")
    plt.show()

    return fcccalc, error_function_matrix





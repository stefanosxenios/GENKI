from pytfa.io.json import load_json_model
import sys
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.sampling.ga_parameter_sampler import GaParameterSampler
from skimpy.utils.general import get_stoichiometry
from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
import pandas as pd
import numpy as np
from skimpy.core.parameters import ParameterValuePopulation, load_parameter_population

from sympy import Matrix
from scipy.sparse import csc_matrix as sparse_matrix
from sys import argv
import matplotlib.pyplot as plt
from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
from skimpy.utils.general import get_stoichiometry
from skimpy.utils.namespace import QSSA
from skimpy.utils.tensor import Tensor
from skimpy.utils.tabdict import TabDict


def fetch_km_values(kmodel,parameter_population):
    parameters = kmodel.parameters
    km_parameters = {key: value for key, value in parameters.items() if key.startswith('km')} 
    parameter_names = [key for key in km_parameters.keys()]
    calculated_parameters=pd.DataFrame(data=[dict(parameter_population._data[parameter_population._index[i]]) for i in parameter_population._index])
    Km=calculated_parameters[parameter_names]
    Km.index=parameter_population._index
    return Km






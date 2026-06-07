"""Parameters for MNL expansions and simulations."""

from math import sqrt

import numpy as np

from simuls_misspecif.MNL_utils import DataParams, TrueParams

# a starting set of parameter values, modified in the simulation scenarii
true_pars = TrueParams(beta0=0.0, beta1=1.0, sigma=0.5)

# the parameters of the model; do_exo is modified in the simulations
data_pars = DataParams(
    sigxi=1.0,
    sigx=1.0,
    rhox_z=sqrt(0.5),
    rhox_xi=sqrt(0.5),
    do_exo=True,
)

# True to use the second derivative (fourth order W regressor)
do_a_second = False

# whether we compute SPE bounds for the semi elasticities (costly)
do_bounds_semi_elast = False


# ranges of values of sigma and pi
basic_sigma_range = np.sqrt(np.arange(0.1, 2.05, 0.1))
large_sigma_range = np.arange(1.00, 2.00, 0.05)

"""Parameters for MNL expansions and simulations."""

from math import sqrt

import numpy as np

from simuls_misspecif.MNL_utils import DataParams, TrueParams

# a starting set of parameter values, modified in the simulation scenarii;
#  the standard errors of the random coefficients are sigma * sigma_profile
#  for sigma in the sigma range
true_pars = TrueParams(beta0=0.0, beta=np.array([1.0]), sigma_profile=np.array([1.0]))

# the parameters of the model; do_exo is modified in the simulations
data_pars = DataParams(
    sigxi=1.0,
    sigx=1.0,
    rhox_z=sqrt(0.5),
    rhox_xi=sqrt(0.5),
    do_exo=True,
    n_x=1,
)

# True to use the second derivative (fourth order W regressor)
do_a_second = False

# nodes per dimension of the Gauss-Hermite rule for the integrals in the bounds
#  and the true semi-elasticities when there are several random coefficients
n_gh_integrals = 8


# ranges of values of sigma and pi
basic_sigma_range = np.sqrt(np.arange(0.1, 2.05, 0.1))
large_sigma_range = np.arange(1.00, 2.00, 0.05)

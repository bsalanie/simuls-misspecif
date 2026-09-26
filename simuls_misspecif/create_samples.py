"""Create one sample of the normal model with M random coefficients."""

from typing import Optional, cast

import numpy as np
from bs_python_utils.bssputils import describe_array
from bs_python_utils.bsutils import print_stars

from simuls_misspecif.MNL_params import data_pars, true_pars
from simuls_misspecif.MNL_utils import (
    DataParams,
    TrueParams,
    _mean_utils,
    quadrature_nodes,
)


def make_shares(
    mean_utils_xi: np.ndarray,
    x: np.ndarray,
    sigma_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Use Gauss-Hermite quadrature to evaluate market shares.

    Args:
        mean_utils_xi: `(T, J)` mean utilities with the product effects.
        x: `(T, J, M)` covariates.
        sigma_vec: `(M,)` standard errors of the random coefficients, if any.

    Returns:
        `(T, J)` array of market shares.
    """
    if sigma_vec is None:  # shares with non-random coefficients
        exp_utils = np.exp(mean_utils_xi)
        return cast(np.ndarray, exp_utils / (1.0 + np.sum(exp_utils, 1, keepdims=True)))

    # we integrate
    sigma_vec = np.atleast_1d(sigma_vec)
    nodes, weights = quadrature_nodes(sigma_vec.size)
    shares = np.zeros_like(mean_utils_xi)
    for node_l, weight_l in zip(nodes, weights):
        exp_utils = np.exp(mean_utils_xi + x @ (sigma_vec * node_l))
        shares += weight_l * exp_utils / (1.0 + np.sum(exp_utils, 1, keepdims=True))
    return shares


def create_sample(
    nmarkets: int,
    nproducts: int,
    stream: np.random.Generator,
    sigma: float = 0.5,
    pars: TrueParams = true_pars,
    dpars: DataParams = data_pars,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one sample.

    Args:
        nmarkets: Number of markets.
        nproducts: Number of products.
        stream: Random generator.
        sigma: Scale of the standard errors of the random coefficients.
        pars: Parameters for the DGP.
        dpars: Data parameters for the DGP.

    Returns:
        A tuple of xi, x, z, and the simulated market shares.
    """

    draws = dpars.generate_random_draws(nmarkets, nproducts, stream)
    xi, x, z = dpars.generate_exogenous_vars_from_draws(draws)

    sigma_vec = sigma * pars.sigma_profile

    mean_utils_xi = _mean_utils(pars.beta0, pars.beta, x) + xi
    # describe_array(mean_utils_xi, "mxi")
    shares = make_shares(mean_utils_xi, x, sigma_vec)
    # describe_array(shares, "shares")

    return (xi, x, z, shares)


if __name__ == "__main__":
    stream = np.random.default_rng()
    print_stars("Without randomness")
    xi, x, z, shares = create_sample(100, 4, stream, 0.5, true_pars, data_pars)
    describe_array(xi, "xi")
    describe_array(x, "x")
    describe_array(z, "z")
    describe_array(shares, "shares")

    print_stars("Without micromoment")
    xi, x, z, shares = create_sample(100, 4, stream, 0.5, true_pars, data_pars)
    describe_array(xi, "xi")
    describe_array(x, "x")
    describe_array(z, "z")
    describe_array(shares, "shares")

"""Create one sample of the normal model with one random coefficient."""

from typing import Optional, cast

import numpy as np
from bs_python_utils.bsnputils import gaussian_expectation
from bs_python_utils.bssputils import describe_array
from bs_python_utils.bsutils import print_stars

from simuls_misspecif.MNL_params import data_pars, true_pars
from simuls_misspecif.MNL_utils import DataParams, TrueParams, _mean_utils, wgh, xgh


def make_shares(
    mean_utils_xi: np.ndarray, x: np.ndarray, sigma_tot: Optional[float] = None
) -> np.ndarray:
    """Use Gauss-Hermite quadrature to evaluate market shares.

    Args:
        mean_utils_xi: `(T, J)` mean utilities with the product effects.
        x: `(T, J)` covariates.
        sigma_tot: Total standard error of the random coefficient, if any.

    Returns:
        `(T, J)` array of market shares.
    """
    nmarkets, nproducts = mean_utils_xi.shape
    shares = np.zeros((nmarkets, nproducts))
    if sigma_tot is None:  # shares with non-random coefficients
        for t in range(nmarkets):
            exp_utils = np.exp(mean_utils_xi[t, :])
            shares[t, :] = exp_utils / (1.0 + np.sum(exp_utils))
    else:  # we integrate

        def utils(v, pars):
            x_t, means_t, sigma_tot = pars
            utils_v = sigma_tot * v * x_t + means_t
            shares_v = np.exp(utils_v)
            denom_v = 1.0 + np.sum(shares_v)
            shares_v /= denom_v
            return shares_v

        for t in range(nmarkets):
            pars_t = [x[t, :], mean_utils_xi[t, :], sigma_tot]
            shares[t, :] = gaussian_expectation(utils, pars=pars_t, x=xgh, w=wgh)

    return shares


def create_sample(
    nmarkets: int,
    nproducts: int,
    stream: np.random.Generator,
    pars: TrueParams = true_pars,
    dpars: DataParams = data_pars,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one sample.

    Args:
        nmarkets: Number of markets.
        nproducts: Number of products.
        stream: Random generator.
        pars: Parameters for the DGP.
        dpars: Data parameters for the DGP.

    Returns:
        A tuple of xi, x, z, and the simulated market shares.
    """

    beta0, beta1, sigma = (
        cast(float, pars.beta0),
        cast(float, pars.beta1),
        cast(float, pars.sigma),
    )
    draws = dpars.generate_random_draws(nmarkets, nproducts, stream)
    xi, x, z = dpars.generate_exogenous_vars_from_draws(draws)

    sigma_tot = sigma

    mean_utils_xi = _mean_utils(beta0, beta1, x) + xi
    # describe_array(mean_utils_xi, "mxi")
    shares = make_shares(mean_utils_xi, x, sigma_tot)
    # describe_array(shares, "shares")

    return (xi, x, z, shares)


if __name__ == "__main__":
    stream = np.random.default_rng()
    print_stars("Without randomness")
    xi, x, z, shares = create_sample(100, 4, stream, true_pars, data_pars)
    describe_array(xi, "xi")
    describe_array(x, "x")
    describe_array(z, "z")
    describe_array(shares, "shares")

    print_stars("Without micromoment")
    xi, x, z, shares = create_sample(100, 4, stream, true_pars, data_pars)
    describe_array(xi, "xi")
    describe_array(x, "x")
    describe_array(z, "z")
    describe_array(shares, "shares")

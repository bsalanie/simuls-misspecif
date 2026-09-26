"""Utility functions for simulation and numerical operations.

Includes RNG stream generation for parallel workers, logging helpers,
and what-if estimator computations.
"""

from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort, file_print_stars, print_stars
from numpy.random import SeedSequence, default_rng


def generate_RNG_streams(
    nsim: int, initial_seed: int = 13091962
) -> list[np.random.Generator]:
    """
    Generate independent RNG streams for parallel processes.

    Args:
        nsim: Number of independent RNG streams to generate (one per simulation).
        initial_seed: An integer seed to initialize the SeedSequence.

    Returns:
        A list of independent RNG generators that can be used in parallel processes.
    """
    ss = SeedSequence(initial_seed)
    # Spawn off child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(nsim)
    streams = [default_rng(s) for s in child_seeds]
    return streams


def f_print_stars(use_mp: bool, what: str, fout_name: str | None = None):
    """prints to a file or to the screen; to a file under multiprocessing.

    Args:
        use_mp (bool): True if we use `multiprocessing`_
        what (str): what we print
        fout_name (str | None, optional): where we print, if not to screen. Defaults to None.
    """
    if use_mp and fout_name is not None:
        fout_path = Path(fout_name)
        if fout_path.parent:
            fout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fout_path, "a") as fout:
            file_print_stars(fout, what)
    elif use_mp:
        bs_error_abort("use_mp is True but fout_name is None")
    else:
        print_stars(what)


def center_moments(moments_used: np.ndarray, nproducts: int) -> np.ndarray:
    """Center the moments used in the what-if estimation by subtracting the mean across products.

    Args:
        moments_used: A 2D array of shape (TJ, n_instr) containing the moments used in the what-if estimation.
        nproducts: The number of products in each market.

    Returns:
        A 2D array of shape (TJ, n_instr) containing the centered moments.
    """
    moments_used_centered = np.zeros_like(moments_used)
    npts, n_instr = moments_used.shape
    nmarkets = npts // nproducts
    for k in range(n_instr):
        moments_used_k = moments_used[:, k].reshape((nmarkets, nproducts))
        moments_used_k_mean = moments_used_k.mean(axis=0)
        moments_used_centered[:, k] = (moments_used_k - moments_used_k_mean).reshape(
            npts
        )
    return moments_used_centered


def make_omega_inv(moments_used: np.ndarray) -> np.ndarray:
    """Compute the omega_inv matrix based on the moments used in the what-if estimation.

    Args:
        moments_used: A 2D array of shape (TJ, n_instr) containing the moments used in the what-if estimation.
    Returns:
        A 2D array of shape (n_instr, n_instr) representing the omega_inv matrix.
    """
    npts = moments_used.shape[0]
    return cast(np.ndarray, moments_used.T @ moments_used / npts)


def estimate_what_if(
    X: np.ndarray,
    K: np.ndarray,
    W: np.ndarray,
    beta0_0: float,
    beta_0: np.ndarray,
    xi_0_vec: np.ndarray,
    Z_used: np.ndarray,
    Omega: np.ndarray,
) -> np.ndarray:
    """The what-if estimator of `[beta0, beta, sigma2]`.

    We use `xi(sigma2) = xi_0 - X dbeta - sum_m sigma2_m K_m
    + sum_{m,n} sigma2_m sigma2_n W_mn` and linearize the GMM first-order conditions
    around the non-random estimates.

    Args:
        X: `(TJ, M)` covariates.
        K: `(TJ, M)` second-order artificial regressors.
        W: `(TJ, M, M)` fourth-order artificial regressors (half of `frac_blp.make_W`).
        beta0_0: Non-random estimate of beta0.
        beta_0: `(M,)` non-random estimates of beta.
        xi_0_vec: `(TJ,)` residuals of the non-random model.
        Z_used: `(TJ, n_instr)` instruments.
        Omega: `(n_instr, n_instr)` weighting matrix.

    Returns:
        The `(1 + 2M)` what-if estimates.
    """
    npts, n_x = K.shape
    R = np.column_stack((np.ones(npts), X, K))
    Z_R = Z_used.T @ R / npts
    Z_xi0 = Z_used.T @ xi_0_vec / npts
    Omega_Z_xi0 = Omega @ Z_xi0

    lhs_mat = Z_R.T @ Omega @ Z_R
    Z_W = np.einsum("il,imn->lmn", Z_used, W) / npts
    lhs_mat[1 + n_x :, 1 + n_x :] -= 2.0 * np.einsum("lmn,l->mn", Z_W, Omega_Z_xi0)

    rhs_vec = np.zeros(1 + 2 * n_x)
    rhs_vec[1 + n_x :] = Z_R[:, 1 + n_x :].T @ Omega_Z_xi0

    dbeta_s2_whatif = np.linalg.solve(lhs_mat, rhs_vec)
    dbeta_whatif, s2_whatif = (
        dbeta_s2_whatif[: 1 + n_x],
        dbeta_s2_whatif[1 + n_x :],
    )

    beta_whatif = np.concatenate(([beta0_0], beta_0)) + dbeta_whatif
    whatif_vals = np.concatenate((beta_whatif, s2_whatif))

    return whatif_vals


def get_semi_elast_stats(
    own_semi: np.ndarray, cross_semi: np.ndarray, nproducts: int
) -> np.ndarray:
    """Means and standard deviations across markets of the semi-elasticities.

    Args:
        own_semi: `(T, M)` own semi-elasticities.
        cross_semi: `(T, M)` cross semi-elasticities.
        nproducts: Number of products.

    Returns:
        An `(M, 4)` array (mean own, std own, mean cross, std cross),
        or `(M, 2)` if `nproducts = 1`.
    """
    stats = [np.mean(own_semi, 0), np.std(own_semi, 0)]
    if nproducts > 1:
        stats += [np.mean(cross_semi, 0), np.std(cross_semi, 0)]
    return np.column_stack(stats)

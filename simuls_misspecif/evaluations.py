"""evaluating and printing the various stats we collect in the simulations"""

from math import sqrt
from typing import Iterable, List, Tuple, cast

import numpy as np
import scipy.linalg as spla
from bs_python_utils.bsnputils import (
    ThreeArrays,
    check_vector,
    check_vector_or_matrix,
    npexp,
    nplog,
    npmaxabs,
)
from bs_python_utils.bsstats import flexible_reg
from bs_python_utils.bsutils import bs_error_abort, print_stars
from frac_blp.artificial_regressors import (
    make_K_and_y,
    make_V,
    make_W,
)

from simuls_misspecif.MNL_integrals import (
    _dshares_dx,
    _exp_stj_eps,
    _exp_stj_stk,
    _exp_stj_stk_eps,
)

#
# def simulated_mean_shares(utils: np.ndarray) -> np.ndarray:
#     pass
#
#
#
# def berry_xis(
#     shares: np.ndarray,
#     mean_u: np.ndarray,
#     x: np.ndarray,
#     xi: np.ndarray,
#     s2: float,
#     tol: float = 1e-9,
#     maxiter: int = 10000,
#     ndraws: int = 10000,
#     verbose: bool = False,
# ) -> Tuple[np.ndarray, int, int]:
#     """Invert product effects xi from market shares on one market.
#
#     Args:
#         shares: `nproducts` vector of observed shares.
#         mean_u: `nproducts` vector of mean utilities.
#         x: `nproducts` vector of covariates.
#         xi: `nproducts` vector of an initial estimate of xi.
#         s2: Variance of the random coefficient.
#         tol: Tolerance.
#         maxiter: Maximum number of iterations.
#         ndraws: Number of draws for simulation.
#         verbose: Whether to print progress information.
#
#     Returns:
#         A tuple containing the estimated xi vector, the return code, and the
#         number of evaluations.
#     """
#
#     s = sqrt_kludge(s2)
#     xi_cur = xi.copy()
#     max_err = np.inf
#     retcode = 0
#     iter = 0
#     eps = np.random.normal(size=ndraws)
#     while max_err > tol:
#         utils = s * np.outer(x, eps) + (mean_u + xi_cur).reshape((-1, 1))
#         shares_sim = simulated_mean_shares(utils)
#         err_shares = np.log(shares) - np.log(shares_sim)
#         max_err = npmaxabs(err_shares)
#         if verbose and iter % 1000 == 1:
#             print(f"berry_xis: error {max_err} after {iter - 1} iterations")
#         xi_cur += err_shares
#         iter += 1
#         if iter > maxiter:
#             print_stars(
#                 f"berry_xis: stuck with error {max_err} after {iter} iterations"
#             )
#             retcode = 1
#             break
#     if verbose:
#         print_stars(f"berry_xis: error {max_err} after {iter} iterations")
#     return xi_cur, retcode, iter


def _integrand_shares(values, pars):
    sx, mean_u_xi_cur = pars
    utils = np.outer(values, sx) + mean_u_xi_cur
    max_utils = np.max(utils, 1)
    dutils = utils - max_utils.reshape((-1, 1))
    exp_d = cast(np.ndarray, npexp(dutils))
    denom = np.sum(exp_d, 1) + npexp(-max_utils)
    shares = exp_d.T / denom
    return shares


def sqrt_kludge(sq):
    return sqrt(max(sq, 1e-9))


def berry_xis_GQ(
    shares: np.ndarray,
    mean_u: np.ndarray,
    x: np.ndarray,
    xi: np.ndarray,
    s2: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Invert product effects xi from market shares on one market.

    Gaussian quadrature is used for the integration.

    Args:
        shares: `nproducts` vector of observed shares.
        mean_u: `nproducts` vector of mean utilities.
        x: `nproducts` vector of covariates.
        xi: `nproducts` vector of an initial estimate of xi.
        s2: Variance of the random coefficient.
        nodes: Nodes for Gauss-Hermite integration.
        weights: Weights for Gauss-Hermite integration.
        tol: Tolerance.
        maxiter: Maximum number of iterations.
        verbose: Whether to print progress information.

    Returns:
        A tuple containing the estimated xi vector, the return code, and the
        number of evaluations.
    """

    s = sqrt_kludge(s2)
    sx = s * x
    xi_cur = xi.copy()
    max_err = np.inf
    retcode = 0
    iter = 0

    while max_err > tol:
        mean_u_xi_cur = mean_u + xi_cur
        shares_sim = _integrand_shares(nodes, pars=(sx, mean_u_xi_cur)) @ weights
        err_shares = nplog(shares) - nplog(shares_sim)
        max_err = npmaxabs(err_shares)
        if verbose and iter % 100 == 1:
            print(f"berry_xis_GQ: error {max_err} after {iter - 1} iterations")
        xi_cur += err_shares
        iter += 1
        if iter > maxiter:
            print(f"berry_xis_GQ: stuck with error {max_err} after {iter} iterations")
            retcode = 1
            break

    if verbose:
        print(f"berry_xis_GQ: error {max_err} after {iter} iterations")

    return xi_cur, retcode, iter


def _artificial_regressors(
    observed_shares: np.ndarray, x: np.ndarray, J: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the artificial regressors at order 2 and 4.

    Args:
        observed_shares: Observed market shares, a `T*J` vector
        x: `(T*J, M)` covariates.
        J: Number of products.

    Returns:
        `K` `(T*J, M)`, `y` `(T*J,)`, `V` `(T*J, M)`, and `W` `(T*J, M, M)`.
    """
    K, y = make_K_and_y(x, observed_shares, J)
    V = make_V(x, observed_shares, J)
    W = make_W(x, observed_shares, J)
    return K, y, V, W


def _make_quadratic_instruments(z: np.ndarray):
    """Build quadratic instruments.

    Args:
        z: The `(T, J)` matrix of instruments.

    Returns:
        A `(T, 7)` or `(T, 11)` matrix.
    """
    nmarkets, nproducts = z.shape
    npts = z.size  # we will stack observations in the order of markets
    n_instr = 7
    quad_instr = np.zeros((npts, n_instr))
    mean_z = np.mean(z, axis=1)
    z2 = z * z
    mean_z2 = np.mean(z2, axis=1)
    market_means_z = np.repeat(mean_z, nproducts)
    market_means_z2 = np.repeat(mean_z2, nproducts)
    quad_instr[:, 0] = 1.0
    quad_instr[:, 1] = market_means_z
    quad_instr[:, 2] = market_means_z * market_means_z
    quad_instr[:, 3] = market_means_z2
    for j in range(nproducts):
        slice_j = slice(j, npts, nproducts)
        z_j = z[:, j]
        quad_instr[slice_j, 4] = z_j
        quad_instr[slice_j, 5] = z_j * z_j
        quad_instr[slice_j, 6] = z_j * mean_z
    return quad_instr


def _make_quartic_instruments(z: np.ndarray):
    """Build quartic instruments.

    Args:
        z: The `(T, J)` matrix of instruments.

    Returns:
        A `(T, 20)` or `(T, 49)` matrix.
    """
    nmarkets, nproducts = z.shape
    npts = z.size  # we will stack observations in the order of markets
    n_instr = 20
    quartic_instr = np.zeros((npts, n_instr))
    mean_z = np.mean(z, axis=1)
    z2 = z * z
    mean_z2 = np.mean(z2, axis=1)
    z3 = z2 * z
    mean_z3 = np.mean(z3, axis=1)
    z4 = z2 * z2
    mean_z4 = np.mean(z4, axis=1)
    market_means_z = np.repeat(mean_z, nproducts)
    market_means_z2 = np.repeat(mean_z2, nproducts)
    market_means_z3 = np.repeat(mean_z3, nproducts)
    market_means_z4 = np.repeat(mean_z4, nproducts)
    quartic_instr[:, 0] = 1.0
    quartic_instr[:, 1] = market_means_z
    quartic_instr[:, 2] = market_means_z * market_means_z
    quartic_instr[:, 3] = market_means_z2
    quartic_instr[:, 4] = market_means_z3
    quartic_instr[:, 5] = market_means_z * market_means_z2
    quartic_instr[:, 6] = market_means_z2 * market_means_z2
    quartic_instr[:, 7] = market_means_z * market_means_z3
    quartic_instr[:, 8] = market_means_z4
    for j in range(nproducts):
        slice_j = slice(j, npts, nproducts)
        z_j = z[:, j]
        zj_2 = z_j * z_j
        zj_3 = zj_2 * z_j
        zj_4 = zj_2 * zj_2
        quartic_instr[slice_j, 9] = z_j
        quartic_instr[slice_j, 10] = zj_2
        quartic_instr[slice_j, 11] = z_j * mean_z
        quartic_instr[slice_j, 12] = zj_2 * mean_z2
        quartic_instr[slice_j, 13] = z_j * mean_z2
        quartic_instr[slice_j, 14] = z_j * mean_z * mean_z
        quartic_instr[slice_j, 15] = zj_3
        quartic_instr[slice_j, 16] = zj_4
        quartic_instr[slice_j, 17] = zj_3 * mean_z
        quartic_instr[slice_j, 18] = zj_2 * mean_z * mean_z
        quartic_instr[slice_j, 19] = zj_2 * mean_z2
    return quartic_instr


def _projection_instruments(
    var: np.ndarray, z_instruments: np.ndarray, mode: str = "NP"
):
    check_vector(var, "_projection_instruments")
    ndims_z = check_vector_or_matrix(z_instruments, "_projection_instruments")
    nobs_v = var.size
    if ndims_z == 1:
        nobs_z, n_z = z_instruments.size, 1
    else:
        nobs_z, n_z = z_instruments.shape
    if nobs_v != nobs_z:
        bs_error_abort(
            f"var has {nobs_v} observations, while z_instruments has {nobs_z}"
        )
    return flexible_reg(var, z_instruments, mode=mode)


def _instruments_matrix(z: np.ndarray) -> np.ndarray:
    """Flatten `(T, J, M)` instruments to `(T*J, M)`, or `(T*J,)` if `M = 1`."""
    npts = z.shape[0] * z.shape[1]
    z_mat = z.reshape((npts, -1))
    return z_mat[:, 0] if z_mat.shape[1] == 1 else z_mat


def _project_columns(var: np.ndarray, z_instr: np.ndarray, mode: str) -> np.ndarray:
    """Project each column of a `(T*J, M)` matrix on the instruments."""
    var_proj = np.zeros_like(var)
    for m in range(var.shape[1]):
        var_proj[:, m] = _projection_instruments(var[:, m], z_instr, mode=mode)
    return var_proj


def _project_variables(
    y: np.ndarray,
    X: np.ndarray,
    z: np.ndarray,
    K: np.ndarray,
    V: np.ndarray | None = None,
    W: np.ndarray | None = None,
    mode: str = "NP",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Project the variables onto the instruments z.

    Args:
        y: `(T*J,)` vector.
        X: `(T*J, M)` covariates.
        z: `(T, J, M)` instruments.
        K: `(T*J, M)` second-order artificial regressors.
        V: `(T*J, M)` if not None.
        W: `(T*J, M, M)` if not None.
        mode: Projection mode. Default: `NP`.

    Returns:
        The projections of y `(T*J,)`, X and K `(T*J, M)`, V `(T*J, M)` or None,
        and W `(T*J, M, M)` or None.
    """
    z_instr = _instruments_matrix(z)

    y_proj = _projection_instruments(y, z_instr, mode=mode)
    X_proj = _project_columns(X, z_instr, mode)
    K_proj = _project_columns(K, z_instr, mode)

    V_proj = None if V is None else _project_columns(V, z_instr, mode)
    W_proj = None
    if W is not None:
        n_x = W.shape[1]
        W_proj = np.zeros_like(W)
        for m in range(n_x):
            for n in range(m, n_x):
                W_proj[:, m, n] = _projection_instruments(
                    W[:, m, n], z_instr, mode=mode
                )
                W_proj[:, n, m] = W_proj[:, m, n]

    return y_proj, X_proj, K_proj, V_proj, W_proj


def _reshape_proj(var_proj: np.ndarray, nproducts: int) -> np.ndarray:
    nmarkets = var_proj.size // nproducts
    if var_proj.ndim == 2:
        v_proj = var_proj[:, 0]
        return v_proj.reshape((nmarkets, nproducts))
    else:
        return var_proj.reshape((nmarkets, nproducts))


def _our_tsls0(
    y_proj: np.ndarray,
    X_proj: np.ndarray,
):
    """Regress `y_proj` on a constant and `X_proj` (the non-random model).

    Args:
        y_proj: `(T*J,)` projected LHS.
        X_proj: `(T*J, M)` (or `(T*J,)`) projected covariates.

    Returns:
        The instruments, the `1 + M` coefficients, and the condition number.
    """
    npts = y_proj.size
    Zstar0 = np.column_stack((np.ones(npts), X_proj))

    nonrandom_vals, _, _, s = spla.lstsq(Zstar0, y_proj)
    cond_number = abs(s[0] / s[-1])
    return Zstar0, nonrandom_vals, cond_number


def _our_tsls2(
    y_proj: np.ndarray,
    X_proj: np.ndarray,
    K_proj: np.ndarray,
):
    """Regress `y_proj` on a constant, `X_proj`, and `K_proj` (Salanie-Wolak).

    Args:
        y_proj: `(T*J,)` projected LHS.
        X_proj: `(T*J, M)` (or `(T*J,)`) projected covariates.
        K_proj: `(T*J, M)` (or `(T*J,)`) projected artificial regressors.

    Returns:
        The instruments, the `1 + 2M` coefficients, and the condition number.
    """
    npts = y_proj.size
    Zstar2 = np.column_stack((np.ones(npts), X_proj, K_proj))

    pseudo_vals, _, _, s = spla.lstsq(Zstar2, y_proj)
    cond_number = abs(s[0] / s[-1])
    return Zstar2, pseudo_vals, cond_number


def _print_pseudo_true_errors(
    true_p: np.ndarray,
    pseudo_vals: np.ndarray,
    names_ptv: List[str],
    verbose: bool = False,
):
    if verbose:
        n_params = true_p.size
        n_x = (n_params - 1) // 2
        print_stars(f"Pseudo-true errors for true sigma2={true_p[1 + n_x :]}:")
        for i in range(n_params):
            print(f"on {names_ptv[i]}: {pseudo_vals[i] - true_p[i]: >10.4f}")


def _print_set_semi_elast(
    semi_elasts: tuple[float, float] | tuple[float, float, float, float],
    name_elasts: str,
    verbose=False,
):
    """Print a set of semi-elasticities and returns it.

    Args:
        semi_elasts: the mean and stderr of own semi-elasticities,\
        or a tuple of 4 floats also containing the mean and stderr of the cross semi-elasticities.
        name_elasts: Name of the set of semi-elasticities, for printing.
        verbose: Whether to print the semi-elasticities.

    Returns:

    """
    do_cross = len(semi_elasts) == 4
    if do_cross:
        semi_elasts = cast(tuple[float, float, float, float], semi_elasts)
        resus_semi_elast = np.array(
            [
                semi_elasts[0],
                semi_elasts[1],
                semi_elasts[2],
                semi_elasts[3],
            ]
        )
    else:
        resus_semi_elast = np.array(
            [
                semi_elasts[0],
                semi_elasts[1],
            ]
        )
    if verbose:
        print_stars(name_elasts + " semi-elasticities")
        print(
            f"   own: mean = {semi_elasts[0]: 10.3f} and stderr = {semi_elasts[1]: 10.3f}"
        )
        if do_cross:
            semi_elasts = cast(tuple[float, float, float, float], semi_elasts)
            print(
                f"   cross: mean = {semi_elasts[2]: 10.3f} and stderr = {semi_elasts[3]: 10.3f}"
            )
    return resus_semi_elast


def estimated_xi_infty(
    observed_shares: np.ndarray,
    mean_utils: np.ndarray,
    x: np.ndarray,
    xi: np.ndarray,
    s2: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    verbose: bool = False,
) -> ThreeArrays:
    """Estimate the limit xi values using Berry inversion.

    Args:
        observed_shares: A `(T, J)` matrix.
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        xi: A `(T, J)` matrix, the initial estimate of xi.
        s2: An estimate of the variance of the random coefficient.
        nodes: Nodes for Gauss-Hermite integration.
        weights: Weights for Gauss-Hermite integration.
        verbose: Whether to print progress information.

    Returns:
        A `(T, J)` matrix,  a vector of return codes, and numbers of evaluations.
    """
    nmarkets = observed_shares.shape[0]

    xi_infty_est = np.zeros_like(observed_shares)
    rcodes = np.zeros(nmarkets, int)
    nevals = np.zeros(nmarkets, int)
    for t in range(nmarkets):
        xi_infty_est[t, :], rcodes[t], nevals[t] = berry_xis_GQ(
            observed_shares[t, :],
            mean_utils[t, :],
            x[t, :],
            xi[t, :],
            s2,
            nodes,
            weights,
            tol=1e-4,
            maxiter=10000,
            verbose=verbose,
        )
        if rcodes[t] != 0:
            print_stars(f"for t = {t}, rcode = {rcodes[t]}")
    if np.any(rcodes != 0):
        print_stars("estimated_xi_infty: problem with Berry inversion")

    return xi_infty_est, rcodes, nevals


def _split_params(pars: np.ndarray, n_x: int) -> ThreeArrays:
    """Split `[beta0, beta_1..beta_M, sigma2_1..sigma2_M]` into its three parts."""
    return pars[:1], pars[1 : 1 + n_x], pars[1 + n_x : 1 + 2 * n_x]


def _true_optimal_instruments(
    true_p: np.ndarray,
    true_mean_utils_xi: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
    X_proj: np.ndarray,
    z: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    mode: str = "NP",
):
    """Optimal instruments for `[beta0, beta, sigma2]` at the true values.

    Args:
        true_p: `(1 + 2M)` true values of the parameters.
        true_mean_utils_xi: `(T, J)` true mean utilities with the product effects.
        observed_shares: `(T, J)` market shares.
        x: `(T, J, M)` covariates.
        X_proj: `(T*J, M)` projected covariates.
        z: `(T, J, M)` instruments.
        nodes: `(L, M)` nodes for Gaussian integration.
        weights: `(L,)` weights for Gaussian integration.
        mode: Projection mode.

    Returns:
        A `(T*J, 1 + 2M)` matrix.
    """
    n_params = true_p.size
    n_x = x.shape[2]
    npts = observed_shares.size
    z_instr = _instruments_matrix(z)

    _, _, s2 = _split_params(true_p, n_x)
    sig_vec = np.array([sqrt_kludge(s2_m) for s2_m in s2])

    Zstar = np.zeros((npts, n_params))
    Zstar[:, 0] = -1.0
    Zstar[:, 1 : 1 + n_x] = -X_proj.reshape((npts, n_x))

    E_stj_stk = _exp_stj_stk(true_mean_utils_xi, x, sig_vec, nodes, weights)
    E_stj_stk_eps = _exp_stj_stk_eps(true_mean_utils_xi, x, sig_vec, nodes, weights)
    E_stj_eps = _exp_stj_eps(true_mean_utils_xi, x, sig_vec, nodes, weights)

    # Mbar = ds/dxi, Nbar = -ds/dsigma
    Mbar = -E_stj_stk
    nproducts = x.shape[1]
    for j in range(nproducts):
        Mbar[:, j, j] += observed_shares[:, j]
    Nbar = -x * E_stj_eps + np.einsum("tjkm,tkm->tjm", E_stj_stk_eps, x)
    dxi_ds = np.linalg.solve(Mbar, Nbar)  # (T, J, M)

    for m in range(n_x):
        Edxi_ds_m = flexible_reg(dxi_ds[:, :, m].reshape(npts), z_instr, mode=mode)
        # we want bounds for sigma**2
        Zstar[:, 1 + n_x + m] = Edxi_ds_m / (2.0 * sig_vec[m])

    return Zstar


def _true_semi_elasticities(
    true_p: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
    true_mean_utils_xi: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
):
    """True semi-elasticities of the share of product 0 with respect to each x_m.

    Args:
        true_p: `(1 + 2M)` true values of `[beta0, beta, sigma2]`.
        observed_shares: `(T, J)` market shares.
        x: `(T, J, M)` covariates.
        true_mean_utils_xi: `(T, J)` true mean utilities with the product effects.
        nodes: `(L, M)` nodes for Gaussian integration.
        weights: `(L,)` weights for Gaussian integration.

    Returns:
        Own and cross semi-elasticities, both `(T, M)`, and the `(T, J, J, M)`
        derivatives of the shares in x.
    """
    nmarkets, nproducts, n_x = x.shape
    _, beta, s2 = _split_params(true_p, n_x)
    sig_vec = np.sqrt(np.maximum(s2, 0.0))
    dshares_dx = _dshares_dx(true_mean_utils_xi, x, beta, sig_vec, nodes, weights)
    observed_shares_0 = observed_shares[:, 0].reshape((-1, 1))
    true_own_semi = dshares_dx[:, 0, 0, :] / observed_shares_0
    true_cross_semi = np.zeros((nmarkets, n_x))
    if nproducts > 1:  # cross semi-elasticity
        true_cross_semi = dshares_dx[:, 0, 1, :] / observed_shares_0
    return true_own_semi, true_cross_semi, dshares_dx


def _nonrandom_semi_elasticities(
    nonrandom_vals: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
):
    """Semi-elasticities of the share of product 0 in the non-random model.

    Args:
        nonrandom_vals: `(1 + M)` estimates of `[beta0, beta]`.
        observed_shares: `(T, J)` market shares.
        x: `(T, J, M)` covariates.

    Returns:
        Own and cross semi-elasticities, both `(T, M)`.
    """
    n_x = x.shape[2]
    beta_0 = nonrandom_vals[1 : 1 + n_x]

    nmarkets, nproducts = observed_shares.shape
    nonrandom_own_semi = np.outer(1.0 - observed_shares[:, 0], beta_0)
    nonrandom_cross_semi = np.zeros((nmarkets, n_x))
    if nproducts > 1:  # cross semi-elasticity
        nonrandom_cross_semi = -np.outer(observed_shares[:, 1], beta_0)

    return nonrandom_own_semi, nonrandom_cross_semi


def _pseudo_semi_elasticities_ift(
    pseudo_vals: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
):
    """Semi-elasticities of the share of product 0 in the approximate model.

    The shares solve `log(s_j/s_0) = beta0 + x_j'beta + sum_m sigma2_m K_jm(s, x) + xi_j`;
    we differentiate this equation by the implicit function theorem:
    `A ds = B_m dx_m` with `A = diag(1/s) + 11'/s_0 + sum_m sigma2_m x_m x_m'`
    and `B_m = beta_m I + sigma2_m (diag(x_m - e_m) - x_m s')`, where `e_m = s'x_m`.

    Args:
        pseudo_vals: `(1 + 2M)` values of `[beta0, beta, sigma2]`.
        observed_shares: `(T, J)` market shares.
        x: `(T, J, M)` covariates.

    Returns:
        Own and cross semi-elasticities, both `(T, M)`.
    """
    nmarkets, nproducts, n_x = x.shape
    _, beta, s2 = _split_params(pseudo_vals, n_x)

    shares = observed_shares
    outside_shares = 1.0 - np.sum(shares, 1)
    A = np.einsum("tjm,tkm,m->tjk", x, x, s2)
    A += (1.0 / outside_shares).reshape((-1, 1, 1))
    for j in range(nproducts):
        A[:, j, j] += 1.0 / shares[:, j]
    # A is symmetric: the first row of A^{-1} solves A a0 = e_0
    e_0 = np.zeros((nmarkets, nproducts))
    e_0[:, 0] = 1.0
    a0 = np.linalg.solve(A, e_0[:, :, np.newaxis])[:, :, 0]  # (T, J)

    e_S_x = np.einsum("tj,tjm->tm", shares, x)
    a0_x = np.einsum("tj,tjm->tm", a0, x)
    # row 0 of A^{-1} B_m, column k
    D0 = np.einsum("tk,m->tkm", a0, beta) + s2 * (
        a0[:, :, np.newaxis] * (x - e_S_x[:, np.newaxis, :])
        - a0_x[:, np.newaxis, :] * shares[:, :, np.newaxis]
    )

    shares_0 = shares[:, 0].reshape((-1, 1))
    pseudo_own_semi = D0[:, 0, :] / shares_0
    pseudo_cross_semi = np.zeros((nmarkets, n_x))
    if nproducts > 1:  # cross semi-elasticity
        pseudo_cross_semi = D0[:, 1, :] / shares_0

    return pseudo_own_semi, pseudo_cross_semi


def _mean_squared_residuals(lhs: np.ndarray, regressors: np.ndarray | None = None):
    """Add a constant and compute the mean squared residual.

    Args:
        lhs: `n`-vector of the dependent variable.
        regressors: A matrix with `n` rows; if `None`, the variance of `lhs`
            is returned.

    Returns:
        The mean squared residual of lhs regressed on regressors and a
        constant.
    """
    check_vector(lhs, "_mean_squared_residuals")
    if regressors is None:
        return np.var(lhs)
    _ = check_vector_or_matrix(regressors, "_mean_squared_residuals")
    nobs = lhs.size
    if nobs != regressors.shape[0]:
        bs_error_abort(
            f"lhs has {nobs} observations, while regressors has {regressors.shape[0]}"
        )
    regs = np.column_stack((np.ones(nobs), regressors))
    coeffs, _, _, _ = cast(Iterable, spla.lstsq(regs, lhs))
    resid = lhs - regs @ coeffs
    return np.dot(resid, resid) / nobs

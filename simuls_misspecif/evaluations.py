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
    nprepeat_row,
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
        x: Covariates.
        J: Number of products.

    Returns:
        Artificial regressors of order 2 and 4.
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


def _project_variables(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    K: np.ndarray,
    V: np.ndarray | None = None,
    W: np.ndarray | None = None,
    mode: str = "NP",
) -> (
    ThreeArrays
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None
):
    """Project the variables onto z.

    Args:
        y: Should be `(T, J)`.
        x: Should be `(T, J)`.
        z: Should be `(T, J)`.
        K: Should be `(T, J)`.
        V: Should be `(T, J)` if not None.
        W: Should be `(T, J)` if not None.
        mode: Projection mode without a micromoment. Default: `NP`.

    Returns:
        The projections of the 3, 4, or 5 variables as `T`-vectors.
    """
    npts = y.size
    Kvec = K.reshape(npts)
    xvec = x.reshape(npts)
    yvec = y.reshape(npts)
    zvec = z.reshape(npts)

    y_proj = _projection_instruments(yvec, zvec, mode=mode)
    x_proj = _projection_instruments(xvec, zvec, mode=mode)
    K_proj = _projection_instruments(Kvec, zvec, mode=mode)

    match (V, W):
        case (None, None):
            return y_proj, x_proj, K_proj
        case (np.ndarray, None):
            V = cast(np.ndarray, V)
            Vvec = V.reshape(npts)
            V_proj = _projection_instruments(Vvec, zvec, mode=mode)
            return y_proj, x_proj, K_proj, V_proj
        case (None, np.ndarray):
            W = cast(np.ndarray, W)
            Wvec = W.reshape(npts)
            W_proj = _projection_instruments(Wvec, zvec, mode=mode)
            return y_proj, x_proj, K_proj, W_proj
        case (np.ndarray, np.ndarray):
            V = cast(np.ndarray, V)
            W = cast(np.ndarray, W)
            Vvec = V.reshape(npts)
            V_proj = _projection_instruments(Vvec, zvec, mode=mode)
            Wvec = W.reshape(npts)
            W_proj = _projection_instruments(Wvec, zvec, mode=mode)
            return y_proj, x_proj, K_proj, V_proj, W_proj

    return None


def _reshape_proj(var_proj: np.ndarray, nproducts: int) -> np.ndarray:
    nmarkets = var_proj.size // nproducts
    if var_proj.ndim == 2:
        v_proj = var_proj[:, 0]
        return v_proj.reshape((nmarkets, nproducts))
    else:
        return var_proj.reshape((nmarkets, nproducts))


def _our_tsls0(
    y_proj: np.ndarray,
    x_proj: np.ndarray,
):
    #  the "optimal" instruments will be in Zstar0
    npts = y_proj.size
    Zstar0 = np.zeros((npts, 2))
    Zstar0[:, 0] = np.ones(npts)
    Zstar0[:, 1] = x_proj

    nonrandom_vals, _, _, s = spla.lstsq(Zstar0, y_proj)
    cond_number = abs(s[0] / s[-1])
    return Zstar0, nonrandom_vals, cond_number


def _our_tsls2(
    y_proj: np.ndarray,
    x_proj: np.ndarray,
    K_proj: np.ndarray,
):
    #  the "optimal" instruments will be in Zstar2
    npts = y_proj.size
    Zstar2 = np.zeros((npts, 3))
    Zstar2[:, 0] = np.ones(npts)
    Zstar2[:, 1] = x_proj
    Zstar2[:, -1] = K_proj

    pseudo_vals, _, _, s = spla.lstsq(Zstar2, y_proj)
    cond_number = abs(s[0] / s[-1])
    return Zstar2, pseudo_vals, cond_number


def _reformat_Zstar(Zstar: np.ndarray, nproducts: int) -> np.ndarray:
    """reformats the optimal instruments
    from a `(T*J, n_instr)` matrix to a `(J, n_instr, T)` matrix.

    Args:
        Zstar: The `(T*J, n_instr)` matrix of optimal instruments, rows in
            market-major order: row ``t*J + j`` corresponds to market ``t``,
            product ``j``.
        nproducts: The number of products `J`.

    Returns:
        The `(J, n_instr, T)` array of optimal instruments.
    """
    nmarkets = Zstar.shape[0] // nproducts
    n_instr = Zstar.shape[1]
    return Zstar.reshape(nmarkets, nproducts, n_instr).transpose(1, 2, 0)


def _reformat_varcov(v: np.ndarray) -> np.ndarray:
    """Reformat a variance-covariance matrix from the order of variables to the order of markets.

    Args:
        v: The `(J, n_instr, J, n_instr)` variance-covariance array.

    Returns:
        The `(J*n_instr, J*n_instr)` variance-covariance matrix.
    """
    J, n_instr, _, _ = v.shape
    m = J * n_instr
    v2 = np.zeros((m, m))
    for j in range(J):
        for j2 in range(J):
            block = v[j, :, j2, :]
            v2[j * n_instr : (j + 1) * n_instr, j2 * n_instr : (j2 + 1) * n_instr] = (
                block
            )
    return v2


def _print_pseudo_true_errors(
    true_p: np.ndarray,
    pseudo_vals: np.ndarray,
    names_ptv: List[str],
    verbose: bool = False,
):
    n_params = true_p.size
    if n_params == 4:  # we have a micromoment
        _, _, sig2 = true_p
        if verbose:
            print_stars(f"Pseudo-true errors for true sigma2={sig2: 10.4f}:")
    else:
        sig2 = true_p[2]
        if verbose:
            print_stars(f"Pseudo-true errors for true sigma2={sig2: 10.4f}:")
    if verbose:
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


def _true_optimal_instruments(
    true_p: np.ndarray,
    true_mean_utils_xi: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
    x_proj: np.ndarray,
    z: np.ndarray,
    nodes1: np.ndarray,
    weights1: np.ndarray,
    mode: str = "NP",
):
    n_params = true_p.size

    nmarkets, nproducts = observed_shares.shape
    npts = observed_shares.size
    shares_vec = observed_shares.reshape(npts)
    xvec = x.reshape(npts)
    zvec = z.reshape(npts)

    s2 = true_p[-1]
    sigma_val = sqrt_kludge(s2)

    Zstar = np.zeros((npts, n_params))
    Zstar[:, 0] = -1.0
    Zstar[:, 1] = -x_proj

    sig_val = sqrt_kludge(s2)
    E_stj_stk = _exp_stj_stk(true_mean_utils_xi, x, sig_val, nodes1, weights1)
    E_stj_stk_eps = _exp_stj_stk_eps(true_mean_utils_xi, x, sig_val, nodes1, weights1)
    E_stj_eps = _exp_stj_eps(true_mean_utils_xi, x, sig_val, nodes1, weights1).reshape(
        npts
    )
    dxi_ds = np.zeros(npts)

    Nbar = -xvec * E_stj_eps
    start_t = 0
    for t in range(nmarkets):
        end_t = start_t + nproducts
        sli_t = slice(start_t, end_t)
        Mbar_t = np.diag(shares_vec[sli_t])
        Mbar_t -= E_stj_stk[t, :, :]
        Nbar_t = Nbar[sli_t] + (E_stj_stk_eps[t, :, :] @ xvec[sli_t])
        dxi_ds[sli_t] = spla.solve(Mbar_t, Nbar_t, assume_a="sym")
        start_t = end_t

    Edxi_ds = flexible_reg(dxi_ds, zvec, mode=mode)
    # we want bounds for sigma**2
    Zstar[:, 2] = Edxi_ds / (2.0 * sigma_val)

    return Zstar


def _true_semi_elasticities(
    true_p: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
    true_mean_utils_xi: np.ndarray,
    nodes1: np.ndarray,
    weights1: np.ndarray,
):
    nmarkets, nproducts = observed_shares.shape
    dshares_dx = _dshares_dx(true_mean_utils_xi, x, true_p, nodes1, weights1)
    observed_shares_0 = observed_shares[:, 0]
    dsh_dx0 = dshares_dx[:, 0] if dshares_dx.ndim == 2 else dshares_dx[:, 0, 0]
    true_own_semi = dsh_dx0 / observed_shares_0
    true_cross_semi = np.zeros(nmarkets)
    if nproducts > 1:  # cross semi-elasticity
        true_cross_semi = dshares_dx[:, 0, 1] / observed_shares_0
    return true_own_semi, true_cross_semi, dshares_dx


def _nonrandom_semi_elasticities(
    nonrandom_vals: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
):
    beta1_0 = nonrandom_vals[1]

    nmarkets, nproducts = observed_shares.shape
    nonrandom_own_semi = np.zeros(nmarkets)
    nonrandom_cross_semi = np.zeros(nmarkets)

    for t in range(nmarkets):
        sh_t = observed_shares[t, :]
        # own semi-elasticity
        nonrandom_own_semi[t] = beta1_0 * (1.0 - sh_t[0])
        if nproducts > 1:  # cross semi-elasticity
            nonrandom_cross_semi[t] = -beta1_0 * sh_t[1]

    return nonrandom_own_semi, nonrandom_cross_semi


def _pseudo_semi_elasticities(
    pseudo_vals: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
):
    beta1_2 = pseudo_vals[1]
    s2 = pseudo_vals[-1]

    nmarkets, nproducts = observed_shares.shape
    pseudo_own_semi = np.zeros(nmarkets)
    pseudo_cross_semi = np.zeros(nmarkets)
    e_S_x = np.sum(observed_shares * x, 1)
    observed_shares0 = 1.0 - np.sum(observed_shares, 1)
    exp_y = observed_shares / observed_shares0.reshape((-1, 1))

    for t in range(nmarkets):
        x_t, sh_t, e_t, exp_y_t = (
            x[t, :],
            observed_shares[t, :],
            e_S_x[t],
            exp_y[t, :],
        )
        dK_dx_t = -np.outer(x_t, sh_t)
        dK_dx_t += np.diag(x_t - e_t)
        lhs_mat = nprepeat_row(exp_y_t, nproducts)
        lhs_mat += np.eye(nproducts)
        rhs_mat = s2 * dK_dx_t
        rhs_mat += beta1_2 * np.eye(nproducts)
        semi_elast_t = spla.solve(lhs_mat, rhs_mat)
        # own semi-elasticity
        pseudo_own_semi[t] = semi_elast_t[0, 0]
        if nproducts > 1:  # cross semi-elasticity
            pseudo_cross_semi[t] = semi_elast_t[0, 1]

    return pseudo_own_semi, pseudo_cross_semi


def _pseudo_semi_elasticities_anal(
    pseudo_vals: np.ndarray,
    observed_shares: np.ndarray,
    x: np.ndarray,
):
    beta1_2 = pseudo_vals[1]
    s2 = pseudo_vals[-1]
    s4 = s2 * s2

    nmarkets, nproducts = observed_shares.shape
    pseudo_own_semi = np.zeros(nmarkets)
    pseudo_cross_semi = np.zeros(nmarkets)
    e_S_x = np.sum(observed_shares * x, 1)
    v_S_x = np.sum(observed_shares * x * x, 1) - e_S_x * e_S_x
    tilde_x = x - e_S_x.reshape((-1, 1))

    for t in range(nmarkets):
        tilde_x_t, sh_t, e_t, v_t = (
            tilde_x[t, :],
            observed_shares[t, :],
            e_S_x[t],
            v_S_x[t],
        )
        denom_t = 1.0 + s2 * v_t
        txt0 = tilde_x_t[0]
        sht0 = sh_t[0]
        dvt0 = v_t - txt0 * e_t
        pseudo_own_semi[t] = (
            beta1_2 * (1.0 - sht0)
            + s2 * txt0
            - s2 * sht0 * (2.0 * txt0 + beta1_2 * dvt0) / denom_t
            - s4 * sht0 * (2.0 * txt0 * v_t - (txt0 + v_t) * e_t) / denom_t
        )
        if nproducts > 1:  # cross semi-elasticity
            sht1 = sh_t[1]
            txt1 = tilde_x_t[1]
            dvt1 = v_t - txt1 * e_t
            pseudo_cross_semi[t] = (
                -beta1_2 * sht1
                - s2 * sht1 * (txt0 + txt1 + beta1_2 * dvt1) / denom_t
                - s4 * sht1 * ((txt0 + txt1) * v_t - (txt1 + v_t) * e_t) / denom_t
            )

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

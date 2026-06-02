"""evaluating and printing the various stats we collect in the simulations"""

from math import sqrt
from typing import Iterable, List, Tuple, cast

import numpy as np
import scipy.linalg as spla
from blp_utils import (  # type: ignore[import-not-found]
    make_K_T1,
    make_QW_T1,
    simulated_mean_shares,
)
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

from simuls_misspecif.MNL_integrals import (
    _dshares_dx,
    _exp_stj_eps,
    _exp_stj_stk,
    _exp_stj_stk_eps,
)


def sqrt_kludge(sq):
    return sqrt(max(sq, 1e-9))


def berry_xis(
    shares: np.ndarray,
    mean_u: np.ndarray,
    x: np.ndarray,
    xi: np.ndarray,
    s2: float,
    tol: float = 1e-9,
    maxiter: int = 10000,
    ndraws: int = 10000,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Invert product effects xi from market shares on one market.

    Args:
        shares: `nproducts` vector of observed shares.
        mean_u: `nproducts` vector of mean utilities.
        x: `nproducts` vector of covariates.
        xi: `nproducts` vector of an initial estimate of xi.
        s2: Variance of the random coefficient.
        tol: Tolerance.
        maxiter: Maximum number of iterations.
        ndraws: Number of draws for simulation.
        verbose: Whether to print progress information.

    Returns:
        A tuple containing the estimated xi vector, the return code, and the
        number of evaluations.
    """

    s = sqrt_kludge(s2)
    xi_cur = xi.copy()
    max_err = np.inf
    retcode = 0
    iter = 0
    eps = np.random.normal(size=ndraws)
    while max_err > tol:
        utils = s * np.outer(x, eps) + (mean_u + xi_cur).reshape((-1, 1))
        shares_sim = simulated_mean_shares(utils)
        err_shares = np.log(shares) - np.log(shares_sim)
        max_err = npmaxabs(err_shares)
        if verbose and iter % 1000 == 1:
            print(f"berry_xis: error {max_err} after {iter - 1} iterations")
        xi_cur += err_shares
        iter += 1
        if iter > maxiter:
            print_stars(
                f"berry_xis: stuck with error {max_err} after {iter} iterations"
            )
            retcode = 1
            break
    if verbose:
        print_stars(f"berry_xis: error {max_err} after {iter} iterations")
    return xi_cur, retcode, iter


def _integrand_shares(values, pars):
    sx, mean_u_xi_cur = pars
    utils = np.outer(values, sx) + mean_u_xi_cur
    max_utils = np.max(utils, 1)
    dutils = utils - max_utils.reshape((-1, 1))
    exp_d = npexp(dutils)
    denom = np.sum(exp_d, 1) + npexp(-max_utils)
    shares = exp_d.T / denom
    return shares


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


def _artificial_regressors(observed_shares: np.ndarray, x: np.ndarray):
    """Compute the artificial regressors at order 2 and 4.

    Args:
        observed_shares: Observed market shares.
        x: Covariates.

    Returns:
        Artificial regressors of order 2 and 4.
    """
    K = make_K_T1(x, observed_shares)
    Q, W = make_QW_T1(x, observed_shares)
    return K, Q, W


def _make_quadratic_instruments(z: np.ndarray, Dbar: np.ndarray | None = None):
    """Build quadratic instruments.

    Args:
        z: The `(T, J)` matrix of instruments.
        Dbar: The micromoment market means, if any.

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
    var: np.ndarray, z_instruments: np.ndarray, mode1: str = "NP", mode2: str = "NP"
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
    if n_z == 1:
        return flexible_reg(var, z_instruments, mode=mode1)
    else:
        return flexible_reg(var, z_instruments, mode=mode2)


def _project_variables(
    y: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    Q: np.ndarray,
    W: np.ndarray,
    z: np.ndarray,
    Dbar: np.ndarray | None = None,
    mode1: str = "NP",
    mode2: str = "NP",
):
    """Project the variables onto z, and onto Dbar when present.

    Args:
        y: Should be `(T, J)`.
        x: Should be `(T, J)`.
        K: Should be `(T, J)`.
        Q: Should be `(T, J)`.
        W: Should be `(T, J)`.
        z: Should be `(T, J)`.
        Dbar: If there is a micromoment, its mean as a `T`-vector.
        mode1: Projection mode without a micromoment.
        mode2: Projection mode with a micromoment.

    Returns:
        The projections of the five variables as `T`-vectors.
    """
    nmarkets, nproducts = y.shape
    npts = y.size
    Kvec = K.reshape(npts)
    xvec = x.reshape(npts)
    yvec = y.reshape(npts)
    zvec = z.reshape(npts)
    Qvec = Q.reshape(npts)
    Wvec = W.reshape(npts)

    if Dbar is None:  # no micromoment
        y_proj = _projection_instruments(yvec, zvec, mode1=mode1)
        x_proj = _projection_instruments(xvec, zvec, mode1=mode1)
        K_proj = _projection_instruments(Kvec, zvec, mode1=mode1)
        Q_proj = _projection_instruments(Qvec, zvec, mode1=mode1)
        W_proj = _projection_instruments(Wvec, zvec, mode1=mode1)
    else:  # we have a micromoment
        Dbarvec = np.repeat(Dbar, nproducts)
        zDbar = np.column_stack((zvec, Dbarvec))
        y_proj = _projection_instruments(yvec, zDbar, mode2=mode2)
        x_proj = _projection_instruments(xvec, zDbar, mode2=mode2)
        K_proj = _projection_instruments(Kvec, zDbar, mode2=mode2)
        Q_proj = _projection_instruments(Qvec, zDbar, mode2=mode2)
        W_proj = _projection_instruments(Wvec, zDbar, mode2=mode2)

    return y_proj, x_proj, K_proj, Q_proj, W_proj


def _reshape_proj(var_proj: np.ndarray, nproducts: int) -> np.ndarray:
    nmarkets = var_proj.size // nproducts
    if var_proj.ndim == 2:
        v_proj = var_proj[:, 0]
        return v_proj.reshape((nmarkets, nproducts))
    else:
        return var_proj.reshape((nmarkets, nproducts))


def _our_tsls(
    y_proj: np.ndarray,
    x_proj: np.ndarray,
    K_proj: np.ndarray,
    Dbar: np.ndarray | None = None,
):
    #  the "optimal" instruments will be in Zstar2
    npts = y_proj.size
    if Dbar is None:
        Zstar2 = np.zeros((npts, 3))
    else:  # we have a micromoment
        Zstar2 = np.zeros((npts, 4))
        nproducts = y_proj.size // Dbar.size
        Dbarvec = np.repeat(Dbar, nproducts)
        Zstar2[:, 2] = x_proj * Dbarvec
    Zstar2[:, 0] = np.ones(npts)
    Zstar2[:, 1] = x_proj
    Zstar2[:, -1] = K_proj

    pseudo_vals, _, _, s = spla.lstsq(Zstar2, y_proj)
    cond_number = abs(s[0] / s[-1])
    return Zstar2, pseudo_vals, cond_number


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


def _print_semi_elast(
    mean_pseudo_own_semi_elast,
    stderr_pseudo_own_semi_elast,
    mean_true_own_semi_elast,
    stderr_true_own_semi_elast,
    mean_corrected_own_semi_elast,
    stderr_corrected_own_semi_elast,
    mean_pseudo_cross_semi_elast=None,
    stderr_pseudo_cross_semi_elast=None,
    mean_true_cross_semi_elast=None,
    stderr_true_cross_semi_elast=None,
    mean_corrected_cross_semi_elast=None,
    stderr_corrected_cross_semi_elast=None,
    verbose=False,
):
    do_cross = False if mean_pseudo_cross_semi_elast is None else True
    if do_cross:
        resus_pseudo_semi_elast = np.array(
            [
                mean_pseudo_own_semi_elast,
                stderr_pseudo_own_semi_elast,
                mean_pseudo_cross_semi_elast,
                stderr_pseudo_cross_semi_elast,
            ]
        )
        resus_true_semi_elast = np.array(
            [
                mean_true_own_semi_elast,
                stderr_true_own_semi_elast,
                mean_true_cross_semi_elast,
                stderr_true_cross_semi_elast,
            ]
        )
        resus_corrected_semi_elast = np.array(
            [
                mean_corrected_own_semi_elast,
                stderr_corrected_own_semi_elast,
                mean_corrected_cross_semi_elast,
                stderr_corrected_cross_semi_elast,
            ]
        )
    else:
        resus_pseudo_semi_elast = np.array(
            [mean_pseudo_own_semi_elast, stderr_pseudo_own_semi_elast]
        )
        resus_true_semi_elast = np.array(
            [mean_true_own_semi_elast, stderr_true_own_semi_elast]
        )
        resus_corrected_semi_elast = np.array(
            [mean_corrected_own_semi_elast, stderr_corrected_own_semi_elast]
        )

    if verbose:
        print_stars("True semi-elasticities")
        print(
            f"   own: mean = {mean_true_own_semi_elast: 10.3f} and stderr = {stderr_true_own_semi_elast: 10.3f}"
        )
        if do_cross:
            print(
                f"   cross: mean = {mean_true_cross_semi_elast: > 10.3f} and stderr = {stderr_true_cross_semi_elast: 10.3f}"
            )
        print_stars("Pseudo semi-elasticities")
        print(
            f"   own: mean = {mean_pseudo_own_semi_elast: 10.3f} and stderr = {stderr_pseudo_own_semi_elast: 10.3f}"
        )
        if do_cross:
            print(
                f"   cross: mean = {mean_pseudo_cross_semi_elast: > 10.3f} and stderr = {stderr_pseudo_cross_semi_elast: 10.3f}"
            )
        print_stars("Corrected semi-elasticities")
        print(
            f"   own: mean = {mean_corrected_own_semi_elast: 10.3f} and stderr = {stderr_corrected_own_semi_elast: 10.3f}"
        )
        if do_cross:
            print(
                f"   cross: mean = {mean_corrected_cross_semi_elast: > 10.3f} and stderr = {stderr_corrected_cross_semi_elast: 10.3f}"
            )

    return resus_pseudo_semi_elast, resus_true_semi_elast, resus_corrected_semi_elast


def _newton_raphson_step(
    Q_proj: np.ndarray,
    W_proj: np.ndarray,
    xi_2: np.ndarray,
    Zstar2: np.ndarray,
    pseudo_vals: np.ndarray,
    true_p: np.ndarray,
    verbose: bool = False,
):
    s2_2 = pseudo_vals[-1]
    n_params = true_p.size

    sig2_2 = s2_2
    sig2 = sig2_2
    s4_2 = s2_2 * s2_2

    Zstar2_T = Zstar2.T
    xi2_T = xi_2.T
    ZZ = Zstar2_T @ Zstar2

    f_val = s4_2
    f_der_s2 = 2.0 * s2_2
    add_term_Q = np.zeros(n_params)
    add_term_Q[-1] = f_der_s2
    add_term_W = np.zeros(n_params)
    add_term_W[-1] = 2.0 * s2_2
    xi_prime_Q = xi2_T @ Q_proj
    xi_prime_W = xi2_T @ W_proj
    d4 = spla.solve(ZZ, xi_prime_Q * add_term_Q - f_val * (Zstar2_T @ Q_proj))
    dp4 = spla.solve(ZZ, xi_prime_W * add_term_W - s4_2 * (Zstar2_T @ W_proj))

    if verbose:
        print_stars(f"Correction d4 for sigma2={sig2: 10.4f}:")
        print(d4)
        print_stars(f"Correction dp4 for sigma2={sig2: 10.4f}:")
        print(dp4)

    return d4, dp4


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
    mode1: str = "NP",
    mode2: str = "NP",
    quad_instr: np.ndarray | None = None,
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

    if mode1 == "2":
        coeffs, _, _, _ = spla.lstsq(quad_instr, dxi_ds)
        Edxi_ds = quad_instr @ coeffs
    else:
        Edxi_ds = flexible_reg(dxi_ds, zvec, mode=mode1)
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


def _pseudo_semi_elasticities(
    Dbar: np.ndarray,
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

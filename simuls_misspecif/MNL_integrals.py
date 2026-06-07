"""Compute market-share integrals and derivatives for one normal random coefficient."""

import numpy as np
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from bs_python_utils.bsnputils import npmaxabs
from bs_python_utils.bsutils import bs_error_abort, print_stars


def _exp_stj(
    mean_utils: np.ndarray,
    x: np.ndarray,
    sig_val: float,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Evaluate E(s_tj).

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        sig_val: Standard deviation of the random coefficient.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J)` matrix.
    """
    if mean_utils.ndim != 2:
        bs_error_abort("mean_utils should be (T, J)")
    if x.ndim != 2:
        bs_error_abort("x should be (T, J)")
    if x.shape != mean_utils.shape:
        bs_error_abort("x and mean_utils should both be (T, J)")

    nmarkets, nproducts = x.shape
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    sig_x = sig_val * x
    for ll, node_l in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node_l)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, ll] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
    Estj = s_tjl @ weights1
    return Estj  # type: ignore[no-any-return]


def _exp_stj_eps(
    mean_utils: np.ndarray,
    x: np.ndarray,
    sig_val: float,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Evaluate E(s_tj epsilon).

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        sig_val: Standard deviation of the random coefficient.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J)` matrix.
    """
    if mean_utils.ndim != 2:
        bs_error_abort("mean_utils should be (T, J)")
    if x.ndim != 2:
        bs_error_abort("x should be (T, J)")
    if x.shape != mean_utils.shape:
        bs_error_abort("x and mean_utils should both be (T, J)")

    nmarkets, nproducts = x.shape
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    sig_x = sig_val * x
    for ll, node_l in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node_l)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, ll] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
    Estj_eps = s_tjl @ (nodes1 * weights1)
    return Estj_eps  # type: ignore[no-any-return]


def _exp_stj_stk(
    mean_utils: np.ndarray,
    x: np.ndarray,
    sig_val: float,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Return E(s_tj s_tk).

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        sig_val: Standard deviation of the random coefficient.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J, J)` array.
    """
    if mean_utils.ndim != 2:
        bs_error_abort("mean_utils should be (T, J)")
    if x.ndim != 2:
        bs_error_abort("x should be (T, J)")
    if x.shape != mean_utils.shape:
        bs_error_abort("x and mean_utils should both be (T, J)")

    nmarkets, nproducts = x.shape
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    sig_x = sig_val * x
    for ll, node_l in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node_l)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, ll] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
    Estjstk = np.einsum("tjl, tkl, l->tjk", s_tjl, s_tjl, weights1)
    return Estjstk  # type: ignore[no-any-return]


def _exp_stj_stk_eps(
    mean_utils: np.ndarray,
    x: np.ndarray,
    sig_val: float,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Return E(s_tj s_tk epsilon).

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        sig_val: Standard deviation of the random coefficient.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J, J)` array.
    """
    if mean_utils.ndim != 2:
        bs_error_abort("mean_utils should be (T, J)")
    if x.ndim != 2:
        bs_error_abort("x should be (T, J)")
    if x.shape != mean_utils.shape:
        bs_error_abort("x and mean_utils should both be (T, J)")

    nmarkets, nproducts = x.shape
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    sig_x = sig_val * x
    for ll, node_l in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node_l)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, ll] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
    Estjstk_eps = np.einsum("tjl, tkl, l->tjk", s_tjl, s_tjl, nodes1 * weights1)
    return Estjstk_eps  # type: ignore[no-any-return]


def _dshares_dx(
    mean_utils: np.ndarray,
    x: np.ndarray,
    true_p: np.ndarray,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Compute derivatives of the market shares with respect to x.

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        true_p: The true values of the parameters.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J, J)` array.
    """
    nproducts = x.shape[1]
    beta1 = true_p[1]
    s_val = true_p[-1]
    Estj = _exp_stj(mean_utils, x, s_val, nodes1, weights1)
    Estj_eps = _exp_stj_eps(mean_utils, x, s_val, nodes1, weights1)
    Estjstk = _exp_stj_stk(mean_utils, x, s_val, nodes1, weights1)
    Estjstk_eps = _exp_stj_stk_eps(mean_utils, x, s_val, nodes1, weights1)

    # dsh_dx = -np.einsum("tjk, t->tjk", Estjstk, beta1) - s_val * Estjstk_eps
    dsh_dx = -Estjstk * beta1 - s_val * Estjstk_eps
    diag_plus = Estj * beta1 + s_val * Estj_eps

    for j in range(nproducts):
        dsh_dx[:, j, j] += diag_plus[:, j]

    return dsh_dx  # type: ignore[no-any-return]


def _dshares_dtheta(
    mean_utils: np.ndarray,
    x: np.ndarray,
    true_p: np.ndarray,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Compute derivatives of the market shares with respect to theta.

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        true_p: The true values of the parameters.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J, 3)` array with derivatives in beta0, beta1, and s.
    """
    nmarkets, nproducts = x.shape
    dsh_dth = np.zeros((nmarkets, nproducts, 3))
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    s_t0l = np.zeros((nmarkets, n_nodes))
    x_diff_esx = np.zeros((nmarkets, nproducts, n_nodes))
    sig_tot = true_p[-1]
    sig_x = sig_tot * x
    for inode, node in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, inode] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
        s_t0l[:, inode] = 1.0 - np.sum(s_tjl[:, :, inode], 1)
        esh_x_l = np.sum(s_tjl[:, :, inode] * x, 1)
        x_diff_esx[:, :, inode] = x - esh_x_l.reshape((-1, 1))

    dsh_dth[:, :, 0] = np.einsum("tjl, tl, l->tj", s_tjl, s_t0l, weights1)
    dsh_dth[:, :, 1] = np.einsum("tjl, tjl, l->tj", s_tjl, x_diff_esx, weights1)
    dsh_dth[:, :, 2] = np.einsum(
        "tjl, tjl, l->tj", s_tjl, x_diff_esx, nodes1 * weights1
    )

    return dsh_dth


def _d2shares_dx_dtheta(
    mean_utils: np.ndarray,
    x: np.ndarray,
    true_p: np.ndarray,
    nodes1: np.ndarray,
    weights1: np.ndarray,
) -> np.ndarray:
    """Compute cross-derivatives of the market shares with respect to x and theta.

    Args:
        mean_utils: A `(T, J)` matrix.
        x: A `(T, J)` matrix.
        true_p: The true values of the parameters.
        nodes1: Nodes for 1d Gauss-Hermite integration.
        weights1: Weights for 1d Gauss-Hermite integration.

    Returns:
        A `(T, J, J, 3)` array with derivatives in beta0, beta1, and s.
    """
    beta1 = true_p[1]
    s_val = true_p[-1]
    nmarkets, nproducts = x.shape

    d2sh_dx_dth = np.zeros((nmarkets, nproducts, nproducts, 3))
    n_nodes = nodes1.size
    s_tjl = np.zeros((nmarkets, nproducts, n_nodes))
    s_t0l = np.zeros((nmarkets, n_nodes))
    dsh_dx = np.zeros((nmarkets, nproducts, nproducts, n_nodes))
    dsh0_dx = np.zeros((nmarkets, nproducts, n_nodes))
    desh_dx = np.zeros((nmarkets, nproducts, n_nodes))
    x_diff_esx = np.zeros((nmarkets, nproducts, n_nodes))
    dx_diff_esx = np.zeros((nmarkets, nproducts, nproducts, n_nodes))
    sig_x = s_val * x
    for inode, node in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_x * node)
        s_tl = np.sum(s_tjl_l, 1)
        s_tjl[:, :, inode] = s_tjl_l / (1.0 + s_tl.reshape((-1, 1)))
        s_t0l[:, inode] = 1.0 - np.sum(s_tjl[:, :, inode], 1)
        b1_l = beta1 + s_val * node
        for j in range(nproducts):
            sh_jl = s_tjl[:, j, inode]
            dsh_dx[:, j, :, inode] = -(s_tjl[:, :, inode] * sh_jl.reshape((-1, 1)))
            dsh_dx[:, j, j, inode] += sh_jl
            dsh_dx[:, j, :, inode] *= b1_l.reshape((-1, 1))
        dsh0_dx[:, :, inode] = -np.sum(dsh_dx[:, :, :, inode], 1)
        esh_x_l = np.sum(s_tjl[:, :, inode] * x, 1)
        x_diff_esx[:, :, inode] = x - esh_x_l.reshape((-1, 1))
        for k in range(nproducts):
            desh_dx[:, k, inode] = s_tjl[:, k, inode] + np.sum(
                dsh_dx[:, :, k, inode] * x, 1
            )
            dx_diff_esx[:, :, k, inode] = -desh_dx[:, k, inode].reshape((-1, 1))
            dx_diff_esx[:, k, k, inode] += 1.0

    d2sh_dx_dth[:, :, :, 0] = np.einsum(
        "tl, tjkl, l->tjk", s_t0l, dsh_dx, weights1
    ) + np.einsum("tjl, tkl, l->tjk", s_tjl, dsh0_dx, weights1)
    d2sh_dx_dth[:, :, :, 1] = np.einsum(
        "tjkl, tjl, l->tjk", dsh_dx, x_diff_esx, weights1
    ) + np.einsum("tjl, tjkl, l->tjk", s_tjl, dx_diff_esx, weights1)
    d2sh_dx_dth[:, :, :, 2] = np.einsum(
        "tjkl, tjl, l->tjk", dsh_dx, x_diff_esx, nodes1 * weights1
    ) + np.einsum("tjl, tjkl, l->tjk", s_tjl, dx_diff_esx, nodes1 * weights1)
    return d2sh_dx_dth


if __name__ == "__main__":
    n = 1
    iprec = 17
    nodes1, weights1 = setup_sparse_gaussian(n, iprec)

    true_p = np.array([0.2, 0.4, 0.5])

    tau = 0.8
    b0, b1, sigma = true_p

    nmarkets, nproducts = (10, 5)
    x = np.random.normal(size=(nmarkets, nproducts))
    xi = np.random.normal(size=(nmarkets, nproducts))
    mean_utils = b0 + x * b1 + xi
    shares = _exp_stj(mean_utils, x, sigma, nodes1, weights1)

    dsh_dx = _dshares_dx(mean_utils, x, true_p, nodes1, weights1)
    dsh_dth = _dshares_dtheta(mean_utils, x, true_p, nodes1, weights1)
    d2sh_dx_dth = _d2shares_dx_dtheta(mean_utils, x, true_p, nodes1, weights1)

    d2sh_dx_dth_tjc = np.zeros_like(d2sh_dx_dth)

    EPS = 1e-6

    for t in range(nmarkets):
        for j in range(nproducts):
            xtj = x.copy()
            xtj[t, j] += EPS
            mean_utils_tj = b0 + xtj * b1 + xi
            shares_tj = _exp_stj(mean_utils_tj, xtj, sigma, nodes1, weights1)
            d_shares_tj = (shares_tj[t, :] - shares[t, :]) / EPS
            print_stars(
                f"max error on dsh_dx[{t}, :, {j}]: {npmaxabs(d_shares_tj - dsh_dx[t, :, j])}"
            )

    shares_c = np.zeros((3, nmarkets, nproducts))
    for c in range(3):
        pars_c = true_p.copy()
        pars_c[c] += EPS
        mean_utils_c = pars_c[0] + x * pars_c[1] + xi
        sig_val_c = pars_c[-1]
        shares_c[c, :, :] = _exp_stj(mean_utils_c, x, sig_val_c, nodes1, weights1)
        d_shares_c = (shares_c[c, :, :] - shares) / EPS
        print_stars(
            f"max error on dsh_dth[:, :, {c}]: {npmaxabs(d_shares_c - dsh_dth[:, :, c])}"
        )

    for t in range(nmarkets):
        for j in range(nproducts):
            xtj = x.copy()
            xtj[t, j] += EPS
            mean_utils_tj = b0 + xtj * b1 + xi
            shares_tj = _exp_stj(mean_utils_tj, xtj, sigma, nodes1, weights1)
            for c in range(3):
                pars_c = true_p.copy()
                pars_c[c] += EPS
                sig_val_c = pars_c[-1]
                mean_utils_tjc = pars_c[0] + xtj * pars_c[1] + xi
                shares_tjc = _exp_stj(mean_utils_tjc, xtj, sig_val_c, nodes1, weights1)
                d2sh_dx_dth_tjc[t, :, j, c] = (
                    (
                        shares_tjc[t, :]
                        + shares[t, :]
                        - shares_tj[t, :]
                        - shares_c[c, t, :]
                    )
                    / EPS
                    / EPS
                )
                print_stars(
                    f"max error on d2sh_dx_dth[{t}, :, {j}, {c}]: {npmaxabs(d2sh_dx_dth_tjc[t, :, j, c] - d2sh_dx_dth[t, :, j, c])}"
                )

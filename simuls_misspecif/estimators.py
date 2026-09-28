import numpy as np
import scipy.linalg as spla

from simuls_misspecif.utils import make_omega_inv


def estimate_nonrandom(xmat1, Z_alt, yvec):
    npts = xmat1.shape[0]
    omega_0_inv = make_omega_inv(Z_alt)
    omega_0 = spla.inv(omega_0_inv)
    xpZ = xmat1.T @ Z_alt / npts
    Zpy = Z_alt.T @ yvec / npts
    lhs_0 = xpZ @ omega_0 @ xpZ.T
    rhs_0 = xpZ @ omega_0 @ Zpy
    beta_hat_0 = spla.solve(lhs_0, rhs_0)

    # second stage omega
    resid_0 = yvec - xmat1 @ beta_hat_0
    zxi_0 = Z_alt.T * resid_0
    zxi_0_mean = np.mean(zxi_0, 0)
    zxi_0_centered = zxi_0 - zxi_0_mean
    zxi_0_centered[0, :] = 1.0
    Omega_inv = make_omega_inv(zxi_0_centered.T)
    Omega = spla.inv(Omega_inv)
    lhs_1 = xpZ @ Omega @ xpZ.T
    rhs_1 = xpZ @ Omega @ Zpy
    nonrandom_vals = spla.solve(lhs_1, rhs_1)
    return Omega, nonrandom_vals


def estimate_whatif_just(yvec, xmat1, Z_alt, Kmat, Omega):
    npts = xmat1.shape[0]
    Zpy = Z_alt.T @ yvec / npts
    xmat1_wijust = np.column_stack((xmat1, Kmat))
    xpZ_wijust = xmat1_wijust.T @ Z_alt / npts
    lhs_wijust = xpZ_wijust @ Omega @ xpZ_wijust.T
    rhs_wijust = xpZ_wijust @ Omega @ Zpy
    whatif_just_vals = spla.solve(lhs_wijust, rhs_wijust)
    return whatif_just_vals


def make_U_array(
    nmarkets,
    n_x,
    xmat,
    observed_shares_mat,
):
    npts = xmat.shape[0]
    nproducts = npts // nmarkets
    eS_x = np.zeros((nmarkets, n_x))
    eS_xx = np.zeros((nmarkets, n_x, n_x))
    for m in range(n_x):
        x_m = xmat[:, m].reshape((nmarkets, nproducts))
        eS_x[:, m] = np.sum(x_m * observed_shares_mat, axis=1)
        for n in range(m, n_x):
            x_n = xmat[:, n].reshape((nmarkets, nproducts))
            eS_xx[:, m, n] = np.sum(x_m * x_n * observed_shares_mat, 1)
            eS_xx[:, n, m] = eS_xx[:, m, n]

    W_array = np.zeros((npts, n_x, n_x))
    for mkt in range(nmarkets):
        mkt_slice = slice(mkt * nproducts, (mkt + 1) * nproducts)
        for m in range(n_x):
            x_mt = xmat[mkt_slice, m]
            eS_x_t = eS_x[mkt, :]
            eS_xx_t = eS_xx[mkt, :, :]
            for n in range(m, n_x):
                x_nt = xmat[mkt_slice, n]
                W_array[mkt_slice, m, n] = (
                    x_mt * eS_x_t[n] + x_nt * eS_x_t[m] - x_mt * x_nt
                ) * (eS_xx_t[m, n] - eS_x_t[m] * eS_x_t[n])
                W_array[mkt_slice, n, m] = W_array[mkt_slice, m, n]

    U_array = W_array / 2.0
    return U_array


def estimate_whatif_over(
    xmat1, Z_alt, Kmat, Omega, nonrandom_vals, zxi_nonrandom_mean, U_array
):
    npts = xmat1.shape[0]
    n_x = U_array.shape[1]
    quadratic_term = np.zeros((n_x, n_x))
    for m in range(n_x):
        for n in range(m, n_x):
            UpZ_mn = (U_array[:, m, n].T @ Z_alt) / npts
            quadratic_term[m, n] = -2.0 * UpZ_mn @ Omega @ zxi_nonrandom_mean
            quadratic_term[n, m] = quadratic_term[m, n]

    xpZ = xmat1.T @ Z_alt / npts

    Cmat = xpZ @ Omega @ xpZ.T
    KpZ = Kmat.T @ Z_alt / npts
    Dmat = xpZ @ Omega @ KpZ.T
    Rmat = KpZ @ Omega @ KpZ.T + quadratic_term

    # print(f"{Rmat=}, {quadratic_term=}, {Cmat=}, {Dmat=}")

    r_vec = -KpZ @ Omega @ zxi_nonrandom_mean
    # print(f"{r_vec=}")
    lhs_over = np.block([[Cmat, Dmat], [Dmat.T, Rmat]])
    rhs_over = np.concatenate((np.zeros(n_x + 1), -r_vec))

    whatif_over_vals = spla.solve(lhs_over, rhs_over)
    # print(f"Done {whatif_over_vals=}")
    whatif_over_vals += nonrandom_vals
    return whatif_over_vals

"""Tests for the model with M covariates with random coefficients.

With one covariate, the new code must reproduce the previous one-covariate formulas,
which are copied here as references.
"""

import numpy as np
import pytest
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from bs_python_utils.bsnputils import gaussian_expectation
from frac_blp.artificial_regressors import make_W
from scipy.optimize import fsolve

from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.evaluations import (
    _nonrandom_semi_elasticities,
    _pseudo_semi_elasticities_ift,
    _true_semi_elasticities,
)
from simuls_misspecif.MNL_integrals import (
    _dshares_dx,
    _exp_stj,
    _exp_stj_eps,
    _exp_stj_stk_eps,
)
from simuls_misspecif.MNL_utils import integration_nodes, wgh, xgh
from simuls_misspecif.utils import estimate_what_if, get_semi_elast_stats


def _random_market_data(nmarkets, nproducts, n_x, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(nmarkets, nproducts, n_x))
    mean_utils = rng.normal(size=(nmarkets, nproducts)) - 1.0
    return x, mean_utils


# ---------------------------------------------------------------------------
#  one covariate: compare with the previous implementation
# ---------------------------------------------------------------------------


def _old_make_shares(mean_utils_xi, x, sigma_tot):
    nmarkets, nproducts = mean_utils_xi.shape
    shares = np.zeros((nmarkets, nproducts))

    def utils(v, pars):
        x_t, means_t, sigma_tot = pars
        shares_v = np.exp(sigma_tot * v * x_t + means_t)
        return shares_v / (1.0 + np.sum(shares_v))

    for t in range(nmarkets):
        pars_t = [x[t, :], mean_utils_xi[t, :], sigma_tot]
        shares[t, :] = gaussian_expectation(utils, pars=pars_t, x=xgh, w=wgh)
    return shares


def _old_node_shares(mean_utils, x, sig_val, nodes1):
    s_tjl = np.zeros(x.shape + (nodes1.size,))
    for ll, node_l in enumerate(nodes1):
        s_tjl_l = np.exp(mean_utils + sig_val * x * node_l)
        s_tjl[:, :, ll] = s_tjl_l / (1.0 + np.sum(s_tjl_l, 1).reshape((-1, 1)))
    return s_tjl


def test_make_shares_one_covariate_unchanged():
    x, mean_utils = _random_market_data(6, 4, 1)
    new = make_shares(mean_utils, x, np.array([0.8]))
    old = _old_make_shares(mean_utils, x[:, :, 0], 0.8)
    np.testing.assert_allclose(new, old, rtol=1e-12)


def test_integrals_one_covariate_unchanged():
    x, mean_utils = _random_market_data(6, 4, 1)
    nodes1, weights1 = setup_sparse_gaussian(1, 17)
    s_tjl = _old_node_shares(mean_utils, x[:, :, 0], 0.7, nodes1)
    np.testing.assert_allclose(
        _exp_stj(mean_utils, x, 0.7, nodes1, weights1), s_tjl @ weights1, rtol=1e-12
    )
    np.testing.assert_allclose(
        _exp_stj_eps(mean_utils, x, 0.7, nodes1, weights1)[:, :, 0],
        s_tjl @ (nodes1 * weights1),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        _exp_stj_stk_eps(mean_utils, x, 0.7, nodes1, weights1)[..., 0],
        np.einsum("tjl, tkl, l->tjk", s_tjl, s_tjl, nodes1 * weights1),
        rtol=1e-12,
        atol=1e-15,
    )


def test_make_W_matches_old_W():
    nmarkets, nproducts = 7, 4
    x, mean_utils = _random_market_data(nmarkets, nproducts, 1)
    shares = make_shares(mean_utils, x, np.array([0.5]))
    x1 = x[:, :, 0]
    eS_x = np.sum(shares * x1, axis=1)
    eS_x2 = np.sum(shares * x1 * x1, axis=1)
    old_W = (
        x1 * (-x1 + 2.0 * eS_x.reshape((-1, 1))) * (eS_x2 - eS_x**2).reshape((-1, 1))
    )
    new_W = make_W(x.reshape((-1, 1)), shares.reshape(-1), nproducts)
    np.testing.assert_allclose(new_W[:, 0, 0], old_W.reshape(-1), rtol=1e-12)


def _old_estimate_what_if(xvec, Kvec, Wvec, beta0_0, beta1_0, xi_0_vec, Z, Omega):
    def ap(a, b):
        Z_a = np.array([np.mean(Z[:, k] * a) for k in range(Z.shape[1])])
        Z_b = np.array([np.mean(Z[:, k] * b) for k in range(Z.shape[1])])
        return float(Z_a @ Omega @ Z_b)

    ones = np.ones(xvec.size)
    regs = [ones, xvec, Kvec]
    lhs = np.array([[ap(a, b) for b in regs] for a in regs])
    lhs[2, 2] -= 2.0 * ap(Wvec, xi_0_vec)
    rhs = np.array([0.0, 0.0, ap(Kvec, xi_0_vec)])
    d = np.linalg.solve(lhs, rhs)
    return np.array([beta0_0 + d[0], beta1_0 + d[1], d[2]])


def test_estimate_what_if_one_covariate_unchanged():
    npts, n_instr = 500, 5
    rng = np.random.default_rng(1)
    x, K, W, xi_0 = (rng.normal(size=npts) for _ in range(4))
    Z = np.column_stack((np.ones(npts), rng.normal(size=(npts, n_instr - 1))))
    Omega = np.linalg.inv(Z.T @ Z / npts)
    old = _old_estimate_what_if(x, K, W, 0.3, -2.0, xi_0, Z, Omega)
    new = estimate_what_if(
        x.reshape((-1, 1)),
        K.reshape((-1, 1)),
        W.reshape((-1, 1, 1)),
        0.3,
        np.array([-2.0]),
        xi_0,
        Z,
        Omega,
    )
    np.testing.assert_allclose(new, old, rtol=1e-10)


# ---------------------------------------------------------------------------
#  several covariates: check derivatives by finite differences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nproducts", [1, 3])
def test_dshares_dx_finite_differences(nproducts):
    nmarkets, n_x = 3, 2
    x, xi = _random_market_data(nmarkets, nproducts, n_x)
    beta = np.array([-1.0, 0.5])
    sig_vec = np.array([0.8, 0.4])
    nodes, weights = setup_sparse_gaussian(n_x, 17)

    def shares_at(x_val):
        return _exp_stj(xi + x_val @ beta, x_val, sig_vec, nodes, weights)

    dsh_dx = _dshares_dx(xi + x @ beta, x, beta, sig_vec, nodes, weights)
    assert dsh_dx.shape == (nmarkets, nproducts, nproducts, n_x)
    h = 1e-6
    for k in range(nproducts):
        for m in range(n_x):
            xp, xm = x.copy(), x.copy()
            xp[:, k, m] += h
            xm[:, k, m] -= h
            fd = (shares_at(xp) - shares_at(xm)) / (2.0 * h)
            np.testing.assert_allclose(dsh_dx[:, :, k, m], fd, atol=1e-8)


@pytest.mark.parametrize("n_x, nproducts", [(1, 4), (2, 4), (1, 1), (2, 1)])
def test_pseudo_semi_elasticities_finite_differences(n_x, nproducts):
    nmarkets = 4
    x, mean_utils = _random_market_data(nmarkets, nproducts, n_x, seed=2)
    shares = make_shares(mean_utils, x)
    beta = np.linspace(-2.0, 1.0, n_x)
    s2 = np.linspace(0.7, 0.3, n_x)
    pars = np.concatenate(([0.1], beta, s2))

    own, cross = _pseudo_semi_elasticities_ift(pars, shares, x)
    assert own.shape == cross.shape == (nmarkets, n_x)
    if nproducts == 1:
        np.testing.assert_array_equal(cross, 0.0)

    def equation(s, x_t):
        # log(s_j/s_0) - x_j'beta - sum_m s2_m K_jm(s, x)
        e_S_x = s @ x_t
        K = -(e_S_x - x_t / 2.0) * x_t
        return np.log(s / (1.0 - s.sum())) - x_t @ beta - K @ s2

    h = 1e-6
    for t in range(nmarkets):
        s_t, x_t = shares[t], x[t]
        c_t = equation(s_t, x_t)
        for m in range(n_x):
            products = ((0, own), (1, cross)) if nproducts > 1 else ((0, own),)
            for k, semi in products:
                xp, xm = x_t.copy(), x_t.copy()
                xp[k, m] += h
                xm[k, m] -= h
                sp = fsolve(lambda s: equation(s, xp) - c_t, s_t, xtol=1e-13)
                sm = fsolve(lambda s: equation(s, xm) - c_t, s_t, xtol=1e-13)
                fd = (sp[0] - sm[0]) / (2.0 * h) / s_t[0]
                assert semi[t, m] == pytest.approx(fd, abs=1e-6)


def test_quadrature_several_covariates():
    # with x_2 = 0 the second random coefficient is irrelevant
    nmarkets, nproducts = 5, 3
    x, mean_utils = _random_market_data(nmarkets, nproducts, 2)
    x[:, :, 1] = 0.0
    shares2 = make_shares(mean_utils, x, np.array([0.6, 1.3]))
    shares1 = make_shares(mean_utils, x[:, :, :1], np.array([0.6]))
    np.testing.assert_allclose(shares2, shares1, rtol=1e-12)


def test_shares_positive_with_large_sigma():
    # sparse grids have negative weights and gave negative shares here
    nmarkets, nproducts, n_x = 200, 25, 3
    x, xi = _random_market_data(nmarkets, nproducts, n_x, seed=4)
    mean_utils = -4.9 + x @ np.full(n_x, -4.0) + xi
    shares = make_shares(mean_utils, x, np.full(n_x, np.sqrt(2.0)))
    assert np.all(shares > 0.0)
    assert np.all(np.sum(shares, 1) < 1.0)


def test_integration_nodes():
    for n_x in (1, 2, 3):
        nodes, weights = integration_nodes(n_x, 17, 8)
        assert nodes.shape == (weights.size, n_x)
        assert weights.sum() == pytest.approx(1.0)
        # identity covariance
        cov = (nodes * weights[:, np.newaxis]).T @ nodes
        np.testing.assert_allclose(cov, np.eye(n_x), atol=1e-12)
        if n_x > 1:
            assert np.all(weights > 0.0)


@pytest.mark.parametrize("n_x", [1, 2])
def test_one_product_semi_elasticities(n_x):
    # with J = 1 there are no cross semi-elasticities
    nmarkets = 6
    x, mean_utils = _random_market_data(nmarkets, 1, n_x, seed=5)
    beta = np.full(n_x, -1.5)
    sig_vec = np.full(n_x, 0.6)
    shares = make_shares(mean_utils + x @ beta, x, sig_vec)
    true_p = np.concatenate(([0.0], beta, sig_vec**2))
    nodes, weights = integration_nodes(n_x, 17, 8)
    own, cross, dsh_dx = _true_semi_elasticities(
        true_p, shares, x, mean_utils + x @ beta, nodes, weights
    )
    assert own.shape == cross.shape == (nmarkets, n_x)
    assert dsh_dx.shape == (nmarkets, 1, 1, n_x)
    np.testing.assert_array_equal(cross, 0.0)
    stats = get_semi_elast_stats(own, cross, 1)
    assert stats.shape == (n_x, 2)
    non_own, non_cross = _nonrandom_semi_elasticities(true_p[: 1 + n_x], shares, x)
    np.testing.assert_allclose(non_own, np.outer(1.0 - shares[:, 0], beta))
    np.testing.assert_array_equal(non_cross, 0.0)

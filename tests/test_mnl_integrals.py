"""Tests for Gauss-Hermite MNL integrals."""

import numpy as np
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from simuls_misspecif.MNL_integrals import _exp_stj, _exp_stj_eps


def test_exp_stj_shape_and_bounds():
    nmarkets, nproducts = 5, 3
    mean_utils = np.zeros((nmarkets, nproducts))
    x = np.ones((nmarkets, nproducts, 1))
    sig_val = 0.5
    nodes1, weights1 = setup_sparse_gaussian(1, 17)

    Estj = _exp_stj(mean_utils, x, sig_val, nodes1, weights1)

    assert Estj.shape == (nmarkets, nproducts)
    # Market shares must be between 0 and 1
    assert np.all(Estj >= 0.0)
    assert np.all(Estj <= 1.0)

    # Sum of shares per market must be less than 1 (outside share S0 > 0)
    S_tot = np.sum(Estj, axis=1)
    assert np.all(S_tot < 1.0)


def test_exp_stj_zero_sigma():
    # When sig_val = 0, MNL share is 1/(1 + J*exp(mu))
    nmarkets, nproducts = 4, 2
    mu_val = 0.5
    mean_utils = np.full((nmarkets, nproducts), mu_val)
    x = np.ones((nmarkets, nproducts, 1))
    nodes1, weights1 = setup_sparse_gaussian(1, 17)

    Estj = _exp_stj(mean_utils, x, sig_vec=0.0, nodes=nodes1, weights=weights1)

    expected_share = np.exp(mu_val) / (1.0 + nproducts * np.exp(mu_val))
    np.testing.assert_allclose(Estj, expected_share, rtol=1e-5)


def test_exp_stj_eps_symmetry():
    # For symmetric mean_utils around epsilon=0, E(s_tj * eps) should be close to 0 when sig_val=0
    nmarkets, nproducts = 4, 2
    mean_utils = np.zeros((nmarkets, nproducts))
    x = np.ones((nmarkets, nproducts, 1))
    nodes1, weights1 = setup_sparse_gaussian(1, 17)

    Estj_eps = _exp_stj_eps(mean_utils, x, sig_vec=0.0, nodes=nodes1, weights=weights1)

    assert Estj_eps.shape == (nmarkets, nproducts, 1)
    # At sig_val=0, s_tj is independent of eps (which is N(0,1)), so E(s_tj * eps) = s_tj * E(eps) = 0
    np.testing.assert_allclose(Estj_eps, 0.0, atol=1e-6)

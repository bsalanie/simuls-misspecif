"""Tests for evaluations routines."""

import numpy as np
from simuls_misspecif.evaluations import _our_tsls0, _our_tsls2


def test_our_tsls0_exact_recovery():
    npts = 100
    np.random.seed(42)
    x_proj = np.random.normal(size=npts)
    beta0_true, beta1_true = 1.5, -2.0
    y_proj = beta0_true + beta1_true * x_proj

    Zstar0, nonrandom_vals, cond_num = _our_tsls0(y_proj, x_proj)

    assert Zstar0.shape == (npts, 2)
    np.testing.assert_allclose(nonrandom_vals[0], beta0_true, rtol=1e-5)
    np.testing.assert_allclose(nonrandom_vals[1], beta1_true, rtol=1e-5)
    assert cond_num > 0


def test_our_tsls2_exact_recovery():
    npts = 100
    np.random.seed(42)
    x_proj = np.random.normal(size=npts)
    K_proj = np.random.normal(size=npts)
    b0, b1, b2 = 0.5, 3.0, -1.2
    y_proj = b0 + b1 * x_proj + b2 * K_proj

    Zstar2, pseudo_vals, cond_num = _our_tsls2(y_proj, x_proj, K_proj)

    assert Zstar2.shape == (npts, 3)
    np.testing.assert_allclose(pseudo_vals[0], b0, rtol=1e-5)
    np.testing.assert_allclose(pseudo_vals[1], b1, rtol=1e-5)
    np.testing.assert_allclose(pseudo_vals[2], b2, rtol=1e-5)
    assert cond_num > 0


def test_our_tsls_several_covariates():
    npts, n_x = 200, 2
    rng = np.random.default_rng(0)
    X_proj = rng.normal(size=(npts, n_x))
    K_proj = rng.normal(size=(npts, n_x))
    coeffs = np.array([0.5, 3.0, -1.0, 0.7, 0.2])
    y_proj = coeffs[0] + X_proj @ coeffs[1:3] + K_proj @ coeffs[3:]

    Zstar2, pseudo_vals, _ = _our_tsls2(y_proj, X_proj, K_proj)
    assert Zstar2.shape == (npts, 1 + 2 * n_x)
    np.testing.assert_allclose(pseudo_vals, coeffs, rtol=1e-10)

    y0 = coeffs[0] + X_proj @ coeffs[1:3]
    Zstar0, nonrandom_vals, _ = _our_tsls0(y0, X_proj)
    assert Zstar0.shape == (npts, 1 + n_x)
    np.testing.assert_allclose(nonrandom_vals, coeffs[:3], rtol=1e-10)

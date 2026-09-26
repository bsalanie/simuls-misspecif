"""Tests for MNL_utils data structures and generation logic."""

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from simuls_misspecif.MNL_utils import (
    DataParams,
    ModelData,
    SimulationCase,
    TrueParams,
    _mean_utils,
    make_names_params,
)
from simuls_misspecif.utils import generate_RNG_streams


def test_true_params():
    tp = TrueParams(beta0=-1.0, beta=np.array([2.0, 1.0]), sigma_profile=[1.0, 0.5])
    assert tp.beta0 == -1.0
    np.testing.assert_array_equal(tp.beta, [2.0, 1.0])
    np.testing.assert_array_equal(tp.sigma_profile, [1.0, 0.5])
    assert tp.n_x == 2


def test_data_params_exogenous_generation():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=True)
    streams = generate_RNG_streams(1, initial_seed=42)
    draws = dp.generate_random_draws(nmarkets=100, nproducts=4, stream=streams[0])

    xi, x, z = dp.generate_exogenous_vars_from_draws(draws)
    assert xi.shape == (100, 4)
    assert x.shape == (100, 4, 1)
    assert z.shape == (100, 4, 1)
    # For exogenous case, x should equal z
    np.testing.assert_array_equal(x, z)


def test_data_params_endogenous_generation():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=False)
    streams = generate_RNG_streams(1, initial_seed=42)
    draws = dp.generate_random_draws(nmarkets=100, nproducts=4, stream=streams[0])

    xi, x, z = dp.generate_exogenous_vars_from_draws(draws)
    assert xi.shape == (100, 4)
    assert x.shape == (100, 4, 1)
    assert z.shape == (100, 4, 1)
    # For endogenous case, x and z should be distinct
    assert not np.array_equal(x, z)


def test_simulation_case_dataclass():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=True)
    tp = TrueParams(beta0=0.0, beta=np.array([1.0]), sigma_profile=np.array([1.0]))
    model = ModelData(
        data_pars=dp,
        true_pars=tp,
        names_pars=make_names_params(1),
        model_string="test_model",
        long_name="Test Model",
        nmarkets=100,
        nproducts=2,
        scenario=3,
        sigma_range=np.array([0.1, 0.5]),
        mode="2",
        iprec=17,
    )
    case = SimulationCase(
        stream=np.random.default_rng(123),
        model=model,
        isim=0,
        pickle_dir=Path("/tmp/test_pickle"),
        use_mp=False,
    )

    assert case.isim == 0
    assert case.model.nproducts == 2
    assert case.pickle_dir == Path("/tmp/test_pickle")
    assert not case.use_mp


def test_mean_utils():
    x = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    mu = _mean_utils(beta0=0.5, beta=np.array([2.0, 1.0]), x=x)
    expected = np.array([2.5, 5.5, 5.5])
    np.testing.assert_allclose(mu, expected)


def test_names_params():
    assert make_names_params(2) == [
        "beta0",
        "beta_1",
        "beta_2",
        "sigma2_1",
        "sigma2_2",
    ]


def _old_exogenous_vars(dp, xi_d, z_d, u_d):
    """The one-covariate data generation before M covariates were allowed."""
    xi = dp.sigxi * xi_d
    z = dp.sigx * z_d
    if dp.do_exo:
        return xi, z.copy(), z
    rnorm = sqrt(1 - dp.rhox_xi**2) * u_d
    xi_term = rnorm + (dp.rhox_xi * xi / dp.sigxi)
    x = dp.rhox_z * z + dp.sigx * sqrt(1 - dp.rhox_z**2) * xi_term
    return xi, x, z


@pytest.mark.parametrize("do_exo", [True, False])
def test_one_covariate_draws_unchanged(do_exo):
    dp = DataParams(sigxi=1.0, sigx=1.5, rhox_z=0.7, rhox_xi=0.5, do_exo=do_exo)
    xi, x, z = dp.generate_exogenous_vars_from_draws(
        dp.generate_random_draws(50, 3, np.random.default_rng(7))
    )
    stream = np.random.default_rng(7)
    xi_d, z_d, u_d = (stream.normal(size=(50, 3)) for _ in range(3))
    xi_old, x_old, z_old = _old_exogenous_vars(dp, xi_d, z_d, u_d)
    np.testing.assert_array_equal(xi, xi_old)
    np.testing.assert_array_equal(x[:, :, 0], x_old)
    np.testing.assert_array_equal(z[:, :, 0], z_old)


def test_several_covariates_generation():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=False, n_x=3)
    xi, x, z = dp.generate_exogenous_vars_from_draws(
        dp.generate_random_draws(20_000, 2, np.random.default_rng(3))
    )
    assert x.shape == z.shape == (20_000, 2, 3)
    # each x_m is correlated with its own z_m only
    corr = np.corrcoef(x[:, 0, :].T, z[:, 0, :].T)[:3, 3:]
    np.testing.assert_allclose(np.diag(corr), 0.7, atol=0.02)
    np.testing.assert_allclose(corr - np.diag(np.diag(corr)), 0.0, atol=0.03)


def test_model_data_checks_n_x():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=True, n_x=2)
    tp = TrueParams(beta0=0.0, beta=np.array([1.0]), sigma_profile=np.array([1.0]))
    with pytest.raises(SystemExit):
        ModelData(
            data_pars=dp,
            true_pars=tp,
            names_pars=make_names_params(1),
            model_string="",
            long_name="",
            nmarkets=10,
            nproducts=2,
            scenario=0,
            sigma_range=np.array([0.5]),
            mode="2",
            iprec=17,
        )

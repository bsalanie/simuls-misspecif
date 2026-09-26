"""Tests for MNL_utils data structures and generation logic."""

from pathlib import Path
import numpy as np
from simuls_misspecif.MNL_utils import (
    DataParams,
    ModelData,
    SimulationCase,
    TrueParams,
    _mean_utils,
)
from simuls_misspecif.utils import generate_RNG_streams


def test_true_params():
    tp = TrueParams(beta0=-1.0, beta1=2.0, sigma=0.5)
    assert tp.beta0 == -1.0
    assert tp.beta1 == 2.0
    assert tp.sigma == 0.5


def test_data_params_exogenous_generation():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=True)
    streams = generate_RNG_streams(1, initial_seed=42)
    draws = dp.generate_random_draws(nmarkets=100, nproducts=4, stream=streams[0])

    xi, x, z = dp.generate_exogenous_vars_from_draws(draws)
    assert xi.shape == (100, 4)
    assert x.shape == (100, 4)
    assert z.shape == (100, 4)
    # For exogenous case, x should equal z
    np.testing.assert_array_equal(x, z)


def test_data_params_endogenous_generation():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=False)
    streams = generate_RNG_streams(1, initial_seed=42)
    draws = dp.generate_random_draws(nmarkets=100, nproducts=4, stream=streams[0])

    xi, x, z = dp.generate_exogenous_vars_from_draws(draws)
    assert xi.shape == (100, 4)
    assert x.shape == (100, 4)
    assert z.shape == (100, 4)
    # For endogenous case, x and z should be distinct
    assert not np.array_equal(x, z)


def test_simulation_case_dataclass():
    dp = DataParams(sigxi=1.0, sigx=1.0, rhox_z=0.7, rhox_xi=0.5, do_exo=True)
    tp = TrueParams(beta0=0.0, beta1=1.0, sigma=0.5)
    model = ModelData(
        data_pars=dp,
        true_pars=tp,
        names_pars=["beta0", "beta1", "sigma"],
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
    x = np.array([1.0, 2.0, 3.0])
    mu = _mean_utils(beta0=0.5, beta1=2.0, x=x)
    expected = np.array([2.5, 4.5, 6.5])
    np.testing.assert_allclose(mu, expected)

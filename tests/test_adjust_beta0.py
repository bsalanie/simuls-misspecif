"""Tests for adjust_beta0_S0 root solver."""

from simuls_misspecif.MNL_params import data_pars, true_pars
from simuls_misspecif.simuls_driver import adjust_beta0_S0


def test_adjust_beta0_S0():
    target_s0 = 0.5
    nproducts = 4

    fitted_beta0, achieved_s0 = adjust_beta0_S0(
        target_s0, nproducts, data_pars, true_pars
    )

    # Check achieved S0 is close to target (within tolerance 1e-4)
    assert abs(achieved_s0 - target_s0) < 1e-4
    assert isinstance(fitted_beta0, float)

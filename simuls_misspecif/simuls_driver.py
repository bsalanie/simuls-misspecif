"""MNL normal without a demographic term: evaluate and pickle.

We simulate a large number of markets from the true DGP and compute:

* the error on xi
* the pseudo-true values
* the semiparametric efficiency bounds
"""

import dataclasses as dc
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
from bs_python_utils.bsnputils import TwoArrays, npexp
from bs_python_utils.bsutils import mkdir_if_needed, print_stars
from numpy.random import SeedSequence, default_rng

from simuls_misspecif.compute_stats import get_the_stats
from simuls_misspecif.extract_from_results import extract_from_results
from simuls_misspecif.MNL_params import (
    basic_sigma_range,
    data_pars,
    large_sigma_range,
    true_pars,
)
from simuls_misspecif.MNL_utils import DataParams, ModelData, TrueParams, names_params
from simuls_misspecif.plots_paper import new_plots_paper


# Numpy parallel RNG
def generate_RNG_streams(
    nsim: int, initial_seed: int = 13091962
) -> List[np.random.Generator]:
    ss = SeedSequence(initial_seed)
    # Spawn off child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(nsim)
    streams = [default_rng(s) for s in child_seeds]
    return streams


def run_model(
    model_root: str,
    base_model: ModelData,
    scenario: Dict,
    str_roots: List,
    long_names: List,
) -> Tuple[ModelData, Path]:
    scenario_number = base_model.scenario
    do_exo = True if "exo" in model_root else False
    data_p = dc.replace(scenario["data"], do_exo=do_exo)
    str_long = long_names[0] if do_exo else long_names[2]
    str_root = str_roots[0] if do_exo else str_roots[2]
    str_model = f"{str_root}_J={nproducts}_v{scenario_number}"
    new_model = dc.replace(
        base_model, data_pars=data_p, model_string=str_model, long_name=str_long
    )
    pickle_subdir = Path(f"{str_root}_v{scenario_number}")
    return new_model, pickle_subdir


def adjust_beta0_S0(
    S0: float, nproducts: int, data_pars: DataParams, true_pars: TrueParams
) -> Tuple[float, float]:
    """Find the beta0 that makes the expected outside share equal S0.

    Args:
        S0: Target average outside share.
        nproducts: Number of products `J`.
        data_pars: The data parameters.
        true_pars: The true coefficients.

    Returns:
        The fitted beta0 and the achieved expected outside share.
    """
    beta1 = true_pars.beta1
    ndraws = 1000
    x = np.random.normal(scale=data_pars.sigx, size=ndraws * nproducts).reshape(
        (nproducts, ndraws)
    )
    xi = np.random.normal(scale=data_pars.sigxi, size=ndraws * nproducts).reshape(
        (nproducts, ndraws)
    )
    utils0 = beta1 * x + xi

    def compute_ES0(beta0):
        utils = utils0 + beta0
        exp_utils, der_exp_utils = cast(TwoArrays, npexp(utils, deriv=1))
        S0_draws = 1 / (1 + np.sum(exp_utils, 0))
        ES0 = np.mean(S0_draws)
        der_ES0 = -np.mean(np.sum(der_exp_utils, 0) * S0_draws * S0_draws)
        return ES0, der_ES0

    # Newton iterations to solve `compute_ES0(beta0) = S0`
    beta0i = np.log((1.0 - S0) / (S0 * nproducts))  # solution when utils0 = 0
    errS0 = np.inf
    tol = 1e-6
    while errS0 > tol:
        ES0i, der_ES0i = compute_ES0(beta0i)
        beta0i -= (ES0i - S0) / der_ES0i
        errS0 = abs(ES0i - S0)

    return beta0i, ES0i


if __name__ == "__main__":
    # what we run
    nmarkets = 5000
    number_products = [1, 2, 5, 10, 25]
    selected_scenario_numbers = [3, 4]
    selected_models = ["exo", "endo"]

    # multiprocessing
    use_mp = True
    n_cpus = mp.cpu_count()  # number of CPUs
    nb_cpus = n_cpus - 2  # we reserve 2

    # we may use different scenarii
    # 0 is the central scenario from MNL_params.py;  the others modify some of its parameters
    central_scenario = {
        "data": data_pars,
        "coeffs": true_pars,
        "sigma_range": basic_sigma_range,
    }

    str_roots = ["exo", "endo"]
    long_names = [
        "Exogenous",
        "Endogenous",
    ]

    target_S0 = 0.9

    scenarii = {
        0: central_scenario,
        1: {
            "data": data_pars,
            "coeffs": true_pars,
            "sigma_range": large_sigma_range,
        },
        2: {
            "data": data_pars,
            "coeffs": true_pars,
            "sigma_range": basic_sigma_range,
        },  # we will make S0 close to 1/2
        3: {
            "data": data_pars,
            "coeffs": dc.replace(true_pars, beta1=-4.0),  # to get elasticity about -2
            "sigma_range": basic_sigma_range,
        },
        4: {
            "data": data_pars,
            "coeffs": dc.replace(true_pars, beta1=-4.0),  # to get elasticity about -2
            "sigma_range": basic_sigma_range,
        },  # we will make S0 close to target_S0
    }

    n_scenarii = len(selected_scenario_numbers)
    selected_scenarii = {
        k: v for k, v in scenarii.items() if k in selected_scenario_numbers
    }

    n_types_models = len(selected_models)
    nsim = len(number_products) * n_scenarii * n_types_models

    models: list[ModelData | None] = [None] * nsim
    pickles_dir: list[Path | None] = [None] * nsim

    streams = generate_RNG_streams(nsim, initial_seed=5546757)

    isim = 0

    for nproducts in number_products:
        beta0_3, ES0_3 = adjust_beta0_S0(0.5, nproducts, data_pars, true_pars)
        scenarii[3]["coeffs"] = dc.replace(scenarii[3]["coeffs"], beta0=beta0_3)
        beta0_4, ES0_4 = adjust_beta0_S0(0.9, nproducts, data_pars, true_pars)
        scenarii[4]["coeffs"] = dc.replace(scenarii[4]["coeffs"], beta0=beta0_4)

        root_dir = mkdir_if_needed(Path.cwd() / f"J{nproducts}")

        for scenario_number, scenario in selected_scenarii.items():
            sigma_range = scenario["sigma_range"]
            true_p = scenario["coeffs"]
            base_model = ModelData(
                true_pars=true_p,
                data_pars=scenario["data"],
                scenario=scenario_number,
                model_string="",
                long_name="",
                names_pars=names_params,
                nproducts=nproducts,
                nmarkets=nmarkets,
                mode1="2",
                mode2="2",
                iprec=17,
                sigma_range=sigma_range,
            )
            for model_root in selected_models:
                models[isim], pickle_subdir = run_model(
                    model_root, base_model, scenario, str_roots, long_names
                )
                pickles_dir[isim] = mkdir_if_needed(root_dir / pickle_subdir)
                isim += 1

    list_cases = [
        [streams[isim], models[isim], isim, pickles_dir[isim], use_mp]
        for isim in range(nsim)
    ]

    # run the simulation
    res: list[dict | None] = [None] * nsim
    if use_mp:
        with mp.Pool(processes=nb_cpus) as pool:
            res = pool.map(get_the_stats, list_cases)
    else:
        for isim in range(nsim):
            res[isim] = get_the_stats(list_cases[isim])

    # just to be sure
    with open("res.pkl", "wb") as f:
        pickle.dump(res, f)
    print_stars("saved res")

    # now extract what we need for the plots
    keys_extract = [
        "pseudo true values",
        "correc_d4",
        "correc_dprime4",
        "correc_infty",
        "SPE variance bounds",
        "true semi-elasticities",
        "pseudo semi-elasticities",
        "corrected semi-elasticities",
        "model",
    ]

    for scenario_number, scenario in selected_scenarii.items():
        for nproducts in number_products:
            for model in selected_models:
                extract_from_results(
                    model, nproducts, nmarkets, scenario_number, keys_extract
                )
                new_plots_paper(model, nproducts, nmarkets, selected_scenario_numbers)

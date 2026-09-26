"""MNL normal without a demographic term: evaluate and pickle.

We simulate a large number of markets from the true DGP and compute:

* the error on xi
* the pseudo-true values
* the semiparametric efficiency bounds
"""

import argparse
import dataclasses as dc
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from bs_python_utils.bsnputils import TwoArrays, npexp
from bs_python_utils.bsutils import mkdir_if_needed, print_stars

from simuls_misspecif.compute_stats import get_the_stats
from simuls_misspecif.extract_from_results import (
    KEYS_EXTRACT,
    case_subdir,
    extract_from_results,
    model_string,
)
from simuls_misspecif.MNL_params import (
    basic_sigma_range,
    data_pars,
    large_sigma_range,
    true_pars,
)
from simuls_misspecif.MNL_utils import (
    DataParams,
    ModelData,
    SimulationCase,
    TrueParams,
    make_names_params,
)
from simuls_misspecif.plots_paper import new_plots_paper
from simuls_misspecif.utils import generate_RNG_streams


class ScenarioDict(TypedDict):
    """Configuration for a simulation scenario.

    Attributes:
        data: Data generation parameters.
        coeffs: True coefficients for the DGP.
        sigma_range: Array of sigma values to simulate over.
    """

    data: DataParams
    coeffs: TrueParams
    sigma_range: np.ndarray


def setup_model(
    model_root: str,
    base_model: ModelData,
    scenario: ScenarioDict,
    str_roots: list,
    long_names: list,
) -> tuple[ModelData, Path]:
    """Set up a model instance with exogenous/endogenous specification.

    Creates a new ModelData instance with the specified exogenous or
    endogenous specification and returns the configured model and its
    output subdirectory path.

    Args:
        model_root: Base string for model type ('endo' or 'exo').
        base_model: Base ModelData configuration to copy from.
        scenario: Scenario configuration with data, coefficients, and sigma range.
        str_roots: List of root model strings (['exo', 'endo']).
        long_names: List of long descriptive names for the models.

    Returns:
        Tuple of (configured ModelData instance, pickle output subdirectory Path).
    """
    scenario_number = base_model.scenario
    do_exo = True if "exo" in model_root else False
    data_p = dc.replace(scenario["data"], do_exo=do_exo)
    str_long = long_names[0] if do_exo else long_names[1]
    str_root = str_roots[0] if do_exo else str_roots[1]
    nproducts = base_model.nproducts
    str_model = model_string(str_root, nproducts, scenario_number, data_p.n_x)
    new_model = dc.replace(
        base_model, data_pars=data_p, model_string=str_model, long_name=str_long
    )
    pickle_subdir = Path(case_subdir(str_root, scenario_number, data_p.n_x))
    return new_model, pickle_subdir


def adjust_beta0_S0(
    S0: float,
    nproducts: int,
    data_pars: DataParams,
    true_pars: TrueParams,
    seed: int = 5514557,
) -> tuple[float, float]:
    """Find the beta0 that makes the expected outside share equal S0.

    Args:
        S0: Target average outside share.
        nproducts: Number of products `J`.
        data_pars: The data parameters.
        true_pars: The true coefficients.
        seed: for the random number generator.

    Returns:
        The fitted beta0 and the achieved expected outside share.
    """
    ndraws = 1000
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=data_pars.sigx, size=(nproducts, ndraws, data_pars.n_x))
    xi = rng.normal(scale=data_pars.sigxi, size=ndraws * nproducts).reshape(
        (nproducts, ndraws)
    )
    utils0 = x @ true_pars.beta + xi

    def compute_ES0(beta0):
        utils = utils0 + beta0
        exp_utils, der_exp_utils = cast(TwoArrays, npexp(utils, deriv=1))
        S0_draws = 1 / (1 + np.sum(exp_utils, 0))
        ES0 = np.mean(S0_draws)
        der_ES0 = -np.mean(np.sum(der_exp_utils, 0) * S0_draws * S0_draws)
        return ES0, der_ES0

    # Newton iterations to solve `compute_ES0(beta0) = S0`
    beta0i = np.log((1.0 - S0) / (S0 * nproducts))  # solution when utils0 = 0
    errS0 = ES0i = np.inf
    tol = 1e-6
    while errS0 > tol:
        ES0i, der_ES0i = compute_ES0(beta0i)
        beta0i -= (ES0i - S0) / der_ES0i
        errS0 = abs(ES0i - S0)

    return beta0i, ES0i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run symmetric mixed MNL misspecification simulations."
    )
    parser.add_argument(
        "-s",
        "--scenarios",
        nargs="+",
        type=int,
        default=[3, 4],
        help="Scenario numbers to run (default: 3 4)",
    )
    parser.add_argument(
        "-J",
        "--products",
        nargs="+",
        type=int,
        default=[2, 5, 10, 25, 50, 100],
        help="List of product numbers J (default: 2 5 10 25 50 100)",
    )
    parser.add_argument(
        "-T",
        "--markets",
        type=int,
        default=10_000,
        help="Number of markets T (default: 10000)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=["endo", "exo"],
        default=["endo"],
        help="Model types to run (choices: endo exo, default: endo)",
    )
    parser.add_argument(
        "-M",
        "--n-x",
        type=int,
        default=1,
        help="Number of covariates with random coefficients M (default: 1)",
    )
    parser.add_argument(
        "--sigma-profile",
        nargs="+",
        type=float,
        default=None,
        help="Relative standard errors of the M random coefficients (default: all 1)",
    )
    parser.add_argument("--no-mp", action="store_true", help="Disable multiprocessing")
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Number of CPU cores to use, default: n_cpus-2",
    )
    args = parser.parse_args()

    # what we run
    nmarkets = args.markets
    number_products = args.products
    selected_scenario_numbers = args.scenarios
    selected_models = args.models
    n_x = args.n_x
    sigma_profile = (
        np.ones(n_x) if args.sigma_profile is None else np.array(args.sigma_profile)
    )
    if sigma_profile.size != n_x:
        parser.error(f"--sigma-profile needs {n_x} values")

    # multiprocessing
    use_mp = not args.no_mp
    if args.cpus is not None:
        nb_cpus = args.cpus
    else:
        n_cpus = mp.cpu_count()
        nb_cpus = max(1, n_cpus - 2)

    # create logs directory
    mkdir_if_needed(Path.cwd() / "logs")

    str_roots = ["exo", "endo"]
    long_names = [
        "Exogenous",
        "Endogenous",
    ]

    target_S0 = 0.9

    # expand the default parameters to M covariates
    data_pars = dc.replace(data_pars, n_x=n_x)
    true_pars = TrueParams(
        beta0=true_pars.beta0,
        beta=np.full(n_x, true_pars.beta[0]),
        sigma_profile=sigma_profile,
    )
    beta_minus4 = np.full(n_x, -4.0)

    central_scenario: ScenarioDict = {
        "data": data_pars,
        "coeffs": true_pars,
        "sigma_range": basic_sigma_range,
    }

    scenarii: dict[int, ScenarioDict] = {
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
        },
        3: {
            "data": data_pars,
            "coeffs": dc.replace(true_pars, beta=beta_minus4),
            "sigma_range": basic_sigma_range,
        },
        4: {
            "data": data_pars,
            "coeffs": dc.replace(true_pars, beta=beta_minus4),
            "sigma_range": basic_sigma_range,
        },
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
        if 3 in selected_scenarii:
            beta0_3, ES0_3 = adjust_beta0_S0(0.5, nproducts, data_pars, true_pars)
            scenarii[3]["coeffs"] = dc.replace(scenarii[3]["coeffs"], beta0=beta0_3)
        if 4 in selected_scenarii:
            beta0_4, ES0_4 = adjust_beta0_S0(target_S0, nproducts, data_pars, true_pars)
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
                names_pars=make_names_params(n_x),
                nproducts=nproducts,
                nmarkets=nmarkets,
                mode="2",
                iprec=17,
                sigma_range=sigma_range,
            )
            for model_root in selected_models:
                model_inst, pickle_subdir = setup_model(
                    model_root, base_model, scenario, str_roots, long_names
                )
                models[isim] = model_inst
                pickles_dir[isim] = mkdir_if_needed(root_dir / pickle_subdir)
                isim += 1

    list_cases = [
        SimulationCase(
            stream=streams[i],
            model=cast(ModelData, models[i]),
            isim=i,
            pickle_dir=cast(Path, pickles_dir[i]),
            use_mp=use_mp,
        )
        for i in range(nsim)
    ]

    # run the simulation
    res: list[dict | None] = [None] * nsim
    if use_mp:
        with mp.Pool(processes=nb_cpus) as pool:
            res = pool.map(get_the_stats, list_cases)
    else:
        for i in range(nsim):
            print_stars(f"Calling model {i}")
            res[i] = get_the_stats(list_cases[i])

    # just to be sure
    with open("res.pkl", "wb") as f:
        pickle.dump(res, f)
    print_stars("saved res")

    # now extract what we need for the plots
    for scenario_number in selected_scenarii.keys():
        for nproducts in number_products:
            for model in selected_models:
                resmod: dict = extract_from_results(
                    model, nproducts, nmarkets, scenario_number, KEYS_EXTRACT, n_x=n_x
                )
                new_plots_paper(
                    model, nproducts, nmarkets, selected_scenario_numbers, n_x=n_x
                )

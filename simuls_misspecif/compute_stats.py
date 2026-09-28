"""Worker function for simulation cases.

Computes econometric estimates (non-random, what-if just and over identified) and
semi-elasticity bounds for each simulation case, including semiparametric
efficiency (SPE) variance bounds.
"""

import os
import pickle
import tracemalloc

import numpy as np
import scipy.linalg as spla
from bs_python_utils.bs_mem import memory_display_top, memory_display_top_diffs
from bs_python_utils.bsutils import print_stars

from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.estimators import (
    estimate_nonrandom,
    estimate_whatif_just,
    estimate_whatif_over,
    make_U_array,
)
from simuls_misspecif.evaluations import (
    _artificial_regressors,
    _nonrandom_semi_elasticities,
    _print_pseudo_true_errors,
    _pseudo_semi_elasticities_ift,
    _true_optimal_instruments,
    _true_semi_elasticities,
)
from simuls_misspecif.MNL_params import (
    n_gh_integrals,
)
from simuls_misspecif.MNL_utils import (
    SimulationCase,
    _mean_utils,
    integration_nodes,
    make_names_params,
)
from simuls_misspecif.utils import (
    f_print_stars,
    get_semi_elast_stats,
)


def get_the_stats(
    case: SimulationCase | list,
    do_bounds: bool = False,
) -> dict:
    """Evaluate the various statistics needed for one simulation case.

    Args:
        case: A `SimulationCase` dataclass (or 5-element list for backwards compatibility)
            containing the random generator, model, simulation number, pickle directory,
            and multiprocessing flag.
        do_bounds: Whether to compute SPE bounds and true semi-elasticities (default: False).

    Returns:
        The model and the simulation results in a dictionary.
    """

    do_trace_memory = False

    if do_trace_memory:
        tracemalloc.start()

    verbose = False
    if isinstance(case, SimulationCase):
        stream, model, isim, pickle_dir, use_mp = (
            case.stream,
            case.model,
            case.isim,
            case.pickle_dir,
            case.use_mp,
        )
    else:
        stream, model, isim, pickle_dir, use_mp = case

    if isinstance(stream, np.random.SeedSequence):
        stream = np.random.default_rng(stream)

    if use_mp:
        fout_name = os.path.join("logs", f"{os.getpid()}.out")
    else:
        fout_name = None

    f_print_stars(
        use_mp, f"Doing simulation {isim}, pickled in {pickle_dir}", fout_name
    )

    if verbose:
        model.print()

    mode, nmarkets, nproducts, iprec = (
        model.mode,
        model.nmarkets,
        model.nproducts,
        model.iprec,
    )

    print_stars(
        f"Calling get stats for scenario {model.scenario} with {nmarkets} markets\n"
        + f" and {nproducts} products with {model.n_x} covariates."
    )

    i_scenario, str_long = model.scenario, model.long_name
    sigma_range = model.sigma_range
    npts = nmarkets * nproducts

    true_pars, data_pars = model.true_pars, model.data_pars
    n_x = data_pars.n_x
    sigxi = data_pars.sigxi
    true_beta0, true_beta = true_pars.beta0, true_pars.beta
    sigma_profile = true_pars.sigma_profile

    nodes, weights = integration_nodes(n_x, iprec, n_gh_integrals)

    n_elast = 1 if nproducts == 1 else 2
    n_sigmas = sigma_range.size
    n_instr = 3
    m = nproducts * n_instr

    n_params = 1 + 2 * n_x
    coeffs_shape = (n_sigmas,)
    names_ptv = make_names_params(n_x)
    names_spb = names_ptv

    nonrandom_values = np.zeros(coeffs_shape + (n_params,))
    whatif_just_values = np.zeros(coeffs_shape + (n_params,))
    whatif_over_values = np.zeros(coeffs_shape + (n_params,))
    sp_bounds = np.zeros(coeffs_shape + (n_params, n_params))
    cond_numbers_bounds = np.zeros(coeffs_shape)
    omega_inv_eigenvalues = np.zeros(coeffs_shape + (m,))
    omega_inv_eigenvectors = np.zeros(coeffs_shape + (m, m))

    shape_elast = coeffs_shape + (n_x, 2 * n_elast)

    values_nonrandom_semi_elast = np.zeros(shape_elast)
    values_whatif_just_semi_elast = np.zeros(shape_elast)
    values_whatif_over_semi_elast = np.zeros(shape_elast)
    values_true_semi_elast = np.zeros(shape_elast)

    mean_squared_residuals = np.zeros(coeffs_shape + (5,))

    snapshot1 = None

    if do_trace_memory:
        snapshot1 = tracemalloc.take_snapshot()
        memory_display_top(snapshot1)

    # create the data, except for the shares
    draws = data_pars.generate_random_draws(nmarkets, nproducts, stream)
    true_xi, x, z = data_pars.generate_exogenous_vars_from_draws(draws)
    xmat = x.reshape((npts, n_x))
    zmat = z.reshape((npts, n_x))
    xmat1 = np.column_stack((np.ones(npts), xmat))

    # basis of instruments for product j on market t:
    #   z_jt1,..., z_jtm, sum_k z_kt1^2, ..., sum_k z_ktm^2, sum_k z_kt1^3, ..., sum_k z_kt m^3
    n_Z_alt: int = n_x * 3 + 1
    Z_alt = np.zeros((npts, n_Z_alt))
    for mkt in range(nmarkets):
        mkt_slice = slice(mkt * nproducts, (mkt + 1) * nproducts)
        z_mkt = zmat[mkt_slice, :]
        z_mkt_sq = z_mkt * z_mkt
        z_mkt_cub = z_mkt_sq * z_mkt
        Z_alt[mkt_slice, 0] = 1.0
        Z_alt[mkt_slice, 1 : (n_x + 1)] = z_mkt
        Z_alt[mkt_slice, (n_x + 1) : (2 * n_x + 1)] = z_mkt_sq
        Z_alt[mkt_slice, (2 * n_x + 1) :] = z_mkt_cub

    for isig, sigma_val in enumerate(sigma_range):
        sig_vec = sigma_val * sigma_profile
        sig2_vec = sig_vec * sig_vec

        true_mean_utils = _mean_utils(true_beta0, true_beta, x)
        true_mean_utils_xi = true_mean_utils + true_xi

        # generate the shares
        observed_shares_mat = make_shares(true_mean_utils_xi, x, sig_vec)
        observed_shares_vec = observed_shares_mat.reshape(npts)

        Kmat, yvec, *_ = _artificial_regressors(observed_shares_vec, xmat, nproducts)

        # true_p contains the true values of the coefficients we estimate in TSLS
        true_p = np.concatenate(([true_beta0], true_beta, sig2_vec))

        #################################################################################
        ##                        our TSLS                                             ##
        #################################################################################
        # start = time.time()

        Omega, nonrandom_vals = estimate_nonrandom(xmat1, Z_alt, yvec)

        zxi_nonrandom_mean = np.mean(Z_alt.T * (yvec - xmat1 @ nonrandom_vals), 1)
        whatif_just_vals = estimate_whatif_just(yvec, xmat1, Z_alt, Kmat, Omega)

        nonrandom_vals: np.ndarray = np.concatenate((nonrandom_vals, np.zeros(n_x)))

        if verbose:
            _print_pseudo_true_errors(true_p, whatif_just_vals, names_ptv, verbose=True)

        U_array = make_U_array(
            nmarkets,
            n_x,
            xmat,
            observed_shares_mat,
        )

        whatif_over_vals = estimate_whatif_over(
            xmat1,
            Z_alt,
            Kmat,
            Omega,
            nonrandom_vals,
            zxi_nonrandom_mean,
            U_array,
        )

        print_stars("True ; estimates non-random, what-if just, what-if over:")
        for i in range(n_params):
            print(
                f"{names_ptv[i]:>9}: {true_p[i]: .3f};",
                f"  {nonrandom_vals[i]: .3f},",
                f"  {whatif_just_vals[i]: .3f},",
                f"  {whatif_over_vals[i]: .3f}",
            )
        ##              now we work on the semi-elasticities                           ##
        #################################################################################

        # start = time.time()

        nonrandom_own_semi, nonrandom_cross_semi = _nonrandom_semi_elasticities(
            nonrandom_vals, observed_shares_mat, x
        )

        whatif_just_own_semi, whatif_just_cross_semi = _pseudo_semi_elasticities_ift(
            whatif_just_vals, observed_shares_mat, x
        )

        whatif_over_own_semi, whatif_over_cross_semi = _pseudo_semi_elasticities_ift(
            whatif_over_vals, observed_shares_mat, x
        )

        resus_nonrandom_semi_elast = get_semi_elast_stats(
            nonrandom_own_semi, nonrandom_cross_semi, nproducts
        )
        resus_whatif_just_semi_elast = get_semi_elast_stats(
            whatif_just_own_semi, whatif_just_cross_semi, nproducts
        )
        resus_whatif_over_semi_elast = get_semi_elast_stats(
            whatif_over_own_semi, whatif_over_cross_semi, nproducts
        )

        if do_bounds:
            true_own_semi, true_cross_semi, dshares_dx = _true_semi_elasticities(
                true_p, observed_shares_mat, x, true_mean_utils_xi, nodes, weights
            )
            resus_true_semi_elast = get_semi_elast_stats(
                true_own_semi, true_cross_semi, nproducts
            )
        else:
            true_own_semi = np.zeros((nmarkets, n_x))
            true_cross_semi = np.zeros((nmarkets, n_x))
            resus_true_semi_elast = np.zeros((4,)) if nproducts > 1 else np.zeros((2,))

        # end = time.time()
        # print(f"semi-elast took {end - start} seconds")

        #################################################################################
        ##              now we work on SPE bounds                                      ##
        #################################################################################

        # start = time.time()
        cond_bounds = np.nan  # Initialize before conditional
        if do_bounds:
            X_proj = Z_alt @ spla.solve(Z_alt.T @ Z_alt, Z_alt.T @ xmat, assume_a="pos")
            Zstar = _true_optimal_instruments(
                true_p,
                true_mean_utils_xi,
                observed_shares_mat,
                x,
                X_proj,
                z,
                nodes,
                weights,
                mode=mode,
            )
            Zstar_T = Zstar.T
            exp_dxi_zstar = (Zstar_T @ Zstar) / npts
            s = np.linalg.svd(exp_dxi_zstar, compute_uv=False)
            cond_bounds = abs(s[0] / s[-1])
            exp_dxi_zstar_inv = np.linalg.inv(exp_dxi_zstar)
            spb = sigxi * sigxi * exp_dxi_zstar_inv

            if verbose:
                print_stars(
                    (
                        f"          {model.long_name}\n"
                        f"   variance bounds for true sigma2={sig2_vec}"
                        f" with {nproducts} products:"
                    )
                )
                for i in range(n_params):
                    print(f"on {names_spb[i]}: {spb[i, i]: 10.4f}")
        else:
            spb = np.zeros((n_params, n_params))

        sp_bounds[isig, :, :] = spb
        # end = time.time()
        # print(f"spe bounds took {end - start} seconds")

        nonrandom_values[isig, :] = nonrandom_vals
        whatif_just_values[isig, :] = whatif_just_vals
        whatif_over_values[isig, :] = whatif_over_vals
        sp_bounds[isig, :, :] = spb
        values_nonrandom_semi_elast[isig] = resus_nonrandom_semi_elast
        values_whatif_just_semi_elast[isig] = resus_whatif_just_semi_elast
        values_whatif_over_semi_elast[isig] = resus_whatif_over_semi_elast
        values_true_semi_elast[isig] = resus_true_semi_elast
        cond_numbers_bounds[isig] = cond_bounds

        done_message = f"                     Done with sigma value {isig + 1}/{n_sigmas} for J = {nproducts}, "
        done_message += f"scenario {i_scenario}, model: {str_long}"

        f_print_stars(use_mp, done_message, fout_name)

    dict_results = {
        "model": model,
        "n_x": n_x,
        "non-random values": nonrandom_values,
        "whatif just values": whatif_just_values,
        "whatif over values": whatif_over_values,
        "omega_inv eigenvalues": omega_inv_eigenvalues,
        "omega_inv eigenvectors": omega_inv_eigenvectors,
        "SPE variance bounds": sp_bounds,
        "true semi-elasticities": values_true_semi_elast,
        "non-random semi-elasticities": values_nonrandom_semi_elast,
        "whatif just semi-elasticities": values_whatif_just_semi_elast,
        "whatif over semi-elasticities": values_whatif_over_semi_elast,
        "mean squared residuals": mean_squared_residuals,
        "condition number bounds": cond_numbers_bounds,
    }

    if do_trace_memory:
        snapshotf = tracemalloc.take_snapshot()
        if snapshot1 is not None:
            memory_display_top_diffs(snapshot1, snapshotf)

        tracemalloc.stop()

    str_model, nmarkets = model.model_string, model.nmarkets
    pickle_file = pickle_dir / f"simul_results_{str_model}_T={nmarkets}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(dict_results, f)
    f_print_stars(use_mp, f"saved results of simulation {isim}", fout_name)

    return dict_results

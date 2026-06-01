import os
import pickle
import time
import tracemalloc
from math import sqrt
from typing import Dict, List

import numpy as np
import scipy.linalg as spla
from bs_python_utils.bs_mem import memory_display_top, memory_display_top_diffs
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from bs_python_utils.bsutils import bs_error_abort, file_print_stars, print_stars

from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.evaluations import (
    _artificial_regressors,
    _make_quartic_instruments,
    _mean_squared_residuals,
    _newton_raphson_step,
    _our_tsls,
    _print_pseudo_true_errors,
    _print_semi_elast,
    _project_variables,
    _pseudo_semi_elasticities,
    _true_optimal_instruments,
    _true_semi_elasticities,
    estimated_xi_infty,
)
from simuls_misspecif.MNL_integrals import _d2shares_dx_dtheta, _dshares_dtheta
from simuls_misspecif.MNL_utils import _mean_utils


def _print_stars(use_mp: bool, what: str, fout_name: str | None = None):
    """_summary_

    Args:
        use_mp (bool): _description_
        what (str): _description_
        fout_name (str | None, optional): _description_. Defaults to None.
    """
    if use_mp and fout_name is not None:
        with open(fout_name, "a") as fout:
            file_print_stars(fout, what)
    elif use_mp:
        bs_error_abort("use_mp is True but fout_name is None")
    else:
        print_stars(what)


def get_the_stats(case: List, save_more: bool = False) -> Dict:
    """Evaluate the various statistics needed for one simulation case.

    Args:
        case: A five-element list containing the random generator, model,
            simulation number, pickle directory, and multiprocessing flag.
        save_more: Whether to save the xi values and related intermediates.

    Returns:
        The model and the simulation results in a dictionary.
    """

    do_trace_memory = False

    if do_trace_memory:
        tracemalloc.start()

    verbose = False
    niters = 1  # number of iterated corrections with xi_infty
    stream, model, isim, pickle_dir, use_mp = case

    do_bounds_semi_elast = (
        False  # whether we compute SPE bounds for the semi elasticities (costly)
    )

    if use_mp:
        fout_name = str(os.getpid()) + ".out"
    else:
        fout_name = None

    _print_stars(use_mp, f"Doing simulation {isim}, pickled in {pickle_dir}", fout_name)

    if verbose:
        model.print()

    mode1, mode2, nmarkets, nproducts, iprec = (
        model.mode1,
        model.mode2,
        model.nmarkets,
        model.nproducts,
        model.iprec,
    )
    i_scenario, str_long = model.scenario, model.long_name
    sigma_range = model.sigma_range

    nodes1, weights1 = setup_sparse_gaussian(1, iprec)
    nodes1 = nodes1[:, 0]

    true_pars, data_pars = model.true_pars, model.data_pars
    sigxi = data_pars.sigxi
    true_beta0, true_beta1 = true_pars.beta0, true_pars.beta1

    n_elast = 1 if nproducts == 1 else 2
    n_sigmas = sigma_range.size

    n_params = 3
    coeffs_shape = (n_sigmas,)
    names_ptv = ["beta0", "beta1", "sigma2"]
    names_spb = ["beta0", "beta1", "sigma2"]

    pseudo_true_values = np.zeros(coeffs_shape + (n_params,))
    sp_bounds = np.zeros(coeffs_shape + (n_params, n_params))
    correc_d4 = np.zeros(coeffs_shape + (n_params,))
    correc_dprime4 = np.zeros(coeffs_shape + (n_params,))
    correc_infty = np.zeros(coeffs_shape + (n_params, niters))
    cond_numbers2 = np.zeros(coeffs_shape)
    cond_numbers_bounds = np.zeros(coeffs_shape)

    shape_elast = coeffs_shape + (2 * n_elast,)

    values_pseudo_semi_elast = np.zeros(shape_elast)
    values_true_semi_elast = np.zeros(shape_elast)
    values_corrected_semi_elast = np.zeros(shape_elast)
    if do_bounds_semi_elast:
        sp_bounds_semi_elast = np.zeros(shape_elast)

    mean_squared_residuals = np.zeros(coeffs_shape + (5,))

    if save_more:
        estimated_xi2 = np.zeros(coeffs_shape + (nmarkets, nproducts))
        xi_vals = np.zeros(coeffs_shape + (nmarkets, nproducts))
        errors_xi2 = np.zeros(coeffs_shape + (nmarkets, nproducts))
        errors_xi4 = np.zeros(coeffs_shape + (nmarkets, nproducts))
        ZZ = np.zeros(coeffs_shape + (n_params, n_params))
        Zy = np.zeros(coeffs_shape + (n_params,))
        ZQ = np.zeros(coeffs_shape + (n_params,))
        ZW = np.zeros(coeffs_shape + (n_params,))
        xiQ = np.zeros(coeffs_shape)
        xiW = np.zeros(coeffs_shape)

    if do_trace_memory:
        snapshot1 = tracemalloc.take_snapshot()
        memory_display_top(snapshot1)

    # create the data, except for the shares
    draws = data_pars.generate_random_draws(nmarkets, nproducts, stream)
    true_xi, x, z, Dbar = data_pars.generate_exogenous_vars_from_draws(draws)

    for isig, sigma_val in enumerate(sigma_range):
        true_mean_utils = _mean_utils(true_beta0, true_beta1, x)
        true_mean_utils_xi = true_mean_utils + true_xi

        # generate the shares
        sig2 = sigma_val * sigma_val
        s2 = sig2
        s_tot = sqrt(s2)
        observed_shares = make_shares(true_mean_utils_xi, x, s_tot)

        observed_shares0 = 1.0 - np.sum(observed_shares, 1)
        y = np.log(observed_shares / observed_shares0.reshape((-1, 1)))

        K, Q, W = _artificial_regressors(observed_shares, x)

        npts = y.size
        Kvec = K.reshape(npts)
        xvec = x.reshape(npts)
        yvec = y.reshape(npts)
        Qvec = Q.reshape(npts)
        Wvec = W.reshape(npts)
        yxKQW = np.column_stack((yvec, xvec, Kvec, Qvec, Wvec))

        # project the variables on the instruments
        if mode2 == "2":
            base_instruments = _make_quartic_instruments(z)
            coeffs, _, _, _ = spla.lstsq(base_instruments, yxKQW)
            y_proj = base_instruments @ coeffs[:, 0]
            x_proj = base_instruments @ coeffs[:, 1]
            K_proj = base_instruments @ coeffs[:, 2]
            Q_proj = base_instruments @ coeffs[:, 3]
            W_proj = base_instruments @ coeffs[:, 4]
        else:
            base_instruments = None
            y_proj, x_proj, K_proj, Q_proj, W_proj = _project_variables(
                y, x, K, Q, W, z, mode1=mode1, mode2=mode2
            )

        # true_p contains the true values of the coefficients we estimate in TSLS
        true_p = np.array([true_beta0, true_beta1, sig2])

        # evaluate xi(k) - xi(infty) for k = 2 and 4
        true_xi2 = y - true_mean_utils - s2 * K
        errors2 = true_xi2 - true_xi
        true_xi4 = true_xi2 - (3.0 * Q + W) * s2 * s2
        errors4 = true_xi4 - true_xi

        #################################################################################
        ##                        our TSLS                                             ##
        #################################################################################
        start = time.time()
        Zstar2, pseudo_vals, cond_number2 = _our_tsls(y_proj, x_proj, K_proj)

        Zstar2_T = Zstar2.T

        beta0_2 = pseudo_vals[0]
        beta1_2 = pseudo_vals[1]
        s2_2 = pseudo_vals[-1]

        if verbose:
            _print_pseudo_true_errors(true_p, pseudo_vals, names_ptv, verbose=True)

        # the estimated approximate xi
        xi2_2 = (
            y
            - _mean_utils(
                beta0_2,
                beta1_2,
                x,
            )
            - s2_2 * K
        )

        xi_2 = xi2_2.reshape(npts)

        # the estimated mean utilities
        mean_utils_2 = _mean_utils(beta0_2, beta1_2, x)

        end = time.time()
        print(f"2SLS took {end - start} seconds")

        #################################################################################
        ##              let us do one step of Newton-Raphson using 4th order           ##
        #################################################################################
        start = time.time()
        d4, dp4 = _newton_raphson_step(
            Q_proj, W_proj, xi_2, Zstar2, pseudo_vals, true_p
        )
        end = time.time()
        print(f"4th order correction took {end - start} seconds")

        #################################################################################
        ##          TSLS correction based on xi_infty(theta2)                          ##
        #################################################################################

        start = time.time()
        corrections_infty = np.zeros((n_params, niters))

        xi_cur = xi2_2.copy()
        mean_utils_cur = mean_utils_2.copy()
        s2_cur = s2_2.copy()
        p = pseudo_vals.copy()

        for iter in range(niters):
            print(f"Doing iter {iter + 1}/{niters}")
            xi_infty_est, rcodes, nevals = estimated_xi_infty(
                observed_shares,
                mean_utils_cur,
                x,
                xi_cur,
                s2_cur,
                nodes1,
                weights1,
                verbose=False,
            )
            print(f"   mean number of evals: {np.mean(nevals)}")
            dxi_vec = (xi_infty_est - xi_cur).reshape(npts)
            dtheta, _, _, _ = spla.lstsq(Zstar2, dxi_vec)
            p += dtheta
            beta0_cur, beta1_cur = p[0], p[1]
            s2_cur = p[-1]
            mean_utils_cur = _mean_utils(beta0_cur, beta1_cur, x)
            xi_cur = xi_infty_est
            corrections_infty[:, iter] = p

            corrected_pvals = corrections_infty[:, -1]

            print_stars("True values, pseudo values, corrected pseudo-values:")
            for i in range(n_params):
                print(
                    f"{names_ptv[i]}:     {true_p[i]: > 10.3f}, {pseudo_vals[i]: > 10.3f},   {corrected_pvals[i]: > 10.3f}"
                )

            end = time.time()
            print(f"infty correction took {end - start} seconds")

            #################################################################################
            ##                        the mean squared residuals                           ##
            #################################################################################
            start = time.time()
            msr_null = _mean_squared_residuals(y_proj)
            x_covs = x_proj
            msr_non_random = _mean_squared_residuals(y_proj, x_covs)
            x_K = np.column_stack((x_covs, K_proj))
            msr_order2 = _mean_squared_residuals(y_proj, x_K)

            Zstar4_k1 = np.column_stack((x_K, Q_proj + W_proj))
            msr_order4_k1 = _mean_squared_residuals(y_proj, Zstar4_k1)
            Zstar4_k3 = np.column_stack((x_K, Q_proj + 3.0 * W_proj))
            msr_order4_k3 = _mean_squared_residuals(y_proj, Zstar4_k3)

            msr_values = np.array(
                [msr_null, msr_non_random, msr_order2, msr_order4_k1, msr_order4_k3]
            )

            end = time.time()
            print(f"MSR took {end - start} seconds")

            #################################################################################
            ##              now we work on the semi-elasticities                           ##
            #################################################################################

            start = time.time()
            pseudo_own_semi, pseudo_cross_semi = _pseudo_semi_elasticities(
                Dbar, pseudo_vals, observed_shares, x
            )

            true_own_semi, true_cross_semi, dshares_dx = _true_semi_elasticities(
                true_p, observed_shares, x, true_mean_utils_xi, nodes1, weights1
            )

            corrected_beta0, corrected_beta1 = corrected_pvals[:2]
            corrected_mean_utils = _mean_utils(corrected_beta0, corrected_beta1, x)
            corrected_mean_utils_xi = corrected_mean_utils + xi_cur
            corrected_own_semi, corrected_cross_semi, _ = _true_semi_elasticities(
                corrected_pvals,
                observed_shares,
                x,
                corrected_mean_utils_xi,
                nodes1,
                weights1,
            )

            mean_true_own_semi_elast = np.mean(true_own_semi)
            stderr_true_own_semi_elast = np.std(true_own_semi)
            mean_pseudo_own_semi_elast = np.mean(pseudo_own_semi)
            stderr_pseudo_own_semi_elast = np.std(pseudo_own_semi)
            mean_corrected_own_semi_elast = np.mean(corrected_own_semi)
            stderr_corrected_own_semi_elast = np.std(corrected_own_semi)

            if nproducts > 1:
                mean_true_cross_semi_elast = np.mean(true_cross_semi)
                stderr_true_cross_semi_elast = np.std(true_cross_semi)
                mean_pseudo_cross_semi_elast = np.mean(pseudo_cross_semi)
                stderr_pseudo_cross_semi_elast = np.std(pseudo_cross_semi)
                mean_corrected_cross_semi_elast = np.mean(corrected_cross_semi)
                stderr_corrected_cross_semi_elast = np.std(corrected_cross_semi)
                (
                    resus_pseudo_semi_elast,
                    resus_true_semi_elast,
                    resus_corrected_semi_elast,
                ) = _print_semi_elast(
                    mean_pseudo_own_semi_elast,
                    stderr_pseudo_own_semi_elast,
                    mean_true_own_semi_elast,
                    stderr_true_own_semi_elast,
                    mean_corrected_own_semi_elast,
                    stderr_corrected_own_semi_elast,
                    mean_pseudo_cross_semi_elast,
                    stderr_pseudo_cross_semi_elast,
                    mean_true_cross_semi_elast,
                    stderr_true_cross_semi_elast,
                    mean_corrected_cross_semi_elast,
                    stderr_corrected_cross_semi_elast,
                    verbose=verbose,
                )
            else:
                (
                    resus_pseudo_semi_elast,
                    resus_true_semi_elast,
                    resus_corrected_semi_elast,
                ) = _print_semi_elast(
                    mean_pseudo_own_semi_elast,
                    stderr_pseudo_own_semi_elast,
                    mean_true_own_semi_elast,
                    stderr_true_own_semi_elast,
                    mean_corrected_own_semi_elast,
                    stderr_corrected_own_semi_elast,
                    verbose=verbose,
                )

            end = time.time()
            print(f"semi-elast took {end - start} seconds")

            #################################################################################
            ##              now we work on SPE bounds                                      ##
            #################################################################################

            start = time.time()
            Zstar = _true_optimal_instruments(
                true_p,
                true_mean_utils_xi,
                observed_shares,
                x,
                x_proj,
                z,
                nodes1,
                weights1,
                mode1=mode1,
                mode2=mode2,
                quad_instr=base_instruments,
            )
            Zstar_T = Zstar.T
            exp_dxi_zstar = (Zstar_T @ Zstar) / npts
            s = spla.svdvals(exp_dxi_zstar)
            cond_bounds = abs(s[0] / s[-1])
            exp_dxi_zstar_inv = spla.inv(exp_dxi_zstar)
            spb = sigxi * sigxi * exp_dxi_zstar_inv

            sp_bounds[isig, :, :] = spb
            if verbose:
                print_stars(
                    (
                        f"          {model.long_name}\n"
                        f"   variance bounds for true sigma2={sig2: 10.4f}"
                        f" with {nproducts} products:"
                    )
                )
                for i in range(n_params):
                    print(f"on {names_spb[i]}: {spb[i, i]: 10.4f}")

            end = time.time()
            print(f"spe bounds took {end - start} seconds")

            if do_bounds_semi_elast:
                start = time.time()
                # we also want bounds for the semi-elasticities
                dshares_dth = _dshares_dtheta(
                    true_mean_utils_xi, x, true_p, nodes1, weights1
                )
                d2shares_dx_dth = _d2shares_dx_dtheta(
                    true_mean_utils_xi, x, true_p, nodes1, weights1
                )
                share_0 = observed_shares[:, 0]
                share_0_sq = share_0 * share_0
                dsh0_dth = (
                    dshares_dth[:, 0, :] if dshares_dth.ndim == 3 else dshares_dth
                )
                dsh_dx0 = (
                    dshares_dx[:, 0, 0] if dshares_dx.ndim == 3 else dshares_dx[:, 0]
                )
                d2sh_dx0_dth = (
                    d2shares_dx_dth[:, 0, 0, :]
                    if d2shares_dx_dth.ndim == 4
                    else d2shares_dx_dth[:, 0, :]
                )
                d_own_semi_dth = d2sh_dx0_dth / share_0.reshape((-1, 1)) - dsh0_dth * (
                    dsh_dx0 / share_0_sq
                ).reshape((-1, 1))

                if nproducts > 1:
                    d_cross_semi_dth = d2shares_dx_dth[:, 0, 1, :] / share_0.reshape(
                        (-1, 1)
                    ) - dshares_dth[:, 0, :] * (
                        dshares_dx[:, 0, 1] / share_0_sq
                    ).reshape((-1, 1))

                d_mean_own_semi_elast = np.mean(d_own_semi_dth, 0)
                d_var_own_semi_elast = 2.0 * (
                    np.mean(d_own_semi_dth * true_own_semi.reshape((-1, 1)), 0)
                    - mean_true_own_semi_elast * d_mean_own_semi_elast
                )
                if nproducts > 1:
                    d_mean_cross_semi_elast = np.mean(d_cross_semi_dth, 0)
                    d_var_cross_semi_elast = 2.0 * (
                        np.mean(d_cross_semi_dth * true_cross_semi.reshape((-1, 1)), 0)
                        - mean_true_cross_semi_elast * d_mean_cross_semi_elast
                    )
                    spb_semi_elast = np.array(
                        [
                            (d_mean_own_semi_elast.T @ spb @ d_mean_own_semi_elast),
                            (d_var_own_semi_elast.T @ spb @ d_var_own_semi_elast)
                            / (
                                4.0
                                * stderr_true_own_semi_elast
                                * stderr_true_own_semi_elast
                            ),
                            (d_mean_cross_semi_elast.T @ spb @ d_mean_cross_semi_elast),
                            (d_var_cross_semi_elast.T @ spb @ d_var_cross_semi_elast)
                            / (
                                4.0
                                * stderr_true_cross_semi_elast
                                * stderr_true_cross_semi_elast
                            ),
                        ]
                    )
                else:
                    spb_semi_elast = np.array(
                        [
                            (d_mean_own_semi_elast.T @ spb @ d_mean_own_semi_elast),
                            (d_var_own_semi_elast.T @ spb @ d_var_own_semi_elast)
                            / (
                                4.0
                                * stderr_true_own_semi_elast
                                * stderr_true_own_semi_elast
                            ),
                        ]
                    )

                end = time.time()
                print(f"spb_semi_elast took {end - start} seconds")

                pseudo_true_values[isig, :] = pseudo_vals
                correc_d4[isig, :] = d4
                correc_dprime4[isig, :] = dp4
                correc_infty[isig, :, :] = corrections_infty
                sp_bounds[isig, :, :] = spb
                values_pseudo_semi_elast[isig, :] = resus_pseudo_semi_elast
                values_true_semi_elast[isig, :] = resus_true_semi_elast
                values_corrected_semi_elast[isig, :] = resus_corrected_semi_elast
                if do_bounds_semi_elast:
                    sp_bounds_semi_elast[isig, :] = spb_semi_elast
                mean_squared_residuals[isig, :] = msr_values
                cond_numbers2[isig] = cond_number2
                cond_numbers_bounds[isig] = cond_bounds
                if save_more:
                    ZZ[isig, :, :] = Zstar2_T @ Zstar2
                    Zy[isig, :] = Zstar2_T @ y_proj
                    ZQ[isig, :] = Zstar2_T @ Q_proj
                    ZW[isig, :] = Zstar2_T @ W_proj
                    xiQ[isig] = np.dot(xi_2, Q_proj)
                    xiW[isig] = np.dot(xi_2, W_proj)
                    xi_vals[isig, :, :] = true_xi
                    estimated_xi2[isig, :, :] = xi2_2
                    errors_xi2[isig, :, :] = errors2
                    errors_xi4[isig, :, :] = errors4
                done_message = f"                     Done with {isig + 1}/{n_sigmas} for J = {nproducts}, "
                done_message += f"scenario {i_scenario}, model: {str_long}"

            _print_stars(use_mp, done_message, fout_name)

    dict_results = {
        "model": model,
        "pseudo true values": pseudo_true_values,
        "correc_d4": correc_d4,
        "correc_dprime4": correc_dprime4,
        "correc_infty": correc_infty,
        "SPE variance bounds": sp_bounds,
        "true semi-elasticities": values_true_semi_elast,
        "pseudo semi-elasticities": values_pseudo_semi_elast,
        "corrected semi-elasticities": values_corrected_semi_elast,
        "mean squared residuals": mean_squared_residuals,
        "condition number 2": cond_numbers2,
        "condition number bounds": cond_numbers_bounds,
    }
    if do_bounds_semi_elast:
        dict_results["SPE bounds on semi-elasticities"] = sp_bounds_semi_elast

    if save_more:
        more_results = {
            "ZprimeZ": ZZ,
            "Zprimey": Zy,
            "ZprimeQ": ZQ,
            "ZprimeW": ZW,
            "xiprimeQ": xiQ,
            "xiprimeW": xiW,
            "true_xi": xi_vals,
            "errors_xi2": errors_xi2,
            "estimated_xi2": estimated_xi2,
            "errors_xi4": errors_xi4,
        }
        dict_results |= more_results

    if do_trace_memory:
        snapshotf = tracemalloc.take_snapshot()
        memory_display_top_diffs(snapshot1, snapshotf)

        tracemalloc.stop()

    str_model, nmarkets = model.model_string, model.nmarkets
    pickle_file = pickle_dir / f"simul_results_{str_model}_T={nmarkets}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(dict_results, f)
    _print_stars(use_mp, f"saved results of simulation {isim}", fout_name)

    return dict_results

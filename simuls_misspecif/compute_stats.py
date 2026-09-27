"""Worker function for simulation cases.

Computes econometric estimates (non-random, pseudo-true, what-if) and
semi-elasticity bounds for each simulation case, including semiparametric
efficiency (SPE) variance bounds.
"""

import os
import pickle
import tracemalloc
from typing import cast

import numpy as np
import scipy.linalg as spla
from bs_python_utils.bs_mem import memory_display_top, memory_display_top_diffs
from bs_python_utils.bsutils import bs_error_abort, print_stars

from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.evaluations import (
    _artificial_regressors,
    _nonrandom_semi_elasticities,
    _our_tsls0,
    _our_tsls2,
    _print_pseudo_true_errors,
    _project_variables,
    _pseudo_semi_elasticities_ift,
    _true_optimal_instruments,
    _true_semi_elasticities,
)
from simuls_misspecif.MNL_params import (
    do_a_second,
    n_gh_integrals,
)
from simuls_misspecif.MNL_utils import (
    SimulationCase,
    _mean_utils,
    integration_nodes,
    make_names_params,
)
from simuls_misspecif.utils import (
    estimate_what_if,
    f_print_stars,
    get_semi_elast_stats,
    make_omega_inv,
)


def get_the_stats(
    case: SimulationCase | list,
    save_more: bool = False,
    do_bounds: bool = False,
) -> dict:
    """Evaluate the various statistics needed for one simulation case.

    Args:
        case: A `SimulationCase` dataclass (or 5-element list for backwards compatibility)
            containing the random generator, model, simulation number, pickle directory,
            and multiprocessing flag.
        save_more: Whether to save the xi values and related intermediates.
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
        f"Calling get stats for simulation {isim} with {nmarkets} markets and {nproducts} products"
    )

    i_scenario, str_long = model.scenario, model.long_name
    sigma_range = model.sigma_range
    npts = nmarkets * nproducts
    ones = np.ones(npts)

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
    pseudo_true_values = np.zeros(coeffs_shape + (n_params,))
    whatif_just_values = np.zeros(coeffs_shape + (n_params,))
    whatif_over_values = np.zeros(coeffs_shape + (n_params,))
    sp_bounds = np.zeros(coeffs_shape + (n_params, n_params))
    cond_numbers2 = np.zeros(coeffs_shape)
    cond_numbers_bounds = np.zeros(coeffs_shape)
    omega_inv_eigenvalues = np.zeros(coeffs_shape + (m,))
    omega_inv_eigenvectors = np.zeros(coeffs_shape + (m, m))

    shape_elast = coeffs_shape + (n_x, 2 * n_elast)

    values_nonrandom_semi_elast = np.zeros(shape_elast)
    values_pseudo_semi_elast = np.zeros(shape_elast)
    values_whatif_just_semi_elast = np.zeros(shape_elast)
    values_whatif_over_semi_elast = np.zeros(shape_elast)
    values_true_semi_elast = np.zeros(shape_elast)

    mean_squared_residuals = np.zeros(coeffs_shape + (5,))

    estimated_xi2: np.ndarray | None = None
    xi_vals: np.ndarray | None = None
    errors_xi2: np.ndarray | None = None
    ZZ: np.ndarray | None = None
    Zy: np.ndarray | None = None
    ZV: np.ndarray | None = None
    ZW: np.ndarray | None = None
    xiV: np.ndarray | None = None
    xiW: np.ndarray | None = None

    if save_more:
        estimated_xi2 = np.zeros(coeffs_shape + (nmarkets, nproducts))
        xi_vals = np.zeros(coeffs_shape + (nmarkets, nproducts))
        errors_xi2 = np.zeros(coeffs_shape + (nmarkets, nproducts))
        ZZ = np.zeros(coeffs_shape + (n_params, n_params))
        Zy = np.zeros(coeffs_shape + (n_params,))
        ZV = np.zeros(coeffs_shape + (n_params, n_x))
        ZW = np.zeros(coeffs_shape + (n_params, n_x, n_x))
        xiV = np.zeros(coeffs_shape + (n_x,))
        xiW = np.zeros(coeffs_shape + (n_x, n_x))

    snapshot1 = None

    if do_trace_memory:
        snapshot1 = tracemalloc.take_snapshot()
        memory_display_top(snapshot1)

    # create the data, except for the shares
    draws = data_pars.generate_random_draws(nmarkets, nproducts, stream)
    true_xi, x, z = data_pars.generate_exogenous_vars_from_draws(draws)
    xmat = x.reshape((npts, n_x))
    zmat = z.reshape((npts, n_x))

    # the instruments for the over-identified what-if:
    #   powers 1 to 4 of each z_m and the products z_m z_n
    Z_powers_list = [ones]
    for m_x in range(n_x):
        z_m = zmat[:, m_x]
        Z_powers_list += [z_m, z_m**2, z_m**3, z_m**4]
    for m_x in range(n_x):
        for n_x2 in range(m_x + 1, n_x):
            Z_powers_list.append(zmat[:, m_x] * zmat[:, n_x2])
    Z_powers = np.column_stack(Z_powers_list)

    # an alternative basis of instruments for product j on market t:
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
        V_proj: np.ndarray | None = None
        W_proj: np.ndarray | None = None

        sig_vec = sigma_val * sigma_profile
        sig2_vec = sig_vec * sig_vec

        true_mean_utils = _mean_utils(true_beta0, true_beta, x)
        true_mean_utils_xi = true_mean_utils + true_xi

        # generate the shares
        observed_shares_mat = make_shares(true_mean_utils_xi, x, sig_vec)
        observed_shares_vec = observed_shares_mat.reshape(npts)

        Kmat, yvec, Vmat, Warr = _artificial_regressors(
            observed_shares_vec, xmat, nproducts
        )

        # the what-if uses half of the frac_blp W
        Whalf = Warr / 2.0
        ymat = yvec.reshape((nmarkets, nproducts))
        K_sig2 = (Kmat @ sig2_vec).reshape((nmarkets, nproducts))

        # project the variables on the instruments
        if do_a_second:
            y_proj, X_proj, K_proj, V_proj, W_proj = _project_variables(
                yvec, xmat, z, Kmat, Vmat, Warr, mode=mode
            )
        else:
            y_proj, X_proj, K_proj, _, _ = _project_variables(
                yvec, xmat, z, Kmat, mode=mode
            )

        # true_p contains the true values of the coefficients we estimate in TSLS
        true_p = np.concatenate(([true_beta0], true_beta, sig2_vec))

        # evaluate xi(0, 2) - xi(infty)
        true_xi0 = ymat - true_mean_utils
        true_xi2 = true_xi0 - K_sig2
        errors2 = true_xi2 - true_xi

        #################################################################################
        ##                        our TSLS                                             ##
        #################################################################################
        # start = time.time()
        nonrandom_vals = _our_tsls0(y_proj, X_proj)[1]
        beta0_0 = nonrandom_vals[0]
        beta_0 = nonrandom_vals[1:]

        Zstar2, pseudo_vals, cond_number2 = _our_tsls2(y_proj, X_proj, K_proj)
        Zstar2_T = Zstar2.T

        beta0_2 = pseudo_vals[0]
        beta_2 = pseudo_vals[1 : 1 + n_x]
        s2_2 = pseudo_vals[1 + n_x :]

        # another way
        xmat1 = np.column_stack((np.ones(npts), xmat))
        omega_0_inv = make_omega_inv(Z_alt)
        omega_0 = spla.inv(omega_0_inv)
        print(f"{xmat1.shape=}, {Z_alt.shape=}, {omega_0.shape=}")
        xpZ = xmat1.T @ Z_alt / npts
        Zpy = Z_alt.T @ yvec / npts
        lhs_0 = xpZ @ omega_0 @ xpZ.T
        rhs_0 = xpZ @ omega_0 @ Zpy
        beta_hat_0 = spla.solve(lhs_0, rhs_0)
        print(f"{beta_hat_0=}")
        print("Done first 2SLS")

        # second stage omega
        resid_0 = yvec - xmat1 @ beta_hat_0
        zxi_0 = Z_alt.T * resid_0
        zxi_0_mean = np.mean(zxi_0, 0)
        zxi_0_centered = zxi_0 - zxi_0_mean
        print(f"{zxi_0.shape=}")
        zxi_0_centered[0, :] = 1.0
        omega_1_inv = make_omega_inv(zxi_0_centered.T)
        omega_1 = spla.inv(omega_1_inv)
        lhs_1 = xpZ @ omega_1 @ xpZ.T
        rhs_1 = xpZ @ omega_1 @ Zpy
        beta_hat_1 = spla.solve(lhs_1, rhs_1)
        print(f"{beta_hat_1=}")
        bs_error_abort("Done first 2SLS")

        if verbose:
            _print_pseudo_true_errors(true_p, pseudo_vals, names_ptv, verbose=True)

        # the estimated mean utilities
        mean_utils_0 = _mean_utils(beta0_0, beta_0, x)
        mean_utils_2 = _mean_utils(beta0_2, beta_2, x)

        # the estimated approximate xi
        xi_0 = ymat - mean_utils_0
        xi_0_vec = xi_0.reshape(npts)
        xi2_2 = ymat - mean_utils_2 - (Kmat @ s2_2).reshape((nmarkets, nproducts))
        xi_2 = cast(np.ndarray, xi2_2.reshape(npts))

        # end = time.time()
        # print(f"2SLS took {end - start} seconds")

        #################################################################################
        ##                        the what-if second-order version                     ##
        #################################################################################

        Z_used = Zstar2
        moments_used = Zstar2

        # moments_used_centered = center_moments(moments_used, nproducts)

        omega_inv = make_omega_inv(moments_used)
        Omega = np.linalg.inv(omega_inv)
        if verbose:
            print_stars(f"eigenvalues of Omega:\n{np.linalg.eigvals(Omega)}")

        whatif_just_vals = estimate_what_if(
            xmat, Kmat, Whalf, beta0_0, beta_0, xi_0_vec, Z_used, Omega
        )

        Z_used = Z_powers
        moments_used = Z_powers
        omega_inv = make_omega_inv(moments_used)
        Omega = np.linalg.inv(omega_inv)
        if verbose:
            print_stars(f"eigenvalues of Omega:\n{np.linalg.eigvals(Omega)}")

        whatif_over_vals = estimate_what_if(
            xmat, Kmat, Whalf, beta0_0, beta_0, xi_0_vec, Z_used, Omega
        )

        print_stars("True ; estimates SW, just, over:")
        for i in range(n_params):
            print(
                f"{names_ptv[i]:>9}: {true_p[i]: .3f};",
                f"  {pseudo_vals[i]: .3f},",
                f"  {whatif_just_vals[i]: .3f},",
                f"  {whatif_over_vals[i]: .3f}",
            )
        ##              now we work on the semi-elasticities                           ##
        #################################################################################

        # start = time.time()

        nonrandom_own_semi, nonrandom_cross_semi = _nonrandom_semi_elasticities(
            nonrandom_vals, observed_shares_mat, x
        )

        pseudo_own_semi, pseudo_cross_semi = _pseudo_semi_elasticities_ift(
            pseudo_vals, observed_shares_mat, x
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
        resus_pseudo_semi_elast = get_semi_elast_stats(
            pseudo_own_semi, pseudo_cross_semi, nproducts
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

        nonrandom_values[isig, : 1 + n_x] = nonrandom_vals
        nonrandom_values[isig, 1 + n_x :] = 0.0
        pseudo_true_values[isig, :] = pseudo_vals
        whatif_just_values[isig, :] = whatif_just_vals
        whatif_over_values[isig, :] = whatif_over_vals
        sp_bounds[isig, :, :] = spb
        values_nonrandom_semi_elast[isig] = resus_nonrandom_semi_elast
        values_pseudo_semi_elast[isig] = resus_pseudo_semi_elast
        values_whatif_just_semi_elast[isig] = resus_whatif_just_semi_elast
        values_whatif_over_semi_elast[isig] = resus_whatif_over_semi_elast
        values_true_semi_elast[isig] = resus_true_semi_elast
        cond_numbers2[isig] = cond_number2
        cond_numbers_bounds[isig] = cond_bounds
        if (
            save_more
            and ZZ is not None
            and Zy is not None
            and xi_vals is not None
            and estimated_xi2 is not None
            and errors_xi2 is not None
        ):
            ZZ[isig, :, :] = Zstar2_T @ Zstar2
            Zy[isig, :] = Zstar2_T @ y_proj
            if (
                do_a_second
                and ZV is not None
                and ZW is not None
                and xiV is not None
                and xiW is not None
                and V_proj is not None
                and W_proj is not None
            ):
                ZV[isig] = Zstar2_T @ V_proj
                ZW[isig] = np.einsum("ip,imn->pmn", Zstar2, W_proj)
                xiV[isig] = xi_2 @ V_proj
                xiW[isig] = np.einsum("i,imn->mn", xi_2, W_proj)
            xi_vals[isig, :, :] = true_xi
            estimated_xi2[isig, :, :] = xi2_2
            errors_xi2[isig, :, :] = errors2
        done_message = f"                     Done with sigma value {isig + 1}/{n_sigmas} for J = {nproducts}, "
        done_message += f"scenario {i_scenario}, model: {str_long}"

        f_print_stars(use_mp, done_message, fout_name)

    dict_results = {
        "model": model,
        "n_x": n_x,
        "non-random values": nonrandom_values,
        "pseudo true values": pseudo_true_values,
        "whatif just values": whatif_just_values,
        "whatif over values": whatif_over_values,
        "omega_inv eigenvalues": omega_inv_eigenvalues,
        "omega_inv eigenvectors": omega_inv_eigenvectors,
        "SPE variance bounds": sp_bounds,
        "true semi-elasticities": values_true_semi_elast,
        "non-random semi-elasticities": values_nonrandom_semi_elast,
        "pseudo semi-elasticities": values_pseudo_semi_elast,
        "whatif just semi-elasticities": values_whatif_just_semi_elast,
        "whatif over semi-elasticities": values_whatif_over_semi_elast,
        "mean squared residuals": mean_squared_residuals,
        "condition number 2": cond_numbers2,
        "condition number bounds": cond_numbers_bounds,
    }

    if (
        save_more
        and ZZ is not None
        and Zy is not None
        and ZV is not None
        and ZW is not None
        and xiV is not None
        and xiW is not None
        and xi_vals is not None
        and errors_xi2 is not None
        and estimated_xi2 is not None
    ):
        more_results = {
            "ZprimeZ": ZZ,
            "Zprimey": Zy,
            "ZprimeV": ZV,
            "ZprimeW": ZW,
            "xiprimeV": xiV,
            "xiprimeW": xiW,
            "true_xi": xi_vals,
            "errors_xi2": errors_xi2,
            "estimated_xi2": estimated_xi2,
        }
        dict_results |= more_results

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

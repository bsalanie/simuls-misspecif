import os
import pickle
import tracemalloc
from typing import cast

import numpy as np
from bs_python_utils.bs_mem import memory_display_top, memory_display_top_diffs
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from bs_python_utils.bsnputils import ThreeArrays
from bs_python_utils.bsutils import print_stars

from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.evaluations import (
    _artificial_regressors,
    _nonrandom_semi_elasticities,
    _our_tsls0,
    _our_tsls2,
    _print_pseudo_true_errors,
    _print_set_semi_elast,
    _project_variables,
    # _pseudo_semi_elasticities,
    _pseudo_semi_elasticities_anal,
    _true_optimal_instruments,
    _true_semi_elasticities,
)
from simuls_misspecif.MNL_params import do_a_second, do_bounds_semi_elast
from simuls_misspecif.MNL_utils import _mean_utils
from simuls_misspecif.utils import f_print_stars


def get_the_stats(case: list, save_more: bool = False) -> dict:
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
    stream, model, isim, pickle_dir, use_mp = case
    print_stars(f"Calling get stats for {isim}")

    if use_mp:
        fout_name = str(os.getpid()) + ".out"
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
    i_scenario, str_long = model.scenario, model.long_name
    sigma_range = model.sigma_range
    npts = nmarkets * nproducts

    nodes1, weights1 = setup_sparse_gaussian(1, iprec)

    true_pars, data_pars = model.true_pars, model.data_pars
    sigxi = data_pars.sigxi
    true_beta0, true_beta1 = true_pars.beta0, true_pars.beta1

    n_elast = 1 if nproducts == 1 else 2
    n_sigmas = sigma_range.size
    n_instr = 3
    m = nproducts * n_instr

    n_params = 3
    coeffs_shape = (n_sigmas,)
    names_ptv = ["beta0", "beta1", "sigma2"]
    names_spb = ["beta0", "beta1", "sigma2"]

    nonrandom_values = np.zeros(coeffs_shape + (n_params,))
    pseudo_true_values = np.zeros(coeffs_shape + (n_params,))
    whatif_values = np.zeros(coeffs_shape + (n_params,))
    sp_bounds = np.zeros(coeffs_shape + (n_params, n_params))
    cond_numbers2 = np.zeros(coeffs_shape)
    cond_numbers_bounds = np.zeros(coeffs_shape)
    omega_inv_eigenvalues = np.zeros(coeffs_shape + (m,))
    omega_inv_eigenvectors = np.zeros(coeffs_shape + (m, m))

    shape_elast = coeffs_shape + (2 * n_elast,)

    values_nonrandom_semi_elast = np.zeros(shape_elast)
    values_pseudo_semi_elast = np.zeros(shape_elast)
    values_whatif_semi_elast = np.zeros(shape_elast)
    values_true_semi_elast = np.zeros(shape_elast)
    if do_bounds_semi_elast:
        pass

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
        ZV = np.zeros(coeffs_shape + (n_params,))
        ZW = np.zeros(coeffs_shape + (n_params,))
        xiV = np.zeros(coeffs_shape)
        xiW = np.zeros(coeffs_shape)

    snapshot1 = None

    if do_trace_memory:
        snapshot1 = tracemalloc.take_snapshot()
        memory_display_top(snapshot1)

    # create the data, except for the shares
    draws = data_pars.generate_random_draws(nmarkets, nproducts, stream)
    true_xi, x, z = data_pars.generate_exogenous_vars_from_draws(draws)
    xmat = x.reshape((npts, 1))
    xvec = xmat[:, 0]

    for isig, sigma_val in enumerate(sigma_range):
        V_proj: np.ndarray | None = None
        W_proj: np.ndarray | None = None

        true_mean_utils = _mean_utils(true_beta0, true_beta1, x)
        true_mean_utils_xi = true_mean_utils + true_xi

        # generate the shares
        observed_shares_mat = make_shares(true_mean_utils_xi, x, sigma_val)
        observed_shares_vec = observed_shares_mat.reshape(npts)

        sig2 = sigma_val * sigma_val

        Kmat, yvec, Vmat, Warr = _artificial_regressors(
            observed_shares_vec, xmat, nproducts
        )
        Kvec = Kmat[:, 0]
        Vvec = Vmat[:, 0]
        Wvec = Warr[:, 0, 0]
        ymat = yvec.reshape((nmarkets, nproducts))
        Kmat = Kvec.reshape((nmarkets, nproducts))

        # project the variables on the instruments
        if do_a_second:
            y_proj, x_proj, K_proj, V_proj, W_proj = cast(
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                _project_variables(yvec, xvec, z, Kvec, Vvec, Wvec, mode=mode),
            )
        else:
            y_proj, x_proj, K_proj = cast(
                ThreeArrays, _project_variables(yvec, xvec, z, Kvec, mode=mode)
            )

        # true_p contains the true values of the coefficients we estimate in TSLS
        true_p = np.array([true_beta0, true_beta1, sig2])

        # evaluate xi(0, 2) - xi(infty)
        true_xi0 = ymat - true_mean_utils
        true_xi2 = true_xi0 - sig2 * Kmat
        errors2 = true_xi2 - true_xi

        #################################################################################
        ##                        our TSLS                                             ##
        #################################################################################
        # start = time.time()
        nonrandom_vals = _our_tsls0(y_proj, x_proj)[1]
        beta0_0 = nonrandom_vals[0]
        beta1_0 = nonrandom_vals[1]

        Zstar2, pseudo_vals, cond_number2 = _our_tsls2(y_proj, x_proj, K_proj)
        Zstar2_T = Zstar2.T

        beta0_2 = pseudo_vals[0]
        beta1_2 = pseudo_vals[1]
        s2_2 = pseudo_vals[-1]

        if verbose:
            _print_pseudo_true_errors(true_p, pseudo_vals, names_ptv, verbose=True)

        # the estimated approximate xi
        xi_0 = ymat - _mean_utils(beta0_0, beta1_0, x)
        xi_0_vec = xi_0.reshape(npts)
        xi2_2 = ymat - _mean_utils(beta0_2, beta1_2, x) - s2_2 * Kmat
        xi_2 = cast(np.ndarray, xi2_2.reshape(npts))

        # the estimated mean utilities
        # mean_utils_2 = _mean_utils(beta0_2, beta1_2, x)

        # end = time.time()
        # print(f"2SLS took {end - start} seconds")

        #################################################################################
        ##                        the what-if second-order version                     ##
        #################################################################################
        # Placeholders for future what-if refinement.
        # Z_used = _reformat_Zstar(Zstar2, nproducts)
        # moments = np.einsum("jkt,tj->jkt", Z_used, xi_0)
        # moments_means = moments.mean(axis=2)
        # moments_recentered = moments - moments_means[:, :, None]
        # moments_varcov = _reformat_varcov(
        #     np.einsum("jkt,lmt->jklm", moments_recentered, moments_recentered)
        # )
        # omega = spla.inv(moments_varcov)
        # omega = np.eye(m)
        # Zp_Z = _reformat_varcov(np.einsum("jkt,lmt->jklm", Z_used, Z_used) / nmarkets)
        # omega = spla.inv(Zp_Z)
        # # omega0 = spla.inv(z.T @ z / nmarkets)
        # # omega = omega0
        # # print(f"{omega=}, {omega0=}")
        # # bs_error_abort("Done")
        # zp_1 = Z_used.mean(axis=2).reshape(m)
        # zp_x = np.einsum("jkt,tj->jk", Z_used, X_TJ) / nmarkets
        # zp_x = zp_x.reshape(m)
        # zp_K = np.einsum("jkt,tj->jk", Z_used, Kmat) / nmarkets
        # zp_K = zp_K.reshape(m)
        # zp_W = np.einsum("jkt,tj->jk", Z_used, Wmat) / nmarkets
        # zp_W = zp_W.reshape(m)
        # print(f"{zp_1.shape=}, {zp_x.shape=}, {zp_W.shape}")
        # zp1_zp1 = angle_product(zp_1, zp_1, omega)
        # zp1_zpx = angle_product(zp_1, zp_x, omega)
        # zpx_zpx = angle_product(zp_x, zp_x, omega)
        # zpX_zpX = np.array([[zp1_zp1, zp1_zpx], [zp1_zpx.T, zpx_zpx]])
        # zp1_zpK = angle_product(zp_1, zp_K, omega)
        # zpx_zpK = angle_product(zp_x, zp_K, omega)
        # zpK_zpK = angle_product(zp_K, zp_K, omega)
        # moment_means_vec = moments_means.reshape(m)
        # zpW_zpxi0 = angle_product(zp_W, moment_means_vec, omega)
        # zpK_zpxi0 = angle_product(zp_K, moment_means_vec, omega)
        # lhs_mat = np.zeros((3, 3))
        # lhs_mat[:2, :2] = zpX_zpX
        # lhs_mat[0, 2] = zp1_zpK
        # lhs_mat[1, 2] = zpx_zpK
        # lhs_mat[2, 0] = zp1_zpK
        # lhs_mat[2, 1] = zpx_zpK
        # lhs_mat[2, 2] = zpK_zpK  # - 2.0 * zpW_zpxi0
        # rhs_vec = np.zeros(3)
        # rhs_vec[2] = zpK_zpxi0

        # omega_inv =

        # dbeta_s2 = spla.solve(lhs_mat, rhs_vec)

        # Zarr = np.zeros((nmarkets, nproducts, n_instr))
        # for t in range(nmarkets):
        #     Zarr[t, :, :] = Zstar2_T[:, t * nproducts : (t + 1) * nproducts].T

        #     eS_x =  X_TJ @ observed_shares
        # squared_eS_X2 = np.outer(eS_X2, eS_X2)
        # eS_X2_X2 = np.einsum("j,jm,jn->mn", shares, X2, X2)
        # covar_eS = eS_X2_X2 - squared_eS_X2
        # X2_sq = np.einsum("jm,jn->jmn", X2, X2)
        # Xm_eSXn = np.einsum("jm,n->jmn", X2, eS_X2)
        # Xn_eSXm = np.einsum("jmn->jnm", Xm_eSXn)
        # Wvec = np.einsum("jmn,mn->jmn", Xm_eSXn + Xn_eSXm - X2_sq, covar_eS)

        eS_x = np.sum(observed_shares_mat * x, axis=1)
        sq_eS_x = eS_x * eS_x
        eS_x2 = np.sum(observed_shares_mat * x * x, axis=1)
        Wmat = (
            x * (-x + 2.0 * eS_x.reshape((-1, 1))) * (eS_x2 - sq_eS_x).reshape((-1, 1))
        )
        Wvec = Wmat.reshape(npts) / 2.0

        # Z_xi = Zstar2 * xi_0_vec.reshape((-1, 1))
        # Z_xi_centered = np.zeros_like(Z_xi)
        # Zstar2_centered = np.zeros_like(Zstar2)
        # for k in range(n_instr):
        #     Z_xi_k = Z_xi[:, k].reshape((nmarkets, nproducts))
        #     Zstar2_k = Zstar2[:, k].reshape((nmarkets, nproducts))
        #     Z_xi_k_mean = Z_xi_k.mean(axis=0)
        #     print(f"{k=}, {Z_xi_k_mean=}")
        #     Zstar2_k_mean = Zstar2_k.mean(axis=0)
        #     Z_xi_centered[:, k] = (Z_xi_k - Z_xi_k_mean).reshape(npts)
        #     Zstar2_centered[:, k] = (Zstar2_k - Zstar2_k_mean).reshape(npts)

        omega_inv = np.zeros((n_instr, n_instr))
        for k in range(n_instr):
            for kp in range(n_instr):
                omega_inv[k, kp] = np.mean(Zstar2[:, k] * Zstar2[:, kp])
        print_stars(f"omega_inv:\n{omega_inv}")
        # for k in range(n_instr):
        #     for kp in range(n_instr):
        #         omega_inv[k, kp] = np.mean(Z_xi_centered[:, k] * Z_xi_centered[:, kp])

        omega = np.linalg.inv(omega_inv)

        def ap2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            Z_a = np.zeros(n_instr)
            Z_b = np.zeros(n_instr)
            for k in range(n_instr):
                Z_a[k] = np.mean(Zstar2[:, k] * a)
                Z_b[k] = np.mean(Zstar2[:, k] * b)
            return Z_a.T @ omega @ Z_b

        ones = np.ones(npts)

        lhs_mat = np.zeros((3, 3))
        lhs_mat[0, 0] = ap2(ones, ones)
        lhs_mat[0, 1] = ap2(ones, xvec)
        lhs_mat[1, 0] = ap2(xvec, ones)
        lhs_mat[1, 1] = ap2(xvec, xvec)
        lhs_mat[0, 2] = ap2(ones, Kvec)
        lhs_mat[2, 0] = ap2(Kvec, ones)
        lhs_mat[1, 2] = ap2(xvec, Kvec)
        lhs_mat[2, 1] = ap2(Kvec, xvec)
        lhs_mat[2, 2] = ap2(Kvec, Kvec) - 2.0 * ap2(Wvec, xi_0_vec)

        print_stars(f"{ap2(Kvec, Kvec)=}  and {-2.0 * ap2(Wvec, xi_0_vec)=}")

        rhs_vec = np.zeros(3)
        rhs_vec[2] = ap2(Kvec, xi_0_vec)

        dbeta_s2 = np.linalg.solve(lhs_mat, rhs_vec)
        dbeta, s2_whatif = dbeta_s2[:2], dbeta_s2[2]

        # dbeta_ddelta = -np.linalg.solve(lhs_mat[:2, :2], lhs_mat[:2, 2])
        # s2_whatif = ap2(Kvec, xi_0_vec) / (
        #     ap2(Kvec, Kvec) - 2.0 * ap2(Wvec, xi_0_vec) + lhs_mat[2, :2] @ dbeta_ddelta
        # )
        # dbeta = dbeta_ddelta * s2_whatif

        print_stars(f"{dbeta=}, {s2_whatif=}")

        beta0_whatif = beta0_0 + dbeta[0]
        beta1_whatif = beta1_0 + dbeta[1]
        whatif_vals = np.array([beta0_whatif, beta1_whatif, s2_whatif])

        ##              now we work on the semi-elasticities                           ##
        #################################################################################

        # start = time.time()

        nonrandom_own_semi, nonrandom_cross_semi = _nonrandom_semi_elasticities(
            nonrandom_vals, observed_shares_mat, x
        )

        # pseudo_own_semi, pseudo_cross_semi = _pseudo_semi_elasticities(
        #     pseudo_vals, observed_shares_mat, x
        # )

        pseudo_own_semi, pseudo_cross_semi = _pseudo_semi_elasticities_anal(
            pseudo_vals, observed_shares_mat, x
        )

        whatif_own_semi, whatif_cross_semi = _pseudo_semi_elasticities_anal(
            whatif_vals, observed_shares_mat, x
        )

        true_own_semi, true_cross_semi, dshares_dx = _true_semi_elasticities(
            true_p, observed_shares_mat, x, true_mean_utils_xi, nodes1, weights1
        )

        mean_true_own_semi_elast = float(np.mean(true_own_semi))
        stderr_true_own_semi_elast = float(np.std(true_own_semi))
        mean_pseudo_own_semi_elast = float(np.mean(pseudo_own_semi))
        stderr_pseudo_own_semi_elast = float(np.std(pseudo_own_semi))
        mean_whatif_own_semi_elast = float(np.mean(whatif_own_semi))
        stderr_whatif_own_semi_elast = float(np.std(whatif_own_semi))
        mean_nonrandom_own_semi_elast = float(np.mean(nonrandom_own_semi))
        stderr_nonrandom_own_semi_elast = float(np.std(nonrandom_own_semi))

        if nproducts > 1:
            mean_true_cross_semi_elast = float(np.mean(true_cross_semi))
            stderr_true_cross_semi_elast = float(np.std(true_cross_semi))
            mean_pseudo_cross_semi_elast = float(np.mean(pseudo_cross_semi))
            stderr_pseudo_cross_semi_elast = float(np.std(pseudo_cross_semi))
            mean_whatif_cross_semi_elast = float(np.mean(whatif_cross_semi))
            stderr_whatif_cross_semi_elast = float(np.std(whatif_cross_semi))
            mean_nonrandom_cross_semi_elast = float(np.mean(nonrandom_cross_semi))
            stderr_nonrandom_cross_semi_elast = float(np.std(nonrandom_cross_semi))
            resus_nonrandom_semi_elast = _print_set_semi_elast(
                (
                    mean_nonrandom_own_semi_elast,
                    stderr_nonrandom_own_semi_elast,
                    mean_nonrandom_cross_semi_elast,
                    stderr_nonrandom_cross_semi_elast,
                ),
                "Non-random",
            )
            resus_pseudo_semi_elast = _print_set_semi_elast(
                (
                    mean_pseudo_own_semi_elast,
                    stderr_pseudo_own_semi_elast,
                    mean_pseudo_cross_semi_elast,
                    stderr_pseudo_cross_semi_elast,
                ),
                "With K",
            )
            resus_whatif_semi_elast = _print_set_semi_elast(
                (
                    mean_whatif_own_semi_elast,
                    stderr_whatif_own_semi_elast,
                    mean_whatif_cross_semi_elast,
                    stderr_whatif_cross_semi_elast,
                ),
                "What-if",
            )
            resus_true_semi_elast = _print_set_semi_elast(
                (
                    mean_true_own_semi_elast,
                    stderr_true_own_semi_elast,
                    mean_true_cross_semi_elast,
                    stderr_true_cross_semi_elast,
                ),
                "True",
            )
        else:
            resus_nonrandom_semi_elast = _print_set_semi_elast(
                (
                    mean_nonrandom_own_semi_elast,
                    stderr_nonrandom_own_semi_elast,
                ),
                "Non-random",
            )
            resus_pseudo_semi_elast = _print_set_semi_elast(
                (
                    mean_pseudo_own_semi_elast,
                    stderr_pseudo_own_semi_elast,
                ),
                "With K",
            )
            resus_whatif_semi_elast = _print_set_semi_elast(
                (
                    mean_whatif_own_semi_elast,
                    stderr_whatif_own_semi_elast,
                ),
                "What-if",
            )
            resus_true_semi_elast = _print_set_semi_elast(
                (
                    mean_true_own_semi_elast,
                    stderr_true_own_semi_elast,
                ),
                "True",
            )

        # end = time.time()
        # print(f"semi-elast took {end - start} seconds")

        #################################################################################
        ##              now we work on SPE bounds                                      ##
        #################################################################################

        # start = time.time()
        Zstar = _true_optimal_instruments(
            true_p,
            true_mean_utils_xi,
            observed_shares_mat,
            x,
            x_proj,
            z,
            nodes1,
            weights1,
            mode=mode,
        )
        Zstar_T = Zstar.T
        exp_dxi_zstar = (Zstar_T @ Zstar) / npts
        s = np.linalg.svd(exp_dxi_zstar, compute_uv=False)
        cond_bounds = abs(s[0] / s[-1])
        exp_dxi_zstar_inv = np.linalg.inv(exp_dxi_zstar)
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

        # end = time.time()
        # print(f"spe bounds took {end - start} seconds")

        nonrandom_values[isig, :2] = nonrandom_vals
        nonrandom_values[isig, 2] = 0.0
        pseudo_true_values[isig, :] = pseudo_vals
        whatif_values[isig, :] = whatif_vals
        sp_bounds[isig, :, :] = spb
        values_nonrandom_semi_elast[isig, :] = resus_nonrandom_semi_elast
        values_pseudo_semi_elast[isig, :] = resus_pseudo_semi_elast
        values_whatif_semi_elast[isig, :] = resus_whatif_semi_elast
        values_true_semi_elast[isig, :] = resus_true_semi_elast
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
                ZV[isig, :] = Zstar2_T @ V_proj
                ZW[isig, :] = Zstar2_T @ W_proj
                xiV[isig] = np.dot(xi_2, V_proj)
                xiW[isig] = np.dot(xi_2, W_proj)
            xi_vals[isig, :, :] = true_xi
            estimated_xi2[isig, :, :] = xi2_2
            errors_xi2[isig, :, :] = errors2
        done_message = f"                     Done with sigma value {isig + 1}/{n_sigmas} for J = {nproducts}, "
        done_message += f"scenario {i_scenario}, model: {str_long}"

        f_print_stars(use_mp, done_message, fout_name)

    dict_results = {
        "model": model,
        "non-random values": nonrandom_values,
        "pseudo true values": pseudo_true_values,
        "whatif values": whatif_values,
        "omega_inv eigenvalues": omega_inv_eigenvalues,
        "omega_inv eigenvectors": omega_inv_eigenvectors,
        "SPE variance bounds": sp_bounds,
        "true semi-elasticities": values_true_semi_elast,
        "non-random semi-elasticities": values_nonrandom_semi_elast,
        "pseudo semi-elasticities": values_pseudo_semi_elast,
        "whatif semi-elasticities": values_whatif_semi_elast,
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

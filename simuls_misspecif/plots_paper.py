"""Plotly version of the paper plots, with SPE confidence bands."""

import pickle
from math import sqrt
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
from bs_python_utils.bs_mathstr import uni_beta0, uni_beta1, uni_sigma2
from bs_python_utils.bsnputils import check_matrix, check_vector
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed, print_stars

from simuls_misspecif.extract_from_results import case_paths

fig_fmt = "png"


# the maximum number of covariates whose coefficients are plotted
MAX_X_PLOTTED = 3


def _get_result(dict_results: dict, varname: str):
    """Retrieve a variable from the results dictionary.

    Args:
        dict_results: Dictionary of simulation results.
        varname: Name of the variable to retrieve.

    Returns:
        The requested variable from the results.
    """
    dict_var = dict_results[varname]
    return dict_var


def _stack_cols(mat: np.ndarray) -> np.ndarray:
    """Stack matrix columns into a single 1D array.

    Args:
        mat: 2D array with multiple columns.

    Returns:
        1D array with columns stacked horizontally.
    """
    v = mat[:, 0]
    for i in range(1, mat.shape[1]):
        v = np.hstack((v, mat[:, i]))
    return v


def _stack_estimates(
    estimate_names: Union[str, list[str]], estimates: np.ndarray, df: pd.DataFrame
):
    """Add dataframe columns for various estimates of one coefficient.

    Args:
        estimate_names: One estimate name or a list of names.
        estimates: Estimated values.
        df: Input dataframe.

    Returns:
        A dataframe with the estimates stacked alongside the true value and
        the ordered estimate labels.
    """
    df1 = df.copy()
    if isinstance(estimate_names, str):
        n_estimates = 1
        estimate_names = [estimate_names]
    else:
        n_estimates = len(estimate_names)
    if n_estimates == 1:
        size_est = check_vector(estimates, "_stack_estimates")
        if size_est != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and {size_est} estimators"
            )
        df1[estimate_names[0]] = estimates
        ordered_estimates = [estimate_names[0], "True value"]
    else:
        shape_est = check_matrix(estimates, "_stack_estimates")
        if shape_est[1] != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and {shape_est[1]} estimators"
            )
        for i_est, est_name in enumerate(estimate_names):
            df1[est_name] = estimates[:, i_est]
        ordered_estimates = ["True value"] + estimate_names

    return df1, ordered_estimates


_subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _param_labels(n_x: int) -> list[str]:
    """Plot labels for `[beta0, beta_1..beta_M, sigma2_1..sigma2_M]`."""
    if n_x == 1:
        return [uni_beta0, uni_beta1, uni_sigma2]
    return (
        [uni_beta0]
        + [f"β{str(m + 1).translate(_subscripts)}" for m in range(n_x)]
        + [f"σ²{str(m + 1).translate(_subscripts)}" for m in range(n_x)]
    )


def _true_coeffs(true_pars) -> tuple[float, np.ndarray, np.ndarray]:
    """beta0, beta, and sigma_profile; also works for pickles with one covariate."""
    if hasattr(true_pars, "beta"):
        return true_pars.beta0, true_pars.beta, true_pars.sigma_profile
    return true_pars.beta0, np.array([true_pars.beta1]), np.ones(1)


def _make_suffix(nproducts: int, do_exo: bool) -> str:
    """Create a descriptive suffix for plot titles.

    Args:
        nproducts: Number of products (J).
        do_exo: If True, generate suffix for exogenous model; else endogenous.

    Returns:
        Descriptive string for plot titles and labels.
    """
    if do_exo:
        suffix = f"J = {nproducts}, exogenous"
    else:
        suffix = f"J = {nproducts}, endogenous"
    return suffix


def new_plots_paper(
    str_model: str,
    nproducts: int,
    nmarkets: int,
    selected_scenario_numbers: list[int],
    plot_pseudo_with_bounds: bool = True,
    plot_semi_elast: bool = True,
    simuls_dir: Path | None = None,
    do_bounds: bool = True,
    n_x: int = 1,
    select_seed: int | None = None,
):
    """Plot the pseudo-true values and the semi-elasticities of some cases.

    With more than `MAX_X_PLOTTED` covariates, the pseudo-true values are plotted
    only for `beta0` and for the `beta_m` and `sigma2_m` of `MAX_X_PLOTTED`
    covariates drawn at random.

    Args:
        str_model: `endo` or `exo`.
        nproducts: Number of products J.
        nmarkets: Number of markets T.
        selected_scenario_numbers: The scenarios to plot.
        plot_pseudo_with_bounds: Whether to plot the pseudo-true values.
        plot_semi_elast: Whether to plot the semi-elasticities.
        simuls_dir: The root directory of the results; default: the current directory.
        n_x: Number of covariates M.
        select_seed: Seed for the random choice of covariates; `None` for a fresh draw.
    """
    if simuls_dir is None:
        simuls_dir = Path.cwd()

    spe_bounds_nmarkets = 100  # used for the SPE bounds
    lower_bound_str = f"95% CI- (T = {spe_bounds_nmarkets})"
    upper_bound_str = f"95% CI+ (T = {spe_bounds_nmarkets})"

    for i_scenario in selected_scenario_numbers:
        case_dir, full_str = case_paths(
            str_model, nproducts, nmarkets, i_scenario, simuls_dir, n_x
        )
        with open(case_dir / f"simul_results_{full_str}.pkl", "rb") as f:
            dict_results = pickle.load(f)

        figures_dir = mkdir_if_needed(case_dir / "figures_paper")
        # print_stars(f"Figures will be saved in {figures_dir}")

        model = dict_results["model"]
        data_pars = model.data_pars
        do_exo = data_pars.do_exo
        sigma_range = model.sigma_range

        n_sigmas = sigma_range.size

        print_stars(f"Plotting model {full_str}")
        nonrandom_vals = _get_result(dict_results, "non-random values")
        pseudo_vals = _get_result(dict_results, "pseudo true values")
        whatif_just_vals = _get_result(dict_results, "whatif just values")
        whatif_over_vals = _get_result(dict_results, "whatif over values")
        spb = _get_result(dict_results, "SPE variance bounds")
        nonrandom_semi = _get_result(dict_results, "non-random semi-elasticities")
        true_semi = _get_result(dict_results, "true semi-elasticities")
        pseudo_semi = _get_result(dict_results, "pseudo semi-elasticities")
        whatif_just_semi = _get_result(dict_results, "whatif just semi-elasticities")
        whatif_over_semi = _get_result(dict_results, "whatif over semi-elasticities")
        # old pickles have one covariate and no M axis
        semis = [
            nonrandom_semi,
            true_semi,
            pseudo_semi,
            whatif_just_semi,
            whatif_over_semi,
        ]
        semis = [sem[:, np.newaxis, :] if sem.ndim == 2 else sem for sem in semis]
        nonrandom_semi, true_semi, pseudo_semi, whatif_just_semi, whatif_over_semi = (
            semis
        )

        n_pars = pseudo_vals.shape[-1]
        n_x_res = (n_pars - 1) // 2

        # we compute standard errors for SPE bounds, putting in zero if the variance is negative
        stb = np.zeros((n_sigmas, n_pars))
        for isig in range(n_sigmas):
            spb_isig = np.maximum(np.diag(spb[isig, :, :]), 0.0)
            stb[isig, :] = np.sqrt(spb_isig)

        true_beta0, true_beta, sigma_profile = _true_coeffs(model.true_pars)
        sigma2_range = sigma_range * sigma_range
        true_values = np.zeros_like(pseudo_vals)
        true_values[:, 0] = true_beta0
        true_values[:, 1 : 1 + n_x_res] = true_beta
        true_values[:, 1 + n_x_res :] = np.outer(sigma2_range, sigma_profile**2)

        order_parameters = _param_labels(n_x_res)

        # Add note if bounds were not computed
        # with many covariates, we only plot the coefficients of a few of them
        if n_x_res > MAX_X_PLOTTED:
            rng = np.random.default_rng(select_seed)
            x_plotted = np.sort(rng.choice(n_x_res, MAX_X_PLOTTED, replace=False))
            str_plotted = ", ".join(str(m + 1) for m in x_plotted)
            ptitle_pars = (
                f"{_make_suffix(nproducts, do_exo)}; covariates {str_plotted}"
                f" out of M = {n_x_res}"
            )
        else:
            x_plotted = np.arange(n_x_res)
            ptitle_pars = _make_suffix(nproducts, do_exo)
        pars_plotted = (
            [0] + [1 + m for m in x_plotted] + [1 + n_x_res + m for m in x_plotted]
        )
        n_pars_plotted = len(pars_plotted)

        suffix = _make_suffix(nproducts, do_exo)
        ptitle = suffix
        uni_string2 = uni_sigma2
        margin = 5.0

        ordered_colors = ["black"] * 3 + ["red", "green", "blue", "purple"]
        estimates_names = [
            "True value",
            "Non-random",
            "Salanie-Wolak",
            "What if - just",
            "What if - over",
        ]
        estimated_values = np.zeros((n_sigmas, 7, n_pars))
        estimated_values[:, 2, :] = true_values
        estimated_values[:, 3, :] = np.clip(
            nonrandom_vals,
            true_values - margin,
            true_values + margin,
        )
        estimated_values[:, 4, :] = np.clip(
            pseudo_vals,
            true_values - margin,
            true_values + margin,
        )
        estimated_values[:, 5, :] = np.clip(
            whatif_just_vals,
            true_values - margin,
            true_values + margin,
        )
        estimated_values[:, 6, :] = np.clip(
            whatif_over_vals,
            true_values - margin,
            true_values + margin,
        )

        if plot_pseudo_with_bounds:
            df1 = []
            for ipar in pars_plotted:
                par_name = order_parameters[ipar]
                df_i = pd.DataFrame(
                    {
                        uni_string2: sigma2_range,
                        "True value": true_values[:, ipar],
                    }
                )
                # Only add bounds if they were computed
                if do_bounds:
                    bound_i = stb[:, ipar] / sqrt(spe_bounds_nmarkets)
                    estimated_values[:, 0, ipar] = df_i["True value"] - 1.96 * bound_i
                    estimated_values[:, 1, ipar] = df_i["True value"] + 1.96 * bound_i
                    estimates_to_plot = [
                        lower_bound_str,
                        upper_bound_str,
                    ] + estimates_names
                else:
                    # Skip bounds, just plot the estimates
                    estimates_to_plot = estimates_names

                df1_ipar, ordered_estimates = _stack_estimates(
                    estimates_to_plot,
                    estimated_values[..., ipar],
                    df_i,
                )
                df1_ipar["Coefficient"] = par_name
                df1.append(df1_ipar)

            df2: pd.DataFrame = pd.concat(df1)
            df2m = pd.melt(
                df2,
                id_vars=[uni_string2, "Coefficient"],
                value_vars=ordered_estimates,
                var_name="Estimate",
            )

            # beta0 and the betas in the first row, the sigma2s in the second
            #   (three panels per row with two covariates)
            n_x_plotted = x_plotted.size
            facet_col_wrap = (
                0
                if n_pars_plotted <= 3
                else (3 if n_x_plotted == 2 else n_x_plotted + 1)
            )
            fig = px.line(
                df2m,
                x=uni_string2,
                y="value",
                facet_col="Coefficient",
                facet_col_wrap=facet_col_wrap,
                facet_row_spacing=0.08,
                color="Estimate",
                color_discrete_sequence=ordered_colors,
                line_dash="Estimate",
                line_dash_sequence=["solid"] + ["dot"] * 2 + ["solid"] * 4,
                template="plotly_white",
                facet_col_spacing=0.12,
                title=(
                    f"Pseudo-true values and efficiency bounds<br><sup>{ptitle_pars}</sup>"
                    if do_bounds
                    else f"Pseudo-true values<br><sup>{_make_suffix(nproducts, do_exo)}</sup>"
                ),
            )

            # show only the symbol for the coefficient on top of each panel
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            fig.update_yaxes(
                matches=None, showticklabels=True
            )  # independent y axis with their own ticks
            if n_pars_plotted > 3:
                n_rows = -(-n_pars_plotted // facet_col_wrap)
                width = 1000 if facet_col_wrap == 3 else 1250
                fig.update_layout(width=width, height=150 + 300 * n_rows)

            fig_save_ptv_root = f"{figures_dir}/new_pseudo_vals_{full_str}"
            fig.write_image(f"{fig_save_ptv_root}.{fig_fmt}")

            fig.update_xaxes(rangeslider_visible=True)
            fig.write_html(f"{fig_save_ptv_root}.html")

            if plot_semi_elast:
                if n_x_res == 1:
                    semi_stats = [
                        "Mean own semi-elasticity",
                        "Cross-market dispersion of own semi-elasticity",
                        "Mean cross semi-elasticity",
                        "Cross-market dispersion of cross semi-elasticity",
                    ]
                else:  # shorter titles, as the panels are narrower
                    semi_stats = [
                        "Mean own",
                        "Dispersion own",
                        "Mean cross",
                        "Dispersion cross",
                    ]
                if nproducts == 1:
                    semi_stats = semi_stats[:2]
                list_df_semi = []
                for m_x in range(n_x_res):
                    for i_stat, stat_name in enumerate(semi_stats):
                        list_df_semi.append(
                            pd.DataFrame(
                                {
                                    uni_string2: sigma2_range,
                                    estimates_names[0]: true_semi[:, m_x, i_stat],
                                    estimates_names[1]: nonrandom_semi[:, m_x, i_stat],
                                    estimates_names[2]: pseudo_semi[:, m_x, i_stat],
                                    estimates_names[3]: whatif_just_semi[
                                        :, m_x, i_stat
                                    ],
                                    estimates_names[4]: whatif_over_semi[
                                        :, m_x, i_stat
                                    ],
                                    "Statistic": stat_name,
                                    "Variable": f"x{str(m_x + 1).translate(_subscripts)}",
                                }
                            )
                        )
                df_semi = pd.concat(list_df_semi)

                dfm_semi = pd.melt(
                    df_semi,
                    [uni_string2, "Statistic", "Variable"],
                    var_name="Estimate",
                )

                # with several covariates: one row of panels per covariate
                facet_args: dict = (
                    {"facet_col_wrap": 2}
                    if n_x_res == 1
                    else {"facet_row": "Variable", "facet_row_spacing": 0.05}
                )
                fig = px.line(
                    dfm_semi,
                    x=uni_string2,
                    y="value",
                    title=f"Semi-elasticities<br><sup>{ptitle}</sup>",
                    facet_col="Statistic",
                    facet_col_spacing=0.2 if n_x_res == 1 else 0.06,
                    **facet_args,
                    color="Estimate",
                    color_discrete_map={
                        estimates_names[0]: "black",
                        estimates_names[1]: "red",
                        estimates_names[2]: "green",
                        estimates_names[3]: "blue",
                        estimates_names[4]: "purple",
                    },
                    template="plotly_white",
                )
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig.update_yaxes(matches=None, showticklabels=True)
                if n_x_res > 1:
                    fig.update_layout(width=1100, height=150 + 280 * n_x_res)

                fig_save_semis_root = f"{figures_dir}/new_semi_elast_{full_str}"
                fig.write_image(f"{fig_save_semis_root}.{fig_fmt}")
                fig.write_html(f"{fig_save_semis_root}.html")

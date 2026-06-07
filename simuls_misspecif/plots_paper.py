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

fig_fmt = "png"


def _get_result(dict_results: dict, varname: str):
    dict_var = dict_results[varname]
    return dict_var


def _stack_cols(mat: np.ndarray) -> np.ndarray:
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


def _make_suffix(nproducts: int, do_exo: bool) -> str:
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
):
    if simuls_dir is None:
        simuls_dir = Path.cwd()
    root_dir = simuls_dir / f"J{nproducts}"

    spe_bounds_nmarkets = 100  # used for the SPE bounds
    lower_bound_str = f"95% CI- (T = {spe_bounds_nmarkets})"
    upper_bound_str = f"95% CI+ (T = {spe_bounds_nmarkets})"

    for i_scenario in selected_scenario_numbers:
        full_str = f"{str_model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
        case_dir = root_dir / f"{str_model}_v{i_scenario}"
        with open(
            case_dir
            / f"simul_results_{str_model}_J={nproducts}_v{i_scenario}_T={nmarkets}.pkl",
            "rb",
        ) as f:
            dict_results = pickle.load(f)

        figures_dir = mkdir_if_needed(case_dir / "figures_paper")

        model = dict_results["model"]
        data_pars = model.data_pars
        do_exo = data_pars.do_exo
        sigma_range = model.sigma_range

        n_sigmas = sigma_range.size

        print_stars(f"Model {str_model}_J={nproducts}_v{i_scenario}_T={nmarkets}")
        nonrandom_vals = _get_result(dict_results, "non-random values")
        pseudo_vals = _get_result(dict_results, "pseudo true values")
        whatif_vals = _get_result(dict_results, "whatif values")
        spb = _get_result(dict_results, "SPE variance bounds")
        nonrandom_semi = _get_result(dict_results, "non-random semi-elasticities")
        true_semi = _get_result(dict_results, "true semi-elasticities")
        pseudo_semi = _get_result(dict_results, "pseudo semi-elasticities")
        whatif_semi = _get_result(dict_results, "whatif semi-elasticities")

        n_pars = pseudo_vals.shape[-1]

        # we compute standard errors for SPE bounds, putting in zero if the variance is negative
        stb = np.zeros((n_sigmas, n_pars))
        for isig in range(n_sigmas):
            spb_isig = np.maximum(np.diag(spb[isig, :, :]), 0.0)
            stb[isig, :] = np.sqrt(spb_isig)

        true_values = np.zeros_like(pseudo_vals)
        true_values[:, 0] = model.true_pars.beta0
        true_values[:, 1] = model.true_pars.beta1
        true_values[:, 2] = sigma_range * sigma_range

        order_parameters = [uni_beta0, uni_beta1, uni_sigma2]

        suffix = _make_suffix(nproducts, do_exo)
        ptitle = suffix
        uni_string2 = uni_sigma2

        ordered_colors = ["black"] * 3 + ["red", "green", "blue"]
        estimated_values = np.zeros((n_sigmas, 5, n_pars))
        estimated_values[:, 2, :] = nonrandom_vals
        estimated_values[:, 3, :] = pseudo_vals
        estimated_values[:, 4, :] = whatif_vals

        if plot_pseudo_with_bounds:
            df1 = [None] * n_pars
            for ipar, par_name in enumerate(order_parameters):
                df_i = pd.DataFrame(
                    {
                        uni_string2: true_values[:, -1],
                        "True value": true_values[:, ipar],
                    }
                )
                bound_i = stb[:, ipar] / sqrt(spe_bounds_nmarkets)
                estimated_values[:, 0, ipar] = df_i["True value"] - 1.96 * bound_i
                estimated_values[:, 1, ipar] = df_i["True value"] + 1.96 * bound_i
                df1_ipar, ordered_estimates = _stack_estimates(
                    [
                        lower_bound_str,
                        upper_bound_str,
                        "Non-random",
                        "With K",
                        "What if",
                    ],
                    estimated_values[..., ipar],
                    df_i,
                )
                df1_ipar["Coefficient"] = par_name
                df1[ipar] = df1_ipar

            df2: pd.DataFrame = pd.concat((df1[ipar] for ipar in range(n_pars)))
            df2m = pd.melt(
                df2,
                id_vars=[uni_string2, "Coefficient"],
                value_vars=ordered_estimates,
                var_name="Estimate",
            )

            fig = px.line(
                df2m,
                x=uni_string2,
                y="value",
                facet_col="Coefficient",
                color="Estimate",
                color_discrete_sequence=ordered_colors,
                line_dash="Estimate",
                line_dash_sequence=["solid"] + ["dot"] * 2 + ["solid"] * 4,
                template="plotly_white",
                facet_col_spacing=0.12,
                title=f"Pseudo-true values and efficiency bounds<br><sup>{ptitle}</sup>",
            )

            # show only the symbol for the coefficient on top of each panel
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            fig.update_yaxes(
                matches=None, showticklabels=True
            )  # independent y axis with their own ticks

            fig_save_ptv_root = f"{figures_dir}/new_pseudo_vals_{full_str}"
            fig.write_image(f"{fig_save_ptv_root}.{fig_fmt}")

            fig.update_xaxes(rangeslider_visible=True)
            fig.write_html(f"{fig_save_ptv_root}.html")

            if plot_semi_elast:
                true_semi_elast = true_semi
                nonrandom_semi_elast = nonrandom_semi
                pseudo_semi_elast = pseudo_semi
                whatif_semi_elast = whatif_semi

                df_mean_own = pd.DataFrame(
                    {
                        uni_string2: true_values[:, -1],
                        "True value": true_semi_elast[:, 0],
                        "Non-random value": nonrandom_semi_elast[:, 0],
                        "With K": pseudo_semi_elast[:, 0],
                        "What-if": whatif_semi_elast[:, 0],
                        "Statistic": "Mean own semi-elasticity",
                    }
                )
                df_disp_own = pd.DataFrame(
                    {
                        uni_string2: true_values[:, -1],
                        "True value": true_semi_elast[:, 1],
                        "Non-random value": nonrandom_semi_elast[:, 1],
                        "With K": pseudo_semi_elast[:, 1],
                        "What-if": whatif_semi_elast[:, 1],
                        "Statistic": "Cross-market dispersion of own semi-elasticity",
                    }
                )
                df_semi = pd.concat((df_mean_own, df_disp_own))
                if nproducts > 1:
                    df_mean_cross = pd.DataFrame(
                        {
                            uni_string2: true_values[:, -1],
                            "True value": true_semi_elast[:, 2],
                            "Non-random value": nonrandom_semi_elast[:, 2],
                            "With K": pseudo_semi_elast[:, 2],
                            "What-if": whatif_semi_elast[:, 2],
                            "Statistic": "Mean cross semi-elasticity",
                        }
                    )
                    df_disp_cross = pd.DataFrame(
                        {
                            uni_string2: true_values[:, -1],
                            "True value": true_semi_elast[:, 3],
                            "Non-random value": nonrandom_semi_elast[:, 3],
                            "With K": pseudo_semi_elast[:, 3],
                            "What-if": whatif_semi_elast[:, 3],
                            "Statistic": "Cross-market dispersion of cross semi-elasticity",
                        }
                    )
                    df_semi = pd.concat((df_semi, df_mean_cross, df_disp_cross))

                dfm_semi = pd.melt(
                    df_semi, [uni_string2, "Statistic"], var_name="Estimate"
                )

                fig = px.line(
                    dfm_semi,
                    x=uni_string2,
                    y="value",
                    title=f"Semi-elasticities<br><sup>{ptitle}</sup>",
                    facet_col="Statistic",
                    facet_col_wrap=2,
                    facet_col_spacing=0.2,
                    color="Estimate",
                    color_discrete_map={
                        "True value": "black",
                        "Non-random": "red",
                        "With K": "green",
                        "What-if": "blue",
                    },
                    template="plotly_white",
                )
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig.update_yaxes(matches=None, showticklabels=True)

                fig_save_semis_root = f"{figures_dir}/new_semi_elast_{full_str}"
                fig.write_image(f"{fig_save_semis_root}.{fig_fmt}")
                fig.write_html(f"{fig_save_semis_root}.html")


if __name__ == "__main__":
    model_strings = ["exo", "endo"]

    selected_scenario_numbers = [3, 4]
    nmarkets = 5000
    number_products = [1, 2, 5, 10, 25, 50, 100]

    for nproducts in number_products:
        for str_model in model_strings:
            new_plots_paper(str_model, nproducts, nmarkets, selected_scenario_numbers)

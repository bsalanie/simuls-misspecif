import pickle
from pathlib import Path
from typing import Dict, List, Tuple, cast

import altair as alt
import numpy as np
import pandas as pd
from altair_saver import save as alt_save
from bs_python_utils.bs_mathstr import (
    uni_beta0,
    uni_beta1,
    uni_R2,
    uni_sigma2,
)
from bs_python_utils.bsutils import mkdir_if_needed, print_stars

figs_fmt = ".png"


def _get_result(
    dict_results: Dict,
    varname: str,
):
    dict_var = dict_results[varname]
    return dict_var


def _matrix_summary(
    mat: np.ndarray, title: str | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Summarize a matrix `(T, J)`.

    Returns:
        The mean and total standard error.
    """
    mat_market_means = mat.mean(axis=1)
    mat_mean = mat_market_means.mean()
    mat_market_variances = mat.var(axis=1)
    mat_ev = mat_market_variances.mean()
    mat_ve = mat_market_means.var()
    mat_sdtot = np.sqrt(mat_ev + mat_ve)

    if title is not None:
        print_stars(
            f"{title}: mean = {mat_mean: > 8.4f} and stderrs = within: {np.sqrt(mat_ev): > 8.4f},"
            + f" between: {np.sqrt(mat_ve): > 8.4f} -> total: {mat_sdtot: > 8.4f}"
        )

    return mat_mean, mat_sdtot


def _proj_summary(
    proj: np.ndarray, nproducts: int, title: str | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Summarize projected values by reshaping them into markets.

    Returns:
        The mean and total standard error.
    """
    nmarkets = proj.size // nproducts
    proj_mat = proj.reshape((nmarkets, nproducts))
    return _matrix_summary(proj_mat, title)


def _errors_xi_summary(
    errors_xi2: np.ndarray, errors_xi4: np.ndarray, sigma_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize xi errors across the sigma grid.

    Returns:
        Summary arrays for the second- and fourth-order xi errors.
    """
    errors_xi2_mean = np.zeros_like(sigma_range)
    errors_xi2_sdtot = np.zeros_like(sigma_range)
    errors_xi4_mean = np.zeros_like(sigma_range)
    errors_xi4_sdtot = np.zeros_like(sigma_range)
    for isig, sigma in enumerate(sigma_range):
        errors_xi2_mean[isig], errors_xi2_sdtot[isig] = _matrix_summary(
            errors_xi2[isig, :, :]
        )
        # f"sigma = {sigma}: Errors on xi at true values, second order")
        errors_xi4_mean[isig], errors_xi4_sdtot[isig] = _matrix_summary(
            errors_xi4[isig, :, :]
        )
        # f"sigma = {sigma}: Errors on xi at true values, fourth order")
    return errors_xi2_mean, errors_xi2_sdtot, errors_xi4_mean, errors_xi4_sdtot


def _stack_cols(mat: np.ndarray):
    v = mat[:, 0]
    for i in range(1, mat.shape[1]):
        v = np.hstack((v, mat[:, i]))
    return v


def _data_for_pseudo(
    sigma_range,
    true_values,
    cond_2,
    pseudo_vals,
    pseudo_vals_k1,
    pseudo_vals_k3,
    pseudo_vals_infty,
):
    # ill-conditioned TSLS
    ill_cond_2 = cond_2 > 1e6

    n_params = 3

    n_sigmas = sigma_range.size

    sigma2_range = sigma_range * sigma_range

    df = pd.DataFrame({uni_sigma2: np.tile(sigma2_range, n_params)})
    df["true value"] = _stack_cols(true_values)
    df["pseudo-true value"] = _stack_cols(pseudo_vals)
    df["corrected (k=1)"] = _stack_cols(pseudo_vals_k1)
    df["corrected (k=3)"] = _stack_cols(pseudo_vals_k3)
    for iter in [1, 5]:
        df[f"corrected (infty {iter})"] = _stack_cols(pseudo_vals_infty[:, :, iter - 1])
    df["efficiency bound"] = _stack_cols(stb)
    df["Ill-conditioned"] = np.repeat(ill_cond_2, n_params)

    df["Parameter"] = np.hstack(
        (
            np.repeat(uni_beta0, n_sigmas),
            np.repeat(uni_beta1, n_sigmas),
            np.repeat(uni_sigma2, n_sigmas),
        )
    )

    return df


def _data_for_bounds(sigma_range, true_values, cond_2, stb):
    # ill-conditioned TSLS
    ill_cond_2 = cond_2 > 1e6

    n_params = 3

    n_sigmas = sigma_range.size

    sigma2_range = sigma_range * sigma_range

    df = pd.DataFrame({uni_sigma2: np.tile(sigma2_range, n_params)})
    df["true value"] = _stack_cols(true_values)
    df["efficiency bound"] = _stack_cols(stb)
    df["Ill-conditioned"] = np.repeat(ill_cond_2, n_params)

    df["Parameter"] = np.hstack(
        (
            np.repeat(uni_beta0, n_sigmas),
            np.repeat(uni_beta1, n_sigmas),
            np.repeat(uni_sigma2, n_sigmas),
        )
    )

    return df


def _make_pseudo_val_plots(
    x_spec: alt.X, y_str: str, order_parameters: List
) -> alt.Chart:
    pseudo_plots = (
        alt.Chart()
        .mark_point(filled=True)
        .encode(x=x_spec, y=y_str, color=alt.Color("Parameter", sort=order_parameters))
        .properties(width=200, height=200)
    )
    return cast(alt.Chart, pseudo_plots)


def _make_pseudo_multi_plot(
    df: pd.DataFrame,
    list_vars: List,
    x_str: str,
    order_parameters: List,
) -> alt.Chart:
    true_values = df["true value"]
    dfe = df[["Parameter", x_str] + list_vars]
    n_vars = len(list_vars)
    dfem = dfe.melt(["Parameter", x_str])

    true_vec = np.tile(true_values, n_vars)

    dfem["true value"] = true_vec

    x_spec = alt.X(x_str, axis=alt.Axis(format=".2"))

    true_vals = (
        alt.Chart()
        .mark_line(interpolate="linear")
        .encode(
            x=x_spec,
            y=alt.Y("true value:Q", axis=alt.Axis(title=None)),
            color=alt.Color("Parameter:N", sort=order_parameters),
        )
    )

    pseudo_vals = (
        alt.Chart()
        .mark_line()
        .encode(
            x=x_spec,
            y=alt.Y("value:Q", axis=alt.Axis(title=None)),
            color=alt.Color("Parameter:N", sort=order_parameters),
            shape=alt.Shape("variable:N", sort=list_vars),
        )
    )

    ch = (
        alt.layer(true_vals, pseudo_vals, data=dfem)
        .facet(
            row=alt.Row("variable:N", sort=list_vars, title="Estimator"),
            column=alt.Column("Parameter:N", sort=order_parameters),
        )
        .resolve_scale(y="independent")
    )

    return cast(alt.Chart, ch)


def _make_true_val_plots(x_spec: alt.X, order_parameters: List) -> alt.Chart:
    true_plots = (
        alt.Chart()
        .mark_line()
        .encode(
            x=x_spec,
            y="true value",
            color=alt.Color("Parameter", sort=order_parameters),
        )
        .properties(width=200, height=200)
    )
    return cast(alt.Chart, true_plots)


def _make_pseudo_chart(
    df: pd.DataFrame,
    pseudo_plots: alt.Chart,
    true_plots: alt.Chart,
    order_parameters: List,
    subtitle: str | None = None,
) -> alt.Chart:
    ch = (
        alt.layer(true_plots, pseudo_plots, data=df)
        .facet(column=alt.Column("Parameter", sort=order_parameters))
        .resolve_scale(y="independent")
    )
    if subtitle is not None:
        ch = ch.properties(title=subtitle)
    return cast(alt.Chart, ch)


def _make_suffix(nproducts: int, do_exo: bool) -> str:
    if do_exo:
        suffix = f"J = {nproducts}, exogenous"
    else:
        suffix = f"J = {nproducts}, endogenous"
    return suffix


def _plot_bounds(df: pd.DataFrame, suffix: str, fig_save_bounds: str):
    uni_string2 = uni_sigma2

    x_spec = alt.X(f"{uni_string2}:N", axis=alt.Axis(format=".2"))

    order_parameters = [uni_beta0, uni_beta1, uni_sigma2]

    ptitle = suffix

    title_bounds = "Semiparametric efficiency bounds   :  " + ptitle
    ch_bounds = (
        alt.Chart(df)
        .mark_point(filled=True)
        .encode(
            x=x_spec,
            y="efficiency bound:Q",
            color=alt.Color("Parameter:N", sort=order_parameters),
            facet=alt.Facet("Parameter:N", sort=order_parameters),
        )
        .resolve_scale(y="independent")
        .properties(width=200, height=200, title=title_bounds)
    )

    alt_save(ch_bounds, f"{fig_save_bounds}")


def _plot_pseudo(df: pd.DataFrame, suffix: str, fig_save_ptv: str):
    uni_string2 = uni_sigma2

    x_spec = alt.X(f"{uni_string2}:N", axis=alt.Axis(format=".2"))

    order_parameters = [uni_beta0, uni_beta1, uni_sigma2]

    true_plots = _make_true_val_plots(x_spec, order_parameters)
    pseudo_plots = _make_pseudo_val_plots(x_spec, "pseudo-true value", order_parameters)
    pseudo_plots_k1 = _make_pseudo_val_plots(
        x_spec, "corrected (k=1)", order_parameters
    )
    pseudo_plots_k3 = _make_pseudo_val_plots(
        x_spec, "corrected (k=3)", order_parameters
    )

    pseudo_plots_infty1 = _make_pseudo_val_plots(
        x_spec, "corrected (infty 1)", order_parameters
    )
    pseudo_plots_infty5 = _make_pseudo_val_plots(
        x_spec, "corrected (infty 5)", order_parameters
    )
    # pseudo_plots_infty10 = _make_pseudo_val_plots(
    #     x_spec, f'corrected (infty 10)', order_parameters)

    ptitle = suffix

    ch2 = _make_pseudo_chart(
        df,
        pseudo_plots,
        true_plots,
        order_parameters,
        subtitle=("order 2  :  " + ptitle),
    )
    ch4_k1 = _make_pseudo_chart(
        df,
        pseudo_plots_k1,
        true_plots,
        order_parameters,
        subtitle=("corrected (k=1)  :  " + ptitle),
    )
    ch4_k3 = _make_pseudo_chart(
        df,
        pseudo_plots_k3,
        true_plots,
        order_parameters,
        subtitle=("corrected (k=3)  :  " + ptitle),
    )
    ch4_infty1 = _make_pseudo_chart(
        df,
        pseudo_plots_infty1,
        true_plots,
        order_parameters,
        subtitle=("corrected (infty, iteration 1)  :  " + ptitle),
    )
    ch4_infty5 = _make_pseudo_chart(
        df,
        pseudo_plots_infty5,
        true_plots,
        order_parameters,
        subtitle=("corrected (infty, iteration 5)  :  " + ptitle),
    )
    # ch4_infty10 = _make_pseudo_chart(df, pseudo_plots_infty10, true_plots, order_parameters,
    #                                subtitle=('corrected (infty, iteration 10)  :  ' + ptitle))

    ch_multi = _make_pseudo_multi_plot(
        df,
        ["pseudo-true value", "corrected (k=3)", "corrected (infty 1)"],
        uni_string2,
        order_parameters,
    )

    alt_save(ch2, f"{fig_save_ptv}_2{figs_fmt}")
    alt_save(ch4_k1, f"{fig_save_ptv}_k1{figs_fmt}")
    alt_save(ch4_k3, f"{fig_save_ptv}_k3{figs_fmt}")
    alt_save(ch4_infty1, f"{fig_save_ptv}_infty1{figs_fmt}")
    alt_save(ch4_infty5, f"{fig_save_ptv}_infty5{figs_fmt}")
    alt_save(ch_multi, f"{fig_save_ptv}_multi{figs_fmt}")
    # alt_save(ch4_infty10, f"{fig_save_ptv}_infty10{figs_fmt}")


def _data_for_xis(
    sigma_range: np.ndarray,
    true_xi: np.ndarray,
    estimated_xi2: np.ndarray,
):
    sigma2_range = sigma_range * sigma_range
    df = pd.DataFrame({uni_sigma2: sigma2_range})

    err_xi = estimated_xi2 - true_xi
    err_xi_mean = np.zeros_like(sigma_range)
    err_xi_sdtot = np.zeros_like(sigma_range)
    for isig, _ in enumerate(sigma_range):
        err_xi_mean[isig], err_xi_sdtot[isig] = _matrix_summary(err_xi[isig, :, :])

    df["estimation error on xi: mean"] = err_xi_mean
    df["estimation error on xi: stderr"] = err_xi_sdtot

    return df


def _plot_xis(df_xis: pd.DataFrame, suffix: str, fig_save: str):
    uni_string2 = uni_sigma2

    melt_vars = [uni_sigma2]

    dfm_xis = pd.melt(df_xis, id_vars=melt_vars)
    dfm_xis.loc[dfm_xis.variable.str.contains("mean"), "statistic"] = "mean"
    dfm_xis.loc[dfm_xis.variable.str.contains("stderr"), "statistic"] = "dispersion"

    dfm_xis.dropna(inplace=True)

    errxis_title = "Estimation errors on \N{GREEK SMALL LETTER XI} :  " + suffix

    ch = (
        alt.Chart(dfm_xis)
        .mark_point(filled=True)
        .encode(
            x=f"{uni_string2}:Q",
            y="value:Q",
            facet=alt.Facet("statistic:N", sort=["mean", "dispersion"]),
        )
        .resolve_scale(y="independent")
        .properties(title=errxis_title)
    )

    alt_save(ch, fig_save)


def _data_for_semi_elasticities(
    sigma_range: np.ndarray,
    true_semi: np.ndarray,
    pseudo_semi: np.ndarray,
    stb_semi: np.ndarray,
):
    df = pd.DataFrame({uni_sigma2: sigma_range * sigma_range})

    df["Mean of true own semi-elasticity"] = true_semi[:, 0]
    df["Stderr of true own semi-elasticity"] = true_semi[:, 1]
    df["Mean of pseudo own semi-elasticity"] = pseudo_semi[:, 0]
    df["Stderr of pseudo own semi-elasticity"] = pseudo_semi[:, 1]
    df["SPE bound for mean own semi-elasticity"] = stb_semi[:, 0]
    df["SPE bound for stderr of own semi-elasticity"] = stb_semi[:, 1]

    if true_semi.shape[1] > 2:  # nproducts > 1, we also have cross elasticities
        df["Mean of true cross semi-elasticity"] = true_semi[:, 2]
        df["Stderr of true cross semi-elasticity"] = true_semi[:, 3]
        df["Mean of pseudo cross semi-elasticity"] = pseudo_semi[:, 2]
        df["Stderr of pseudo cross semi-elasticity"] = pseudo_semi[:, 3]
        df["SPE bound for mean cross semi-elasticity"] = stb_semi[:, 2]
        df["SPE bound for stderr of cross semi-elasticity"] = stb_semi[:, 3]

    return df


def _plot_semi_elasticities(
    df_semi: pd.DataFrame,
    suffix: str,
    nproducts: int,
    fig_save: str,
):
    uni_string2 = uni_sigma2
    xspec = alt.X(f"{uni_string2}:Q", axis=alt.Axis(format=".2"))
    melt_vars = [uni_sigma2]

    dfm_semi = df_semi[
        melt_vars
        + [
            "Mean of true own semi-elasticity",
            "Mean of pseudo own semi-elasticity",
            "SPE bound for mean own semi-elasticity",
        ]
    ]
    if nproducts > 1:  # also cross elasticities
        cross_semis = [
            "Mean of true cross semi-elasticity",
            "Mean of pseudo cross semi-elasticity",
            "SPE bound for mean cross semi-elasticity",
        ]
        for cross_var in cross_semis:
            dfm_semi[cross_var] = df_semi[cross_var]

    dfm_semi = dfm_semi.melt(melt_vars)
    dfm_semi.loc[dfm_semi.variable.str.contains("true"), "True or pseudo-true"] = "True"
    dfm_semi.loc[dfm_semi.variable.str.contains("pseudo"), "True or pseudo-true"] = (
        "Pseudo"
    )

    ptitle = "Semi-elasticities  :  " + suffix

    if nproducts > 1:
        dfm_semi.loc[
            dfm_semi.variable.str.contains("own"), "Own or cross semi-elasticity"
        ] = "Own"
        dfm_semi.loc[
            dfm_semi.variable.str.contains("cross"), "Own or cross semi-elasticity"
        ] = "Cross"
        dfm_semi_elast = dfm_semi[
            dfm_semi["True or pseudo-true"].isin(["True", "Pseudo"])
        ]

        ch1 = (
            alt.Chart(dfm_semi_elast)
            .mark_line()
            .encode(
                x=f"{uni_string2}:Q",
                y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
                color=alt.Color("True or pseudo-true:N"),
                facet=alt.Facet(
                    "Own or cross semi-elasticity:N", columns=2, sort=["Own", "Cross"]
                ),
            )
        )

        ch2 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean own semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch3 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean cross semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch = ch1 & (ch3 | ch2)
    else:  # nproducts = 1
        dfm_semi_elast = dfm_semi[
            dfm_semi["True or pseudo-true"].isin(["True", "Pseudo"])
        ]
        ch1 = (
            alt.Chart(dfm_semi_elast)
            .mark_line()
            .encode(
                x=f"{uni_string2}:Q",
                y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
                color=alt.Color("True or pseudo-true:N"),
            )
        )
        ch2 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean own semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch = ch1 & ch2

    ch = ch.properties(title=ptitle)

    alt_save(ch, fig_save)


def _plot_semi_elasticities_meanstd(
    df_semi: pd.DataFrame,
    suffix: str,
    nproducts: int,
    fig_save: str,
):
    uni_string2 = uni_sigma2
    xspec = alt.X(f"{uni_string2}:Q", axis=alt.Axis(format=".2"))
    melt_vars = [uni_sigma2]

    dfm_semi = df_semi[
        melt_vars
        + [
            "Mean of true own semi-elasticity",
            "Mean of pseudo own semi-elasticity",
            "Stderr of true own semi-elasticity",
            "Stderr of pseudo own semi-elasticity",
        ]
    ]
    if nproducts > 1:  # also cross elasticities
        cross_semis = [
            "Mean of true cross semi-elasticity",
            "Mean of pseudo cross semi-elasticity",
            "Stderr of true cross semi-elasticity",
            "Stderr of pseudo cross semi-elasticity",
        ]
        for cross_var in cross_semis:
            dfm_semi[cross_var] = df_semi[cross_var]

    dfm_semi = dfm_semi.melt(melt_vars)
    dfm_semi.loc[dfm_semi.variable.str.contains("true"), "True or pseudo-true"] = "True"
    dfm_semi.loc[dfm_semi.variable.str.contains("pseudo"), "True or pseudo-true"] = (
        "Pseudo"
    )

    ptitle = "Semi-elasticities  :  " + suffix

    if nproducts > 1:
        dfm_semi.loc[
            dfm_semi.variable.str.contains("own"), "Own or cross semi-elasticity"
        ] = "Own"
        dfm_semi.loc[
            dfm_semi.variable.str.contains("cross"), "Own or cross semi-elasticity"
        ] = "Cross"
        dfm_semi_elast = dfm_semi[
            dfm_semi["True or pseudo-true"].isin(["True", "Pseudo"])
        ]

        ch1 = (
            alt.Chart(dfm_semi_elast)
            .mark_line()
            .encode(
                x=f"{uni_string2}:Q",
                y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
                color=alt.Color("True or pseudo-true:N"),
                facet=alt.Facet(
                    "Own or cross semi-elasticity:N", columns=2, sort=["Own", "Cross"]
                ),
            )
        )

        ch2 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean own semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch3 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean cross semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch = ch1 & (ch3 | ch2)
    else:  # nproducts = 1
        dfm_semi_elast = dfm_semi[
            dfm_semi["True or pseudo-true"].isin(["True", "Pseudo"])
        ]
        ch1 = (
            alt.Chart(dfm_semi_elast)
            .mark_line()
            .encode(
                x=f"{uni_string2}:Q",
                y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
                color=alt.Color("True or pseudo-true:N"),
            )
        )
        ch2 = (
            alt.Chart(df_semi)
            .mark_line()
            .encode(
                x=xspec,
                y=alt.Y(
                    "SPE bound for mean own semi-elasticity",
                    axis=alt.Axis(title="Semiparametric efficiency bound"),
                ),
            )
        )
        ch = ch1 & ch2

    ch = ch.properties(title=ptitle)

    alt_save(ch, fig_save)


def _data_for_msr(
    sigma_range: np.ndarray,
    msr_vals: np.ndarray,
):
    df = pd.DataFrame({uni_sigma2: sigma_range * sigma_range})

    df["non-random"] = 1.0 - msr_vals[:, 1] / msr_vals[:, 0]
    df["order 2"] = (msr_vals[:, 1] - msr_vals[:, 2]) / msr_vals[:, 0]
    df["order 4 (k=3)"] = (msr_vals[:, 2] - msr_vals[:, 4]) / msr_vals[:, 0]

    return df


def _plot_msr(df_msr: pd.DataFrame, suffix: str, fig_save: str):
    uni_string2 = uni_sigma2
    melt_vars = [uni_sigma2]

    dfm_msr = pd.melt(df_msr, id_vars=melt_vars)

    ptitle = f"Contributions to {uni_R2}  :  " + suffix

    ch = (
        alt.Chart(dfm_msr)
        .mark_area()
        .encode(
            x=uni_string2,
            y=alt.Y("value", axis=alt.Axis(title=uni_R2)),  # , stack="normalize"),
            color=alt.Color("variable", title="Model"),
        )
        .properties(title=ptitle)
    )

    alt_save(ch, fig_save)


def _data_for_errors(e2_mean, e2_sdtot, e4_mean, e4_sdtot, sigma_range):
    df = pd.DataFrame({uni_sigma2: sigma_range * sigma_range})

    df["error on xi2: mean"] = e2_mean
    df["error on xi2: stderr"] = e2_sdtot
    df["error on xi4: mean"] = e4_mean
    df["error on xi4: stderr"] = e4_sdtot
    df["zero mean"] = 0.0
    df["zero stderr"] = 0.0

    return df


def _plot_errors(df_err: pd.DataFrame, suffix: str, fig_save: str):
    uni_string2 = uni_sigma2

    melt_vars = [uni_sigma2]

    dfm_err = pd.melt(df_err, id_vars=melt_vars)
    dfm_err.loc[dfm_err.variable.str.contains("mean"), "statistic"] = "mean"
    dfm_err.loc[dfm_err.variable.str.contains("stderr"), "statistic"] = "dispersion"
    dfm_err.loc[dfm_err.variable.str.contains("xi2"), "order"] = "2"
    dfm_err.loc[dfm_err.variable.str.contains("xi4"), "order"] = "4"
    dfm_err.loc[dfm_err.variable.str.contains("zero"), "order"] = "\N{INFINITY}"

    dfm_err.dropna(inplace=True)

    err_title = "Errors on \N{GREEK SMALL LETTER XI} at true values :  " + suffix

    ch = (
        alt.Chart(dfm_err)
        .mark_point(filled=True)
        .encode(
            x=f"{uni_string2}:Q",
            y="value:Q",
            color="order:N",
            facet=alt.Facet("statistic:N", sort=["mean", "dispersion"]),
        )
        .resolve_scale(y="independent")
        .properties(title=err_title)
    )

    alt_save(ch, fig_save)


if __name__ == "__main__":
    simuls_dir = Path.home() / "Documents" / "Github" / "simuls_MNL"

    plot_pseudo = True
    plot_bounds = True
    plot_semi = True
    plot_msr = True
    plot_err = True
    plot_xis = True

    model_strings = ["exo", "endo"]

    selected_scenario_numbers = [3, 4]

    nmarkets = 10000
    number_products = [2]

    for nproducts in number_products:
        n_elast = 1 if nproducts == 1 else 2
        root_dir = simuls_dir / f"J{nproducts}"

        for i_scenario in selected_scenario_numbers:
            for do_exo in [True, False]:
                str_model = model_strings[0] if do_exo else model_strings[2]

                full_str = f"{str_model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
                case_dir = root_dir / f"{str_model}_v{i_scenario}"
                with open(
                    case_dir
                    / f"simul_results_{str_model}_J={nproducts}_v{i_scenario}_T={nmarkets}.pkl",
                    "rb",
                ) as f:
                    dict_results = pickle.load(f)

                figures_dir = mkdir_if_needed(case_dir / "figures")

                model = dict_results["model"]
                data_pars = model.data_pars
                sigma_range = model.sigma_range

                n_sigmas = sigma_range.size

                print_stars(f"Model {str_model}")
                pseudo_vals = _get_result(dict_results, "pseudo true values")
                corrd4 = _get_result(dict_results, "correc_d4")
                corrdp4 = _get_result(dict_results, "correc_dprime4")
                corr_infty = _get_result(dict_results, "correc_infty")
                spb = _get_result(dict_results, "SPE variance bounds")
                true_semi = _get_result(dict_results, "true semi-elasticities")
                pseudo_semi = _get_result(dict_results, "pseudo semi-elasticities")
                spb_semi = _get_result(
                    dict_results,
                    "SPE bounds on semi-elasticities",
                )
                msr = _get_result(dict_results, "mean squared residuals")
                cond_2 = _get_result(dict_results, "condition number 2")
                cond_bounds = _get_result(dict_results, "condition number bounds")

                true_xi = _get_result(dict_results, "true_xi")
                errors_xi2 = _get_result(dict_results, "errors_xi2")
                errors_xi4 = _get_result(dict_results, "errors_xi4")
                estimated_xi2 = _get_result(dict_results, "estimated_xi2")

                n_pars = pseudo_vals.shape[-1]

                e2_mean, e2_sdtot, e4_mean, e4_sdtot = _errors_xi_summary(
                    errors_xi2, errors_xi4, sigma_range
                )

                # we compute standard errors for SPE bounds, putting in zero if the variance is negative
                stb = np.zeros((n_sigmas, n_pars))
                for isig in range(n_sigmas):
                    spb_isig = np.maximum(np.diag(spb[isig, :, :]), 0.0)
                    stb[isig, :] = np.sqrt(spb_isig)

                stb_semi = np.zeros((n_sigmas, 2 * n_elast))
                for isig in range(n_sigmas):
                    spb_semi_isig = np.maximum(spb_semi[isig, :], 0.0)
                    stb_semi[isig, :] = np.sqrt(spb_semi_isig)

                true_values = np.zeros_like(pseudo_vals)
                true_values[:, 0] = model.true_pars.beta0
                true_values[:, 1] = model.true_pars.beta1
                true_values[:, 2] = sigma_range * sigma_range

                # corrected pseudo-true values
                pseudo_vals_k1 = pseudo_vals + corrd4 + corrdp4
                pseudo_vals_k3 = pseudo_vals + 3.0 * corrd4 + corrdp4
                pseudo_vals_infty = corr_infty

                # problematic cases
                # cond_2 is the condition number of Zstar2; affects pseudo vals and pseudo elast
                # cond_bounds is the condition number of Zstar; affects efficiency bounds
                print_stars("Condition numbers for second order")
                print(cond_2)
                print_stars("Condition numbers for bounds")
                print(cond_bounds)

                suffix = _make_suffix(nproducts, do_exo)

                if plot_pseudo:
                    df_pseudo = _data_for_pseudo(
                        sigma_range,
                        true_values,
                        cond_2,
                        pseudo_vals,
                        pseudo_vals_k1,
                        pseudo_vals_k3,
                        pseudo_vals_infty,
                    )
                    fig_save_ptv = f"{figures_dir}/pseudo_vals_{full_str}"
                    _plot_pseudo(df_pseudo, suffix, fig_save_ptv)

                    if plot_bounds:
                        df_bounds = _data_for_bounds(
                            sigma_range, true_values, cond_2, stb
                        )
                        fig_save_bounds = f"{figures_dir}/bounds_{full_str}{figs_fmt}"
                        _plot_bounds(df_bounds, suffix, fig_save_bounds)

                    if plot_xis:
                        # now we plot the true xi vs our second order estimate
                        df_xis = _data_for_xis(
                            sigma_range,
                            true_xi,
                            estimated_xi2,
                        )
                        fig_save = f"{figures_dir}/xis_{full_str}{figs_fmt}"
                        _plot_xis(df_xis, suffix, fig_save)

                    if plot_semi:
                        # now semi-elasticity plots
                        df_semi = _data_for_semi_elasticities(
                            sigma_range,
                            true_semi,
                            pseudo_semi,
                            stb_semi,
                        )
                        fig_save = (
                            f"{figures_dir}/semi_elasticities_{full_str}{figs_fmt}"
                        )
                        _plot_semi_elasticities(df_semi, suffix, nproducts, fig_save)

                    # now plot expansion errors
                    if plot_err:
                        df_err = _data_for_errors(
                            e2_mean,
                            e2_sdtot,
                            e4_mean,
                            e4_sdtot,
                            sigma_range,
                        )

                        fig_save = f"{figures_dir}/errors_xi_{full_str}{figs_fmt}"

                        _plot_errors(df_err, suffix, fig_save)

                    if plot_msr:
                        # and plot mean squared residuals
                        df_msr = _data_for_msr(sigma_range, msr)

                        fig_save = (
                            f"{figures_dir}/mean_squared_residuals_{full_str}{figs_fmt}"
                        )

                        _plot_msr(df_msr, suffix, fig_save)

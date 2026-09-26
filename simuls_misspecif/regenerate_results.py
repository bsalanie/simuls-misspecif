"""Regenerate stored simulation results with the current code, on the same draws.

The result pickles do not store the simulated data, so quantities computed from the
data (for instance the semi-elasticities, after a bug fix) can only be updated by
re-running `get_the_stats`. For each pickle this tool:

1. rebuilds the model from the pickle, keeping its `beta0` (which `adjust_beta0_S0`
   draws at random); pickles written before the code allowed several covariates
   are converted;
2. finds the RNG stream of the original run: the stream of the case in position `i`
   is `generate_RNG_streams(nsim, SEED)[i]`, whatever `nsim`, and we match the stored
   non-random estimates at the first sigma;
3. re-runs `get_the_stats` in a temporary directory;
4. checks that the outputs that should not change agree with the stored ones (by
   default, everything except the semi-elasticities);
5. if they do, replaces the pickle and re-extracts and re-plots the case.

Usage:
    uv run python simuls_misspecif/regenerate_results.py                  # all pickles
    uv run python simuls_misspecif/regenerate_results.py "J10/*/simul_results_*.pkl"
    uv run python simuls_misspecif/regenerate_results.py --check-only -p 4
"""

import argparse
import multiprocessing as mp
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from simuls_misspecif.compute_stats import get_the_stats
from simuls_misspecif.create_samples import make_shares
from simuls_misspecif.evaluations import (
    _artificial_regressors,
    _our_tsls0,
    _project_variables,
)
from simuls_misspecif.extract_from_results import KEYS_EXTRACT, extract_from_results
from simuls_misspecif.MNL_utils import (
    DataParams,
    ModelData,
    SimulationCase,
    TrueParams,
    _mean_utils,
    make_names_params,
)
from simuls_misspecif.plots_paper import new_plots_paper
from simuls_misspecif.utils import generate_RNG_streams

# the seed used in `simuls_driver.py`
SEED = 5546757

# keys that are not compared by default: they may change with the code
DEFAULT_CHANGING = ("semi-elasticities",)

# keys never compared
NOT_COMPARED = ("model", "n_x")


@dataclass
class RegenOptions:
    root_dir: Path
    max_stream: int
    tol: float
    changing: tuple[str, ...]
    check_only: bool
    plot: bool


def convert_model(old_model) -> ModelData:
    """Rebuild a `ModelData` with the current dataclasses.

    Args:
        old_model: The model stored in a pickle, possibly from the one-covariate code
            (`TrueParams(beta0, beta1, sigma)`, `DataParams` without `n_x`).

    Returns:
        An equivalent `ModelData`.
    """
    tp = old_model.true_pars
    if hasattr(tp, "beta"):
        true_pars = TrueParams(
            beta0=tp.beta0, beta=tp.beta, sigma_profile=tp.sigma_profile
        )
    else:
        true_pars = TrueParams(
            beta0=tp.beta0, beta=np.array([tp.beta1]), sigma_profile=np.ones(1)
        )
    n_x = true_pars.n_x
    dp = old_model.data_pars
    data_pars = DataParams(
        sigxi=dp.sigxi,
        sigx=dp.sigx,
        rhox_z=dp.rhox_z,
        rhox_xi=dp.rhox_xi,
        do_exo=dp.do_exo,
        n_x=n_x,
    )
    return ModelData(
        data_pars=data_pars,
        true_pars=true_pars,
        names_pars=make_names_params(n_x),
        model_string=old_model.model_string,
        long_name=old_model.long_name,
        nmarkets=old_model.nmarkets,
        nproducts=old_model.nproducts,
        scenario=old_model.scenario,
        sigma_range=old_model.sigma_range,
        mode=old_model.mode,
        iprec=old_model.iprec,
    )


def _nonrandom_first_sigma(model: ModelData, stream: np.random.Generator) -> np.ndarray:
    """The non-random estimates at the first sigma, with data drawn from `stream`."""
    T, J, n_x = model.nmarkets, model.nproducts, model.n_x
    dp, tp = model.data_pars, model.true_pars
    xi, x, z = dp.generate_exogenous_vars_from_draws(
        dp.generate_random_draws(T, J, stream)
    )
    sig_vec = model.sigma_range[0] * tp.sigma_profile
    shares = make_shares(_mean_utils(tp.beta0, tp.beta, x) + xi, x, sig_vec)
    xmat = x.reshape((T * J, n_x))
    K, y, _, _ = _artificial_regressors(shares.reshape(-1), xmat, J)
    y_proj, X_proj, _, _, _ = _project_variables(y, xmat, z, K, mode=model.mode)
    return cast(np.ndarray, _our_tsls0(y_proj, X_proj)[1])


def find_stream(model: ModelData, stored_nonrandom: np.ndarray, max_stream: int):
    """Find the index of the RNG stream that produced a stored case.

    Args:
        model: The model of the case.
        stored_nonrandom: The stored non-random estimates at the first sigma.
        max_stream: How many stream indices we try.

    Returns:
        The index, or None if no stream matches.
    """
    for i in range(max_stream):
        stream = generate_RNG_streams(i + 1, SEED)[i]
        vals = _nonrandom_first_sigma(model, stream)
        if np.allclose(vals, stored_nonrandom, rtol=1e-9, atol=1e-12):
            return i
    return None


def compare_results(
    old: dict, new: dict, changing: tuple[str, ...]
) -> tuple[float, str, dict[str, float]]:
    """Compare stored and regenerated results.

    Args:
        old: Stored results.
        new: Regenerated results.
        changing: Substrings of the keys that are allowed to change.

    Returns:
        The largest relative difference on the keys that should not change, its key,
        and the largest relative differences on the keys that may change.
    """
    worst, worst_key = 0.0, ""
    changes: dict[str, float] = {}
    for key, old_val in old.items():
        if key in NOT_COMPARED:
            continue
        if key not in new:
            return np.inf, f"{key} (missing)", changes
        a, b = np.asarray(old_val), np.asarray(new[key])
        if b.ndim == a.ndim + 1 and b.shape[1] == 1:  # old pickles have no M axis
            b = b[:, 0]
        if a.shape != b.shape:
            if any(c in key for c in changing):
                changes[key] = np.inf
                continue
            return np.inf, f"{key} (shape {a.shape} vs {b.shape})", changes
        if a.size == 0:
            continue
        diff = float(np.max(np.abs(a - b) / (1.0 + np.abs(a))))
        if any(c in key for c in changing):
            changes[key] = diff
        elif diff > worst:
            worst, worst_key = diff, key
    return worst, worst_key, changes


def _parse_case(pkl: Path) -> tuple[str, int, int, int, int]:
    """model, nproducts, n_x, scenario, nmarkets from a pickle name."""
    match = re.match(
        r"simul_results_(\w+?)_J=(\d+)(?:_M=(\d+))?_v(\d+)_T=(\d+)\.pkl", pkl.name
    )
    if match is None:
        raise ValueError(f"cannot parse {pkl.name}")
    model, J, M, scenario, T = match.groups()
    return model, int(J), int(M or 1), int(scenario), int(T)


def regenerate(pkl: Path, opts: RegenOptions) -> str:
    """Regenerate one stored case; see the module docstring.

    Returns:
        A one-line report.
    """
    name = pkl.relative_to(opts.root_dir) if pkl.is_relative_to(opts.root_dir) else pkl
    case = _parse_case(pkl)
    with open(pkl, "rb") as f:
        old = pickle.load(f)
    model = convert_model(old["model"])
    n_x = model.n_x

    i_stream = find_stream(
        model, old["non-random values"][0, : 1 + n_x], opts.max_stream
    )
    if i_stream is None:
        return f"FAIL {name}: no stream among the first {opts.max_stream} matches"

    stream = generate_RNG_streams(i_stream + 1, SEED)[i_stream]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        new = get_the_stats(
            SimulationCase(
                stream=stream,
                model=model,
                isim=i_stream,
                pickle_dir=tmp_path,
                use_mp=False,
            )
        )
        worst, worst_key, changes = compare_results(old, new, opts.changing)
        str_changes = ", ".join(f"{k}: {v:.1e}" for k, v in changes.items())
        report = (
            f"{name} (stream {i_stream}): max rel. diff. {worst:.1e} ({worst_key});"
            f" changed: {str_changes or 'none'}"
        )
        if worst > opts.tol:
            return f"FAIL {report} > tol {opts.tol:.0e}; not replaced"
        if opts.check_only:
            return f"OK   {report}; not replaced (--check-only)"

        (new_pkl,) = tmp_path.glob("simul_results_*.pkl")
        shutil.copyfile(new_pkl, pkl)

    model_str, nproducts, n_x_name, scenario, nmarkets = case
    root = pkl.parents[2]
    extract_from_results(
        model_str,
        nproducts,
        nmarkets,
        scenario,
        KEYS_EXTRACT,
        root_dir=root,
        n_x=n_x_name,
    )
    if opts.plot:
        new_plots_paper(
            model_str, nproducts, nmarkets, [scenario], simuls_dir=root, n_x=n_x_name
        )
    return f"OK   {report}; replaced"


def _regenerate_star(args: tuple[Path, RegenOptions]) -> str:
    """Regenerate one case; report an exception as a failure."""
    pkl, opts = args
    try:
        return regenerate(pkl, opts)
    except Exception as e:
        return f"FAIL {pkl}: {type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate stored simulation results on the same draws."
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["J*/*/simul_results_*.pkl"],
        help="Pickles or glob patterns, relative to --root"
        " (default: all J*/*/simul_results_*.pkl)",
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)"
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Maximum relative difference allowed on the outputs that should not"
        " change (default: 1e-8)",
    )
    parser.add_argument(
        "--changing",
        nargs="*",
        default=list(DEFAULT_CHANGING),
        help="Substrings of the result keys that may change"
        " (default: semi-elasticities)",
    )
    parser.add_argument(
        "--max-stream",
        type=int,
        default=100,
        help="Number of RNG stream indices to try (default: 100)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Rerun and compare, but do not replace pickles or re-plot",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not re-plot")
    args = parser.parse_args()

    root = args.root.resolve()
    # only the full results; patterns may also match the extract_results pickles
    pickles = sorted(
        {
            p.resolve()
            for pattern in args.patterns
            for p in root.glob(pattern)
            if p.name.startswith("simul_results_")
        }
    )
    if not pickles:
        parser.error(f"no pickles match {args.patterns} under {root}")
    # the largest cases first, to balance the load
    pickles.sort(key=lambda p: -p.stat().st_size)

    opts = RegenOptions(
        root_dir=root,
        max_stream=args.max_stream,
        tol=args.tol,
        changing=tuple(args.changing),
        check_only=args.check_only,
        plot=not args.no_plot,
    )
    print(f"Regenerating {len(pickles)} cases")
    tasks = [(p, opts) for p in pickles]
    if args.processes > 1:
        with mp.Pool(processes=args.processes) as pool:
            reports = list(pool.imap_unordered(_regenerate_star, tasks))
    else:
        reports = [_regenerate_star(t) for t in tasks]

    # print the reports together, after the output of get_the_stats
    print("\n".join(["", "=" * 70, "Summary:"] + sorted(reports)))
    n_fail = sum(r.startswith("FAIL") for r in reports)
    if n_fail:
        raise SystemExit(f"{n_fail} case(s) failed")


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Simulations for a "what-if misspecification" paper. The DGP is a symmetric mixed multinomial logit with `M` covariates, each with a random coefficient:

```
U_ijt = β₀ + Σ_m x_jtm (β_m + ν_im) + ξ_jt + u_ijt,   m = 1..M
```

where `ν_im = σ·ω_m·ε_im`, `ε_i ~ N(0, I_M)` (independent random coefficients). `σ` scans `sigma_range`; `ω` is `TrueParams.sigma_profile`. `M` is `DataParams.n_x` (default 1). The estimated parameters are `[β₀, β_1..β_M, σ²_1..σ²_M]` (`1 + 2M`). The project studies pseudo-true parameter values, what-if estimators, and semiparametric efficiency (SPE) bounds when the random-coefficient MNL is misspecified as a plain MNL.

`multiple_covariates.md` documents the extension from one to `M` covariates, the bugs fixed along the way, and remaining open issues.

## Commands

### Environment
This project uses `uv` for dependency management.

```bash
uv sync                   # install dependencies
uv run python <script>    # run a script in the venv
```

### Linting and type checking
```bash
uv run ruff check simuls_misspecif/ tests/   # lint
uv run ruff format simuls_misspecif/ tests/  # format (line length 88)
uv run mypy simuls_misspecif/                # type check
```

### Tests
```bash
uv run pytest                                  # run all tests
uv run pytest tests/test_multi_covariates.py   # single test file
```
`tests/test_multi_covariates.py` holds reference copies of the old one-covariate formulas (the `M = 1` code must reproduce them) and finite-difference checks of the share derivatives and semi-elasticities, including `J = 1`.

### Running simulations
The main entry point is `simuls_misspecif/simuls_driver.py`:
```bash
uv run python simuls_misspecif/simuls_driver.py -s 3 4 -J 2 5 10 -T 10000 -m endo exo -M 2 --sigma-profile 1 0.5
```
Options: `-s/--scenarios`, `-J/--products`, `-T/--markets`, `-m/--models` (`endo`, `exo`), `-M/--n-x` (number of covariates, default 1), `--sigma-profile` (`ω`, default all ones), `--no-mp`, `--cpus`. `J = 1` is supported (no cross semi-elasticities).

To re-extract from existing pickles without re-running simulations, use `simuls_misspecif/extract_from_results.py` as a script.

### Documentation
```bash
uv run mkdocs serve       # local preview
uv run mkdocs build       # build static site
```

### Configuration
The main configuration file is **`MNL_params.py`**, which defines:
- `true_pars` — `TrueParams(beta0, beta, sigma_profile)`; `beta` and `sigma_profile` are `(M,)` arrays. The driver expands them to `M` covariates.
- `data_pars` — `DataParams(sigxi, sigx, rhox_z, rhox_xi, do_exo, n_x)`
- `basic_sigma_range`, `large_sigma_range` — values of σ to simulate over
- `n_gh_integrals` — Gauss-Hermite nodes per dimension for the integrals behind the SPE bounds and true semi-elasticities when `M > 1` (cost grows as `n_gh_integrals**M`)
- Flags `do_a_second`, `do_bounds_semi_elast` to control which computations run

## Architecture

### Data flow
1. **`MNL_params.py`** — global constants (see above).
2. **`MNL_utils.py`** — dataclasses (`TrueParams`, `DataParams`, `ModelData`, `SimulationCase`) and helpers. `DataParams.generate_random_draws` and `generate_exogenous_vars_from_draws` build `(xi, x, z)`; each `x_m` has its own instrument `z_m`. Exogenous case: `x = z`; endogenous case: `x` is correlated with `xi` via `rhox_xi`. `make_names_params(M)` gives parameter names. Quadrature: `quadrature_nodes(M)` (share generation) and `integration_nodes(M, iprec, n_per_dim)` (other integrals).
3. **`create_samples.py`** — `make_shares(mean_utils_xi, x, sigma_vec)` integrates out the random coefficients and produces `(T, J)` market shares.
4. **`simuls_driver.py`** — parses the CLI, builds `ModelData` instances for each `(nproducts, scenario, exo/endo)`, runs `get_the_stats` (with a `multiprocessing.Pool` unless `--no-mp`), saves a top-level `res.pkl`, then calls `extract_from_results` and `new_plots_paper`.
5. **`compute_stats.py`** — `get_the_stats(case)` is the per-simulation worker. For each value of `sigma` in `sigma_range` it:
   - generates shares with standard deviations `sigma * sigma_profile`,
   - computes artificial regressors (`K`, `y`, `V`, `W`) from `evaluations._artificial_regressors` (wrapping `frac_blp`),
   - runs two-stage least squares (`_our_tsls0` for non-random, `_our_tsls2` for Salanie-Wolak pseudo-true values),
   - computes the "what-if" estimates with `utils.estimate_what_if`, just-identified (instruments `Zstar2`) and over-identified (constant, powers 1–4 of each `z_m`, products `z_m z_n`),
   - evaluates semi-elasticities (true, non-random, pseudo-true, what-if),
   - computes SPE variance bounds via `_true_optimal_instruments`,
   - pickles results (see Output structure).
6. **`evaluations.py`** — the econometric estimators: `_artificial_regressors`, `_project_variables`, `_our_tsls0/2`, `_true_optimal_instruments`, `_true_semi_elasticities`, `_nonrandom_semi_elasticities`, `_pseudo_semi_elasticities_ift` (derivative of the approximate share equation by the implicit function theorem; it replaced an analytic formula that did not match finite differences). `berry_xis_GQ` and `estimated_xi_infty` are old one-covariate code, not used by the pipeline.
7. **`MNL_integrals.py`** — quadrature integrals of share-related quantities: `_exp_stj`, `_exp_stj_eps`, `_exp_stj_stk`, `_exp_stj_stk_eps`, `_dshares_dx`. All take `(T, J)` mean utilities, `(T, J, M)` covariates, `(M,)` standard deviations, and `(L, M)` nodes with `(L,)` weights. `_dshares_dtheta` and `_d2shares_dx_dtheta` only support `M = 1` (used only in the module's `__main__`).
8. **`extract_from_results.py`** — path helpers (`model_string`, `case_subdir`, `case_paths`) and `extract_from_results`, which writes a slimmer `extract_results_*.pkl` with only the keys needed for plotting.
9. **`plots_paper.py`** — `new_plots_paper` reads the per-case pickles and produces Plotly PNG and HTML figures: parameter values with SPE confidence bands, and semi-elasticities (one row of panels per covariate when `M > 1`). When `M > MAX_X_PLOTTED` (3), the parameter figure shows only `β₀` and the `β_m`, `σ²_m` of 3 covariates drawn at random (`select_seed` argument; `None` gives a fresh draw each call); the subtitle names them. It also reads pickles written by the old one-covariate code. `plot_simuls_results.py` is legacy (Altair, `M = 1` only, not called).
10. **`utils.py`** — `generate_RNG_streams` (independent RNG streams via `SeedSequence`), `f_print_stars` (file vs. screen logging under multiprocessing), `angle_product_with_Z`, `make_omega_inv`, `estimate_what_if`, `get_semi_elast_stats`.

### Key conventions
- `xi`, `y`, and shares are `(T, J)`; covariates `x` and instruments `z` are `(T, J, M)`. Flattened, they are `(T*J,)` and `(T*J, M)`. `K` and `V` are `(T*J, M)`, `W` is `(T*J, M, M)`.
- Parameter vectors are ordered `[β₀, β_1..β_M, σ²_1..σ²_M]` (`evaluations._split_params`).
- Quadrature for `M = 1` is unchanged from the original code (16-node Gauss-Hermite for shares, 1-d sparse grid with `iprec` for integrals), so `M = 1` reproduces earlier results. For `M > 1`, use tensor-product Gauss-Hermite rules (`MNL_utils.gauss_hermite_tensor`): 16 nodes per dimension for shares, `n_gh_integrals` for integrals. **Do not use sparse grids for `M > 1`**: their negative weights make shares negative at large σ.
- Integrals accumulate over nodes; never build `(T, J, L)` arrays (memory).
- The what-if uses half of `frac_blp.make_W`, i.e. `ξ(σ²) ≈ ξ₀ − Σ_m σ²_m K_m + Σ_{m,n} σ²_m σ²_n W_mn`.
- Semi-elasticity results are `(n_sigmas, M, 4)`: mean and std across markets of own and cross semi-elasticities of product 0 with respect to each `x_m`, or `(n_sigmas, M, 2)` when `J = 1`.
- `mode="2"` controls the flexible projection on instruments in `_project_variables` (`bs_python_utils.flexible_reg`).
- Multiprocessing: each worker process logs to `logs/{pid}.out` instead of stdout.
- Scenarios 0–4 differ in `true_pars` (β values) and `sigma_range`; scenarios 3 and 4 set `β_m = −4` and call `adjust_beta0_S0` to target a specific outside share (it uses the unseeded global `np.random`, so β₀ varies across runs).
- The `frac_blp` package provides `make_K_and_y`, `make_V`, `make_W` (artificial regressors for the BLP expansion).

## Output structure

- **Top-level result**: `res.pkl` — saved by `simuls_driver.py`, contains all cases
- **Per-case results**: `J{nproducts}/{model}_v{scenario}/simul_results_{model}_J={nproducts}_v{scenario}_T={nmarkets}.pkl`. When `M > 1`: directory `{model}_M{M}_v{scenario}` and `_M={M}` after `J={nproducts}` in file names. `extract_from_results.case_paths` is the single source for this rule. The results dict has an `"n_x"` key.
- **Extracted results**: `extract_results_*.pkl` in the same directory, with only the keys needed for plotting
- **Plots**: `figures_paper/new_pseudo_vals_*.{png,html}` and `figures_paper/new_semi_elast_*.{png,html}` in the case directory

The result pickles do not store the simulated shares or data, so anything computed from them (such as the semi-elasticities) can only be changed by re-running `get_the_stats`. All stored one-covariate results (`J{2..100}/endo_v{3,4}/`) were regenerated in September 2026 after two semi-elasticity bugs were fixed. The true semi-elasticities had used σ² as the standard deviation, and the pseudo/what-if ones had used a wrong analytic formula. Their other outputs were reproduced unchanged; see `multiple_covariates.md`.

### Reproducing a stored case exactly
- The case's data come from `generate_RNG_streams(nsim, 5546757)[i]`, where `i` is the case's position in its original driver run. Child `i` depends only on the seed and `i`, not on `nsim`.
- β₀ for scenarios 3 and 4 is random, but it is saved in `dict_results["model"].true_pars`.
- **`simuls_misspecif/regenerate_results.py`** does all of this. For each stored case it rebuilds the model (converting old one-covariate pickles), finds the stream by matching the stored `"non-random values"` at the first σ, and reruns `get_the_stats` in a temporary directory. It replaces the pickle only if all outputs except those allowed to change agree within `--tol`, then re-extracts and re-plots. Use it after any fix that changes outputs computed from the simulated data:
  ```bash
  uv run python simuls_misspecif/regenerate_results.py -p 6                      # all J*/*/simul_results_*.pkl
  uv run python simuls_misspecif/regenerate_results.py "J10/*/*T=100000.pkl" --check-only
  uv run python simuls_misspecif/regenerate_results.py --changing semi-elasticities "SPE variance bounds"
  ```
  `--changing` lists substrings of the result keys that may change (default: `semi-elasticities`). `--check-only` reruns and compares without writing. A failed case is reported, left untouched, and makes the exit code non-zero.

## Debugging

- **Multiprocessing logs**: when multiprocessing is on, worker logs go to `logs/{pid}.out` rather than stdout. Check these for per-worker errors, or rerun with `--no-mp`.
- **Type checking**: run `uv run mypy simuls_misspecif/` before running simulations.
- **Checking a change at `M = 1`**: run the driver on the old and new code with the same seed (`np.random.seed` before running the driver, because of `adjust_beta0_S0`) and compare the pickles. Parameter values and SPE bounds should agree to about 1e-10. At `J = 1` the bounds matrix is badly conditioned, so expect up to about 1e-6.
- **Checking derivatives**: validate any new share derivative or semi-elasticity formula against finite differences (see `tests/test_multi_covariates.py`); both semi-elasticity bugs were found this way.
- **Runtime**: the integrals for the bounds and true semi-elasticities cost O(T·J²·M) per node; with `M = 3` at T = 2000, J = 25 one case takes about 5 minutes and 4.4 GB. Regenerating all 21 stored `M = 1` cases (up to T = 100,000) took about 10 minutes on 6 processes.

## Key dependencies

- **`frac_blp`**: artificial regressors (`make_K_and_y`, `make_V`, `make_W`) for the BLP expansion
- **`bs_python_utils`**: flexible regressions and projections on instruments (`flexible_reg`, `mode="2"`), Gauss-Hermite nodes (`gauher`), sparse grids (`setup_sparse_gaussian`), printing helpers
- **`numpy` / `scipy`**: linear algebra (`lstsq`, batched `solve`); `scipy.optimize.fsolve` in the tests
- **`plotly` / `pandas`**: figures in `plots_paper.py`

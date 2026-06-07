# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Simulations for a "what-if misspecification" paper. The DGP is a symmetric mixed multinomial logit:

```
U_ijt = β₀ + x_jt (β₁ + ν_i) + ξ_jt + u_ijt
```

where `ν_i = σ·ε_i`, `ε_i ~ N(0,1)`. The project studies pseudo-true parameter values, what-if estimators, and semiparametric efficiency bounds when the random-coefficient MNL is misspecified as a plain MNL.

## Commands

### Environment
This project uses `uv` for dependency management.

```bash
uv sync                   # install dependencies
uv run python <script>    # run a script in the venv
```

### Linting and type checking
```bash
uv run ruff check simuls_misspecif/   # lint
uv run ruff format simuls_misspecif/  # format (line length 88)
uv run mypy simuls_misspecif/         # type check
```

### Tests
```bash
uv run pytest             # run all tests
uv run pytest tests/test_evaluations_reformat_varcov.py  # single test file
```

### Running simulations
The main entry point is `simuls_misspecif/simuls_driver.py`, run as a module/script:
```bash
uv run python simuls_misspecif/simuls_driver.py
```
Results are pickled under `J{nproducts}/{model}_v{scenario}/simul_results_*.pkl`. To re-extract from existing pickles without re-running simulations, use `simuls_misspecif/extract_from_results.py` as a script.

### Documentation
```bash
uv run mkdocs serve       # local preview
uv run mkdocs build       # build static site
```

## Architecture

### Data flow
1. **`MNL_params.py`** — global constants: `true_pars` (`TrueParams`), `data_pars` (`DataParams`), sigma ranges, flags (`do_a_second`, `do_bounds_semi_elast`).
2. **`MNL_utils.py`** — dataclasses (`TrueParams`, `DataParams`, `ModelData`) and data-generation helpers. `DataParams.generate_random_draws` and `generate_exogenous_vars_from_draws` build `(xi, x, z)`. Exogenous case: `x = z`; endogenous case: `x` is correlated with `xi` via `rhox_xi`.
3. **`create_samples.py`** — `make_shares` uses Gauss-Hermite quadrature (nodes from `MNL_utils.{xgh, wgh}`) to integrate out the random coefficient and produce `(T, J)` market shares.
4. **`simuls_driver.py`** — orchestrates everything: builds `ModelData` instances for each combination of `(nproducts, scenario, exo/endo)`, spawns `multiprocessing.Pool`, calls `get_the_stats` for each case, saves a top-level `res.pkl`, then calls `extract_from_results` and `plots_paper`.
5. **`compute_stats.py`** — `get_the_stats(case)` is the per-simulation worker. For each value of `sigma` in `sigma_range` it:
   - generates data and shares,
   - computes artificial regressors (`K`, `y`, `V`, `W`) from `evaluations._artificial_regressors`,
   - runs two-stage least squares (`_our_tsls0` for non-random, `_our_tsls2` for pseudo-true),
   - computes the "what-if" correction via a weighted GMM solve,
   - evaluates semi-elasticities (true, non-random, pseudo-true, what-if),
   - computes SPE variance bounds via `_true_optimal_instruments`.
   - Pickles results to `J{J}/{model}_v{scenario}/simul_results_{str_model}_T={nmarkets}.pkl`.
6. **`evaluations.py`** — all the econometric estimators: `_artificial_regressors`, `_our_tsls0/2`, `_project_variables`, `_true_optimal_instruments`, `_true_semi_elasticities`, `_pseudo_semi_elasticities_anal`, `_nonrandom_semi_elasticities`, plus reshape helpers `_reformat_Zstar` and `_reformat_varcov`.
7. **`MNL_integrals.py`** — Gauss-Hermite integrals of share-related quantities: `_exp_stj`, `_exp_stj_eps`, `_exp_stj_stk`, `_exp_stj_stk_eps`, `_dshares_dx`. All take `(T, J)` arrays plus sparse-grid nodes/weights.
8. **`extract_from_results.py`** — loads pickled simulation output and writes a slimmer `extract_results_*.pkl` with only the keys needed for plotting.
9. **`plots_paper.py`** — `new_plots_paper` reads extracted pickles and produces Plotly PNG figures.
10. **`utils.py`** — small utilities: `generate_RNG_streams` (independent RNG streams for parallel workers via `SeedSequence`), `f_print_stars` (file vs. screen logging under multiprocessing), `angle_product`, `bracket_product`.

### Key conventions
- Arrays are `(nmarkets, nproducts)` = `(T, J)` shaped throughout; flattened to `(T*J,)` vectors only when passed to regression helpers.
- `mode="2"` controls which instruments are used in `_project_variables` (flexible regression mode from `bs_python_utils`).
- Multiprocessing: each worker process logs to `{pid}.out` instead of stdout. The `use_mp` flag in `simuls_driver.py` controls this.
- Scenarios 0–4 differ in `true_pars` (β values) and `sigma_range`; scenarios 3 and 4 also call `adjust_beta0_S0` to target a specific outside share.
- The `frac_blp` package provides `make_K_and_y`, `make_V`, `make_W` (artificial regressors for the BLP expansion).

# Multiple covariates with random coefficients

## Model

The simulations used to handle one covariate `x` with a random coefficient. They now handle `M` covariates:

```
U_ijt = β₀ + Σ_m x_jtm (β_m + σ ω_m ε_im) + ξ_jt + u_ijt,    ε_i ~ N(0, I_M)
```

- **Random coefficients.** They are independent: the covariance matrix is diagonal, with standard deviations `σ·ω_m`.
- **`σ` and `ω`.** `σ` runs over `sigma_range`, as before. `ω` is `TrueParams.sigma_profile`, which defaults to all ones. The x-axis of the plots is still `σ²`.
- **Estimated parameters.** They are `[β₀, β_1..β_M, σ²_1..σ²_M]`, so `1 + 2M` parameters.
- **Instruments.** Each `x_m` has its own instrument `z_m`. It is built exactly as before, from the common `sigx`, `rhox_z` and `rhox_xi`, with independent draws for each `m`.
- **Semi-elasticities.** They are computed for the share of product 0 with respect to each `x_m`, both own and cross.

With `M = 1` the code reproduces the previous parameter estimates, what-if values and SPE bounds to about 1e-12. It also draws the same random numbers.

## Usage

```bash
uv run python simuls_misspecif/simuls_driver.py -M 2 --sigma-profile 1 0.5 -s 3 -J 5 -T 2000
```

- **`-M/--n-x`** sets the number of covariates. The default is 1.
- **`--sigma-profile`** sets `ω`. The default is all ones.
- **Scenarios 3 and 4** use `β_m = −4` for every `m`.
- **Output paths for `M > 1`.** Results go to `J{J}/{model}_M{M}_v{s}/`, and file names contain `_M={M}`. `M = 1` keeps the old paths, so existing results remain valid.
- **Accuracy setting.** `MNL_params.n_gh_integrals` (default 8) sets the number of Gauss–Hermite nodes per dimension for the integrals behind the SPE bounds and the true semi-elasticities when `M > 1`.

## Changes by file

- **Array shapes.** `x` and `z` are now `(T, J, M)`, or `(T*J, M)` when flattened. `ξ`, `y` and the shares remain `(T, J)`. `K` is `(T*J, M)` and `W` is `(T*J, M, M)`.
- **`MNL_utils.py`**
  - `TrueParams(beta0, beta, sigma_profile)` replaces `(beta0, beta1, sigma)`. The old `sigma` field was never used.
  - `DataParams` gains `n_x`.
  - `ModelData` checks that the two dataclasses agree on `n_x`.
  - `_mean_utils` computes `beta0 + x @ beta`.
  - New helpers: `make_names_params(M)`, `quadrature_nodes`, `integration_nodes` and `gauss_hermite_tensor`.
- **`create_samples.py`**
  - `make_shares(mean_utils_xi, x, sigma_vec)` is vectorized over markets and loops over the quadrature nodes.
- **`MNL_integrals.py`**
  - All integrals take `(T, J, M)` covariates, `(M,)` standard deviations and `(L, M)` nodes.
  - They accumulate over nodes, so memory stays O(T·J²).
  - `_exp_stj_eps` returns an array of shape `(T, J, M)`.
  - `_exp_stj_stk_eps` returns `(T, J, J, M)`.
  - `_dshares_dx` returns `(T, J, J, M)`.
  - `_dshares_dtheta` and `_d2shares_dx_dtheta` still handle one covariate only. They are used only in the module's `__main__`, and they stop with an error if called with `M > 1`.
- **`evaluations.py`**
  - `_project_variables` projects every column on all `M` instruments.
  - `_our_tsls0` and `_our_tsls2` return `1 + M` and `1 + 2M` coefficients.
  - `_true_optimal_instruments` produces one column per `σ²_m`.
  - The semi-elasticity functions return arrays of shape `(T, M)`.
  - New function `_pseudo_semi_elasticities_ift` (see bug 2 below).
- **`utils.py`**
  - `estimate_what_if` is rewritten in matrix form for `M` covariates.
  - `get_semi_elast_stats` returns an array of shape `(M, 4)`.
- **`compute_stats.py`**
  - All result arrays are sized for `1 + 2M` parameters.
  - Semi-elasticity results have shape `(n_sigmas, M, 4)`.
  - `W` is now taken from `frac_blp.make_W` instead of a hand-coded formula. The two are identical at `M = 1`.
  - The over-identified what-if uses these instruments: a constant, powers 1–4 of each `z_m`, and the products `z_m z_n`.
  - The results dictionary has a new `"n_x"` entry.
- **`simuls_driver.py` and `extract_from_results.py`**
  - The command-line options described above.
  - The path rule lives in one place: `extract_from_results.case_paths`.
- **`plots_paper.py`**
  - Parameter panels are generated for `β₀`, `β₁..β_M` and `σ²₁..σ²_M`.
  - When `M > 3`, only `β₀` and the `β_m` and `σ²_m` of 3 covariates drawn at random are plotted. The subtitle names the covariates shown. The `select_seed` argument of `new_plots_paper` makes the draw reproducible; by default it changes each time. The semi-elasticity figure still has one row per covariate.
  - The semi-elasticity figure has one row of panels per covariate.
  - It still reads result files written by the old one-covariate code.
- **Tests.**
  - `tests/test_multi_covariates.py` is new. It checks against copies of the old one-covariate formulas, and checks derivatives against finite differences.
  - The existing tests are updated for the new function signatures.
  - All 32 tests pass, and `ruff` and `mypy` report no issues.

## Bugs found and fixed

Both fixes are adopted, and the existing results have been regenerated (see "Regenerated results" below). They change the **semi-elasticity** results even at `M = 1`. Parameter estimates and SPE bounds are unaffected.

1. **True semi-elasticities used σ² where σ was needed.**
   - `_true_semi_elasticities` passed `[β₀, β₁, σ²]` to `_dshares_dx`, which read the last element as the standard deviation.
   - The results were correct only at σ = 1: the old and new values agree exactly at σ² = 1.
   - The fixed values match finite differences of `make_shares`.
2. **The analytic pseudo semi-elasticities (`_pseudo_semi_elasticities_anal`) do not match finite differences.**
   - The share equation of the approximate model is `log(s_j/s_0) = β₀ + x_j'β + Σ_m σ²_m K_jm(s, x) + ξ_j`. Finite differences of this equation disagree with the analytic formula by up to about 20%.
   - They are replaced by the implicit-function-theorem derivative of that equation (`_pseudo_semi_elasticities_ift`), which matches finite differences to about 1e-6.
   - The old functions `_pseudo_semi_elasticities_anal` and `_pseudo_semi_elasticities` have been removed. They are still available in git history.
3. **Negative quadrature weights gave negative shares.**
   - An early version of this work used sparse grids when `M > 1`. At `M = 3` their weights are often negative, and shares went negative from σ² ≈ 1 upward, which crashed the run.
   - Integration for `M > 1` now uses a Gauss–Hermite grid built separately in each dimension, whose weights are all positive:
     - 16 nodes per dimension for generating shares.
     - `n_gh_integrals` nodes per dimension (default 8) for the other integrals.
   - `M = 1` keeps its previous integration rules.

## Checks

- **`M = 1` against the previous code.** Both versions were run with the same seed on scenarios 3 and 4, J = 5, T = 2000, exo and endo. Estimates, what-if values and SPE bounds agree to about 3e-12. Only the semi-elasticities differ, because of bugs 1 and 2.
- **`M = 2`** (ω = (1, 0.5), J = 5). At small σ² the Salanié–Wolak estimates are close to the truth, including both σ²'s, and the pseudo semi-elasticities track the true ones.
- **`M = 3`** (J = 25, T = 2000, endo).
  - All 20 values of σ run, and every output is finite.
  - At σ² = 0.1 the estimates are close to the truth.
  - At σ² = 2, the Salanié–Wolak σ²'s are about 1.1 against a true 2.0.
  - The Salanié–Wolak own semi-elasticities stay close to the true ones: −1.93 against −1.94 at σ² = 2.
  - The run took 5.4 minutes and peaked at 4.4 GB of memory.
- **`J = 1`** (M = 1 and M = 2; scenarios 3 and 4; exo and endo; T = 2000).
  - Every case runs, all outputs are finite, and the figures are produced. Only own semi-elasticities are computed and plotted, since there are no cross semi-elasticities.
  - At `M = 1` the results match the previous code to about 1e-11. The one exception is 1.5e-6 on the SPE bounds in one case, where the bounds matrix is badly conditioned (condition number up to about 1e9).
  - With a single product per market, the σ²'s are weakly identified. At `M = 2` and T = 2000 the efficiency bound gives standard errors of about 0.6 to 1.0 on each σ².
  - The unit tests now include J = 1 cases.

## Regenerated results

All 21 existing one-covariate result files (`J{2,5,10,25,50,100}/endo_v{3,4}/`, with T = 100, 10,000 and 100,000) have been regenerated with the fixed semi-elasticities. Their extract files and figures have been redrawn.

- **Same draws.** For each file, the saved model was reused, including its β₀. The original random stream was identified by matching the saved non-random estimates at the first σ.
- **Only the semi-elasticities change.** All other outputs reproduce the previous files to at most 5e-10: parameter estimates, what-if values, SPE bounds and condition numbers. Only the semi-elasticities and their figures differ.
- **`J5/endo_M3_v4`** was already produced by the new code, so it was not regenerated.
- **Reusable tool.** The procedure is now `simuls_misspecif/regenerate_results.py`; see CLAUDE.md. A `--check-only` run on stored cases, including `J5/endo_M3_v4`, reproduces every output exactly.

## Remaining issues

1. **The what-if formula for `M > 1` needs checking.**
   - It assumes `ξ(σ²) ≈ ξ₀ − Σ_m σ²_m K_m + Σ_{m,n} σ²_m σ²_n W_mn`, where `W` is half of `frac_blp.make_W`.
   - This reduces exactly to the previous code at `M = 1`, but the cross terms `W_mn` for `m ≠ n` have not been derived independently.
2. **Choice of over-identifying instruments for `M > 1`.** The current set (powers 1–4 of each `z_m` plus the products `z_m z_n`) is an ad hoc choice. At `M = 1` it is the same set as before.
3. **Integration accuracy at large σ.**
   - With 16 nodes per dimension, the tiniest shares have a relative error of up to about 10% at σ² = 2. This is the same accuracy the previous one-covariate code had.
   - Weighted by share size, the error is about 0.02%.
   - With the default 8 nodes per dimension, the other integrals have a typical relative error of about 0.3% at σ² = 2. Raise `n_gh_integrals` if more accuracy is needed.
4. **Runtime at production sizes.**
   - The integrals behind the bounds and the true semi-elasticities cost O(T·J²·M) per node, with 8^M nodes.
   - For `M = 3` at T = 10,000 and J = 100, a rough estimate is a few hours per case. This has not been run.
   - Options if it is too slow: fewer nodes, running the bounds only for small J, or processing markets in chunks.
   - **`M ≥ 4` is impractical as things stand.** Share generation uses 16^M nodes: about 65,000 at `M = 4` and 1 million at `M = 5`. The other integrals use 8^M nodes: 4,096 at `M = 4` and 32,768 at `M = 5`. An `M ≥ 4` run needs a cheaper rule with positive weights, such as fewer nodes per dimension or quasi-Monte Carlo draws. The plotting code for `M > 3` was tested only on a synthetic `M = 5` result file.
5. **Random draws in `adjust_beta0_S0` are unseeded.** It uses `np.random` without a seed, so β₀ in scenarios 3 and 4 varies from run to run. It also targets the outside share with β = +1, not the scenario's β = −4. Both behaviours are unchanged.
6. **The legacy `plot_simuls_results.py`** is not called anywhere and still handles one covariate only.
7. **Unused one-covariate helpers.**
   - `berry_xis_GQ` and `estimated_xi_infty` in `evaluations.py` still expect one covariate of shape `(T, J)`.
   - `_dshares_dtheta` and `_d2shares_dx_dtheta` in `MNL_integrals.py` do too.
   - None of them are used by the main pipeline.
8. **Nothing is committed yet.** This includes the regenerated result files and figures.

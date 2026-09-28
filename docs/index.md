Bernard Salanie

Started June 1, 2026

## simuls_misspecif

Simulations of a symmetric mutinomial logit for the what-if misspecif paper.

The specification has
$$
U_{ijt} = \beta_0 + x_{jt} (\beta_1 + \nu_{i}) + \xi_{jt} + u_{ijt}
$$


where 

* $\nu_i=\sigma \varepsilon_i$ with $\varepsilon_i = N(0,1)$,

* $x_{jt}=N(0,\sigma_x^2)$ iid,

* $\xi_{jt} = N(0,\sigma_{\xi}^2)$ iid in the exogenous case; in the endogenous case, $x$ and $\xi$ are correlated and we use an instrument $z$. 

$\beta_0, \beta_1, \sigma$ and the correlations are specified in `MNL_params.py`.

### updates

#### June 8, 2026
Working version (if $W$ is correct). Just identified
model, so the Salanie-Wolak correction works best.
Done also the overidentified version with $4$ powers of $z$.
Need to make sure the nonrandom estimates use the same Z.
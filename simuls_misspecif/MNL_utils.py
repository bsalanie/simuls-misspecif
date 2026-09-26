"""Utilities for MNL simulations."""

from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from pprint import pprint
from typing import List, Union, cast

import numpy as np
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian
from bs_python_utils.bsnputils import ThreeArrays, gauher
from bs_python_utils.bsutils import bs_error_abort, print_stars


def make_names_params(n_x: int) -> list[str]:
    """Names of the `1 + 2 n_x` parameters: beta0, the beta_m, and the sigma2_m."""
    return (
        ["beta0"]
        + [f"beta_{m + 1}" for m in range(n_x)]
        + [f"sigma2_{m + 1}" for m in range(n_x)]
    )


max_order = 4

n_normal = np.zeros(max_order + 1)
n_normal[0] = 1.0
for i in range(2, n_normal.size, 2):
    n_normal[i] = n_normal[i - 2] * (i - 1.0)

# for Gauss-Hermite integration
n_gauher = 16
xgh, wgh = gauher(n_gauher)


def gauss_hermite_tensor(
    n_x: int, n_per_dim: int = n_gauher
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Hermite rule for `epsilon ~ N(0, I_{n_x})`.

    Unlike sparse grids, all weights are positive, so integrated shares stay positive.

    Args:
        n_x: Number of random coefficients.
        n_per_dim: Number of nodes in each dimension.

    Returns:
        Nodes of shape `(n_per_dim**n_x, n_x)` and weights of shape `(n_per_dim**n_x,)`.
    """
    x1, w1 = gauher(n_per_dim)
    x1 = sqrt(2.0) * x1
    w1 = w1 / sqrt(np.pi)
    grids = np.meshgrid(*([x1] * n_x), indexing="ij")
    nodes = np.column_stack([g.reshape(-1) for g in grids])
    wgrids = np.meshgrid(*([w1] * n_x), indexing="ij")
    weights = np.prod(np.column_stack([g.reshape(-1) for g in wgrids]), axis=1)
    return nodes, weights


def quadrature_nodes(n_x: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights to generate the market shares.

    We use the tensor-product Gauss-Hermite rule with `n_gauher` nodes per dimension,
    as sparse grids have negative weights that can make shares negative.

    Args:
        n_x: Number of random coefficients.

    Returns:
        Nodes of shape `(L, n_x)` and weights of shape `(L,)`.
    """
    if n_x == 1:
        nodes = (sqrt(2.0) * xgh).reshape((-1, 1))
        weights = wgh / sqrt(np.pi)
        return nodes, weights
    return gauss_hermite_tensor(n_x, n_gauher)


def integration_nodes(
    n_x: int, iprec: int, n_per_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for the integrals in `MNL_integrals` (bounds, semi-elasticities).

    Args:
        n_x: Number of random coefficients.
        iprec: Precision of the one-dimensional sparse grid used when `n_x = 1`.
        n_per_dim: Nodes per dimension of the tensor-product Gauss-Hermite rule
            used when `n_x > 1`.

    Returns:
        Nodes of shape `(L, n_x)` and weights of shape `(L,)`.
    """
    if n_x == 1:
        nodes, weights = setup_sparse_gaussian(1, iprec)
        return nodes.reshape((-1, 1)), weights
    return gauss_hermite_tensor(n_x, n_per_dim)


@dataclass
class TrueParams:
    """Parameters to be estimated.

    Attributes:
        beta0: Coefficient of the constant.
        beta: `(M,)` mean coefficients of the covariates x.
        sigma_profile: `(M,)` relative standard errors of the random coefficients;
            the standard errors are `sigma * sigma_profile` for `sigma` in `sigma_range`.
    """

    beta0: float
    beta: np.ndarray
    sigma_profile: np.ndarray

    def __post_init__(self):
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=float))
        self.sigma_profile = np.atleast_1d(np.asarray(self.sigma_profile, dtype=float))
        if self.beta.shape != self.sigma_profile.shape:
            bs_error_abort("beta and sigma_profile should have the same size")

    @property
    def n_x(self) -> int:
        return int(self.beta.size)

    def print(self):
        pprint(self.__dict__)


@dataclass
class DataParams:
    """Parameters for the data.

    A fraction of the variance of x comes from z = N(0, 1), and a fraction of
    the remaining variance comes from xi.

    Attributes:
        sigxi: Standard deviation of xi.
        sigx: Standard deviation of x.
        rhox_z: Correlation of x and z.
        rhox_xi: Correlation of x and xi conditional on z.
        do_exo: Whether the exogenous case z = x is used.
        n_x: Number of covariates M, each with its own instrument.
    """

    sigxi: float
    sigx: float
    rhox_z: float
    rhox_xi: float
    do_exo: bool
    n_x: int = 1

    def generate_random_draws(
        self, nmarkets: int, nproducts: int, stream: np.random.Generator
    ) -> ThreeArrays:
        """Generate random draws used to construct the data.

        Args:
            nmarkets: Number of markets.
            nproducts: Number of products.
            stream: Random generator.

        Returns:
            A tuple `(xi_d, z_d, u_d)` of `N(0,1)` arrays; `xi_d` is `(nmarkets, nproducts)`,
            `z_d` and `u_d` are `(nmarkets, nproducts, n_x)`.
        """
        xi_d = stream.normal(size=(nmarkets, nproducts))
        z_d = stream.normal(size=(nmarkets, nproducts, self.n_x))
        u_d = stream.normal(size=(nmarkets, nproducts, self.n_x))
        return xi_d, z_d, u_d

    def generate_exogenous_vars_from_draws(self, draws: ThreeArrays) -> ThreeArrays:
        """Build the exogenous variables from random draws.

        Args:
            draws: Random draws.

        Returns:
            A tuple `(xi, x, z)`; `xi` is `(nmarkets, nproducts)`,
            `x` and `z` are `(nmarkets, nproducts, n_x)`.
        """
        xi_d, z_d, u_d = draws
        xi = self.sigxi * xi_d
        z = self.sigx * z_d
        if self.do_exo:
            x = z.copy()
        else:
            rhox_z2 = self.rhox_z * self.rhox_z
            rhox_xi2 = self.rhox_xi * self.rhox_xi
            rnorm = sqrt(1 - rhox_xi2) * u_d
            xi_term = rnorm + (self.rhox_xi * xi / self.sigxi)[..., np.newaxis]
            x = self.rhox_z * z + self.sigx * sqrt(1 - rhox_z2) * xi_term

        return xi, x, z

    def print(self):
        pprint(self.__dict__)


@dataclass
class ModelData:
    """The full model.

    Attributes:
        data_pars: Parameters for the exogenous variables.
        true_pars: Parameters to be estimated.
        names_pars: Their names.
        model_string: The name of the model.
        long_name: A longer name.
        nmarkets: Number of markets.
        nproducts: Number of products.
        scenario: Scenario number.
        sigma_range: Values of sigma explored.
        mode: How flexible regressions are run.
        iprec: Precision for sparse integration.
    """

    data_pars: DataParams
    true_pars: TrueParams
    names_pars: List[str]
    model_string: str
    long_name: str
    nmarkets: int
    nproducts: int
    scenario: int
    sigma_range: np.ndarray
    mode: str
    iprec: int

    def __post_init__(self):
        if self.true_pars.n_x != self.data_pars.n_x:
            bs_error_abort(
                f"true_pars has {self.true_pars.n_x} covariates"
                f" but data_pars has {self.data_pars.n_x}"
            )

    @property
    def n_x(self) -> int:
        return self.data_pars.n_x

    def print(self):
        pprint(self.__dict__)


@dataclass
class SimulationCase:
    """A single simulation configuration case.

    Attributes:
        stream: Random generator or seed sequence.
        model: Model parameter dataclass.
        isim: Simulation index.
        pickle_dir: Path to directory where results are stored.
        use_mp: Whether multiprocessing is enabled.
    """

    stream: Union[np.random.SeedSequence, np.random.Generator]
    model: ModelData
    isim: int
    pickle_dir: Path
    use_mp: bool


def _mean_utils(beta0: float, beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the mean utilities without the product effects.

    Args:
        beta0: Coefficient of the constant.
        beta: `(M,)` mean coefficients of x.
        x: `(..., M)` covariates.

    Returns:
        Mean utilities without the product effects, of shape `x.shape[:-1]`.
    """
    return cast(np.ndarray, beta0 + x @ np.atleast_1d(beta))


if __name__ == "__main__":
    m = ModelData(
        data_pars=DataParams(
            sigxi=1.0,
            sigx=1.0,
            rhox_z=sqrt(0.5),
            rhox_xi=sqrt(0.5),
            do_exo=True,
        ),
        true_pars=TrueParams(
            beta0=-1.0, beta=np.array([1.0]), sigma_profile=np.array([1.0])
        ),
        names_pars=make_names_params(1),
        model_string="youi",
        long_name="youpee",
        scenario=0,
        sigma_range=np.arange(0.01, 1.00, 0.02),
        nmarkets=1000,
        nproducts=4,
        mode="NP",
        iprec=17,
    )

    print_stars(f"We start with {m.nmarkets} markets")

    m.print()

    m2 = replace(m, nmarkets=12)

    print_stars(f"Now we have {m2.nmarkets}")

    m2.print()

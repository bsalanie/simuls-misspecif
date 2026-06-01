"""Utilities for MNL simulations."""

from dataclasses import dataclass, replace
from math import sqrt
from pprint import pprint
from typing import List, Union

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays, gauher
from bs_python_utils.bsutils import print_stars

names_params = ["beta0", "beta1", "sigma"]

max_order = 4

n_normal = np.zeros(max_order + 1)
n_normal[0] = 1.0
for i in range(2, n_normal.size, 2):
    n_normal[i] = n_normal[i - 2] * (i - 1.0)

# for Gauss-Hermite integration
n_gauher = 16
xgh, wgh = gauher(n_gauher)


@dataclass
class TrueParams:
    """Parameters to be estimated.

    Attributes:
        beta0: Coefficient of the constant.
        beta1: Coefficient of x, the single covariate when there is no micromoment.
        sigma: Scaling coefficient for epsilon.
    """

    beta0: float
    beta1: float
    sigma: float

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
    """

    sigxi: float
    sigx: float
    rhox_z: float
    rhox_xi: float
    do_exo: bool

    def generate_random_draws(
        self, nmarkets: int, nproducts: int, stream: np.random.Generator
    ) -> ThreeArrays:
        """Generate random draws used to construct the data.

        Args:
            nmarkets: Number of markets.
            nproducts: Number of products.
            stream: Random generator.

        Returns:
            A tuple `(xi_d, z_d, u_d, dbar)`; if there is no micromoment,
            `dbar = 0`.
        """
        xi_d = stream.normal(size=(nmarkets, nproducts))
        z_d = stream.normal(size=(nmarkets, nproducts))
        u_d = stream.normal(size=(nmarkets, nproducts))
        return xi_d, z_d, u_d

    def generate_exogenous_vars_from_draws(self, draws: ThreeArrays) -> ThreeArrays:
        """Build the exogenous variables from random draws.

        Args:
            draws: Random draws.

        Returns:
            A tuple `(xi, x, z)`.
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
            xi_term = rnorm + (self.rhox_xi * xi / self.sigxi)
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
        mode1: How flexible regressions are run in dimension 1.
        mode2: How flexible regressions are run in dimension 2.
        iprec: Precision for sparse integration.
    """

    data_pars: Union[DataParams, None]
    true_pars: Union[TrueParams, None]
    names_pars: Union[List[str], None]
    model_string: Union[str, None]
    long_name: Union[str, None]
    nmarkets: Union[int, None]
    nproducts: Union[int, None]
    scenario: Union[int, None]
    sigma_range: Union[np.ndarray, None]
    mode1: Union[str, None]
    mode2: Union[str, None]
    iprec: Union[int, None]

    def print(self):
        pprint(self.__dict__)


def _mean_utils(beta0: float, beta1: float, x: np.ndarray) -> np.ndarray:
    """Compute the mean utilities without the product effects.

    Args:
        beta0: Coefficient of the constant.
        beta1: Mean coefficient of x.
        x: Covariates.

    Returns:
        Mean utilities without the product effects.
    """
    return beta0 + beta1 * x


if __name__ == "__main__":
    m = ModelData(
        data_pars=DataParams(
            sigxi=1.0,
            sigx=1.0,
            rhox_z=sqrt(0.5),
            rhox_xi=sqrt(0.5),
            do_exo=True,
        ),
        true_pars=TrueParams(beta0=-1.0, beta1=1.0, sigma=0.5),
        names_pars=["beta0", "beta1", "sigma"],
        model_string="youi",
        long_name="youpee",
        scenario=0,
        sigma_range=np.arange(0.01, 1.00, 0.02),
        nmarkets=1000,
        nproducts=4,
        mode1="NP",
        mode2="NP",
        iprec=17,
    )

    print_stars(f"We start with {m.nmarkets} markets")

    m.print()

    m2 = replace(m, nmarkets=12)

    print_stars(f"Now we have {m2.nmarkets}")

    m2.print()

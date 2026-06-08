import numpy as np
from bs_python_utils.bsutils import bs_error_abort, file_print_stars, print_stars
from numpy.random import SeedSequence, default_rng


def generate_RNG_streams(
    nsim: int, initial_seed: int = 13091962
) -> list[np.random.Generator]:
    """
    Generate independent RNG streams for parallel processes.

    Args:
        nsim: Number of independent RNG streams to generate (one per simulation).
        initial_seed: An integer seed to initialize the SeedSequence.

    Returns:
        A list of independent RNG generators that can be used in parallel processes.
    """
    ss = SeedSequence(initial_seed)
    # Spawn off child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(nsim)
    streams = [default_rng(s) for s in child_seeds]
    return streams


def f_print_stars(use_mp: bool, what: str, fout_name: str | None = None):
    """prints to a file or to the screen; to a file under multiprocessing.

    Args:
        use_mp (bool): True if we use `multiprocessing`_
        what (str): what we print
        fout_name (str | None, optional): where we print, if not to screen. Defaults to None.
    """
    if use_mp and fout_name is not None:
        with open(fout_name, "a") as fout:
            file_print_stars(fout, what)
    elif use_mp:
        bs_error_abort("use_mp is True but fout_name is None")
    else:
        print_stars(what)


def angle_product_with_Z(
    a: np.ndarray, b: np.ndarray, omega: np.ndarray, Z_used: np.ndarray
) -> float:
    """Compute the angle product of a and b with respect to the Z_used moments and omega.
    Args:
        a: A 1D array of shape (npts,) representing the first vector.
        b: A 1D array of shape (npts,) representing the second vector.
        omega: A
        n_instr = Z_used.shape[1]2D array of shape (n_instr, n_instr) representing the weighting matrix.
        Z_used: A 2D array of shape (npts, n_instr) containing the moments used in the what-if estimation.

    Returns:
        A scalar representing the angle product of a and b with respect to Z_used and omega.
    """
    n_instr = Z_used.shape[1]
    Z_a = np.zeros(n_instr)
    Z_b = np.zeros(n_instr)
    for k in range(n_instr):
        Z_a[k] = np.mean(Z_used[:, k] * a)
        Z_b[k] = np.mean(Z_used[:, k] * b)
    return float(Z_a.T @ omega @ Z_b)


def center_moments(moments_used: np.ndarray, nproducts: int) -> np.ndarray:
    """Center the moments used in the what-if estimation by subtracting the mean across products.

    Args:
        moments_used: A 2D array of shape (TJ, n_instr) containing the moments used in the what-if estimation.
        nproducts: The number of products in each market.

    Returns:
        A 2D array of shape (TJ, n_instr) containing the centered moments.
    """
    moments_used_centered = np.zeros_like(moments_used)
    npts, n_instr = moments_used.shape
    nmarkets = npts // nproducts
    for k in range(n_instr):
        moments_used_k = moments_used[:, k].reshape((nmarkets, nproducts))
        moments_used_k_mean = moments_used_k.mean(axis=0)
        moments_used_centered[:, k] = (moments_used_k - moments_used_k_mean).reshape(
            npts
        )
    return moments_used_centered


def make_omega_inv(moments_used: np.ndarray) -> np.ndarray:
    """Compute the omega_inv matrix based on the moments used in the what-if estimation.

    Args:
        moments_used: A 2D array of shape (TJ, n_instr) containing the moments used in the what-if estimation.
    Returns:
        A 2D array of shape (n_instr, n_instr) representing the omega_inv matrix.
    """
    n_instr = moments_used.shape[1]
    omega_inv = np.zeros((n_instr, n_instr))
    for k in range(n_instr):
        for kp in range(n_instr):
            omega_inv[k, kp] = np.mean(moments_used[:, k] * moments_used[:, kp])
    return omega_inv


def estimate_what_if(
    xvec: np.ndarray,
    Kvec: np.ndarray,
    Wvec: np.ndarray,
    beta0_0: float,
    beta1_0: float,
    xi_0_vec: np.ndarray,
    Z_used: np.ndarray,
    Omega: np.ndarray,
) -> np.ndarray:
    npts = xvec.shape[0]
    ones = np.ones(npts)

    lhs_mat = np.zeros((3, 3))
    lhs_mat[0, 0] = angle_product_with_Z(ones, ones, Omega, Z_used)
    lhs_mat[0, 1] = angle_product_with_Z(ones, xvec, Omega, Z_used)
    lhs_mat[1, 0] = angle_product_with_Z(xvec, ones, Omega, Z_used)
    lhs_mat[1, 1] = angle_product_with_Z(xvec, xvec, Omega, Z_used)
    lhs_mat[0, 2] = angle_product_with_Z(ones, Kvec, Omega, Z_used)
    lhs_mat[2, 0] = angle_product_with_Z(Kvec, ones, Omega, Z_used)
    lhs_mat[1, 2] = angle_product_with_Z(xvec, Kvec, Omega, Z_used)
    lhs_mat[2, 1] = angle_product_with_Z(Kvec, xvec, Omega, Z_used)
    lhs_mat[2, 2] = angle_product_with_Z(
        Kvec, Kvec, Omega, Z_used
    ) - 2.0 * angle_product_with_Z(Wvec, xi_0_vec, Omega, Z_used)

    # print_stars(
    #     f"{angle_product_with_Z(Kvec, Kvec, Omega, Z_used)=}  and {-2.0 * angle_product_with_Z(Wvec, xi_0_vec, Omega, Z_used)=}"
    # )

    rhs_vec = np.zeros(3)
    rhs_vec[2] = angle_product_with_Z(Kvec, xi_0_vec, Omega, Z_used)

    dbeta_s2_whatif = np.linalg.solve(lhs_mat, rhs_vec)
    dbeta_whatif, s2_whatif = (
        dbeta_s2_whatif[:2],
        dbeta_s2_whatif[2],
    )

    # print_stars(f"{dbeta_whatif=}, {s2_whatif=}")

    beta0_whatif = beta0_0 + dbeta_whatif[0]
    beta1_whatif = beta1_0 + dbeta_whatif[1]
    whatif_vals = np.array([beta0_whatif, beta1_whatif, s2_whatif])

    return whatif_vals


def get_semi_elast_stats(
    own_semi: np.ndarray, cross_semi: np.ndarray, nproducts: int
) -> tuple[float, float] | tuple[float, float, float, float]:
    if nproducts > 1:
        mean_own_semi_elast = float(np.mean(own_semi))
        stderr_own_semi_elast = float(np.std(own_semi))
        mean_cross_semi_elast = float(np.mean(cross_semi))
        stderr_cross_semi_elast = float(np.std(cross_semi))
        return (
            mean_own_semi_elast,
            stderr_own_semi_elast,
            mean_cross_semi_elast,
            stderr_cross_semi_elast,
        )
    else:
        mean_own_semi_elast = float(np.mean(own_semi))
        stderr_own_semi_elast = float(np.std(own_semi))
        return mean_own_semi_elast, stderr_own_semi_elast

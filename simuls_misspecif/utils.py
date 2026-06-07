import numpy as np
from numpy.random import SeedSequence, default_rng
from typing import cast

from bs_python_utils.bsutils import print_stars, file_print_stars, bs_error_abort


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


def angle_product(um: np.ndarray, vm: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, um.T @ omega @ vm)


def bracket_product(f: np.ndarray, v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    prodfv = np.zeros(f.shape[1:])
    for kk in range(f.shape[1]):
        for ll in range(f.shape[2]):
            prodfv[kk, ll] = angle_product(f[:, kk, ll], v, omega)
    return prodfv

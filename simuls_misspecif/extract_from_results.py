"""Extract the pickled results needed for the main plots."""

import pickle
from pathlib import Path
from typing import Dict, cast


# the keys of the simulation results that the plots need
KEYS_EXTRACT = [
    "non-random values",
    "pseudo true values",
    "whatif just values",
    "whatif over values",
    "SPE variance bounds",
    "true semi-elasticities",
    "pseudo semi-elasticities",
    "whatif just semi-elasticities",
    "whatif over semi-elasticities",
]


def model_string(model: str, nproducts: int, i_scenario: int, n_x: int = 1) -> str:
    """Name of a case, e.g. `endo_J=5_v3`; `_M={n_x}` is inserted when `n_x > 1`."""
    str_M = f"_M={n_x}" if n_x > 1 else ""
    return f"{model}_J={nproducts}{str_M}_v{i_scenario}"


def case_subdir(model: str, i_scenario: int, n_x: int = 1) -> str:
    """Subdirectory of `J{nproducts}` for a case, e.g. `endo_v3` or `endo_M2_v3`."""
    str_M = f"_M{n_x}" if n_x > 1 else ""
    return f"{model}{str_M}_v{i_scenario}"


def case_paths(
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    root_dir: Path,
    n_x: int = 1,
) -> tuple[Path, str]:
    """The directory of a case and its full name including `T`."""
    case_dir = root_dir / f"J{nproducts}" / case_subdir(model, i_scenario, n_x)
    full_str = f"{model_string(model, nproducts, i_scenario, n_x)}_T={nmarkets}"
    return case_dir, full_str


def load_results(
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    root_dir: Path,
    n_x: int = 1,
) -> Dict:
    case_dir, full_str = case_paths(
        model, nproducts, nmarkets, i_scenario, root_dir, n_x
    )
    with open(case_dir / f"simul_results_{full_str}.pkl", "rb") as f:
        dict_results = cast(Dict, pickle.load(f))
    # pprint(dict_results)
    return dict_results


def write_extract_results(
    extract_results: Dict,
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    root_dir: Path,
    n_x: int = 1,
):
    case_dir, full_str = case_paths(
        model, nproducts, nmarkets, i_scenario, root_dir, n_x
    )
    # pprint(extract_results)
    with open(case_dir / f"extract_results_{full_str}.pkl", "wb") as f:
        pickle.dump(extract_results, f)


def extract_from_results(
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    keys_extract: list[str],
    root_dir: Path = Path.cwd(),
    n_x: int = 1,
) -> dict:
    dict_results = load_results(model, nproducts, nmarkets, i_scenario, root_dir, n_x)
    extract_results = {k: dict_results[k] for k in keys_extract}
    write_extract_results(
        extract_results, model, nproducts, nmarkets, i_scenario, root_dir, n_x
    )
    return extract_results


if __name__ == "__main__":
    nmarkets = 5_000

    scenarii = [3, 4]
    J_vals = [5]
    models = ["exo", "endo"]

    root_dir = Path.home() / "Documents" / "Github" / "simuls-misspecif"

    for scenario in scenarii:
        for J in J_vals:
            for model in models:
                res = extract_from_results(
                    model, J, nmarkets, scenario, KEYS_EXTRACT, root_dir=root_dir
                )
                # print(res)

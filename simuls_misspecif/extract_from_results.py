"""Extract the pickled results needed for the main plots."""

import pickle
from pathlib import Path
from typing import Dict, List, cast


def load_results(
    model: str, nproducts: int, nmarkets: int, i_scenario: int, root_dir: Path
) -> Dict:
    case_dir = root_dir / f"J{nproducts}/{model}_v{i_scenario}"
    full_str = f"{model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
    with open(case_dir / f"simul_results_{full_str}.pkl", "rb") as f:
        dict_results = cast(Dict, pickle.load(f))
    return dict_results


def write_extract_results(
    extract_results: Dict,
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    root_dir: Path,
) -> None:
    case_dir = root_dir / f"J{nproducts}/{model}_v{i_scenario}"
    full_str = f"{model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
    with open(case_dir / f"extract_results_{full_str}.pkl", "wb") as f:
        pickle.dump(extract_results, f)


def extract_from_results(
    model: str,
    nproducts: int,
    nmarkets: int,
    i_scenario: int,
    keys_extract: List[str],
    root_dir: Path = Path.cwd(),
):
    dict_results = load_results(model, nproducts, nmarkets, i_scenario, root_dir)
    extract_results = {k: dict_results[k] for k in keys_extract}
    write_extract_results(
        extract_results, model, nproducts, nmarkets, i_scenario, root_dir
    )


if __name__ == "__main__":
    nmarkets = 1000

    scenarii = [3, 4]
    J_vals = [3]
    models = ["exo", "endo"]

    keys_extract = [
        "pseudo true values",
        "correc_d4",
        "correc_dprime4",
        "correc_infty",
        "SPE variance bounds",
        "true semi-elasticities",
        "pseudo semi-elasticities",
        "corrected semi-elasticities",
        "model",
    ]

    root_dir = Path.home() / "Documents" / "Github" / "simuls_MNL"

    for scenario in scenarii:
        for J in J_vals:
            for model in models:
                extract_from_results(
                    model, J, nmarkets, scenario, keys_extract, root_dir=root_dir
                )

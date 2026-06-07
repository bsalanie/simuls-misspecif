"""Extract the pickled results needed for the main plots."""

import pickle
from pathlib import Path
from typing import Dict, cast


def load_results(
    model: str, nproducts: int, nmarkets: int, i_scenario: int, root_dir: Path
) -> Dict:
    case_dir = root_dir / f"J{nproducts}/{model}_v{i_scenario}"
    full_str = f"{model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
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
):
    case_dir = root_dir / f"J{nproducts}/{model}_v{i_scenario}"
    full_str = f"{model}_J={nproducts}_v{i_scenario}_T={nmarkets}"
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
) -> dict:
    dict_results = load_results(model, nproducts, nmarkets, i_scenario, root_dir)
    extract_results = {k: dict_results[k] for k in keys_extract}
    write_extract_results(
        extract_results, model, nproducts, nmarkets, i_scenario, root_dir
    )
    return extract_results


if __name__ == "__main__":
    nmarkets = 5_000

    scenarii = [3, 4]
    J_vals = [5]
    models = ["exo", "endo"]

    keys_extract = [
        "pseudo true values",
        "whatif values",
        "SPE variance bounds",
        "true semi-elasticities",
        "pseudo semi-elasticities",
        "whatif semi-elasticities",
        "model",
    ]

    root_dir = Path.home() / "Documents" / "Github" / "simuls-misspecif"

    for scenario in scenarii:
        for J in J_vals:
            for model in models:
                res = extract_from_results(
                    model, J, nmarkets, scenario, keys_extract, root_dir=root_dir
                )
                # print(res)

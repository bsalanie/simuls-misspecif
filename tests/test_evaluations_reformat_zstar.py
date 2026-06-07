import numpy as np

from simuls_misspecif.evaluations import _reformat_Zstar


def test_reformat_zstar_shape_and_values():
    """_reformat_Zstar should restack rows into (J, n_instr, T)."""
    nmarkets = 3
    nproducts = 2
    n_instr = 2

    # Rows are stacked in market-major order: (t=0,j=0), (t=0,j=1), (t=1,j=0), ...
    zstar = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )

    reformatted = _reformat_Zstar(zstar, nproducts)

    assert reformatted.shape == (nproducts, n_instr, nmarkets)

    # result[j, k, t] == zstar[t*J + j, k]
    expected = np.array(
        [
            # j=0: instrument values across markets t=0,1,2
            [[1.0, 3.0, 5.0], [10.0, 30.0, 50.0]],
            # j=1
            [[2.0, 4.0, 6.0], [20.0, 40.0, 60.0]],
        ]
    )
    np.testing.assert_allclose(reformatted, expected)

import numpy as np

from simuls_misspecif.evaluations import _reformat_varcov


def test_reformat_varcov_output_shape():
    J, n_instr = 3, 4
    v = np.random.default_rng(0).standard_normal((J, n_instr, J, n_instr))
    result = _reformat_varcov(v)
    assert result.shape == (J * n_instr, J * n_instr)


def test_reformat_varcov_single_product():
    J, n_instr = 1, 5
    v = np.random.default_rng(1).standard_normal((J, n_instr, J, n_instr))
    result = _reformat_varcov(v)
    assert result.shape == (n_instr, n_instr)


def test_reformat_varcov_single_instrument():
    J, n_instr = 4, 1
    v = np.random.default_rng(2).standard_normal((J, n_instr, J, n_instr))
    result = _reformat_varcov(v)
    assert result.shape == (J, J)


def test_reformat_varcov_element_mapping():
    """v[j1, k1, j2, k2] should map to result[j1*n_instr+k1, j2*n_instr+k2]."""
    J, n_instr = 2, 3
    v = np.arange(float(J * n_instr * J * n_instr)).reshape(J, n_instr, J, n_instr)
    result = _reformat_varcov(v)
    for j1 in range(J):
        for k1 in range(n_instr):
            for j2 in range(J):
                for k2 in range(n_instr):
                    assert (
                        result[j1 * n_instr + k1, j2 * n_instr + k2]
                        == v[j1, k1, j2, k2]
                    )


def test_reformat_varcov_symmetric_input_gives_symmetric_output():
    """A symmetric variance-covariance array should produce a symmetric matrix."""
    J, n_instr = 3, 2
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((J * n_instr, J * n_instr))
    sym_matrix = raw @ raw.T
    v = sym_matrix.reshape(J, n_instr, J, n_instr)
    result = _reformat_varcov(v)
    np.testing.assert_allclose(result, result.T)


def test_reformat_varcov_zero_input():
    J, n_instr = 2, 2
    v = np.zeros((J, n_instr, J, n_instr))
    result = _reformat_varcov(v)
    np.testing.assert_allclose(result, np.zeros((J * n_instr, J * n_instr)))


def test_reformat_varcov_preserves_diagonal_blocks():
    """The diagonal block (j, j) in the output should equal v[j, :, j, :]."""
    J, n_instr = 3, 2
    v = np.random.default_rng(4).standard_normal((J, n_instr, J, n_instr))
    result = _reformat_varcov(v)
    for j in range(J):
        block = result[j * n_instr : (j + 1) * n_instr, j * n_instr : (j + 1) * n_instr]
        np.testing.assert_allclose(block, v[j, :, j, :])

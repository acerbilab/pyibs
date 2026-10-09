import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyibs._estimates import (
    ibs_loglik,
    ibs_var,
    repeat_estimates,
    trial_weights,
)

EXACT = dict(rtol=1e-12, atol=1e-12)


def test_count_formulas_match_harmonic_sums():
    K = np.arange(1, 51)
    loglik = [-sum(1 / k for k in range(1, int(k_))) for k_ in K]
    var = [sum(1 / k**2 for k in range(1, int(k_))) for k_ in K]
    assert_allclose(ibs_loglik(K), loglik, **EXACT)
    assert_allclose(ibs_var(K), var, **EXACT)
    for k_, a, b in zip(K, loglik, var):
        assert_allclose(ibs_loglik(int(k_)), a, **EXACT)
        assert_allclose(ibs_var(int(k_)), b, **EXACT)


def test_count_formulas_are_zero_at_one():
    assert ibs_loglik(1) == 0.0
    assert ibs_var(1) == 0.0
    assert np.all(ibs_loglik(np.ones(3, dtype=np.int64)) == 0.0)
    assert np.all(ibs_var(np.ones(3, dtype=np.int64)) == 0.0)


def test_trial_weights_forms():
    assert_allclose(trial_weights(None, 3), np.ones(3), **EXACT)
    assert_allclose(trial_weights(2.5, 3), np.full(3, 2.5), **EXACT)
    assert_allclose(trial_weights(np.float64(0.0), 2), np.zeros(2), **EXACT)
    w = [0.0, 1.0, 3.0]
    out = trial_weights(w, 3)
    assert out.dtype == np.float64 and out.shape == (3,)
    assert_allclose(out, w, **EXACT)
    arr = np.array(w)
    out = trial_weights(arr, 3)
    out[0] = 7.0
    assert arr[0] == 0.0  # a copy


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 1.0],
        np.ones((3, 1)),
        [1.0, np.nan, 1.0],
        [1.0, np.inf, 1.0],
        [1.0, -0.5, 1.0],
        np.nan,
        -1.0,
    ],
)
def test_trial_weights_rejected(weights):
    with pytest.raises(ValueError):
        trial_weights(weights, 3)


@pytest.mark.parametrize(
    "weights", [True, [True, False, True], "1.5", ["1", "2", "3"], [1, "2", 3]]
)
def test_trial_weights_refuse_booleans_and_strings(weights):
    with pytest.raises(TypeError, match="booleans or strings"):
        trial_weights(weights, 3)


def test_trial_weights_rejects_bad_n_trials():
    for n in (0, 1.5, True):
        with pytest.raises(ValueError):
            trial_weights(None, n)


def test_repeat_estimates_against_direct_sums():
    K = np.random.default_rng(0).geometric(0.3, size=(6, 4))
    w = np.array([0.0, 0.5, 1.0, 2.0])
    values, var_estimates, trial_values, trial_vars = repeat_estimates(K, w)
    direct_values = [
        sum(w[i] * ibs_loglik(int(K[r, i])) for i in range(4))
        for r in range(6)
    ]
    direct_vars = [
        sum(w[i] ** 2 * ibs_var(int(K[r, i])) for i in range(4))
        for r in range(6)
    ]
    assert_allclose(values, direct_values, **EXACT)
    assert_allclose(var_estimates, direct_vars, **EXACT)
    assert_allclose(trial_values, ibs_loglik(K).sum(axis=0), **EXACT)
    assert_allclose(trial_vars, ibs_var(K).sum(axis=0), **EXACT)


@pytest.mark.parametrize(
    "K",
    [
        # Every count within the table.
        np.random.default_rng(2).geometric(0.05, size=(40, 25)),
        # A table of 4 entries for 5 of the counts; 3 evaluated directly.
        np.array([[1, 3, 10**6, 2], [2, 10**12, 7, 1]]),
        np.array([[1, 3, 10**6, 2], [2, 9, 7, 1]], dtype=np.uint32),
        # No count within the table; all evaluated directly.
        np.array([[10**6, 5, 10**9]]),
    ],
)
def test_repeat_estimates_terms_are_the_formulas_bitwise(K):
    w = np.linspace(0.5, 2, K.shape[1])
    values, var_estimates, trial_values, trial_vars = repeat_estimates(K, w)
    assert np.array_equal(values, np.sum(ibs_loglik(K) * w, axis=1))
    assert np.array_equal(var_estimates, np.sum(ibs_var(K) * w**2, axis=1))
    assert np.array_equal(trial_values, np.sum(ibs_loglik(K), axis=0))
    assert np.array_equal(trial_vars, np.sum(ibs_var(K), axis=0))


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.uint64],
)
def test_count_terms_do_not_depend_on_the_integer_type(dtype):
    # Two counts beyond the table of 4 entries.
    K = np.array([[1, 2, 3, 100], [127, 1, 1, 2]])
    assert np.array_equal(ibs_loglik(K.astype(dtype)), ibs_loglik(K))
    assert np.array_equal(ibs_var(K.astype(dtype)), ibs_var(K))
    w = np.linspace(0.5, 2, 4)
    typed = repeat_estimates(K.astype(dtype), w)
    for a, b in zip(typed, repeat_estimates(K, w)):
        assert np.array_equal(a, b)


def test_repeat_estimates_without_repeats():
    values, var_estimates, trial_values, trial_vars = repeat_estimates(
        np.ones((0, 3), dtype=np.int64), np.ones(3)
    )
    assert values.shape == var_estimates.shape == (0,)
    assert np.all(trial_values == 0.0) and trial_values.shape == (3,)
    assert np.all(trial_vars == 0.0) and trial_vars.shape == (3,)


@pytest.mark.parametrize(
    "K", [np.ones((2, 3)), np.array([[1, 0, 2]]), np.array([[1, -3, 2]])]
)
def test_repeat_estimates_rejects_counts(K):
    with pytest.raises(ValueError):
        repeat_estimates(K, np.ones(3))


def test_repeat_estimates_rows_do_not_depend_on_blocks():
    K = np.random.default_rng(1).geometric(0.05, size=(9, 37))
    w = np.linspace(0, 2, 37)
    whole = repeat_estimates(K, w)
    parts = [repeat_estimates(K[lo : lo + 2], w) for lo in range(0, 9, 2)]
    for k in (0, 1):
        assert np.array_equal(np.concatenate([p[k] for p in parts]), whole[k])


@pytest.mark.parametrize(
    "shape, weights", [((3,), np.ones(3)), ((2, 3), np.ones(2))]
)
def test_repeat_estimates_rejects_shapes(shape, weights):
    with pytest.raises(ValueError):
        repeat_estimates(np.ones(shape, dtype=np.int64), weights)

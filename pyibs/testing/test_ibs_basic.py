"""Tests of :func:`pyibs.ibs_basic`, the didactic loop."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyibs import ibs_basic
from pyibs.testing._exact import exact_loglik, exact_var

SEED = 20261009
THETA = np.zeros(1)
# Match probabilities of ten Bernoulli trials, whose responses are all ones.
P = np.linspace(0.1, 1, 10)
N_CALLS = 1000


def by_index(theta, i, rng):
    return rng.random() < P[i]


def by_design(theta, s, rng):
    return rng.random() < s[0]


def test_agrees_with_the_exact_loglik():
    # The mean of many calls against log L, within 4.5 of its exact SE, and
    # the variance of the calls against the exact variance of one repeat,
    # with the standard error of a sample variance from the squared
    # deviations.
    rng = np.random.default_rng(SEED)
    L = np.array(
        [
            ibs_basic(by_index, THETA, np.ones(P.size), random_seed=rng)
            for _ in range(N_CALLS)
        ]
    )
    se_mean = math.sqrt(exact_var(P) / N_CALLS)
    assert abs(L.mean() - exact_loglik(P)) < 4.5 * se_mean
    sq = (L - L.mean()) ** 2
    se_var = sq.std(ddof=1) / math.sqrt(N_CALLS)
    assert abs(L.var(ddof=1) - exact_var(P)) < 4.5 * se_var


def test_design_gives_the_draws_of_the_trial_index():
    # The probabilities as a design column: the same draws as by index.
    for seed in range(SEED, SEED + 20):
        by_trial = ibs_basic(
            by_index, THETA, np.ones(P.size), random_seed=seed
        )
        assert by_trial == ibs_basic(
            by_design, THETA, np.ones(P.size), P[:, None], random_seed=seed
        )


def test_without_design_the_simulator_receives_the_trial_index():
    seen = []

    def simulator(theta, s):
        seen.append((theta, s))
        return 1.0

    L = ibs_basic(simulator, THETA, np.ones(4))
    assert type(L) is float and L == 0.0
    assert [s for _, s in seen] == [0, 1, 2, 3]
    assert all(theta is THETA for theta, _ in seen)


def test_design_rows_are_passed():
    S = np.arange(8).reshape(4, 2)
    seen = []

    def simulator(theta, s):
        seen.append(np.array(s))
        return 1.0

    assert ibs_basic(simulator, THETA, np.ones(4), S) == 0.0
    assert len(seen) == 4
    for i, s in enumerate(seen):
        assert np.array_equal(s, S[i])


def test_every_column_must_match():
    # Trial 0 matches [1, 2] at its third sample: L = -(1 + 1/2). Trial 1
    # matches [3, 4] at its first.
    R = np.array([[1, 2], [3, 4]])
    outputs = {0: [[1, 0], [0, 2], [1, 2]], 1: [[3, 4]]}

    def simulator(theta, i):
        return outputs[i].pop(0)

    assert ibs_basic(simulator, THETA, R) == -1.5
    assert outputs == {0: [], 1: []}


@pytest.mark.parametrize(
    "observed, output",
    [
        ([1], []),
        ([1], [1, 1]),
        ([[1, 1]], 1),
        ([[1, 1]], [1]),
        ([[1, 1]], [[1], [1]]),
        ([[1, 1]], [[[1, 1]]]),
    ],
)
def test_invalid_simulator_shape_raises(observed, output):
    calls = 0

    def simulator(theta, s):
        nonlocal calls
        calls += 1
        assert calls == 1, "Invalid responses must fail on the first draw."
        return output

    with pytest.raises(ValueError, match="shape"):
        ibs_basic(simulator, THETA, observed)


@pytest.mark.parametrize("output", [1, [1], [[1]]])
@pytest.mark.parametrize("observed", [[1], [[1]]])
def test_single_response_shapes(observed, output):
    assert ibs_basic(lambda theta, s: output, THETA, observed) == 0.0


@pytest.mark.parametrize("output", [[1, 2], [[1, 2]]])
def test_multicolumn_response_shapes(output):
    assert ibs_basic(lambda theta, s: output, THETA, [[1, 2]]) == 0.0


def test_generator_reaches_the_simulator():
    seen = []

    def simulator(theta, s, rng):
        seen.append(rng)
        return rng.random() < 0.5

    gen = np.random.default_rng(SEED)
    ibs_basic(simulator, THETA, np.ones(5), random_seed=gen)
    assert seen and all(rng is gen for rng in seen)
    # A seed gives the estimate of the draws of its generator, in order.
    rng = np.random.default_rng(SEED)
    expected = 0.0
    for _ in range(20):
        K = 1
        while not rng.random() < 0.5:
            K += 1
        expected -= sum(1 / k for k in range(1, K))
    first, second = (
        ibs_basic(simulator, THETA, np.ones(20), random_seed=SEED)
        for _ in range(2)
    )
    assert first == second
    assert_allclose(first, expected, rtol=1e-12, atol=1e-12)


def test_keyword_only_rng():
    def simulator(theta, s, *, rng):
        return rng.random() < 0.5

    first = ibs_basic(simulator, THETA, np.ones(20), random_seed=SEED)
    assert first == ibs_basic(simulator, THETA, np.ones(20), random_seed=SEED)


def never_called(theta, s):
    raise AssertionError("The simulator must not be called.")


@pytest.mark.parametrize(
    "R, S, match",
    [
        (np.array([1.0, np.nan]), None, "R holds a NaN"),
        (np.array([[1.0, 2.0], [np.nan, 1.0]]), None, "R holds a NaN"),
        (np.array(["a", np.nan], dtype=object), None, "R holds a NaN"),
        (np.ones((2, 2, 2)), None, "R must be"),
        (np.ones(0), None, "R must be"),
        (np.ones(3), np.ones(2), "S must have one row per trial"),
    ],
)
def test_invalid_inputs_raise(R, S, match):
    with pytest.raises(ValueError, match=match):
        ibs_basic(never_called, THETA, R, S)


@pytest.mark.parametrize(
    "R, output",
    [
        (np.ones(3), "1"),
        (np.array(["a", "b"]), b"a"),
        (np.array(["1", "0"]), 1.0),
        # NumPy makes text of a row that mixes numbers and text.
        (np.array([[1, "a"], [2, "b"]], dtype=object), [1, "a"]),
    ],
)
def test_responses_that_never_match_raise(R, output):
    # NumPy finds text, bytes and numbers unequal to one another whatever
    # their values, so no sample of such a simulator can match; without the
    # check, ibs_basic would sample forever, which the guard turns into
    # RuntimeError.
    calls = 0

    def simulator(theta, s):
        nonlocal calls
        calls += 1
        if calls > 50:
            raise RuntimeError("ibs_basic is still sampling after 50 calls")
        return output

    with pytest.raises(TypeError, match="never finds equal"):
        ibs_basic(simulator, THETA, R)
    assert calls == 1


def test_responses_that_mix_numbers_and_text():
    # Trial 0 matches at its second sample, trial 1 at its first.
    R = np.array([[1, "a"], [2, "b"]], dtype=object)
    outputs = {0: [[1, "b"], [1, "a"]], 1: [[2, "b"]]}

    def simulator(theta, i):
        return np.array(outputs[i].pop(0), dtype=object)

    assert ibs_basic(simulator, THETA, R) == -1.0

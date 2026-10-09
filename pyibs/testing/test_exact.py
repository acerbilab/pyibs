import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import spence

from pyibs._estimates import ibs_loglik, ibs_var, repeat_estimates
from pyibs.testing import _exact
from pyibs.testing._exact import exact_draw, exact_loglik, exact_var
from pyibs.testing._helpers import summary

SEED = 20260929
N_REPEATS = 20_000
P = np.linspace(0.05, 1, 20)
W = np.linspace(0, 2, 20)
EXACT = dict(rtol=1e-12, atol=1e-12)


def draw(n, weights=None, seed=SEED, p=P):
    return exact_draw(p, n, np.random.default_rng(seed), weights)


# ---------------------------------------------------------------------------
# Exact moments


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_exact_var_against_series(p):
    k = np.arange(1, 10**5 + 1, dtype=float)
    series = np.sum((1 - p) ** k / k**2)
    assert_allclose(exact_var(p), series, rtol=1e-9)
    assert_allclose(exact_var(np.array([p])), series, rtol=1e-9)


def test_exact_moments_at_one():
    assert exact_loglik(np.ones(4)) == 0.0
    assert exact_var(np.ones(4)) == 0.0


def test_exact_var_bound():
    # Li2(1) = pi**2 / 6 bounds the variance of a trial.
    assert exact_var(1e-12) < math.pi**2 / 6
    assert_allclose(exact_var(1e-12), math.pi**2 / 6, rtol=1e-9)


def test_weighted_exact_moments():
    w = np.linspace(0, 2, 20)
    direct_loglik = sum(wi * math.log(pi) for wi, pi in zip(w, P))
    direct_var = sum(wi**2 * spence(pi) for wi, pi in zip(w, P))
    assert_allclose(exact_loglik(P, w), direct_loglik, **EXACT)
    assert_allclose(exact_var(P, w), direct_var, **EXACT)
    assert_allclose(exact_loglik(P, 3.0), 3 * exact_loglik(P), **EXACT)
    assert_allclose(exact_var(P, 3.0), 9 * exact_var(P), **EXACT)


@pytest.mark.parametrize(
    "p", [[0.5, 0.0], [0.5, 1.5], [np.nan], [], np.ones((2, 2))]
)
def test_exact_moments_reject_bad_probs(p):
    with pytest.raises(ValueError):
        exact_loglik(p)
    with pytest.raises(ValueError):
        exact_var(p)


# ---------------------------------------------------------------------------
# Exact draws


@pytest.fixture(scope="module")
def batch():
    return draw(N_REPEATS)


@pytest.fixture(scope="module")
def weighted_batch():
    return draw(N_REPEATS, weights=W, seed=SEED + 1)


def assert_calibrated(batch, loglik, var):
    n = batch.n
    values = batch.values
    se_mean = values.std(ddof=1) / math.sqrt(n)
    assert abs(values.mean() - loglik) < 4.5 * se_mean
    sample_var = np.var(values, ddof=1)
    assert abs(sample_var - var) < 4.5 * var * math.sqrt(2 / (n - 1))
    # IBS paper Eq 16: the variance estimate is calibrated on average.
    v_hat = batch.var_estimates
    se_v_hat = v_hat.std(ddof=1) / math.sqrt(n)
    assert abs(v_hat.mean() - var) < 4.5 * se_v_hat


def test_calibrated(batch):
    assert batch.n == N_REPEATS
    assert_calibrated(batch, exact_loglik(P), exact_var(P))


def test_calibrated_weighted(weighted_batch):
    assert_calibrated(weighted_batch, exact_loglik(P, W), exact_var(P, W))


@pytest.mark.parametrize("fixture", ["batch", "weighted_batch"])
def test_per_trial_outputs(fixture, request):
    b = request.getfixturevalue(fixture)
    s = summary(b)
    assert s.trial_repeats == N_REPEATS
    trial_loglik = s.trial_loglik
    assert trial_loglik.shape == P.shape
    se = np.sqrt(spence(P) / N_REPEATS)
    assert P[-1] == 1.0 and trial_loglik[-1] == 0.0
    assert s.trial_nominal_var[-1] == 0.0
    assert np.all(np.abs(trial_loglik[:-1] - np.log(P[:-1])) < 4.5 * se[:-1])


def test_trial_sums_add_up_to_values(batch):
    assert_allclose(batch.trial_value_sums.sum(), batch.values.sum(), **EXACT)
    assert_allclose(
        batch.trial_var_sums.sum(), batch.var_estimates.sum(), **EXACT
    )


def test_all_ones_give_zero():
    b = draw(50, p=np.ones(5))
    assert np.all(b.values == 0.0)
    assert np.all(b.var_estimates == 0.0)
    assert b.samples == 50 * 5


@pytest.mark.parametrize("weights", [None, W])
def test_samples_and_values_from_counts(weights):
    n = 5
    b = draw(n, weights=weights)
    K = np.random.default_rng(SEED).geometric(P, size=(n, P.size))
    w = np.ones(P.size) if weights is None else weights
    assert b.samples == K.sum()
    assert_allclose(b.values, ibs_loglik(K) @ w, **EXACT)
    # The draws reduce their counts through the shared helper.
    values, var_estimates, _, _ = repeat_estimates(K, w)
    assert np.array_equal(b.values, values)
    assert np.array_equal(b.var_estimates, var_estimates)
    assert_allclose(b.var_estimates, ibs_var(K) @ w**2, **EXACT)
    assert_allclose(b.trial_value_sums, ibs_loglik(K).sum(axis=0), **EXACT)
    assert_allclose(b.trial_var_sums, ibs_var(K).sum(axis=0), **EXACT)


@pytest.mark.parametrize("chunk", [7, 45])
def test_chunking_is_bitwise_invariant(monkeypatch, chunk):
    n = 53
    reference = draw(n, weights=W)
    monkeypatch.setattr(_exact, "_CHUNK_ELEMENTS", chunk)
    chunked = draw(n, weights=W)
    assert np.array_equal(chunked.values, reference.values)
    assert np.array_equal(chunked.var_estimates, reference.var_estimates)
    assert chunked.samples == reference.samples
    assert_allclose(
        chunked.trial_value_sums, reference.trial_value_sums, **EXACT
    )


@pytest.mark.parametrize(
    "probs", [[0.5, 0.0], [0.5, 1.2], [np.nan, 0.5], [], [[0.5, 0.5]]]
)
def test_rejects_bad_probs(probs):
    with pytest.raises(ValueError):
        draw(2, p=np.array(probs))


def test_weights_validated():
    for weights in ([1.0, -1.0], np.nan, np.ones((2, 2)), np.ones(3)):
        with pytest.raises(ValueError):
            draw(2, weights=weights)


def test_rejects_bad_n():
    for n in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            draw(n)


def test_scalar_weight_scales_values():
    plain, scaled = draw(10), draw(10, weights=2.0)
    assert_allclose(scaled.values, 2 * plain.values, **EXACT)
    assert_allclose(scaled.var_estimates, 4 * plain.var_estimates, **EXACT)
    assert_allclose(scaled.trial_value_sums, plain.trial_value_sums, **EXACT)


class FixedCounts:
    """A generator stub whose geometric draws all return ``count``."""

    def __init__(self, count):
        self.count = count

    def geometric(self, p, size):
        return np.full(size, self.count, dtype=np.int64)


def test_smallest_probability():
    with pytest.raises(ValueError, match="int64"):
        draw(2, p=np.array([0.5, 9e-16]))
    p = np.array([0.5, 1e-15])
    b = draw(3, p=p)
    K = np.random.default_rng(SEED).geometric(p, size=(3, 2))
    assert b.samples == sum(int(k) for k in K.ravel())
    assert np.all(np.isfinite(b.values))


def test_samples_are_exact_beyond_int64():
    # Six counts of 2**62 sum to 1.5 * 2**64, which an int64 sum wraps.
    b = exact_draw(np.full(3, 0.5), 2, FixedCounts(2**62))
    assert b.samples == 6 * 2**62
    assert_allclose(b.values, 3 * ibs_loglik(2**62), **EXACT)


def test_clamped_count_raises():
    with pytest.raises(OverflowError, match="int64"):
        exact_draw(np.full(3, 0.5), 2, FixedCounts(np.iinfo(np.int64).max))

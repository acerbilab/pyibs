import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import spence

from pyibs import _estimates, _sampler
from pyibs._estimates import ibs_loglik, ibs_var
from pyibs._sampler import IBSSamplingError, _Settings
from pyibs.testing._exact import exact_loglik, exact_var
from pyibs.testing._helpers import (
    P,
    ScriptedSimulator,
    bernoulli,
    cost,
    draw,
    ibslike_samples,
    matlab_round,
    never_matches,
    summary,
)

SEED = 20260929
N_REPEATS = 20_000
W = np.linspace(0, 2, 20)
EXACT = dict(rtol=1e-12, atol=1e-12)


class Counting:
    """Wrap a simulator and count its calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, theta, design_rows, rng):
        self.calls += 1
        return self.fn(theta, design_rows, rng)


@pytest.fixture
def captured_counts(monkeypatch):
    """Capture every count matrix the sampler reduces."""
    captured = []
    original = _estimates.repeat_estimates

    def spy(K, weights):
        captured.append(np.array(K))
        return original(K, weights)

    monkeypatch.setattr(_estimates, "repeat_estimates", spy)
    return captured


def stream_counts(stream, n):
    """The first n gaps between the ones of a 0/1 stream."""
    hits = np.flatnonzero(np.asarray(stream) == 1)
    return np.diff(np.concatenate([[-1], hits]))[:n]


# ---------------------------------------------------------------------------
# Gap extraction on scripted streams

# Trial 0 matches at every sample, trial 1 needs 5 then 2 samples, trial 2
# needs 2 then 6; each stream is padded with matches.
STREAMS = [
    [1] * 12,
    [0, 0, 0, 0, 1, 0, 1, 1] + [1] * 4,
    [0, 1, 0, 0, 0, 0, 0, 1, 1] + [1] * 3,
]
STREAM_K = np.array([[1, 5, 2], [1, 2, 6]])
STREAM_W = np.array([0.5, 1.0, 2.0])


@pytest.mark.parametrize(
    "initial, acceleration, sizes",
    [
        # One sample per open trial per call: trial 0 is done after two
        # calls, trial 1 after seven and trial 2 after eight; no surplus.
        (1, 1, [3, 3, 2, 2, 2, 2, 2, 1]),
        # Three samples per trial per call. Trial 0 is done in the first
        # call with one surplus sample. Trial 1's first count spans the
        # first two calls, and its second match in call 3 leaves two
        # surplus samples; trial 2's second count spans calls 1-3 and
        # leaves one.
        (3, 1, [9, 6, 6]),
        # Two, three, then five samples per trial per call: MATLAB's round
        # takes 4.5 to 5, where Python's and NumPy's give 4.
        (2, 1.5, [6, 6, 10]),
    ],
)
def test_scripted_counts(captured_counts, initial, acceleration, sizes):
    sim = ScriptedSimulator(STREAMS)
    settings = _Settings(
        sim,
        np.ones(3),
        trial_weights=STREAM_W,
        initial_samples=initial,
        acceleration=acceleration,
    )
    b = draw(settings, 2, SEED)
    (K,) = captured_counts
    assert np.array_equal(K, STREAM_K)
    assert np.array_equal(b.K, STREAM_K)
    for i, stream in enumerate(STREAMS):
        assert np.array_equal(K[:, i], stream_counts(stream, 2))
    # Trial-major requests, with the trials still open.
    assert [r.size for r in sim.requests] == sizes
    m = sizes[0] // 3
    assert np.array_equal(sim.requests[0], np.repeat([0, 1, 2], m))
    assert b.calls == len(sizes)
    assert b.samples == sum(sizes)
    assert b.n == 2
    assert b.samples >= K.sum()
    assert_allclose(b.values, ibs_loglik(STREAM_K) @ STREAM_W, **EXACT)
    assert_allclose(
        b.var_estimates, ibs_var(STREAM_K) @ STREAM_W**2, **EXACT
    )
    assert_allclose(
        b.trial_value_sums, ibs_loglik(STREAM_K).sum(axis=0), **EXACT
    )
    assert_allclose(b.trial_var_sums, ibs_var(STREAM_K).sum(axis=0), **EXACT)
    assert summary(b).trial_repeats == 2
    assert b.n_thresholded == 0
    assert not b.ended.any()


def reference_sampler(streams, n, initial, acceleration, max_samples, mem):
    """Rows-first IBS sampling, one trial and one sample at a time.

    The level is not bounded, as in ``ibslike.m``; ``mem`` is ``max_mem``.
    """
    n_trials = len(streams)
    pos = [0] * n_trials
    counts = [[] for _ in range(n_trials)]
    open_count = [0] * n_trials
    level = initial
    sizes = []
    while True:
        open_trials = [i for i in range(n_trials) if len(counts[i]) < n]
        if not open_trials:
            break
        m = ibslike_samples(level, len(open_trials), max_samples, mem)
        sizes.append(len(open_trials) * m)
        for i in open_trials:
            for _ in range(m):
                hit = streams[i][pos[i]] == 1
                pos[i] += 1
                if len(counts[i]) < n:
                    open_count[i] += 1
                    if hit:
                        counts[i].append(open_count[i])
                        open_count[i] = 0
        level *= acceleration
    return np.array(counts).T, sizes


# The default max_mem of 6 trials is 1000.
@pytest.mark.parametrize(
    "initial, acceleration, max_samples, max_mem",
    [
        (1, 1, 10**4, 1000),
        (3, 1, 10**4, 1000),
        (2, 1.5, 10**4, 1000),
        (None, 1.5, 10**4, 1000),
        (5, 2.0, 10**4, 13),
        (1, 1.5, 10**4, 4),
        (None, 1.5, 3, 1000),
        (5, 2.0, 7, 10),
    ],
)
def test_counts_match_reference_sampler(
    captured_counts, initial, acceleration, max_samples, max_mem
):
    rng = np.random.default_rng(SEED)
    probs = np.linspace(0.1, 0.9, 6)
    streams = (rng.random((6, 20_000)) < probs[:, None]).astype(int)
    n = 7
    sim = ScriptedSimulator(streams)
    settings = _Settings(
        sim,
        np.ones(6),
        initial_samples=initial,
        acceleration=acceleration,
        max_samples=max_samples,
        max_mem=None if max_mem == 1000 else max_mem,
    )
    assert settings.max_mem == max_mem
    b = draw(settings, n, SEED)
    K_ref, sizes_ref = reference_sampler(
        streams,
        n,
        n if initial is None else initial,
        acceleration,
        max_samples,
        max_mem,
    )
    (K,) = captured_counts
    assert np.array_equal(K, K_ref)
    for i in range(6):
        assert np.array_equal(K[:, i], stream_counts(streams[i], n))
    assert [r.size for r in sim.requests] == sizes_ref
    assert b.samples == sum(sizes_ref)
    assert b.calls == len(sizes_ref)


def test_match_counts_state():
    # Two repeats of three trials; the open count is the part of the
    # current count sampled so far, and 0 once a trial is done.
    counts = _sampler._MatchCounts(2, 3)
    trials = counts.open_trials()
    assert np.array_equal(trials, [0, 1, 2])
    counts.absorb(
        trials, np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], bool)
    )
    assert np.array_equal(counts.K, [[1, 3, 0], [1, 0, 0]])
    assert np.array_equal(counts.completed, counts.K > 0)
    assert np.array_equal(counts.repeat, [2, 1, 0])
    assert np.array_equal(counts.open_count, [0, 1, 4])
    trials = counts.open_trials()
    assert np.array_equal(trials, [1, 2])
    # Trial 1 needs one more match and discards its third sample; trial 2
    # closes its open count of 4 at its first sample.
    counts.absorb(trials, np.array([[0, 1, 1], [1, 0, 1]], bool))
    assert np.array_equal(counts.K, [[1, 3, 5], [1, 3, 2]])
    assert counts.completed.all()
    assert np.array_equal(counts.repeat, [2, 2, 2])
    assert np.array_equal(counts.open_count, [0, 0, 0])
    assert counts.open_trials().size == 0


def test_multi_column_responses_match_on_every_column(captured_counts):
    responses = np.array([[1, 2], [3, 4]])
    streams = [
        [[1, 0], [0, 2], [1, 2], [1, 2], [2, 1], [1, 2]],
        [[3, 4], [4, 3], [3, 0], [3, 4], [3, 4], [3, 4]],
    ]
    settings = _Settings(
        ScriptedSimulator(streams),
        responses,
        initial_samples=1,
        acceleration=1,
    )
    b = draw(settings, 2, SEED)
    (K,) = captured_counts
    assert np.array_equal(K, [[3, 1], [1, 3]])
    assert b.samples == K.sum()


def test_design_rows_are_passed(captured_counts):
    n_trials = 3
    design = np.column_stack([np.arange(n_trials), 10.0 * np.arange(n_trials)])
    seen = []
    scripted = ScriptedSimulator(STREAMS, design=True)

    def simulator(theta, design_rows, rng):
        seen.append(np.array(design_rows))
        return scripted(theta, design_rows, rng)

    settings = _Settings(
        simulator, np.ones(n_trials), design, initial_samples=2
    )
    draw(settings, 2, SEED)
    (K,) = captured_counts
    assert np.array_equal(K, STREAM_K)
    for rows, idx in zip(seen, scripted.requests):
        assert np.array_equal(rows, design[idx])


def test_simulator_receives_theta_and_rng():
    seen = []

    def simulator(theta, design_rows, rng):
        seen.append((theta, rng))
        return bernoulli(theta, design_rows, rng)

    settings = _Settings(simulator, np.ones(P.size))
    theta = np.array([0.5, -1.0])
    rng = np.random.default_rng(SEED)
    _sampler.sample(settings, theta, 3, rng)
    assert seen
    assert all(t is theta and r is rng for t, r in seen)


def test_all_ones_give_zero():
    settings = _Settings(lambda t, idx, rng: np.ones(len(idx)), np.ones(5))
    b = draw(settings, 4, SEED)
    assert np.all(b.values == 0.0)
    assert np.all(b.var_estimates == 0.0)
    assert b.samples == 4 * 5
    assert b.calls == 1


# ---------------------------------------------------------------------------
# Distribution against the exact IBS moments


@pytest.fixture(scope="module")
def batch():
    return draw(_Settings(bernoulli, np.ones(P.size)), N_REPEATS, SEED)


@pytest.fixture(scope="module")
def weighted_batch():
    settings = _Settings(bernoulli, np.ones(P.size), trial_weights=W)
    return draw(settings, N_REPEATS, seed=SEED + 1)


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
    # Successive repeats are independent.
    lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
    assert abs(lag1) < 4.5 / math.sqrt(n)


def test_calibrated(batch):
    assert batch.n == N_REPEATS
    assert batch.n_thresholded == 0
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


# ---------------------------------------------------------------------------
# Reproducibility and the acceleration schedule


def test_seeded_draws_reproduce():
    settings = _Settings(bernoulli, np.ones(P.size), trial_weights=W)
    first, second = draw(settings, 50, SEED), draw(settings, 50, SEED)
    assert np.array_equal(first.values, second.values)
    assert np.array_equal(first.var_estimates, second.var_estimates)
    assert np.array_equal(first.trial_value_sums, second.trial_value_sums)
    assert cost(first) == cost(second)
    other = draw(settings, 50, seed=SEED + 1)
    assert not np.array_equal(first.values, other.values)


class FakeTime:
    """Stands in for the ``time`` module; the simulator advances it."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


# One trial whose first match is its 41st sample.
LATE_MATCH = [0] * 40 + [1] * 60


@pytest.mark.parametrize(
    "threshold, durations, sizes",
    [
        # Default: the level doubles after every call.
        (None, [1.0], [2, 4, 8, 16, 32]),
        # It doubles only after the fast calls (0.01 s < 0.1 s).
        (0.1, [0.01, 1.0], [2, 4, 4, 8, 8, 16]),
        # It never grows when every call is slow.
        (0.1, [1.0], [2] * 21),
    ],
)
def test_acceleration_threshold_follows_timing(
    monkeypatch, threshold, durations, sizes
):
    clock = FakeTime()
    monkeypatch.setattr(_sampler, "time", clock)
    scripted = ScriptedSimulator([LATE_MATCH])
    elapsed = []

    def simulator(theta, design_rows, rng):
        elapsed.append(durations[len(elapsed) % len(durations)])
        clock.now += elapsed[-1]
        return scripted(theta, design_rows, rng)

    settings = _Settings(
        simulator,
        np.ones(1),
        initial_samples=2,
        acceleration=2.0,
        acceleration_threshold=threshold,
    )
    b = draw(settings, 1, SEED)
    assert [r.size for r in scripted.requests] == sizes
    assert b.seconds == pytest.approx(sum(elapsed))
    assert b.values[0] == ibs_loglik(41)


@pytest.mark.parametrize(
    "level, expected",
    [
        (1.0, 1),
        (1.4999999999999998, 1),
        (1.5, 2),
        (2.5, 3),
        (4.5, 5),
        (22.5, 23),
        (22.499999999999996, 22),
        (10**4 - 0.5, 10**4),
        (10**4, 10**4),
    ],
)
def test_samples_per_trial_rounds_as_matlab(level, expected):
    # MATLAB's round takes halves away from zero; round and np.round take
    # 2.5, 4.5 and 22.5 to 2, 4 and 22.
    settings = _Settings(bernoulli, np.ones(1), max_mem=10**9)
    assert _sampler._samples_per_trial(level, 1, settings) == expected
    assert expected == matlab_round(level)


@pytest.mark.parametrize(
    "max_samples, max_mem, n_open, expected",
    [
        # ceil(max_mem / n_open) per trial: ceil(10 / 3) = 4.
        (10**4, 10, 3, 4),
        (10**4, 9, 3, 3),
        # At least one sample per trial with more open trials than max_mem.
        (10**4, 10, 25, 1),
        # max_samples bounds the samples of one trial.
        (5, 10**6, 2, 5),
    ],
)
def test_samples_per_trial_bounds(max_samples, max_mem, n_open, expected):
    settings = _Settings(
        bernoulli, np.ones(1), max_samples=max_samples, max_mem=max_mem
    )
    level = 1000.0
    assert _sampler._samples_per_trial(level, n_open, settings) == expected
    assert expected == ibslike_samples(level, n_open, max_samples, max_mem)


@pytest.mark.parametrize(
    "n_trials, expected",
    [(1, 1000), (10, 1000), (11, 1100), (10**5, 10**6)],
)
def test_default_max_mem(n_trials, expected):
    # ibslike.m's MaxMem, max(min(N, 1e4), 10) * 100 (line 137).
    settings = _Settings(bernoulli, np.ones(n_trials))
    assert settings.max_mem == expected
    assert _Settings(bernoulli, np.ones(n_trials), max_mem=7).max_mem == 7


def test_level_is_bounded_by_max_samples():
    # The request of one open trial is capped at max_samples.
    scripted = ScriptedSimulator([[0] * 99 + [1] * 30])
    settings = _Settings(
        scripted,
        np.ones(1),
        initial_samples=8,
        acceleration=3.0,
        max_samples=30,
    )
    draw(settings, 1, SEED)
    assert [r.size for r in scripted.requests] == [8, 24, 30, 30, 30]


def test_level_stays_finite_under_a_large_acceleration():
    # The level is bounded by max_samples after every call: multiplied by
    # 1e200 without the bound, it would reach inf at the third call, where
    # math.floor raises.
    scripted = ScriptedSimulator([LATE_MATCH])
    settings = _Settings(
        scripted,
        np.ones(1),
        initial_samples=1,
        acceleration=1e200,
        max_samples=7,
    )
    b = draw(settings, 1, SEED)
    assert [r.size for r in scripted.requests] == [1] + [7] * 6
    assert b.values[0] == ibs_loglik(41)


# ---------------------------------------------------------------------------
# The one-sample schedule and the first round


def test_unvectorized_requests_one_sample_per_open_trial(captured_counts):
    # vectorized=False ignores initial_samples and acceleration.
    sim = ScriptedSimulator(STREAMS)
    settings = _Settings(sim, np.ones(3), initial_samples=5, acceleration=2.0)
    b = _sampler.sample(
        settings, np.zeros(1), 2, np.random.default_rng(SEED), vectorized=False
    )
    (K,) = captured_counts
    assert np.array_equal(K, STREAM_K)
    assert [r.size for r in sim.requests] == [3, 3, 2, 2, 2, 2, 2, 1]
    assert b.samples == K.sum()


def test_first_round_is_taken_on_one_sample_per_trial(captured_counts):
    # The call of first_round is the draw's first round: one call less, the
    # same counts, and the cost counts it.
    sim = ScriptedSimulator(STREAMS)
    settings = _Settings(sim, np.ones(3))
    rng = np.random.default_rng(SEED)
    first = _sampler.first_round(settings, np.zeros(1), rng)
    assert first.hits.shape == (3, 1)
    assert np.array_equal(first.hits[:, 0], [True, False, False])
    b = _sampler.sample(
        settings, np.zeros(1), 2, rng, vectorized=False, first=first
    )
    (K,) = captured_counts
    assert np.array_equal(K, STREAM_K)
    assert [r.size for r in sim.requests] == [3, 3, 2, 2, 2, 2, 2, 1]
    assert b.calls == 8
    assert b.samples == K.sum()


def test_first_round_precedes_more_samples_per_trial(captured_counts):
    # The schedule's first round requests n = 2 samples of each trial, so
    # the call of first_round is an extra round before it, after which the
    # level does not grow: the next round requests 2 samples of each trial,
    # not round(2 * 1.5) = 3. Every sample matches.
    sim = ScriptedSimulator([[1] * 20] * 3)
    settings = _Settings(sim, np.ones(3))
    rng = np.random.default_rng(SEED)
    first = _sampler.first_round(settings, np.zeros(1), rng)
    b = _sampler.sample(settings, np.zeros(1), 2, rng, first=first)
    (K,) = captured_counts
    assert np.array_equal(K, np.ones((2, 3)))
    assert [r.size for r in sim.requests] == [3, 6]
    assert b.calls == 2
    assert b.samples == 9


@pytest.mark.parametrize("vectorized", [False, True])
def test_first_round_reproduces_the_draw_without_it(vectorized):
    # A first round drawn from the draw's generator gives the draw that the
    # same generator gives without it, when the schedule's first round
    # requests one sample of every trial (initial_samples=1).
    settings = _Settings(
        bernoulli, np.ones(P.size), trial_weights=W, initial_samples=1
    )
    plain = _sampler.sample(
        settings,
        np.zeros(1),
        5,
        np.random.default_rng(SEED),
        vectorized=vectorized,
    )
    rng = np.random.default_rng(SEED)
    first = _sampler.first_round(settings, np.zeros(1), rng)
    taken = _sampler.sample(
        settings, np.zeros(1), 5, rng, vectorized=vectorized, first=first
    )
    assert np.array_equal(taken.K, plain.K)
    assert np.array_equal(taken.values, plain.values)
    assert cost(taken) == cost(plain)


def test_first_round_checks_the_output():
    settings = _Settings(lambda t, idx, rng: np.ones(len(idx) + 1), np.ones(3))
    with pytest.raises(ValueError, match="simulator"):
        _sampler.first_round(settings, np.zeros(1), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Time limit


def clocked(streams, clock, seconds_per_call=1.0):
    """A scripted simulator that advances the clock at every call."""
    scripted = ScriptedSimulator(streams)

    def simulator(theta, design_rows, rng):
        clock.now += seconds_per_call
        return scripted(theta, design_rows, rng)

    simulator.requests = scripted.requests
    return simulator


def timed_draw(monkeypatch, streams, n, max_time, start=0.0, **kwargs):
    """A one-sample-per-call draw of scripted streams; each call lasts 1 s."""
    clock = FakeTime()
    monkeypatch.setattr(_sampler, "time", clock)
    sim = clocked(streams, clock)
    settings = _Settings(
        sim, np.ones(len(streams)), max_time=max_time, **kwargs
    )
    b = _sampler.sample(
        settings,
        np.zeros(1),
        n,
        np.random.default_rng(SEED),
        vectorized=False,
        start=start,
    )
    return b, sim


# After three calls, trial 0 has completed its three counts (1, 1, 1),
# trial 1 one count (1), and trial 2 two counts (2, 1).
TIMED_STREAMS = [[1] * 10, [1, 0, 0, 0] + [1] * 6, [0, 1, 1] + [1] * 7]
TIMED_W = np.array([0.5, 1.0, 2.0])


def test_time_limit_averages_the_completed_counts(monkeypatch):
    # The time is checked after every call: the fourth is not made, since
    # 3 s > 2.5 s have passed after the third.
    b, sim = timed_draw(
        monkeypatch, TIMED_STREAMS, 3, 2.5, trial_weights=TIMED_W
    )
    assert b.timed_out
    assert len(sim.requests) == 3
    assert b.calls == 3
    assert np.array_equal(b.trial_counts, [3, 1, 2])
    # Each trial's value averages its completed counts.
    a = np.array([0.0, 0.0, (ibs_loglik(2) + ibs_loglik(1)) / 2])
    S = np.array([0.0, 0.0, ibs_var(2) + ibs_var(1)])
    assert_allclose(b.loglik, TIMED_W @ a, **EXACT)
    assert_allclose(b.loglik_var, TIMED_W**2 @ (S / [9, 1, 4]), **EXACT)
    assert_allclose(b.trial_value_sums, a * [3, 1, 2], **EXACT)
    assert_allclose(b.trial_var_sums, S, **EXACT)
    # Repeat 0 is complete; the others have no value.
    assert_allclose(b.values[0], TIMED_W[2] * ibs_loglik(2), **EXACT)
    assert np.all(np.isnan(b.values[1:]))
    assert np.all(np.isnan(b.var_estimates[1:]))
    assert b.n_thresholded == 0


def test_time_limit_raises_for_a_trial_without_a_count(monkeypatch):
    # After one call, trial 2 has no completed count.
    with pytest.raises(IBSSamplingError) as e:
        timed_draw(monkeypatch, TIMED_STREAMS, 3, 0.5)
    message = str(e.value)
    assert "max_time = 0.5 s" in message
    assert "no completed count in the n = 3 repeats for trial 2." in message


def test_time_limit_follows_the_first_call(monkeypatch):
    # A draw makes its first call even when the time has run out before it.
    b, sim = timed_draw(monkeypatch, [[1] * 5, [1] * 5], 3, 1.0, start=-10.0)
    assert b.timed_out
    assert len(sim.requests) == 1
    assert np.array_equal(b.trial_counts, [1, 1])
    assert b.loglik == 0.0


def test_time_limit_does_not_flag_a_complete_draw(monkeypatch):
    # The third call completes the draw after 3 s > 2.5 s: nothing is left
    # to stop.
    b, sim = timed_draw(monkeypatch, [[1] * 5], 3, 2.5)
    assert not b.timed_out
    assert len(sim.requests) == 3
    assert np.array_equal(b.trial_counts, [3])


def test_time_limit_under_the_threshold(monkeypatch):
    # Two trials of unit weight, T = 1.7. Trial 0 misses three times: the
    # bound h(4) = 1.83 of repeat 0 exceeds T after call 3, which ends it,
    # and trial 0 matches at once in repeat 1 at call 4. Trial 1 completes
    # 1 in repeat 0, 2 in repeat 1 (call 3) and 1 in repeat 2 (call 4).
    # The time runs out after call 4, with trial 0 open in repeat 2.
    T = 1.7
    b, sim = timed_draw(
        monkeypatch,
        [[0, 0, 0, 1] + [1] * 5, [1, 0, 1, 1] + [1] * 5],
        3,
        3.5,
        neg_loglik_threshold=T,
    )
    assert b.timed_out
    assert len(sim.requests) == 4
    assert np.array_equal(b.ended, [True, False, False])
    assert b.n_thresholded == 1
    assert np.array_equal(b.trial_counts, [1, 2])
    # (n_e / n)(-T) + (1 - n_e / n) sum_i w_i a_i, with a = (0, -1/2).
    assert_allclose(b.loglik, -T / 3 + 2 / 3 * (-0.5), **EXACT)
    # The ended repeat's variance estimate, ibs_var(4), over n**2, and
    # (2/3)**2 times trial 1's ibs_var(2) / 2**2.
    assert_allclose(
        b.loglik_var, ibs_var(4) / 9 + 4 / 9 * ibs_var(2) / 4, **EXACT
    )
    assert_allclose(b.values[:2], [-T, ibs_loglik(2)], **EXACT)
    assert np.isnan(b.values[2])


def limited(K, completed, ended, ended_var, w, T=None):
    return _sampler._limited_estimates(
        np.array(K),
        np.array(completed, bool),
        np.array(ended, bool),
        np.array(ended_var, float),
        np.array(w, float),
        T,
        1.0,
    )


def test_limited_estimates_of_a_complete_draw_are_the_mean():
    # With every count completed and no repeat ended, the reduction is the
    # mean of the repeat values and their variance estimates over n**2.
    b = draw(_Settings(bernoulli, np.ones(P.size), trial_weights=W), 7, SEED)
    loglik, var, tv, ts, m = limited(
        b.K, np.ones(b.K.shape), np.zeros(7), np.zeros(7), W
    )
    assert_allclose(loglik, b.values.mean(), rtol=1e-13)
    assert_allclose(var, b.var_estimates.sum() / 49, rtol=1e-13)
    assert_allclose(tv, b.trial_value_sums, **EXACT)
    assert_allclose(ts, b.trial_var_sums, **EXACT)
    assert np.array_equal(m, np.full(P.size, 7))


def test_limited_estimates_limits():
    K = [[3, 1], [0, 2], [5, 0]]
    completed = [[True, True], [False, True], [True, False]]
    w = [0.5, 2.0]
    # No ended repeat: the weighted sum of the trials' averages.
    loglik, var, _, _, m = limited(K, completed, [0, 0, 0], [0, 0, 0], w)
    assert np.array_equal(m, [2, 2])
    a = [(ibs_loglik(3) + ibs_loglik(5)) / 2, ibs_loglik(2) / 2]
    S = [ibs_var(3) + ibs_var(5), ibs_var(2)]
    assert_allclose(loglik, np.dot(w, a), **EXACT)
    assert_allclose(var, np.dot(np.square(w), np.divide(S, 4)), **EXACT)
    # Every repeat ended: -T, whatever the counts.
    T = 4.0
    loglik, var, _, _, _ = limited(
        K, np.zeros((3, 2)), [1, 1, 1], [1.0, 2.0, 3.0], w, T
    )
    assert loglik == -T
    assert_allclose(var, 6.0 / 9, **EXACT)
    # Repeat 2 ended: trial 0 keeps one count and trial 1 two.
    loglik, var, _, _, m = limited(K, completed, [0, 0, 1], [0, 0, 3.0], w, T)
    assert np.array_equal(m, [1, 2])
    a = [ibs_loglik(3), (ibs_loglik(1) + ibs_loglik(2)) / 2]
    S = [ibs_var(3), ibs_var(1) + ibs_var(2)]
    assert_allclose(loglik, -T / 3 + 2 / 3 * np.dot(w, a), **EXACT)
    assert_allclose(
        var,
        3.0 / 9 + 4 / 9 * np.dot(np.square(w), np.divide(S, [1, 4])),
        **EXACT,
    )


def test_limited_estimates_raise_for_a_trial_without_a_count():
    with pytest.raises(IBSSamplingError, match="for trial 1"):
        limited([[3, 0], [0, 0]], [[1, 0], [0, 0]], [0, 0], [0, 0], [1, 1])
    # A count in an ended repeat does not count.
    with pytest.raises(IBSSamplingError, match="for trial 1"):
        limited(
            [[3, 2], [4, 0]], [[1, 1], [1, 0]], [1, 0], [1.0, 0], [1, 1], 2.0
        )


# ---------------------------------------------------------------------------
# Cost


@pytest.mark.parametrize(
    "initial, acceleration", [(None, 1.5), (4, 1.0), (1, 1.0)]
)
def test_cost(captured_counts, initial, acceleration):
    counting = Counting(bernoulli)
    settings = _Settings(
        counting,
        np.ones(P.size),
        initial_samples=initial,
        acceleration=acceleration,
    )
    b = draw(settings, 50, SEED)
    (K,) = captured_counts
    assert b.calls == counting.calls
    assert b.n == 50
    assert b.seconds > 0
    if initial == 1 and acceleration == 1:
        assert b.samples == K.sum()
    else:
        assert b.samples > K.sum()


# ---------------------------------------------------------------------------
# Errors


def test_max_samples_per_trial_raises():
    counting = Counting(never_matches)
    settings = _Settings(
        counting,
        np.ones(2),
        initial_samples=1,
        acceleration=1,
        max_samples_per_trial=10,
    )
    assert settings.max_samples_per_trial == 10
    assert issubclass(IBSSamplingError, RuntimeError)
    # A draw of 3 repeats allows 30 samples per trial: the call that
    # brings both trials to 31 raises and names them.
    with pytest.raises(IBSSamplingError, match="max_samples_per_trial") as e:
        draw(settings, 3, SEED)
    assert counting.calls == 31
    message = str(e.value)
    assert "In a draw of 3 repeats, 2 trials drew more than" in message
    assert "= 30 samples: trial 0 drew 31, trial 1 drew 31." in message
    assert "2 of 2 trials still need matches" in message


@pytest.mark.parametrize(
    "responses, expected",
    [
        # Trials 1 and 2 never match.
        (
            [1.0, 1.0, 1.0],
            [
                "In a draw of 1 repeat, 2 trials drew more than "
                "max_samples_per_trial * n = 5 samples: trial 1 drew 6, "
                "trial 2 drew 6.",
                "2 of 3 trials still need matches",
            ],
        ),
        # Only trial 1 never matches.
        (
            [1.0, 1.0, 0.0],
            [
                "In a draw of 1 repeat, trial 1 drew 6 samples, more than "
                "max_samples_per_trial * n = 5.",
                "1 of 3 trials still need matches",
            ],
        ),
    ],
)
def test_max_samples_per_trial_names_the_trials_over_it(responses, expected):
    # The simulator returns 1 for trial 0 and 0 for the others; trial 0
    # is done after its first sample.
    settings = _Settings(
        lambda t, idx, rng: (idx == 0).astype(float),
        np.array(responses),
        initial_samples=1,
        acceleration=1,
        max_samples_per_trial=5,
    )
    with pytest.raises(IBSSamplingError) as e:
        draw(settings, 1, SEED)
    for text in expected:
        assert text in str(e.value)


def test_max_samples_per_trial_names_five_trials_and_counts_the_rest():
    settings = _Settings(
        never_matches,
        np.ones(8),
        initial_samples=1,
        acceleration=1,
        max_samples_per_trial=2,
    )
    with pytest.raises(IBSSamplingError) as e:
        draw(settings, 1, SEED)
    assert (
        "8 trials drew more than max_samples_per_trial * n = 2 samples: "
        "trial 0 drew 3, trial 1 drew 3, trial 2 drew 3, trial 3 drew 3, "
        "trial 4 drew 3, and 3 more." in str(e.value)
    )
    assert type(e.value).__module__ == "pyibs"


def test_max_samples_per_trial_counts_the_surplus():
    # One call of 5 samples completes a draw of one repeat with 4 surplus
    # samples, more than the cap of 4: the draw raises, and the message
    # does not claim that trials are still sampling.
    counting = Counting(lambda t, idx, rng: np.ones(len(idx)))
    settings = _Settings(
        counting,
        np.ones(1),
        initial_samples=5,
        acceleration=1,
        max_samples_per_trial=4,
    )
    with pytest.raises(IBSSamplingError) as e:
        draw(settings, 1, SEED)
    assert counting.calls == 1
    message = str(e.value)
    assert "trial 0 drew 5 samples" in message
    assert "completed the draw" in message
    assert "still need" not in message


def test_max_samples_per_trial_scales_with_n():
    # The first match of the trial is its 30th sample, then every sample
    # matches. With a cap of 10 per repeat and one sample per call, a draw
    # of 3 repeats needs 32 samples (> 30) and raises, while a draw of 4
    # repeats needs 33 (<= 40).
    def settings():
        return _Settings(
            ScriptedSimulator([[0] * 29 + [1] * 30]),
            np.ones(1),
            initial_samples=1,
            acceleration=1,
            max_samples_per_trial=10,
        )

    with pytest.raises(IBSSamplingError, match="drew 31 samples"):
        draw(settings(), 3, SEED)
    assert draw(settings(), 4, SEED).samples == 33


def test_max_samples_per_trial_none_disables_the_cap():
    scripted = ScriptedSimulator([[0] * 999 + [1]])
    settings = _Settings(
        scripted,
        np.ones(1),
        initial_samples=1,
        acceleration=1,
        max_samples_per_trial=None,
    )
    b = draw(settings, 1, SEED)
    assert b.samples == 1000
    assert b.calls == 1000


def test_default_max_samples_per_trial():
    settings = _Settings(bernoulli, np.ones(P.size))
    assert settings.max_samples_per_trial == 10**5


@pytest.mark.parametrize(
    "responses, output",
    [
        (np.array(["a", "b"]), lambda n: np.zeros(n)),
        (np.ones(2), lambda n: np.full(n, "1")),
        (np.array([b"a", b"b"]), lambda n: np.full(n, "a")),
        (np.array([True, False]), lambda n: np.full(n, b"1")),
        (np.array([["a", "b"], ["c", "d"]]), lambda n: np.zeros((n, 2))),
    ],
)
def test_responses_of_another_kind_raise(responses, output):
    # Such arrays compare as unequal elementwise, so without the check the
    # draw would sample until the cap.
    counting = Counting(lambda t, idx, rng: output(len(idx)))
    settings = _Settings(counting, responses)
    with pytest.raises(TypeError, match="dtype"):
        draw(settings, 2, SEED)
    assert counting.calls == 1


@pytest.mark.parametrize(
    "responses, output",
    [
        # Booleans and numbers compare by value.
        (np.array([True, True]), lambda n: np.ones(n)),
        (np.ones(2, dtype=int), lambda n: np.ones(n, dtype=bool)),
        # Object arrays compare element by element.
        (np.array(["a", "a"], dtype=object), lambda n: np.full(n, "a")),
        (np.array(["a", "a"]), lambda n: np.full(n, "a", dtype=object)),
        (np.ones(2), lambda n: np.full(n, 1, dtype=object)),
        # Text of different lengths is one kind.
        (np.array(["ab", "ab"]), lambda n: np.full(n, "ab", dtype="<U5")),
    ],
)
def test_responses_of_a_comparable_kind_match(responses, output):
    settings = _Settings(lambda t, idx, rng: output(len(idx)), responses)
    b = draw(settings, 3, SEED)
    assert np.all(b.values == 0.0)
    assert b.samples == 3 * 2


@pytest.mark.parametrize(
    "responses, output",
    [
        (np.ones(3), lambda n: np.ones(n - 1)),
        (np.ones(3), lambda n: np.ones(n + 1)),
        (np.ones(3), lambda n: np.ones((n, 2))),
        (np.ones(3), lambda n: np.ones((n, 1, 1))),
        (np.ones((3, 1)), lambda n: np.ones((n + 1, 1))),
        (np.ones((3, 1)), lambda n: np.ones((n, 2))),
        (np.ones((3, 2)), lambda n: np.ones(n)),
        (np.ones((3, 2)), lambda n: np.ones((n, 1))),
        (np.ones((3, 2)), lambda n: np.ones((n, 3))),
        (np.ones(3), lambda n: 1.0),
    ],
)
def test_wrong_simulator_output_raises(responses, output):
    settings = _Settings(lambda t, idx, rng: output(len(idx)), responses)
    with pytest.raises(ValueError, match="simulator"):
        draw(settings, 2, SEED)


@pytest.mark.parametrize("responses_shape", [(3,), (3, 1)])
@pytest.mark.parametrize("output_column", [False, True])
def test_one_column_responses_take_both_output_shapes(
    captured_counts, responses_shape, output_column
):
    # Responses of shape (N,) or (N, 1) take an output of shape (r,) or
    # (r, 1), compared with their one column.
    scripted = ScriptedSimulator(STREAMS)

    def simulator(theta, design_rows, rng):
        out = scripted(theta, design_rows, rng)
        return out[:, None] if output_column else out

    settings = _Settings(
        simulator, np.ones(responses_shape), initial_samples=1, acceleration=1
    )
    draw(settings, 2, SEED)
    (K,) = captured_counts
    assert np.array_equal(K, STREAM_K)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, -1.0, 1.0],
        [1.0, np.nan, 1.0],
        np.inf,
        [1.0, 1.0],
        np.ones((3, 1)),
    ],
)
def test_invalid_weights_raise(weights):
    with pytest.raises(ValueError):
        _Settings(bernoulli, np.ones(3), trial_weights=weights)


def test_scalar_weight_scales_values():
    plain = draw(_Settings(bernoulli, np.ones(P.size)), 10, SEED)
    scaled = draw(
        _Settings(bernoulli, np.ones(P.size), trial_weights=2.0),
        10,
        SEED,
    )
    assert_allclose(scaled.values, 2 * plain.values, **EXACT)
    assert_allclose(scaled.var_estimates, 4 * plain.var_estimates, **EXACT)
    assert_allclose(scaled.trial_value_sums, plain.trial_value_sums, **EXACT)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(responses=np.ones((2, 2, 2))),
        dict(responses=np.ones(0)),
        dict(responses=np.ones((3, 0))),
        dict(responses=1.0),
        dict(design=np.ones(2)),
        dict(design=1.0),
        dict(initial_samples=0),
        dict(initial_samples=1.5),
        dict(initial_samples=math.inf),
        dict(acceleration=0.5),
        dict(acceleration=math.inf),
        dict(acceleration=math.nan),
        dict(acceleration_threshold=0.0),
        dict(acceleration_threshold=-1.0),
        dict(acceleration_threshold=math.nan),
        dict(max_samples=0),
        dict(max_samples=2.5),
        dict(max_mem=0),
        dict(max_mem=-10),
        dict(max_samples_per_trial=0),
        dict(max_samples_per_trial=2.5),
        dict(max_time=0.0),
        dict(max_time=-1.0),
        dict(max_time=math.nan),
    ],
)
def test_invalid_settings_raise(kwargs):
    kwargs = {"responses": np.ones(3), **kwargs}
    with pytest.raises(ValueError):
        _Settings(bernoulli, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(initial_samples=True),
        dict(initial_samples="3"),
        dict(acceleration="fast"),
        dict(acceleration=np.bool_(True)),
        dict(acceleration_threshold="0.1"),
        dict(max_samples=np.array([5])),
        dict(max_mem=False),
        dict(max_samples_per_trial="10"),
        dict(max_time="1"),
    ],
)
def test_settings_of_the_wrong_type_raise(kwargs):
    with pytest.raises(TypeError):
        _Settings(bernoulli, np.ones(3), **kwargs)


def test_whole_number_floats_are_counts():
    settings = _Settings(
        bernoulli,
        np.ones(3),
        initial_samples=3.0,
        max_samples=1e4,
        max_mem=np.float64(100),
        max_samples_per_trial=1e5,
    )
    for name, value in [
        ("initial_samples", 3),
        ("max_samples", 10**4),
        ("max_mem", 100),
        ("max_samples_per_trial", 10**5),
    ]:
        assert getattr(settings, name) == value
        assert type(getattr(settings, name)) is int


def test_simulator_must_be_callable():
    with pytest.raises(TypeError):
        _Settings(None, np.ones(3))


def test_sample_rejects_bad_n():
    settings = _Settings(bernoulli, np.ones(3))
    rng = np.random.default_rng(0)
    for n in (0, -1, 1.5):
        with pytest.raises(ValueError):
            _sampler.sample(settings, np.zeros(1), n, rng)
    with pytest.raises(TypeError):
        _sampler.sample(settings, np.zeros(1), True, rng)
    assert _sampler.sample(settings, np.zeros(1), 2.0, rng).n == 2


def test_inputs_are_copied():
    responses = np.ones(P.size)
    design = np.arange(P.size)
    settings = _Settings(
        lambda t, rows, rng: bernoulli(t, rows, rng), responses, design
    )
    responses[:] = 0
    design[:] = 0
    assert np.all(settings.responses == 1)
    assert np.array_equal(settings.design, np.arange(P.size))
    assert draw(settings, 5, SEED).n == 5
    assert not settings.responses.flags.writeable
    assert not settings.design.flags.writeable
    assert not settings.trial_weights.flags.writeable

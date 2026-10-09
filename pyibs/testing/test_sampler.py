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
        # Two, three, then four samples per trial per call (floor(4.5)).
        (2, 1.5, [6, 6, 8]),
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


def reference_sampler(streams, n, initial, acceleration, cap):
    """Rows-first IBS sampling, one trial and one sample at a time."""
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
        m = max(1, min(math.floor(level), cap // len(open_trials)))
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


@pytest.mark.parametrize(
    "initial, acceleration, cap",
    [
        (1, 1, 10**6),
        (3, 1, 10**6),
        (2, 1.5, 10**6),
        (None, 1.5, 10**6),
        (5, 2.0, 13),
        (1, 1.5, 4),
    ],
)
def test_counts_match_reference_sampler(
    captured_counts, initial, acceleration, cap
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
        max_samples_per_call=cap,
    )
    b = draw(settings, n, SEED)
    K_ref, sizes_ref = reference_sampler(
        streams, n, n if initial is None else initial, acceleration, cap
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


def test_level_is_bounded_by_max_samples_per_call():
    # The request of one open trial is capped at max_samples_per_call.
    scripted = ScriptedSimulator([[0] * 99 + [1] * 30])
    settings = _Settings(
        scripted,
        np.ones(1),
        initial_samples=8,
        acceleration=3.0,
        max_samples_per_call=30,
    )
    draw(settings, 1, SEED)
    assert [r.size for r in scripted.requests] == [8, 24, 30, 30, 30]


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
        (np.ones(3), lambda n: np.ones((n, 1))),
        (np.ones((3, 2)), lambda n: np.ones(n)),
        (np.ones((3, 2)), lambda n: np.ones((n, 3))),
        (np.ones(3), lambda n: 1.0),
    ],
)
def test_wrong_simulator_output_raises(responses, output):
    settings = _Settings(lambda t, idx, rng: output(len(idx)), responses)
    with pytest.raises(ValueError, match="simulator"):
        draw(settings, 2, SEED)


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
        dict(initial_samples=True),
        dict(acceleration=0.5),
        dict(acceleration=math.inf),
        dict(acceleration=math.nan),
        dict(acceleration="fast"),
        dict(acceleration_threshold=0.0),
        dict(acceleration_threshold=-1.0),
        dict(acceleration_threshold=math.nan),
        dict(max_samples_per_call=0),
        dict(max_samples_per_call=10.0),
        dict(max_samples_per_trial=0),
        dict(max_samples_per_trial=2.5),
    ],
)
def test_invalid_settings_raise(kwargs):
    kwargs = {"responses": np.ones(3), **kwargs}
    with pytest.raises(ValueError):
        _Settings(bernoulli, **kwargs)


def test_simulator_must_be_callable():
    with pytest.raises(TypeError):
        _Settings(None, np.ones(3))


def test_sample_rejects_bad_n():
    settings = _Settings(bernoulli, np.ones(3))
    for n in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            _sampler.sample(settings, np.zeros(1), n, np.random.default_rng(0))


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

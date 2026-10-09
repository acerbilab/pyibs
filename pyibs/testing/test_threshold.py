import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyibs import _sampler
from pyibs._estimates import ibs_loglik, ibs_var
from pyibs._sampler import IBSSamplingError, _Settings
from pyibs.testing._exact import exact_draw
from pyibs.testing._helpers import (
    P,
    ScriptedSimulator,
    bernoulli,
    cost,
    draw,
    never_matches,
    summary,
)

SEED = 20260930
N_REPEATS = 20_000
W = np.linspace(0, 2, 20)
EXACT = dict(rtol=1e-12, atol=1e-12)


def h(k):
    """A trial's term in the bound, ``digamma(k) - digamma(1)``."""
    return -ibs_loglik(k)


@pytest.fixture
def checks(monkeypatch):
    """Record the bounds and the ended repeats at every per-round check."""
    recorded = []
    original = _sampler._MatchCounts.end_above

    def spy(self, threshold):
        bounds = self.bounds()
        rows = original(self, threshold)
        recorded.append((bounds, rows))
        return rows

    monkeypatch.setattr(_sampler._MatchCounts, "end_above", spy)
    return recorded


# ---------------------------------------------------------------------------
# Scripted streams


def test_scripted_bound_ends_and_skips_a_repeat(checks):
    # Three repeats of two trials with weights 0.5 and 1, one sample per
    # trial per round, T = 1.7. Trial 0 needs 5 samples in repeat 0. Trial 1
    # matches at once in repeat 0, then misses in repeat 1 until the bound
    # H_3 = 1.83 exceeds T in round 4. Trial 1 drops its open count of 3
    # and matches at its first sample of repeat 2; trial 0 closes repeat 0
    # in round 5, skips repeat 1 and matches at once in repeat 2.
    streams = [[0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1]]
    sim = ScriptedSimulator(streams)
    w = np.array([0.5, 1.0])
    T = 1.7
    settings = _Settings(
        sim,
        np.ones(2),
        trial_weights=w,
        initial_samples=1,
        acceleration=1,
        neg_loglik_threshold=T,
    )
    b = draw(settings, 3, SEED)
    # Bound of each repeat after each round, before its check.
    expected = [
        [w[0] * h(2), 0.0, 0.0],
        [w[0] * h(3), h(2), 0.0],
        [w[0] * h(4), h(3), 0.0],
        [w[0] * h(5), h(4), 0.0],
        [w[0] * h(5), None, 0.0],
        [w[0] * h(5), None, 0.0],
    ]
    assert len(checks) == len(expected)
    for (bounds, _), row in zip(checks, expected):
        live = [r for r, x in enumerate(row) if x is not None]
        assert_allclose(bounds[live], [row[r] for r in live], **EXACT)
    # Bounds never decrease while a repeat is not ended.
    for r in (0, 2):
        series = [bounds[r] for bounds, _ in checks]
        assert np.all(np.diff(series) >= 0)
    assert h(3) <= T < h(4)
    assert [rows.tolist() for _, rows in checks] == [[], [], [], [1], [], []]
    # One sample per open trial per round; in round 6 only trial 0 samples.
    assert [r.tolist() for r in sim.requests] == [[0, 1]] * 5 + [[0]]
    assert b.calls == 6
    assert b.samples == 11
    assert b.n == 3
    # Repeat 1 takes -T and the variance estimate of its partial counts:
    # trial 1 open with c = 3, trial 0 not reached. Repeat 2's counts are
    # both 1: trial 1's open count was dropped.
    assert_allclose(b.values, [-w[0] * h(5), -T, 0.0], **EXACT)
    assert_allclose(
        b.var_estimates, [w[0] ** 2 * ibs_var(5), ibs_var(4), 0.0], **EXACT
    )
    assert b.values[1] == -T
    assert b.n_thresholded == 1
    assert np.array_equal(b.ended, [False, True, False])
    # The per-trial sums cover repeats 0 and 2.
    assert summary(b).trial_repeats == 2
    assert_allclose(b.trial_value_sums, [ibs_loglik(5), 0.0], **EXACT)
    assert_allclose(b.trial_var_sums, [ibs_var(5), 0.0], **EXACT)


def test_match_counts_skip_an_ended_repeat_within_a_round():
    # Four repeats, weights 0.5 and 1, T = 1.6, four samples per round.
    w = np.array([0.5, 1.0])
    T = 1.6
    counts = _sampler._MatchCounts(4, 2, w)
    trials = counts.open_trials()
    counts.absorb(trials, np.array([[0, 0, 0, 0], [1, 0, 0, 0]], bool))
    assert np.array_equal(counts.repeat, [0, 1])
    assert np.array_equal(counts.open_count, [4, 3])
    assert_allclose(counts.bounds(), [w[0] * h(5), h(4), 0, 0], **EXACT)
    # Repeat 1 ends; trial 0 has not reached it, trial 1 moves on to 2.
    assert np.array_equal(counts.end_above(T), [1])
    assert np.array_equal(counts.ended, [False, True, False, False])
    assert_allclose(counts.ended_var, [0, ibs_var(4), 0, 0], **EXACT)
    assert np.array_equal(counts.repeat, [0, 2])
    assert np.array_equal(counts.open_count, [4, 0])
    # Trial 0's first match closes repeat 0 (count 4 + 2), its next two
    # skip repeat 1 and close repeats 2 and 3. Trial 1 closes 2 and 3.
    trials = counts.open_trials()
    counts.absorb(trials, np.array([[0, 1, 1, 1], [0, 1, 0, 1]], bool))
    assert np.array_equal(counts.K, [[6, 1], [0, 0], [1, 2], [1, 2]])
    assert np.array_equal(counts.completed, counts.K > 0)
    assert np.array_equal(counts.repeat, [4, 4])
    assert np.array_equal(counts.open_count, [0, 0])
    assert counts.end_above(T).size == 0
    values, var, tv, ts = counts.clipped_estimates(T)
    assert_allclose(values, [-w[0] * h(6), -T, -h(2), -h(2)], **EXACT)
    assert_allclose(
        var,
        [w[0] ** 2 * ibs_var(6), ibs_var(4), ibs_var(2), ibs_var(2)],
        **EXACT,
    )
    assert_allclose(tv, [ibs_loglik(6), 2 * ibs_loglik(2)], **EXACT)
    assert_allclose(ts, [ibs_var(6), 2 * ibs_var(2)], **EXACT)


def test_complete_repeat_below_threshold_is_ended():
    # One round of 10 samples completes both repeats: counts 5 and 1. The
    # first's complete bound h(5) = 2.08 exceeds T, so it is ended, with
    # the variance estimate of its complete count.
    sim = ScriptedSimulator([[0, 0, 0, 0, 1] + [1] * 5])
    T = 1.7
    settings = _Settings(
        sim,
        np.ones(1),
        initial_samples=10,
        acceleration=1,
        neg_loglik_threshold=T,
    )
    b = draw(settings, 2, SEED)
    assert b.calls == 1
    assert b.samples == 10
    assert_allclose(b.values, [-T, 0.0], **EXACT)
    assert_allclose(b.var_estimates, [ibs_var(5), 0.0], **EXACT)
    assert b.n_thresholded == 1
    assert summary(b).trial_repeats == 1
    assert np.array_equal(b.trial_value_sums, [0.0])
    assert np.array_equal(b.trial_var_sums, [0.0])


@pytest.mark.parametrize("samples_per_round", [1, 10])
def test_threshold_is_strict(checks, samples_per_round):
    # T equals the bound h(5) of a count of 5, reached by an open count of
    # 4 or by a complete count of 5: a repeat at the bound is not ended.
    T = float(h(5))
    sim = ScriptedSimulator([[0, 0, 0, 0, 1] + [1] * 5])
    settings = _Settings(
        sim,
        np.ones(1),
        initial_samples=samples_per_round,
        acceleration=1,
        neg_loglik_threshold=T,
    )
    b = draw(settings, 1, SEED)
    assert b.n_thresholded == 0
    assert b.values[0] == -T
    assert b.values[0] == ibs_loglik(5)
    assert all(rows.size == 0 for _, rows in checks)


def test_open_count_one_above_ends_the_repeat(checks):
    # The same stream with T just below h(5): one sample at a time, the
    # bound h(c + 1) first exceeds T after the fourth miss.
    T = float(np.nextafter(h(5), 0))
    sim = ScriptedSimulator([[0, 0, 0, 0, 1] + [1] * 5])
    settings = _Settings(
        sim,
        np.ones(1),
        initial_samples=1,
        acceleration=1,
        neg_loglik_threshold=T,
    )
    b = draw(settings, 1, SEED)
    assert [rows.tolist() for _, rows in checks] == [[], [], [], [0]]
    assert b.samples == 4
    assert b.values[0] == -T
    assert_allclose(b.var_estimates, [ibs_var(5)], **EXACT)


def test_threshold_ends_an_impossible_response():
    # A response the simulator never produces: every repeat ends once the
    # bound h(c + 1) exceeds T (after 3 misses for T = 1.7).
    T = 1.7
    settings = _Settings(
        never_matches,
        np.ones(1),
        initial_samples=1,
        acceleration=1,
        neg_loglik_threshold=T,
    )
    b = draw(settings, 3, SEED)
    assert b.n_thresholded == 3
    assert np.all(b.values == -T)
    assert_allclose(b.var_estimates, np.full(3, ibs_var(4)), **EXACT)
    assert b.samples == 9
    assert summary(b).trial_repeats == 0
    assert np.array_equal(b.trial_value_sums, [0.0])
    assert summary(b).trial_loglik is None
    # A trial of weight 0 adds nothing to the bound, so the cap on its
    # samples applies: the call that brings it to 51 samples raises.
    zero_weight = _Settings(
        never_matches,
        np.ones(1),
        trial_weights=0.0,
        initial_samples=1,
        acceleration=1,
        max_samples_per_trial=50,
        neg_loglik_threshold=T,
    )
    with pytest.raises(IBSSamplingError, match="trial 0 drew 51 samples"):
        draw(zero_weight, 1, SEED)


# ---------------------------------------------------------------------------
# Reference implementation, one sample and one repeat at a time


def reference_sampler(streams, n, initial, acceleration, cap, w, T):
    """Rows-first IBS with the threshold rule, written as plain loops.

    After every round, every repeat that is not ended, complete or not, is
    checked, and the trials sampling an ended repeat move to the next one
    that is not ended.
    """
    n_trials = len(streams)
    pos = [0] * n_trials
    K = [[0] * n for _ in range(n_trials)]
    current = [0] * n_trials
    open_count = [0] * n_trials
    ended = [False] * n
    ended_var = [0.0] * n

    def next_repeat(r):
        while r < n and ended[r]:
            r += 1
        return r

    level = initial
    sizes = []
    while True:
        open_trials = [i for i in range(n_trials) if current[i] < n]
        if not open_trials:
            break
        m = max(1, min(math.floor(level), cap // len(open_trials)))
        sizes.append(len(open_trials) * m)
        for i in open_trials:
            for _ in range(m):
                hit = streams[i][pos[i]] == 1
                pos[i] += 1
                if current[i] < n:
                    open_count[i] += 1
                    if hit:
                        K[i][current[i]] = open_count[i]
                        open_count[i] = 0
                        current[i] = next_repeat(current[i] + 1)
        for r in range(n):
            if ended[r]:
                continue
            k = [
                (
                    K[i][r]
                    if K[i][r] > 0
                    else open_count[i] + 1
                    if current[i] == r
                    else 1
                )
                for i in range(n_trials)
            ]
            if sum(w[i] * h(k[i]) for i in range(n_trials)) > T:
                ended[r] = True
                ended_var[r] = sum(
                    w[i] ** 2 * ibs_var(k[i]) for i in range(n_trials)
                )
                for i in range(n_trials):
                    if current[i] == r:
                        open_count[i] = 0
                        current[i] = next_repeat(r + 1)
        level *= acceleration
    values = np.full(n, -T)
    var = np.array(ended_var)
    tv = np.zeros(n_trials)
    ts = np.zeros(n_trials)
    for r in range(n):
        if not ended[r]:
            k = np.array([K[i][r] for i in range(n_trials)])
            values[r] = np.sum(w * ibs_loglik(k))
            var[r] = np.sum(w**2 * ibs_var(k))
            tv += ibs_loglik(k)
            ts += ibs_var(k)
    return values, var, int(sum(ended)), tv, ts, sizes


@pytest.mark.parametrize(
    "initial, acceleration, cap, T",
    [
        (1, 1, 10**6, 4.1414),
        (3, 1, 10**6, 3.6789),
        (2, 1.5, 10**6, 4.6692),
        (None, 1.5, 10**6, 4.1414),
        (5, 2.0, 13, 3.6789),
        (1, 1.5, 4, 4.6692),
    ],
)
def test_matches_reference_sampler(initial, acceleration, cap, T):
    rng = np.random.default_rng(SEED)
    probs = np.linspace(0.1, 0.9, 6)
    w = np.linspace(0.5, 1.5, 6)
    streams = (rng.random((6, 20_000)) < probs[:, None]).astype(int)
    n = 12
    sim = ScriptedSimulator(streams)
    settings = _Settings(
        sim,
        np.ones(6),
        trial_weights=w,
        initial_samples=initial,
        acceleration=acceleration,
        max_samples_per_call=cap,
        neg_loglik_threshold=T,
    )
    b = draw(settings, n, SEED)
    values, var, n_ended, tv, ts, sizes = reference_sampler(
        streams, n, n if initial is None else initial, acceleration, cap, w, T
    )
    # Some repeats end and some do not.
    assert 0 < n_ended < n
    assert b.n_thresholded == n_ended
    assert summary(b).trial_repeats == n - n_ended
    assert_allclose(b.values, values, **EXACT)
    assert_allclose(b.var_estimates, var, **EXACT)
    assert_allclose(b.trial_value_sums, tv, **EXACT)
    assert_allclose(b.trial_var_sums, ts, **EXACT)
    assert [r.size for r in sim.requests] == sizes
    assert b.samples == sum(sizes)
    assert np.all(b.values >= -T)


# ---------------------------------------------------------------------------
# A threshold that no repeat reaches


@pytest.mark.parametrize(
    "initial, acceleration", [(None, 1.5), (1, 1), (3, 2.0)]
)
def test_unreachable_threshold_changes_nothing(initial, acceleration):
    def run(threshold):
        settings = _Settings(
            bernoulli,
            np.ones(P.size),
            trial_weights=W,
            initial_samples=initial,
            acceleration=acceleration,
            neg_loglik_threshold=threshold,
        )
        # Draws of 1, 3, 10 and 50 repeats from one generator: a threshold
        # that ends no repeat changes neither a draw nor what the generator
        # gives the draws after it.
        rng = np.random.default_rng(SEED)
        return [
            _sampler.sample(settings, np.zeros(1), k, rng)
            for k in (1, 3, 10, 50)
        ]

    for plain, far in zip(run(None), run(1e6)):
        assert far.n_thresholded == 0
        assert summary(far).trial_repeats == far.n
        assert np.array_equal(far.values, plain.values)
        assert np.array_equal(far.var_estimates, plain.var_estimates)
        assert np.array_equal(far.trial_value_sums, plain.trial_value_sums)
        assert np.array_equal(far.trial_var_sums, plain.trial_var_sums)
        assert cost(far) == cost(plain)


# ---------------------------------------------------------------------------
# Clipping identity against exact draws

BLOCK = 10


def exact_values(seed):
    return exact_draw(P, N_REPEATS, np.random.default_rng(seed)).values


def simulator_draws(threshold, seed):
    """20 000 repeats, as 2000 draws of 10 repeats from one generator.

    Each draw's per-trial sums are checked on their own, at the size of
    an estimate with ``ibslike.m``'s default number of repeats.
    """
    settings = _Settings(
        bernoulli, np.ones(P.size), neg_loglik_threshold=threshold
    )
    rng = np.random.default_rng(seed)
    return [
        _sampler.sample(settings, np.zeros(1), BLOCK, rng)
        for _ in range(N_REPEATS // BLOCK)
    ]


@pytest.fixture(scope="module")
def clipping():
    # T at the 70th percentile of -Y, so that about 30 % of repeats end.
    T = float(np.quantile(-exact_values(SEED + 1), 0.7))
    exact = exact_values(SEED + 2)
    draws = simulator_draws(T, SEED + 3)
    plain = simulator_draws(None, SEED + 3)
    return T, exact, draws, plain


def test_clipping_identity(clipping):
    T, exact, draws, _ = clipping
    values = np.concatenate([b.values for b in draws])
    assert values.size == N_REPEATS
    assert np.all(values >= -T)
    clipped = np.maximum(exact, -T)
    se = math.sqrt(
        values.var(ddof=1) / values.size + clipped.var(ddof=1) / clipped.size
    )
    diff = values.mean() - clipped.mean()
    print(
        f"clipping: T={T:.4f}, mean {values.mean():.4f} vs exact "
        f"{clipped.mean():.4f}, z {diff / se:.2f}"
    )
    assert abs(diff) < 4.5 * se
    # The fraction of ended repeats is the probability of Y < -T.
    ended = sum(b.n_thresholded for b in draws) / N_REPEATS
    below = np.mean(exact < -T)
    se = math.sqrt(
        ended * (1 - ended) / N_REPEATS + below * (1 - below) / N_REPEATS
    )
    print(
        f"clipping: ended {ended:.4f} vs exact {below:.4f}, "
        f"z {(ended - below) / se:.2f}"
    )
    assert 0.25 < ended < 0.35
    assert abs(ended - below) < 4.5 * se


def test_threshold_saves_samples(clipping):
    _, _, draws, plain = clipping
    with_threshold = sum(b.samples for b in draws) / N_REPEATS
    without = sum(b.samples for b in plain) / N_REPEATS
    print(
        f"samples per repeat: {with_threshold:.2f} with the threshold, "
        f"{without:.2f} without"
    )
    assert with_threshold < without


def test_per_trial_outputs_exclude_ended_repeats(clipping):
    T, _, draws, _ = clipping
    for b in draws:
        # Unit weights: the per-trial sums add up to the kept values, the
        # total less the ended repeats' -T.
        assert_allclose(
            b.trial_value_sums.sum(),
            b.values.sum() + T * b.n_thresholded,
            **EXACT,
        )
    # Over all the draws, the per-trial means cover the repeats whose
    # values are above -T, so they add up to more than the mean of the
    # clipped values.
    n_thresholded = sum(b.n_thresholded for b in draws)
    trial_repeats = sum(summary(b).trial_repeats for b in draws)
    assert trial_repeats == N_REPEATS - n_thresholded
    trial_loglik = sum(b.trial_value_sums for b in draws) / trial_repeats
    mean = sum(float(np.sum(b.values)) for b in draws) / N_REPEATS
    assert trial_loglik.sum() > mean


# ---------------------------------------------------------------------------
# Settings


@pytest.mark.parametrize(
    "threshold", [0.0, -1.0, math.inf, math.nan, True, "5", np.array([5.0])]
)
def test_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="neg_loglik_threshold"):
        _Settings(bernoulli, np.ones(3), neg_loglik_threshold=threshold)


def test_threshold_is_stored_as_float():
    settings = _Settings(bernoulli, np.ones(3), neg_loglik_threshold=5)
    assert settings.neg_loglik_threshold == 5.0
    assert isinstance(settings.neg_loglik_threshold, float)
    assert _Settings(bernoulli, np.ones(3)).neg_loglik_threshold is None

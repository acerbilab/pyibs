"""Tests of the interface: :class:`pyibs.IBS` and its outputs."""

import copy
import inspect
import math
import pickle
import time
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pyibs
from pyibs import IBS, EstimateResult, IBSSamplingError, _sampler
from pyibs import ibs as ibs_module
from pyibs._estimates import ibs_loglik, ibs_var
from pyibs.testing._exact import exact_loglik, exact_var
from pyibs.testing._helpers import P, ScriptedSimulator, bernoulli

SEED = 20261009
THETA = np.zeros(1)
W = np.linspace(0, 2, P.size)
EXACT = dict(rtol=1e-12, atol=1e-12)

# 0.1.0's messages of the exit flags.
EXIT_MESSAGES = {
    0: "Correct termination (the estimate is unbiased).",
    1: (
        "Termination after negative log-likelihood threshold was reached "
        "(the estimate is biased)."
    ),
    2: (
        "Termination after maximum execution time was reached (the "
        "estimate can be arbitrarily biased)."
    ),
}

# The fields of "full": 0.1.0's, then the per-trial arrays.
FIELDS = (
    "neg_logl",
    "neg_logl_var",
    "neg_logl_std",
    "exit_flag",
    "message",
    "elapsed_time",
    "num_samples_per_trial",
    "fun_count",
    "neg_logl_trials",
    "neg_logl_var_trials",
)

# Scripted streams of three trials, padded with matches: with one sample
# per open trial per call, the counts of two repeats are STREAM_K (repeat
# by trial).
STREAMS = [
    [1] * 12,
    [0, 0, 0, 0, 1, 0, 1, 1] + [1] * 4,
    [0, 1, 0, 0, 0, 0, 0, 1, 1] + [1] * 3,
]
STREAM_K = np.array([[1, 5, 2], [1, 2, 6]])


def never_called(params, design_rows):
    raise AssertionError("The simulator must not be called.")


def bernoulli_ibs(**kwargs):
    """IBS of the 20 Bernoulli trials of P, whose responses are all ones."""
    kwargs = {"vectorized": True, "random_seed": SEED, **kwargs}
    return IBS(bernoulli, np.ones(P.size), **kwargs)


class FakeTime:
    """Stands in for the ``time`` module; the simulators advance it."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    """A fake clock for the sampler and for ``IBS.__call__``.

    ``IBS.__call__`` takes the start of the time limit from the clock of
    ``pyibs.ibs``, and the sampler times the simulator with its own.
    """
    fake = FakeTime()
    monkeypatch.setattr(_sampler, "time", fake)
    monkeypatch.setattr(ibs_module, "time", fake)
    return fake


class Timed:
    """Wrap a simulator: record its calls and rows, and advance the clock.

    Every call lasts ``seconds`` of the fake clock; ``seconds`` can be
    changed between calls.
    """

    def __init__(self, fn, clock, seconds):
        self.fn = fn
        self.clock = clock
        self.seconds = seconds
        self.requests = []

    @property
    def calls(self):
        return len(self.requests)

    @property
    def rows(self):
        return sum(r.size for r in self.requests)

    def __call__(self, params, design_rows, rng):
        self.requests.append(np.array(design_rows))
        self.clock.now += self.seconds
        return self.fn(params, design_rows, rng)


# ---------------------------------------------------------------------------
# Output forms


def test_output_forms_and_their_types():
    # Objects with one seed make the same draw, whatever the form.
    def call(additional_output, **kwargs):
        return bernoulli_ibs()(
            THETA,
            num_reps=5,
            additional_output=additional_output,
            **kwargs,
        )

    value = call(None)
    assert type(value) is float
    assert value > 0
    none = call("none")
    assert type(none) is float and none == value
    for form in ("var", "std"):
        res = call(form)
        assert type(res) is tuple and len(res) == 2
        assert type(res[0]) is float and type(res[1]) is float
        assert res[0] == value
    var, std = call("var")[1], call("std")[1]
    assert var > 0
    assert std == math.sqrt(var)
    full = call("full")
    assert isinstance(full, EstimateResult) and isinstance(full, dict)
    assert set(full) == set(FIELDS)
    for name in FIELDS:
        assert getattr(full, name) is full[name]
    assert type(full.neg_logl) is float and full.neg_logl == value
    assert type(full.neg_logl_var) is float and full.neg_logl_var == var
    assert type(full.neg_logl_std) is float and full.neg_logl_std == std
    assert type(full.exit_flag) is int and full.exit_flag == 0
    assert full.message == EXIT_MESSAGES[0]
    assert type(full.elapsed_time) is float and full.elapsed_time >= 0
    assert type(full.num_samples_per_trial) is float
    assert full.num_samples_per_trial >= 5
    assert type(full.fun_count) is int and full.fun_count >= 1
    for name in ("neg_logl_trials", "neg_logl_var_trials"):
        assert isinstance(full[name], np.ndarray)
        assert full[name].shape == (P.size,)
        assert np.all(np.isfinite(full[name]))


def test_full_result_repr_shows_every_field_in_order():
    full = bernoulli_ibs()(THETA, num_reps=5, additional_output="full")
    text = repr(full)
    positions = [text.index(f"{name}: ") for name in FIELDS]
    assert positions == sorted(positions)
    assert repr(full.neg_logl) in text
    assert full.message in text
    assert repr(EstimateResult()) == "EstimateResult()"


def test_full_result_reads_its_keys_as_attributes():
    full = bernoulli_ibs()(THETA, num_reps=5, additional_output="full")
    assert full.fun_count is full["fun_count"]
    with pytest.raises(AttributeError, match="no_such_field"):
        full.no_such_field


@pytest.mark.parametrize(
    "additional_output", ["variance", "FULL", "", 1, True, ("var",)]
)
def test_invalid_additional_output_raises(additional_output):
    ibs = IBS(never_called, np.ones(3))
    with pytest.raises(ValueError, match="additional_output"):
        ibs(THETA, additional_output=additional_output)


@pytest.mark.parametrize("additional_output", [None, "var", "std", "full"])
def test_return_positive_changes_the_sign_of_the_total_only(
    additional_output,
):
    def call(**kwargs):
        return bernoulli_ibs()(
            THETA,
            num_reps=5,
            trial_weights=W,
            additional_output=additional_output,
            **kwargs,
        )

    negative, positive = call(), call(return_positive=True)
    if additional_output is None:
        assert positive == -negative
    elif additional_output != "full":
        assert positive[0] == -negative[0]
        assert positive[1] == negative[1]
    else:
        assert positive.neg_logl == -negative.neg_logl
        for name in FIELDS[1:]:
            if name != "elapsed_time":
                assert np.array_equal(positive[name], negative[name])


# ---------------------------------------------------------------------------
# Trial weights


def test_scalar_weight_scales_the_total():
    # Scaling by 2 is exact in floating point: the estimate doubles and its
    # variance quadruples bitwise; the unweighted per-trial arrays stay.
    plain = bernoulli_ibs()(THETA, num_reps=5, additional_output="full")
    scaled = bernoulli_ibs()(
        THETA, num_reps=5, trial_weights=2, additional_output="full"
    )
    assert scaled.neg_logl == 2 * plain.neg_logl
    assert scaled.neg_logl_var == 4 * plain.neg_logl_var
    assert np.array_equal(scaled.neg_logl_trials, plain.neg_logl_trials)
    assert np.array_equal(
        scaled.neg_logl_var_trials, plain.neg_logl_var_trials
    )
    assert scaled.fun_count == plain.fun_count


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("return_positive", [False, True])
def test_per_trial_arrays_add_up_to_the_total(vectorized, return_positive):
    # The unweighted per-trial arrays, weighted, add up to the negative
    # log-likelihood and, with the weights squared, to its variance, to
    # rounding, whatever return_positive.
    res = bernoulli_ibs(vectorized=vectorized)(
        THETA,
        num_reps=7,
        trial_weights=W,
        additional_output="full",
        return_positive=return_positive,
    )
    sign = -1 if return_positive else 1
    assert_allclose(np.sum(W * res.neg_logl_trials), sign * res.neg_logl)
    assert_allclose(np.sum(W**2 * res.neg_logl_var_trials), res.neg_logl_var)
    # Trial 0 has weight 0 and keeps its own estimate; trial 19 has p = 1.
    assert res.neg_logl_trials[0] > 0
    assert res.neg_logl_trials[-1] == 0.0
    assert res.neg_logl_var_trials[-1] == 0.0


@pytest.mark.parametrize(
    "weights",
    [
        True,
        [True, False, True],
        np.ones(3, dtype=bool),
        "1",
        ["1", "2", "3"],
        np.array(["1.0", "1.0", "1.0"]),
    ],
)
def test_trial_weights_of_the_wrong_type_raise(weights):
    ibs = IBS(never_called, np.ones(3))
    with pytest.raises(TypeError, match="Trial weights"):
        ibs(THETA, trial_weights=weights)


@pytest.mark.parametrize(
    "weights", [-1.0, [1.0, np.nan, 1.0], np.inf, [1.0, 1.0], np.ones((3, 1))]
)
def test_invalid_trial_weights_raise(weights):
    ibs = IBS(never_called, np.ones(3))
    with pytest.raises(ValueError, match="Trial weights"):
        ibs(THETA, trial_weights=weights)


# ---------------------------------------------------------------------------
# Responses and design


def test_design_none_passes_trial_indices():
    seen = []

    def simulator(params, design_rows):
        seen.append(np.array(design_rows))
        return np.ones(len(design_rows))

    IBS(simulator, np.ones(4), vectorized=False)(THETA, num_reps=2)
    IBS(simulator, np.ones(4), vectorized=True)(THETA, num_reps=2)
    # 0-based trial indices: one sample of every trial, then two of each.
    assert seen[0].dtype.kind == "i"
    assert [s.tolist() for s in seen] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 0, 1, 1, 2, 2, 3, 3],
    ]


def test_design_rows_are_passed():
    design = np.column_stack([np.arange(3), 10.0 * np.arange(3)])
    scripted = ScriptedSimulator(STREAMS, design=True)
    seen = []

    def simulator(params, design_rows):
        seen.append(np.array(design_rows))
        return scripted(params, design_rows, None)

    res = IBS(simulator, np.ones(3), design, vectorized=False)(
        THETA, num_reps=2, additional_output="full"
    )
    for rows, idx in zip(seen, scripted.requests):
        assert np.array_equal(rows, design[idx])
    assert_allclose(res.neg_logl, -ibs_loglik(STREAM_K).sum() / 2, **EXACT)


def test_multi_column_responses_match_on_every_column():
    # Trial 0 matches [1, 2] at its third sample, then at once; trial 1
    # matches [3, 4] at once, then at its third sample. A row that agrees
    # on one column only is not a match.
    responses = np.array([[1, 2], [3, 4]])
    streams = [
        [[1, 0], [0, 2], [1, 2], [1, 2], [2, 1], [1, 2]],
        [[3, 4], [4, 3], [3, 0], [3, 4], [3, 4], [3, 4]],
    ]
    res = IBS(ScriptedSimulator(streams), responses, vectorized=False)(
        THETA, num_reps=2, additional_output="full"
    )
    assert_allclose(res.neg_logl_trials, [0.75, 0.75], **EXACT)
    assert_allclose(res.neg_logl, 1.5, **EXACT)


def test_text_responses():
    # Counts (1, 1), (3, 1) and (2, 1).
    responses = np.array(["left", "right", "left"])
    streams = [
        ["left"] * 4,
        ["left", "left", "right", "right"],
        ["right", "left", "left", "left"],
    ]
    res = IBS(ScriptedSimulator(streams), responses, vectorized=False)(
        THETA, num_reps=2, additional_output="full"
    )
    assert_allclose(res.neg_logl_trials, [0.0, 0.75, 0.5], **EXACT)
    assert_allclose(res.neg_logl, 1.25, **EXACT)


@pytest.mark.parametrize("responses_shape", [(3,), (3, 1)])
@pytest.mark.parametrize("output_column", [False, True])
def test_one_column_responses_take_both_output_shapes(
    responses_shape, output_column
):
    # Responses of shape (N,) or (N, 1), and outputs of shape (r,) or
    # (r, 1): all four give the counts of the streams.
    scripted = ScriptedSimulator(STREAMS)

    def simulator(params, design_rows):
        out = scripted(params, design_rows, None)
        return out[:, None] if output_column else out

    res = IBS(simulator, np.ones(responses_shape), vectorized=False)(
        THETA, num_reps=2, additional_output="full"
    )
    assert_allclose(
        res.neg_logl_trials, -ibs_loglik(STREAM_K).sum(axis=0) / 2, **EXACT
    )


@pytest.mark.parametrize(
    "responses, output",
    [
        # One column for responses of two columns.
        (np.ones((3, 2)), lambda r: np.ones(r)),
        (np.ones((3, 2)), lambda r: np.ones((r, 1))),
        # Two columns for responses of one column.
        (np.ones(3), lambda r: np.ones((r, 2))),
        (np.ones((3, 1)), lambda r: np.ones((r, 2))),
    ],
)
def test_output_of_another_number_of_columns_raises(responses, output):
    ibs = IBS(lambda params, rows: output(len(rows)), responses)
    with pytest.raises(ValueError, match="simulator"):
        ibs(THETA, num_reps=2)


@pytest.mark.parametrize(
    "responses, trials",
    [
        (np.array([1.0, np.nan, 0.0]), "trial 1."),
        (np.array([1 + 0j, 0j, complex(0, np.nan)]), "trial 2."),
        (np.array(["2026-10-09", "NaT"], dtype="datetime64[D]"), "trial 1."),
        (
            np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 4.0]]),
            "2 trials: 1, 2.",
        ),
        (np.array(["left", np.nan, "right"], dtype=object), "trial 1."),
        (
            np.r_[0.0, np.full(7, np.nan)],
            "7 trials: 1, 2, 3, 4, 5, and 2 more.",
        ),
    ],
)
def test_nan_responses_raise(responses, trials):
    with pytest.raises(ValueError, match="response_matrix holds a NaN") as e:
        IBS(never_called, responses)
    message = str(e.value)
    assert f"in {trials[:-1]}: no sample of such a trial can match." in (
        message
    )
    assert "Recode the response" in message
    assert "or remove the trial." in message


@pytest.mark.parametrize(
    "responses",
    [
        np.array(["nan", "NaN", "NaT"]),
        np.array([b"nan", b"NaN"]),
        np.array(["nan", None], dtype=object),
    ],
)
def test_text_responses_that_read_nan_do_not_raise(responses):
    def simulator(params, design_rows):
        return responses[design_rows]

    ibs = IBS(simulator, responses, vectorized=True)
    assert ibs(THETA, num_reps=2) == 0.0


def test_design_is_not_checked_for_nan():
    def simulator(params, design_rows):
        return np.ones(len(design_rows))

    design = np.array([np.nan, 1.0, 2.0])
    assert IBS(simulator, np.ones(3), design, vectorized=True)(THETA) == 0.0


def test_scalar_responses_are_one_trial():
    # Counts 2 and 1.
    ibs = IBS(ScriptedSimulator([[0, 1, 1, 1]]), 1.0, vectorized=False)
    assert ibs.response_matrix.shape == (1,)
    res = ibs(THETA, num_reps=2, additional_output="full")
    # One entry for the trial, where ibslike.m gives one per repeat.
    assert res.neg_logl_trials.shape == (1,)
    assert res.neg_logl_var_trials.shape == (1,)
    assert_allclose(res.neg_logl, 0.5, **EXACT)
    # A scalar design is the design of that one trial.
    seen = []

    def simulator(params, design_rows):
        seen.append(np.array(design_rows))
        return np.ones(len(design_rows))

    IBS(simulator, 1.0, 5.0, vectorized=False)(THETA, num_reps=2)
    assert all(np.array_equal(rows, [5.0]) for rows in seen)


# ---------------------------------------------------------------------------
# Exit flags, the threshold and the cap


def test_threshold_ending_every_repeat_gives_exit_flag_1():
    # No sample ever matches: each repeat's bound passes T = 1 and the
    # repeat is worth exactly -T, so the estimate is T.
    ibs = IBS(
        lambda params, rows: np.zeros(len(rows)),
        np.ones(2),
        vectorized=True,
        neg_logl_threshold=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ibs(THETA, num_reps=4, additional_output="full")
    assert res.exit_flag == 1
    assert res.message == EXIT_MESSAGES[1]
    assert res.neg_logl == 1.0
    assert res.neg_logl_var > 0
    assert np.all(np.isnan(res.neg_logl_trials))
    assert np.all(np.isnan(res.neg_logl_var_trials))
    assert ibs(THETA, num_reps=4, return_positive=True) == -1.0


def test_threshold_ending_some_repeats():
    # At T = -log L, about half of the repeats fall below -T. Each repeat
    # is worth at least -T, so the estimate is below T unless every repeat
    # ended; with 40 repeats, none or all of them end with a probability of
    # about 1e-12.
    T = -exact_loglik(P)
    res = bernoulli_ibs(neg_logl_threshold=T)(
        THETA, num_reps=40, additional_output="full"
    )
    assert res.exit_flag == 1
    assert res.message == EXIT_MESSAGES[1]
    assert res.neg_logl < T
    assert np.all(np.isnan(res.neg_logl_trials))
    assert np.all(np.isnan(res.neg_logl_var_trials))


def test_unreached_threshold_changes_nothing():
    # When no repeat ended, the threshold did not act on the draw: exit flag
    # 0, and the per-trial arrays of the same draw without it (D21).
    plain = bernoulli_ibs()(
        THETA, num_reps=5, trial_weights=W, additional_output="full"
    )
    high = bernoulli_ibs(neg_logl_threshold=1e6)(
        THETA, num_reps=5, trial_weights=W, additional_output="full"
    )
    assert high.exit_flag == 0
    for name in FIELDS:
        if name != "elapsed_time":
            assert np.array_equal(high[name], plain[name])


@pytest.mark.parametrize(
    "responses, expected",
    [
        (
            [1.0, 1.0, 0.0],
            "In a draw of 3 repeats, trial 1 drew 31 samples, more than "
            "max_iter * num_reps = 30.",
        ),
        (
            [1.0, 1.0, 1.0],
            "In a draw of 3 repeats, 2 trials drew more than max_iter * "
            "num_reps = 30 samples: trial 1 drew 31, trial 2 drew 31.",
        ),
    ],
)
def test_cap_raises_and_names_the_trials_over_it(responses, expected):
    # The simulator returns 1 for trial 0 and 0 for the others, so a
    # response of 1 is never matched on trials 1 and 2. One sample per open
    # trial per call: the 31st call crosses max_iter * num_reps = 30.
    calls = []

    def simulator(params, design_rows):
        calls.append(design_rows)
        return (design_rows == 0).astype(float)

    ibs = IBS(simulator, np.array(responses), vectorized=False, max_iter=10)
    with pytest.raises(IBSSamplingError) as e:
        ibs(THETA, num_reps=3)
    assert issubclass(IBSSamplingError, RuntimeError)
    assert pyibs.IBSSamplingError is _sampler.IBSSamplingError
    message = str(e.value)
    assert expected in message
    assert message.endswith("or raise max_iter.")
    assert len(calls) == 31


# ---------------------------------------------------------------------------
# A variance of zero


@pytest.mark.parametrize("additional_output", ["var", "std", "full"])
def test_zero_variance_warns(additional_output):
    # Every trial matches at its first sample: the variance estimate is 0.
    ibs = IBS(lambda p, rows: np.ones(len(rows)), np.ones(4), vectorized=True)
    with pytest.warns(UserWarning, match="variance estimate is 0") as record:
        res = ibs(THETA, additional_output=additional_output)
    assert len(record) == 1
    assert ibs_module._FAQ_ZERO_SD in str(record[0].message)
    assert "PyBADS and PyVBMC" in str(record[0].message)
    if additional_output == "full":
        assert (res.neg_logl, res.neg_logl_var, res.neg_logl_std) == (0, 0, 0)
    else:
        assert res == (0.0, 0.0)


def test_zero_variance_does_not_warn_for_the_value_alone():
    ibs = IBS(lambda p, rows: np.ones(len(rows)), np.ones(4), vectorized=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ibs(THETA) == 0.0
        assert ibs(THETA, additional_output="none") == 0.0
        assert ibs(THETA, return_positive=True) == 0.0


# ---------------------------------------------------------------------------
# The time limit, in real time


class Sleeping:
    """A scripted simulator that sleeps at every call."""

    def __init__(self, streams, seconds):
        self.scripted = ScriptedSimulator(streams)
        self.seconds = seconds

    def __call__(self, params, design_rows):
        time.sleep(self.seconds)
        return self.scripted(params, design_rows, None)


# Every call sleeps 0.02 s, and the time limit is 0.05 s: the limit stops a
# draw after a few calls, or after its first on a slow runner. The tests
# assert only what holds whichever call it stops after. The draws below
# never complete: without the limit, the cap of max_iter * num_reps = 200
# samples per trial would stop them after 67 calls (1.3 s), with another
# error.
SLEEP, MAX_TIME = 0.02, 0.05
SLOW = dict(
    vectorized=True,
    num_samples_per_call=3,
    acceleration=1,
    max_iter=100,
    max_time=MAX_TIME,
)


def test_max_time_stops_a_slow_simulator():
    # Three samples per trial per call. In the first call, trial 0
    # completes a count of 2 and trial 1 a count of 1; neither matches
    # again. However many calls the limit allows, each trial has that one
    # completed count, so the estimate is -ibs_loglik(2) = 1, with the
    # variance ibs_var(2) = 1.
    streams = [[0, 1] + [0] * 300, [1] + [0] * 300]
    ibs = IBS(Sleeping(streams, SLEEP), np.ones(2), **SLOW)
    with pytest.warns(UserWarning, match="max_time = 0.05 s") as record:
        res = ibs(THETA, num_reps=2, additional_output="full")
    assert len(record) == 1
    assert res.exit_flag == 2
    assert res.message == EXIT_MESSAGES[2]
    assert_allclose([res.neg_logl, res.neg_logl_var], [1.0, 1.0], **EXACT)
    assert_allclose(res.neg_logl_trials, [1.0, 0.0], **EXACT)
    assert res.elapsed_time > MAX_TIME
    assert res.fun_count >= 1


def test_max_time_raises_for_a_trial_without_a_completed_count():
    streams = [[1] * 300, [0] * 300]
    ibs = IBS(Sleeping(streams, SLEEP), np.ones(2), **SLOW)
    with pytest.raises(IBSSamplingError) as e:
        ibs(THETA, num_reps=2)
    message = str(e.value)
    assert "max_time = 0.05 s" in message
    assert "no completed count in the num_reps = 2 repeats for trial 1." in (
        message
    )
    assert message.endswith("raise max_time.")


# ---------------------------------------------------------------------------
# The time limit's reduction (D3, D24), on a fake clock


# One sample per open trial per call, each call lasting 1 s. After three
# calls, trial 0 has completed its three counts (1, 1, 1), trial 1 one
# count (1), and trial 2 two counts (2, 1).
TIMED_STREAMS = [[1] * 10, [1, 0, 0, 0] + [1] * 6, [0, 1, 1] + [1] * 7]
TIMED_W = np.array([0.5, 1.0, 2.0])
# Under T = 1.7: trial 0 misses three times, and the bound h(4) = 1.83 of
# repeat 0 exceeds T after call 3, which ends it; trial 0 then matches at
# once in repeat 1 (call 4). Trial 1 completes 1 in repeat 0, 2 in repeat 1
# (call 3) and 1 in repeat 2 (call 4).
THRESHOLD_STREAMS = [[0, 0, 0, 1] + [1] * 5, [1, 0, 1, 1] + [1] * 5]
T = 1.7


def timed_ibs(clock, streams, **kwargs):
    sim = Timed(ScriptedSimulator(streams), clock, 1.0)
    return IBS(sim, np.ones(len(streams)), vectorized=False, **kwargs), sim


def test_time_limit_averages_the_completed_counts(clock):
    # The fourth call is not made: 3 s > 2.5 s have passed after the third.
    ibs, sim = timed_ibs(clock, TIMED_STREAMS, max_time=2.5)
    with pytest.warns(UserWarning, match="max_time = 2.5 s") as record:
        res = ibs(
            THETA, num_reps=3, trial_weights=TIMED_W, additional_output="full"
        )
    assert len(record) == 1
    assert "averages its completed repeats" in str(record[0].message)
    assert "threshold" not in str(record[0].message)
    assert sim.calls == res.fun_count == 3
    assert res.exit_flag == 2
    assert res.message == EXIT_MESSAGES[2]
    assert res.elapsed_time == 3.0
    assert res.num_samples_per_trial == 3.0
    # The trials' averages are (0, 0, -1/2) over 3, 1 and 2 counts:
    # weighted, -1. The variance is 2**2 (ibs_var(2) + ibs_var(1)) / 2**2.
    assert_allclose(res.neg_logl_trials, [0.0, 0.0, 0.5], **EXACT)
    assert_allclose(res.neg_logl_var_trials, [0.0, 0.0, 0.25], **EXACT)
    assert_allclose(res.neg_logl, 1.0, **EXACT)
    assert_allclose(res.neg_logl_var, 1.0, **EXACT)


@pytest.mark.parametrize("return_positive", [False, True])
def test_time_limit_under_the_threshold(clock, return_positive):
    # The time runs out after call 4, with repeat 0 ended and trial 0 open
    # in repeat 2. D24: (n_e / n)(-T) + (1 - n_e / n) sum_i w_i a_i, with
    # a = (0, -1/2), and the ended repeat's variance estimate ibs_var(4)
    # over n**2 plus (2/3)**2 times trial 1's ibs_var(2) / 2**2.
    ibs, sim = timed_ibs(
        clock, THRESHOLD_STREAMS, max_time=3.5, neg_logl_threshold=T
    )
    with pytest.warns(UserWarning, match="max_time = 3.5 s") as record:
        res = ibs(
            THETA,
            num_reps=3,
            additional_output="full",
            return_positive=return_positive,
        )
    assert len(record) == 1
    assert (
        "The likelihood threshold ended 1 of the 3 repeats, which count -1.7 "
        "each." in str(record[0].message)
    )
    assert sim.calls == 4
    assert res.exit_flag == 2
    assert res.message == EXIT_MESSAGES[2]
    loglik = -T / 3 + 2 / 3 * (-0.5)
    assert_allclose(
        res.neg_logl, loglik if return_positive else -loglik, **EXACT
    )
    assert_allclose(
        res.neg_logl_var, ibs_var(4) / 9 + 4 / 9 * ibs_var(2) / 4, **EXACT
    )
    assert np.all(np.isnan(res.neg_logl_trials))
    assert np.all(np.isnan(res.neg_logl_var_trials))


@pytest.mark.parametrize(
    "streams, max_time, kwargs, where",
    [
        # After one call, trial 2 has no completed count.
        (TIMED_STREAMS, 0.5, {}, "in the num_reps = 3 repeats for trial 2."),
        # After three calls, repeat 0 is ended, and trial 0 has no count in
        # repeats 1 and 2.
        (
            THRESHOLD_STREAMS,
            2.5,
            dict(neg_logl_threshold=T),
            "in the 2 of the num_reps = 3 repeats that the likelihood "
            "threshold did not end for trial 0.",
        ),
    ],
)
def test_time_limit_raises_for_a_trial_without_a_completed_count(
    clock, streams, max_time, kwargs, where
):
    ibs, _ = timed_ibs(clock, streams, max_time=max_time, **kwargs)
    with pytest.raises(IBSSamplingError) as e:
        ibs(THETA, num_reps=3)
    message = str(e.value)
    assert f"max_time = {max_time:g} s" in message
    assert f"no completed count {where}" in message


def test_limited_estimates_follow_d24_for_every_number_of_ended_repeats():
    # Given counts, some not completed, with the first n_e repeats ended:
    # D24's estimate, computed trial by trial, for n_e = 0 (D3's per-trial
    # average) to n (-T). Every trial has a count in the last repeat, so
    # only n_e = n leaves a trial without one.
    rng = np.random.default_rng(SEED)
    n, n_trials, threshold = 5, 4, 3.0
    K = rng.geometric(0.4, size=(n, n_trials))
    completed = rng.random((n, n_trials)) < 0.6
    completed[-1] = True
    w = np.array([0.0, 0.5, 1.0, 2.0])
    ended_var = rng.random(n)
    for n_e in range(n + 1):
        ended = np.arange(n) < n_e
        loglik, var, _, _, m = _sampler._limited_estimates(
            K, completed, ended, ended_var, w, threshold, 1.0
        )
        ended_part = ended_var[ended].sum() / n**2
        if n_e == n:
            assert loglik == -threshold
            assert_allclose(var, ended_part, **EXACT)
            continue
        a, S = np.empty(n_trials), np.empty(n_trials)
        for i in range(n_trials):
            k = K[~ended & completed[:, i], i]
            assert m[i] == k.size
            a[i], S[i] = np.mean(ibs_loglik(k)), np.sum(ibs_var(k))
        frac = 1 - n_e / n
        assert_allclose(
            loglik, n_e / n * -threshold + frac * np.dot(w, a), **EXACT
        )
        assert_allclose(
            var, ended_part + frac**2 * np.dot(w**2, S / m**2), **EXACT
        )


# ---------------------------------------------------------------------------
# Settings


def test_signature_follows_0_1_0():
    parameters = inspect.signature(IBS).parameters
    assert list(parameters) == [
        "sample_from_model",
        "response_matrix",
        "design_matrix",
        "vectorized",
        "acceleration",
        "num_samples_per_call",
        "max_iter",
        "max_time",
        "max_samples",
        "acceleration_threshold",
        "vectorized_threshold",
        "max_mem",
        "neg_logl_threshold",
        "random_seed",
    ]
    assert parameters["design_matrix"].default is None
    assert parameters["random_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(IBS.__call__).parameters) == [
        "self",
        "params",
        "num_reps",
        "trial_weights",
        "additional_output",
        "return_positive",
    ]


def test_default_settings():
    ibs = IBS(bernoulli, np.ones(P.size))
    assert ibs.sample_from_model is bernoulli
    assert ibs.design_matrix is None
    assert ibs.vectorized is None
    assert ibs.acceleration == 1.5
    assert ibs.num_samples_per_call == 0
    assert ibs.max_iter == 10**5
    assert ibs.max_time == math.inf
    assert ibs.max_samples == 10**4
    assert ibs.acceleration_threshold is None
    assert ibs.vectorized_threshold == 0.1
    # ibslike.m's max(min(N, 1e4), 10) * 100, for N = 20.
    assert ibs.max_mem == 2000
    assert ibs.neg_logl_threshold == math.inf
    assert isinstance(ibs.rng, np.random.Generator)


@pytest.mark.parametrize(
    "n_trials, max_mem",
    [(1, 1000), (10, 1000), (11, 1100), (10**5, 10**6)],
)
def test_default_max_mem_is_ibslike_formula(n_trials, max_mem):
    assert IBS(never_called, np.ones(n_trials)).max_mem == max_mem
    assert IBS(never_called, np.ones(n_trials), max_mem=7).max_mem == 7


def test_given_settings_are_read_back():
    design = np.arange(3)
    ibs = IBS(
        never_called,
        np.ones(3),
        design,
        vectorized=np.bool_(False),
        acceleration=2,
        num_samples_per_call=4,
        max_iter=50,
        max_time=10,
        max_samples=100,
        acceleration_threshold=0.5,
        vectorized_threshold=0.2,
        max_mem=300,
        neg_logl_threshold=5,
    )
    assert ibs.vectorized is False
    assert np.array_equal(ibs.design_matrix, design)
    for name, value in [
        ("acceleration", 2.0),
        ("num_samples_per_call", 4),
        ("max_iter", 50),
        ("max_time", 10.0),
        ("max_samples", 100),
        ("acceleration_threshold", 0.5),
        ("vectorized_threshold", 0.2),
        ("max_mem", 300),
        ("neg_logl_threshold", 5.0),
    ]:
        assert getattr(ibs, name) == value
        assert type(getattr(ibs, name)) is type(value)


@pytest.mark.parametrize(
    "name",
    [
        "sample_from_model",
        "response_matrix",
        "design_matrix",
        "vectorized",
        "acceleration",
        "num_samples_per_call",
        "max_iter",
        "max_time",
        "max_samples",
        "acceleration_threshold",
        "vectorized_threshold",
        "max_mem",
        "neg_logl_threshold",
    ],
)
def test_settings_are_read_only(name):
    ibs = IBS(never_called, np.ones(3))
    with pytest.raises(AttributeError):
        setattr(ibs, name, getattr(ibs, name))


def test_arrays_are_read_only_copies():
    responses, design = np.ones(3), np.arange(3)
    ibs = IBS(never_called, responses, design)
    responses[:] = 0
    design[:] = 0
    assert np.all(ibs.response_matrix == 1)
    assert np.array_equal(ibs.design_matrix, np.arange(3))
    for array in (ibs.response_matrix, ibs.design_matrix):
        with pytest.raises(ValueError):
            array[0] = 5


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(response_matrix=np.ones((2, 2, 2))),
        dict(response_matrix=np.ones(0)),
        dict(response_matrix=np.ones((3, 0))),
        dict(design_matrix=np.ones(2)),
        dict(design_matrix=1.0),
        dict(acceleration=0.5),
        dict(acceleration=math.inf),
        dict(acceleration=math.nan),
        dict(num_samples_per_call=-1),
        dict(num_samples_per_call=1.5),
        dict(max_iter=0),
        dict(max_iter=2.5),
        dict(max_iter=math.inf),
        dict(max_time=0.0),
        dict(max_time=-1.0),
        dict(max_time=math.nan),
        dict(max_samples=0),
        dict(max_samples=2.5),
        dict(acceleration_threshold=0.0),
        dict(acceleration_threshold=-1.0),
        dict(acceleration_threshold=math.nan),
        dict(vectorized_threshold=0.0),
        dict(vectorized_threshold=-1.0),
        dict(vectorized_threshold=math.nan),
        dict(max_mem=0),
        dict(max_mem=-10),
        dict(max_mem=2.5),
        dict(neg_logl_threshold=0.0),
        dict(neg_logl_threshold=-1.0),
        dict(neg_logl_threshold=-math.inf),
        dict(neg_logl_threshold=math.nan),
        dict(random_seed=-1),
        dict(random_seed=-1.0),
    ],
)
def test_settings_out_of_range_raise(kwargs):
    (name,) = kwargs
    kwargs = {"response_matrix": np.ones(3), **kwargs}
    with pytest.raises(ValueError, match=name):
        IBS(never_called, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(vectorized="yes"),
        dict(vectorized=1),
        dict(acceleration="fast"),
        dict(acceleration=True),
        dict(num_samples_per_call=True),
        dict(num_samples_per_call="3"),
        dict(max_iter=np.bool_(True)),
        dict(max_iter="1e5"),
        dict(max_iter=None),
        dict(max_time="1"),
        dict(max_time=True),
        dict(max_time=None),
        dict(max_samples=np.array([5])),
        dict(acceleration_threshold="0.1"),
        dict(acceleration_threshold=True),
        dict(vectorized_threshold="0.1"),
        dict(vectorized_threshold=None),
        dict(max_mem=False),
        dict(max_mem="100"),
        dict(neg_logl_threshold=True),
        dict(neg_logl_threshold="5"),
        dict(neg_logl_threshold=None),
        dict(random_seed="seed"),
        dict(random_seed=2.5),
        dict(random_seed=math.nan),
    ],
)
def test_settings_of_the_wrong_type_raise(kwargs):
    (name,) = kwargs
    with pytest.raises(TypeError, match=name):
        IBS(never_called, np.ones(3), **kwargs)


def test_simulator_must_be_callable():
    with pytest.raises(TypeError, match="sample_from_model"):
        IBS(None, np.ones(3))


@pytest.mark.parametrize(
    "num_reps, error",
    [
        (0, ValueError),
        (-1, ValueError),
        (2.5, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
        (True, TypeError),
        ("10", TypeError),
        (None, TypeError),
        (np.array([3]), TypeError),
    ],
)
def test_invalid_num_reps_raise(num_reps, error):
    ibs = IBS(never_called, np.ones(3))
    with pytest.raises(error, match="num_reps"):
        ibs(THETA, num_reps=num_reps)


def test_whole_number_floats_are_counts():
    floats = dict(
        num_samples_per_call=5.0, max_iter=1e5, max_samples=1e4, max_mem=1e3
    )
    ibs = bernoulli_ibs(**floats)
    for name, value in floats.items():
        assert getattr(ibs, name) == value
        assert type(getattr(ibs, name)) is int
    ints = bernoulli_ibs(**{k: int(v) for k, v in floats.items()})
    expected = ints(THETA, num_reps=10, additional_output="var")
    assert ibs(THETA, num_reps=10.0, additional_output="var") == expected
    # np.float64 and np.int64 are counts too.
    for num_reps in (np.float64(10), np.int64(10)):
        same = bernoulli_ibs(**floats)
        assert same(THETA, num_reps=num_reps, additional_output="var") == (
            expected
        )


# ---------------------------------------------------------------------------
# The generator


def test_random_seed_none_follows_numpy_global_state():
    state = np.random.get_state()
    try:
        np.random.seed(SEED)
        first = bernoulli_ibs(random_seed=None)
        np.random.seed(SEED)
        second = bernoulli_ibs(random_seed=None)
    finally:
        np.random.set_state(state)
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    assert first(THETA) == second(THETA)


@pytest.mark.parametrize(
    "seed",
    [
        SEED,
        float(SEED),
        np.float64(SEED),
        np.int64(SEED),
        np.random.SeedSequence(SEED),
    ],
)
def test_random_seed_seeds_a_new_generator(seed):
    ibs = IBS(never_called, np.ones(3), random_seed=seed)
    expected = np.random.default_rng(SEED)
    assert ibs.rng.bit_generator.state == expected.bit_generator.state


def test_random_seed_generator_is_used_as_given():
    gen = np.random.default_rng(SEED)
    ibs = bernoulli_ibs(random_seed=gen)
    assert ibs.rng is gen
    before = gen.bit_generator.state
    ibs(THETA, num_reps=2)
    assert gen.bit_generator.state != before


@pytest.mark.parametrize("vectorized", [False, True])
def test_one_seed_reproduces_a_sequence_of_calls(vectorized):
    # The simulator draws from the generator it receives.
    def simulator(params, design_rows, rng):
        return rng.random(len(design_rows)) < params[0] * P[design_rows]

    calls = [
        dict(params=np.array([1.0]), num_reps=5),
        dict(params=np.array([0.8]), num_reps=3, trial_weights=W),
        dict(params=np.array([0.9]), num_reps=8, return_positive=True),
    ]

    def run(seed):
        ibs = IBS(
            simulator, np.ones(P.size), vectorized=vectorized, random_seed=seed
        )
        results = [ibs(**c, additional_output="full") for c in calls]
        for res in results:
            del res["elapsed_time"]
        return results

    first, second, other = run(SEED), run(SEED), run(SEED + 1)
    for a, b in zip(first, second):
        assert a.keys() == b.keys()
        for name in a:
            assert np.array_equal(a[name], b[name])
    assert first[0].neg_logl != other[0].neg_logl


def test_ibs_pickles_with_its_generator():
    # A pickled object, whose simulator pickles, continues the generator's
    # stream as the original does.
    ibs = bernoulli_ibs()
    ibs(THETA)
    clone = pickle.loads(pickle.dumps(ibs))
    assert clone(THETA) == ibs(THETA)
    assert clone.vectorized is True


@pytest.mark.parametrize(
    "duplicate", [lambda x: pickle.loads(pickle.dumps(x)), copy.deepcopy]
)
def test_copies_keep_the_arrays_read_only(duplicate):
    ibs = IBS(bernoulli, np.ones(P.size), np.arange(P.size), random_seed=SEED)
    clone = duplicate(ibs)
    assert not clone.response_matrix.flags.writeable
    assert not clone.design_matrix.flags.writeable
    assert not clone._settings.trial_weights.flags.writeable


def test_simulator_without_rng_is_called_with_two_arguments():
    seen = []

    def simulator(params, design_rows):
        seen.append(params)
        return np.ones(len(design_rows))

    ibs = IBS(simulator, np.ones(3), vectorized=True)
    assert ibs(THETA, num_reps=2) == 0.0
    assert seen and all(params is THETA for params in seen)


def positional_rng(params, design_rows, rng):
    positional_rng.seen.append(rng)
    return np.ones(len(design_rows))


def defaulted_rng(params, design_rows, rng=None):
    defaulted_rng.seen.append(rng)
    return np.ones(len(design_rows))


def keyword_only_rng(params, design_rows, *, rng):
    keyword_only_rng.seen.append(rng)
    return np.ones(len(design_rows))


@pytest.mark.parametrize(
    "simulator", [positional_rng, defaulted_rng, keyword_only_rng]
)
def test_simulator_with_rng_receives_the_generator(simulator):
    simulator.seen = []
    ibs = IBS(simulator, np.ones(3), vectorized=True, random_seed=SEED)
    assert ibs(THETA, num_reps=2) == 0.0
    assert simulator.seen
    assert all(rng is ibs.rng for rng in simulator.seen)


class UnreadableSignature:
    """A simulator whose signature ``inspect.signature`` cannot read."""

    def __init__(self):
        self.seen = []

    @property
    def __signature__(self):
        raise ValueError("no signature")

    def __call__(self, params, design_rows, rng="not given"):
        self.seen.append(rng)
        return np.ones(len(design_rows))


def test_simulator_with_an_unreadable_signature_gets_two_arguments():
    simulator = UnreadableSignature()
    with pytest.raises(ValueError):
        inspect.signature(simulator)
    ibs = IBS(simulator, np.ones(3), vectorized=True)
    assert ibs(THETA, num_reps=2) == 0.0
    assert simulator.seen and set(simulator.seen) == {"not given"}


# ---------------------------------------------------------------------------
# vectorized


@pytest.mark.parametrize(
    "seconds, vectorized_threshold, decided",
    [
        (0.2, 0.1, False),
        # At the threshold, the decision is False.
        (0.1, 0.1, False),
        (0.05, 0.1, True),
        (0.2, 0.5, True),
    ],
)
def test_vectorized_none_is_decided_at_the_first_call(
    clock, seconds, vectorized_threshold, decided
):
    sim = Timed(bernoulli, clock, seconds)
    ibs = IBS(
        sim,
        np.ones(P.size),
        vectorized_threshold=vectorized_threshold,
        random_seed=SEED,
    )
    assert ibs.vectorized is None
    ibs(THETA, num_reps=3)
    assert ibs.vectorized is decided
    # The timing call simulates every trial once, in order.
    assert np.array_equal(sim.requests[0], np.arange(P.size))
    # A later call keeps the decision, whatever the time of a simulation,
    # and samples as decided: one sample of every trial in its first call,
    # or num_reps of each.
    sim.seconds = 10.0 if decided else 0.0
    first = sim.calls
    ibs(THETA, num_reps=3)
    assert ibs.vectorized is decided
    m = 3 if decided else 1
    assert np.array_equal(sim.requests[first], np.repeat(np.arange(P.size), m))


def test_vectorized_none_waits_for_a_call_with_several_repeats(clock):
    sim = Timed(bernoulli, clock, 1.0)
    ibs = IBS(sim, np.ones(P.size), random_seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ibs(THETA, num_reps=1, additional_output="full")
    assert ibs.vectorized is None
    # One sample per open trial per call, and no timing call.
    assert all(np.unique(r).size == r.size for r in sim.requests)
    assert res.fun_count == sim.calls
    ibs(THETA, num_reps=2)
    assert ibs.vectorized is False


@pytest.mark.parametrize("vectorized", [None, False])
def test_one_repeat_does_not_warn_unless_vectorized_was_given(vectorized):
    ibs = IBS(
        lambda p, rows: np.ones(len(rows)), np.ones(3), vectorized=vectorized
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ibs(THETA, num_reps=1)


def test_one_repeat_after_a_decision_of_true_does_not_warn(clock):
    # vectorized=None decided True: a call with num_reps=1 samples one at a
    # time, as the warning of vectorized=True says, but does not warn.
    ibs = IBS(Timed(bernoulli, clock, 0.0), np.ones(P.size), random_seed=SEED)
    ibs(THETA, num_reps=2)
    assert ibs.vectorized is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ibs(THETA, num_reps=1)


def test_max_time_counts_the_timing_call(clock):
    # The timing call (0.04 s, fast: True) counts toward max_time = 0.07 s,
    # from the start of the call: after the next round, 0.08 s have passed,
    # and the sampling stops. The timing call's sample, a miss, is the first
    # round; the next round, of two samples, completes a count of 3 and
    # leaves the second repeat open.
    sim = Timed(ScriptedSimulator([[0, 0, 1] + [0] * 50]), clock, 0.04)
    ibs = IBS(sim, np.ones(1), max_time=0.07)
    with pytest.warns(UserWarning, match="max_time = 0.07 s"):
        res = ibs(THETA, num_reps=2, additional_output="full")
    assert ibs.vectorized is True
    assert sim.calls == res.fun_count == 2
    assert res.exit_flag == 2
    assert_allclose([res.neg_logl, res.neg_logl_var], [1.5, 1.25], **EXACT)


def test_vectorized_true_with_one_repeat_warns_and_samples_one_at_a_time():
    sim = ScriptedSimulator(STREAMS)
    ibs = IBS(sim, np.ones(3), vectorized=True)
    with pytest.warns(UserWarning, match="vectorized=True needs num_reps > 1"):
        res = ibs(THETA, num_reps=1, additional_output="full")
    # Trial 0 matches in call 1, trial 2 in call 2 and trial 1 in call 5.
    assert [r.tolist() for r in sim.requests] == [
        [0, 1, 2],
        [1, 2],
        [1],
        [1],
        [1],
    ]
    assert res.fun_count == 5
    assert ibs.vectorized is True
    assert_allclose(res.neg_logl, -ibs_loglik(STREAM_K[0]).sum(), **EXACT)


@pytest.mark.parametrize(
    "seconds, num_samples_per_call, extra",
    [
        # Slow: False, whose first round, one sample of every trial, is the
        # timing call.
        (1.0, 0, False),
        # Fast: True, whose first round requests num_samples_per_call = 1
        # sample of every trial: the timing call again.
        (0.0, 1, False),
        # Fast: True, whose first round requests num_reps = 4 samples of
        # every trial: the timing call is an extra round before it.
        (0.0, 0, True),
    ],
)
def test_vectorized_none_counts_the_timing_call(
    clock, seconds, num_samples_per_call, extra
):
    sim = Timed(bernoulli, clock, seconds)
    ibs = IBS(
        sim,
        np.ones(P.size),
        num_samples_per_call=num_samples_per_call,
        random_seed=SEED,
    )
    res = ibs(THETA, num_reps=4, additional_output="full")
    # Every call and every row of the simulator, the timing call included,
    # once.
    assert res.fun_count == sim.calls
    assert res.num_samples_per_trial == sim.rows / P.size
    if not extra:
        # The draw is that of the decided setting given explicitly.
        explicit = bernoulli_ibs(
            vectorized=ibs.vectorized,
            num_samples_per_call=num_samples_per_call,
        )(THETA, num_reps=4, additional_output="full")
        for name in FIELDS:
            if name != "elapsed_time":
                assert np.array_equal(res[name], explicit[name])
    else:
        # The level does not grow after the extra round: the next round
        # requests num_reps = 4 samples of every trial, all still open.
        assert np.array_equal(sim.requests[1], np.repeat(np.arange(P.size), 4))


def test_cap_counts_the_timing_call(clock):
    # vectorized=None decides True, and the rounds request two samples of
    # the trial, so the timing call is an extra first round. With its
    # sample, the trial drew 21 samples, more than max_iter * num_reps =
    # 20, although the last call completed both repeats.
    sim = Timed(ScriptedSimulator([[0] * 19 + [1] * 100]), clock, 0.0)
    ibs = IBS(sim, np.ones(1), acceleration=1, max_iter=10)
    with pytest.raises(IBSSamplingError, match="trial 0 drew 21 samples"):
        ibs(THETA, num_reps=2)
    assert ibs.vectorized is True


# Three Bernoulli trials, and estimates of two repeats each.
P_DIST = np.array([0.2, 0.5, 0.8])
DIST_REPS = 2
N_ESTIMATES = 800


@pytest.mark.filterwarnings("ignore:The IBS variance estimate is 0")
@pytest.mark.parametrize(
    "vectorized, seconds, seed",
    [(True, 0.0, 1), (False, 0.0, 2), (None, 0.0, 3), (None, 1.0, 4)],
    ids=["True", "False", "None-fast", "None-slow"],
)
def test_vectorized_settings_agree_in_distribution(
    clock, vectorized, seconds, seed
):
    # Each estimate comes from a new object, so that vectorized=None times
    # every estimate's first call, whose first round is the timing call
    # whatever the decision.
    def simulator(params, design_rows, rng):
        clock.now += seconds
        return rng.random(len(design_rows)) < P_DIST[design_rows]

    rng = np.random.default_rng([SEED, seed])
    estimates = np.empty((N_ESTIMATES, 2))
    for k in range(N_ESTIMATES):
        ibs = IBS(
            simulator,
            np.ones(P_DIST.size, bool),
            vectorized=vectorized,
            random_seed=rng,
        )
        estimates[k] = ibs(THETA, num_reps=DIST_REPS, additional_output="var")
    if vectorized is None:
        assert ibs.vectorized is (seconds == 0.0)
    values, var_estimates = estimates.T
    target_var = exact_var(P_DIST) / DIST_REPS
    # The mean against the negative log-likelihood.
    se_mean = math.sqrt(target_var / N_ESTIMATES)
    assert abs(values.mean() + exact_loglik(P_DIST)) < 4.5 * se_mean
    # The variance of the estimates, with the standard error of a sample
    # variance from the squared deviations.
    sq = (values - values.mean()) ** 2
    se_var = sq.std(ddof=1) / math.sqrt(N_ESTIMATES)
    assert abs(values.var(ddof=1) - target_var) < 4.5 * se_var
    # The variance estimates are calibrated on average.
    se_v_hat = var_estimates.std(ddof=1) / math.sqrt(N_ESTIMATES)
    assert abs(var_estimates.mean() - target_var) < 4.5 * se_v_hat


def test_first_call_is_unbiased_when_the_timing_tracks_the_outcomes(clock):
    # One trial, matched with probability 0.5, whose simulation of one row,
    # the timing call of vectorized=None, lasts 0.2 s when it matches and
    # 0 s when it misses: the decision is False exactly when the timing call
    # matched. Each estimate is the first call of a new object. A sampler
    # that used the timing call's sample only when it decides False would
    # give 0.75 log 2 on average, about 14 standard errors from log 2.
    def simulator(params, design_rows, rng):
        out = rng.random(len(design_rows)) < 0.5
        if len(design_rows) == 1:
            clock.now += 0.2 * np.count_nonzero(out)
        return out

    p = np.array([0.5])
    n_estimates, n_reps = 2000, 2
    rng = np.random.default_rng([SEED, 5])
    estimates = np.array(
        [
            IBS(simulator, np.ones(1, bool), random_seed=rng)(
                THETA, num_reps=n_reps
            )
            for _ in range(n_estimates)
        ]
    )
    se = math.sqrt(exact_var(p) / n_reps / n_estimates)
    assert abs(estimates.mean() + exact_loglik(p)) < 4.5 * se

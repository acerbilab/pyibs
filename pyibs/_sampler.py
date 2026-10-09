"""IBS repeats from a simulator of the model's responses.

:func:`sample` runs inverse binomial sampling (IBS) with a user-supplied
simulator ([1]). For every trial it samples responses until one matches
the observed response; the number of samples up to and including the
match is the trial's matching count K. The algorithm is the sampling of
MATLAB ``ibslike.m`` (https://github.com/acerbilab/ibs), with its two
schedules, accelerated and one sample at a time, in one sampler that
returns per-repeat estimates.

A draw of n repeats is sampled rows-first. Each trial's samples form one
i.i.d. stream of match and no-match outcomes, and the stream is split at
its matches: the r-th gap between matches is the trial's count in repeat r.
Consecutive gaps of such a stream are i.i.d. geometric, and the n-th match
is a stopping time, so the n repeats are independent complete IBS repeats,
and the samples a trial draws after its n-th match can be discarded without
bias. Each round asks the simulator, in one call, for several samples of
every trial that still needs matches, so one call can complete several
counts of a trial, or advance one count that spans several calls.

An optional likelihood threshold ends a repeat as soon as its sampling
shows that its log-likelihood lies below a lower bound, and returns the
bound as the repeat's value, as in [1], Appendix C.1. An optional time
limit stops the sampling of a draw, which then averages each trial's
completed counts.

References
----------
.. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
   efficient log-likelihood estimation with inverse binomial sampling.
   PLOS Computational Biology 16(12): e1008483.
   https://doi.org/10.1371/journal.pcbi.1008483
"""

import copy
import math
import numbers
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass

import numpy as np

from pyibs import _estimates


class IBSSamplingError(RuntimeError):
    """IBS sampling ended without an estimate.

    A draw raises it once a trial has drawn more samples than its cap
    allows, after the simulator call that crossed the cap, even when that
    call completed the draw: IBS gives no estimate for a repeat in which
    some trial has not matched, since its partial count would bias the
    estimate. The usual cause is an observed response that the simulator
    never, or almost never, produces at the parameter vector. A draw that
    its time limit stops raises it when a trial has no completed count to
    average.
    """


IBSSamplingError.__module__ = "pyibs"


def _check_count(n, name, minimum=1):
    """Return ``n`` as an int, which must be an integer >= ``minimum``.

    A whole-number float is taken as the integer it equals; a boolean is
    not a count.

    Raises
    ------
    TypeError
        If ``n`` is a boolean or not a real number.
    ValueError
        If ``n`` is not a whole number, or is below ``minimum``.
    """
    message = f"{name} must be an integer >= {minimum}, got {n!r}."
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Real):
        raise TypeError(message)
    if not isinstance(n, numbers.Integral) and not float(n).is_integer():
        raise ValueError(message)
    if n < minimum:
        raise ValueError(message)
    return int(n)


def _check_real(x, name):
    """Return ``x`` as a float, which must be a real number.

    Raises
    ------
    TypeError
        If ``x`` is a boolean or not a real number.
    """
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {x!r}.")
    return float(x)


def _name_trials(trials):
    """Name trials by their 0-based indices, at most five of them."""
    trials = list(trials)
    if len(trials) == 1:
        return f"trial {trials[0]}"
    listed = ", ".join(str(i) for i in trials[:5])
    if len(trials) > 5:
        listed += f", and {len(trials) - 5} more"
    return f"{len(trials)} trials: {listed}"


def _check_responses(responses, name):
    """Return the responses as a read-only copy, checked.

    Raises
    ------
    ValueError
        If the responses are not a non-empty array of shape (N,) or
        (N, C), or hold a NaN: an element not equal to itself, such as a
        float or complex NaN, ``NaT``, or a NaN in an object array.
    """
    responses = np.array(responses)
    if responses.ndim not in (1, 2) or responses.size == 0:
        raise ValueError(
            f"{name} must be a non-empty array of shape (N,) or (N, C), got "
            f"shape {responses.shape}."
        )
    nan = np.asarray(responses != responses, dtype=bool)
    if nan.ndim == 2:
        nan = nan.any(axis=1)
    if nan.any():
        raise ValueError(
            f"{name} holds a NaN, which no simulated response equals, in "
            f"{_name_trials(np.flatnonzero(nan))}: no sample of such a trial "
            "can match. Recode the response as a value that the simulator "
            "returns, or remove the trial."
        )
    responses.setflags(write=False)
    return responses


def _check_design(design, n_trials, name):
    """Return the design as a read-only copy, or None, checked.

    Raises
    ------
    ValueError
        If the design is not None and does not have one row per trial.
    """
    if design is None:
        return None
    design = np.array(design)
    if design.ndim == 0 or design.shape[0] != n_trials:
        raise ValueError(
            f"{name} must have one row per trial ({n_trials}), got shape "
            f"{design.shape}."
        )
    design.setflags(write=False)
    return design


def default_max_mem(n_trials):
    """``ibslike.m``'s ``MaxMem``: ``max(min(N, 10**4), 10) * 100``."""
    return max(min(n_trials, 10**4), 10) * 100


# Array kinds that NumPy compares by value: text, bytes, and numbers or
# booleans. Arrays of two different kinds among these, one of them text or
# bytes, compare as unequal whatever their values.
_TEXT_KINDS = "US"
_VALUE_KINDS = "USbiufc"


def _check_kinds(simulated, observed):
    """Raise TypeError if simulated rows can never equal the responses.

    Object arrays, and kinds outside text, bytes, numbers and booleans,
    are not checked.
    """
    a, b = simulated.dtype.kind, observed.dtype.kind
    if (
        a != b
        and (a in _TEXT_KINDS or b in _TEXT_KINDS)
        and a in _VALUE_KINDS
        and b in _VALUE_KINDS
    ):
        raise TypeError(
            f"The simulator returned responses of dtype {simulated.dtype}, "
            "which NumPy never finds equal to the observed responses of "
            f"dtype {observed.dtype}, so no sample would match. The "
            "simulator must return responses of the observed kind: text, "
            "bytes, or numbers and booleans."
        )


@dataclass(frozen=True, eq=False)
class _Settings:
    """The model, the data and the sampling settings of :func:`sample`.

    The arrays are stored as read-only copies.

    Parameters
    ----------
    simulator : callable
        ``simulator(theta, design_rows, rng)`` returns one simulated
        response row per requested trial. ``design_rows`` is
        ``design[idx]`` for the indices ``idx`` of the requested trials, or
        ``idx`` itself (0-based trial indices) when ``design`` is None. A
        trial appears in ``idx`` once per requested sample, and every
        requested row must be an independent draw. For r requested rows,
        the simulator returns an array of shape (r,) or (r, 1) when the
        responses have one column, of shape (N,) or (N, 1), and of shape
        (r, C) when they have C > 1 columns. ``rng`` is the
        :class:`numpy.random.Generator` of the call: a simulator that draws
        from NumPy's global state instead makes runs irreproducible.
    responses : array_like of shape (N,) or (N, C)
        The observed responses of the N trials. A simulated row matches a
        response only when every column agrees, as in ``ibslike.m``. NumPy
        finds text, bytes, and numbers or booleans unequal to one another
        whatever their values, so the simulated rows must be of the
        responses' kind; an object array on either side is compared element
        by element. A NaN response, an element not equal to itself, never
        matches, and is refused.
    design : array_like of shape (N, ...), optional
        Per-trial design; the simulator receives its rows for the
        requested trials.
    trial_weights : None, float or array_like of shape (N,), optional
        Trial weights (``ibslike.m``'s ``TrialWeights``), finite and >= 0;
        checked by :func:`pyibs._estimates.trial_weights`. None gives unit
        weights and a scalar applies to every trial.
    initial_samples : int or None, optional
        The level of samples per open trial at the first simulator call of
        a draw, at least 1 (``ibslike.m``'s ``NsamplesPerCall``). None, the
        default, uses the number of repeats the draw asks for.
    acceleration : float, optional
        Factor >= 1 by which the level grows from one call to the next
        (``ibslike.m``'s ``Acceleration``).
    acceleration_threshold : float or None, optional
        None, the default, grows the level after every call. A time in
        seconds > 0 grows it only after calls to the simulator that took
        less than this, as ``ibslike.m`` does. Which samples a call
        requests then depends on the wall-clock time, and so does the
        assignment of random numbers to trials: a seed no longer
        reproduces a run.
    max_samples : int, optional
        Bound, at least 1, on the samples of one trial in one simulator
        call (``ibslike.m``'s ``MaxSamples``, 10**4, also the default).
    max_mem : int or None, optional
        Bound, at least 1, on the samples of one simulator call, which a
        call exceeds by less than its number of open trials: each open
        trial gets at most ``ceil(max_mem / n_open)`` samples
        (``ibslike.m``'s ``MaxMem``). None, the default, takes
        ``ibslike.m``'s ``max(min(N, 10**4), 10) * 100``.
    max_samples_per_trial : int or None, optional
        Cap, at least 1, on the samples of one trial per repeat, the
        counterpart of ``ibslike.m``'s ``MaxIter`` (10**5 per trial and
        estimate, also the default here). A draw of n repeats raises
        :class:`IBSSamplingError` once a trial has drawn more than
        ``max_samples_per_trial * n`` samples in it, surplus included. The
        check follows every simulator call, whether or not that call
        completed the draw. None disables the cap: an observed
        response that the simulator cannot produce then makes a draw run
        forever.
    max_time : float, optional
        Time limit in seconds, > 0, of a draw (``ibslike.m``'s
        ``MaxTime``), counted from the ``start`` given to :func:`sample`.
        ``math.inf``, the default, sets none. See the Notes of
        :func:`sample`.
    neg_loglik_threshold : float or None, optional
        A likelihood threshold T, finite and > 0, on the scale of one
        repeat's weighted negative log-likelihood (``ibslike.m``'s
        ``NegLogLikeThreshold``). A repeat whose sampling shows that its
        value lies below -T is ended, and its value is -T, so every
        repeat's value is ``max(Y_r, -T)``, where ``Y_r`` is the value its
        complete sampling would give ([1], Appendix C.1). This saves the
        samples of poor parameter vectors at the price of an upward bias;
        the negative log-likelihood of chance responding is the usual
        choice (see the Notes of :func:`sample`). None, the default,
        samples every repeat to completion.
    names : tuple of two str, optional
        The names that the messages of :class:`IBSSamplingError` give the
        cap's setting and the number of repeats of a draw; by default
        ``("max_samples_per_trial", "n")``.

    Raises
    ------
    TypeError
        If ``simulator`` is not callable, or a setting is a boolean or of a
        type that its value cannot have.
    ValueError
        If ``responses`` is not a non-empty array of shape (N,) or (N, C)
        or holds a NaN, the design has a length other than N, the weights
        are invalid, or a sampling setting or the threshold is out of
        range.
    """

    simulator: Callable
    responses: np.ndarray
    design: np.ndarray | None = None
    _: KW_ONLY
    trial_weights: np.ndarray | None = None
    initial_samples: int | None = None
    acceleration: float = 1.5
    acceleration_threshold: float | None = None
    max_samples: int = 10**4
    max_mem: int | None = None
    max_samples_per_trial: int | None = 10**5
    max_time: float = math.inf
    neg_loglik_threshold: float | None = None
    names: tuple = ("max_samples_per_trial", "n")

    def __post_init__(self):
        if not callable(self.simulator):
            raise TypeError("simulator must be callable.")
        responses = _check_responses(self.responses, "responses")
        n_trials = responses.shape[0]
        design = _check_design(self.design, n_trials, "design")
        weights = _estimates.trial_weights(self.trial_weights, n_trials)
        weights.setflags(write=False)
        initial_samples = self.initial_samples
        if initial_samples is not None:
            initial_samples = _check_count(initial_samples, "initial_samples")
        acceleration = _check_real(self.acceleration, "acceleration")
        if not 1 <= acceleration < math.inf:
            raise ValueError(
                f"acceleration must be finite and >= 1, got {acceleration}."
            )
        acceleration_threshold = self.acceleration_threshold
        if acceleration_threshold is not None:
            acceleration_threshold = _check_real(
                acceleration_threshold, "acceleration_threshold"
            )
            if not acceleration_threshold > 0:
                raise ValueError(
                    "acceleration_threshold must be None or > 0 seconds, "
                    f"got {acceleration_threshold}."
                )
        max_samples = _check_count(self.max_samples, "max_samples")
        max_mem = self.max_mem
        if max_mem is None:
            max_mem = default_max_mem(n_trials)
        else:
            max_mem = _check_count(max_mem, "max_mem")
        max_samples_per_trial = self.max_samples_per_trial
        if max_samples_per_trial is not None:
            max_samples_per_trial = _check_count(
                max_samples_per_trial, "max_samples_per_trial"
            )
        max_time = _check_real(self.max_time, "max_time")
        if not max_time > 0:
            raise ValueError(
                f"max_time must be > 0 seconds or inf, got {max_time}."
            )
        threshold = self.neg_loglik_threshold
        if threshold is not None:
            threshold = _check_real(threshold, "neg_loglik_threshold")
            if not 0 < threshold < math.inf:
                raise ValueError(
                    "neg_loglik_threshold must be None or finite and > 0, "
                    f"got {threshold}."
                )
        for name, value in [
            ("responses", responses),
            ("design", design),
            ("trial_weights", weights),
            ("initial_samples", initial_samples),
            ("acceleration", acceleration),
            ("acceleration_threshold", acceleration_threshold),
            ("max_samples", max_samples),
            ("max_mem", max_mem),
            ("max_samples_per_trial", max_samples_per_trial),
            ("max_time", max_time),
            ("neg_loglik_threshold", threshold),
            ("names", tuple(self.names)),
        ]:
            object.__setattr__(self, name, value)

    def __setstate__(self, state):
        # Pickling and deep copies give writable arrays.
        self.__dict__.update(state)
        for array in (self.responses, self.design, self.trial_weights):
            if array is not None:
                array.setflags(write=False)

    @property
    def n_trials(self):
        """Number of trials N."""
        return self.responses.shape[0]

    def with_trial_weights(self, trial_weights):
        """These settings with other trial weights.

        The arrays are shared with these settings, not copied.

        Parameters
        ----------
        trial_weights : None, float or array_like of shape (N,)
            The weights, checked by :func:`pyibs._estimates.trial_weights`.

        Returns
        -------
        settings : _Settings
        """
        weights = _estimates.trial_weights(trial_weights, self.n_trials)
        weights.setflags(write=False)
        settings = copy.copy(self)
        object.__setattr__(settings, "trial_weights", weights)
        return settings


@dataclass(frozen=True, eq=False)
class _Draw:
    """What :func:`sample` returns for a draw of n repeats of N trials.

    Attributes
    ----------
    K : ndarray of int64, shape (n, N)
        ``K[r, i]`` is trial i's matching count in repeat r. A count that
        was not completed, in a repeat that the likelihood threshold ended
        or that the time limit left unfinished, is 0.
    values : ndarray of shape (n,)
        Each repeat's log-likelihood estimate, -T for an ended repeat, and
        NaN for a repeat that the time limit left unfinished.
    var_estimates : ndarray of shape (n,)
        Each repeat's variance estimate, NaN where ``values`` is.
    loglik, loglik_var : float
        The draw's log-likelihood estimate and its variance estimate: the
        mean of ``values`` and the sum of ``var_estimates`` over n**2, or,
        when the time limit stopped the draw, the estimates of
        :func:`_limited_estimates`.
    trial_value_sums, trial_var_sums : ndarray of shape (N,)
        The unweighted sums of each trial's ``ibs_loglik(K)`` and
        ``ibs_var(K)`` over its completed counts in the repeats that the
        threshold did not end.
    trial_counts : ndarray of int64, shape (N,)
        The number of counts in each trial's sums: n less the ended
        repeats, or fewer when the time limit stopped the draw.
    ended : ndarray of bool, shape (n,)
        Whether the likelihood threshold ended each repeat.
    n_thresholded : int
        The number of ended repeats.
    timed_out : bool
        Whether the time limit stopped the draw.
    calls : int
        Calls of the simulator in the draw, the call given as ``first``
        included.
    samples : int
        Simulated response rows of those calls, surplus included.
    seconds : float
        Wall time spent inside the simulator in those calls. It is not
        reproducible from a seed.
    """

    K: np.ndarray
    values: np.ndarray
    var_estimates: np.ndarray
    loglik: float
    loglik_var: float
    trial_value_sums: np.ndarray
    trial_var_sums: np.ndarray
    trial_counts: np.ndarray
    ended: np.ndarray
    n_thresholded: int
    timed_out: bool
    calls: int
    samples: int
    seconds: float

    @property
    def n(self):
        """Number of repeats in the draw."""
        return self.values.size


@dataclass(frozen=True, eq=False)
class _FirstRound:
    """A simulator call for one sample of every trial, in trial order.

    Attributes
    ----------
    hits : ndarray of bool, shape (N, 1)
        Whether each trial's sample matches its response.
    elapsed : float
        Seconds spent inside the simulator.
    """

    hits: np.ndarray
    elapsed: float


def first_round(settings, theta, rng):
    """Simulate one sample of every trial, in trial order, and time it.

    This is the call that ``ibslike.m`` times to choose its sampling path.
    :func:`sample` takes it as its first round, whatever its outcomes.

    Parameters
    ----------
    settings : _Settings
        The simulator, the data and the sampling settings.
    theta : ndarray of shape (D,)
        The parameter vector.
    rng : numpy.random.Generator
        The generator, passed to the simulator.

    Returns
    -------
    first : _FirstRound

    Raises
    ------
    ValueError, TypeError
        As the checks of every simulator call in :func:`sample`.
    """
    hits, elapsed = _simulate(
        settings, theta, rng, np.arange(settings.n_trials), 1
    )
    return _FirstRound(hits=hits, elapsed=elapsed)


class _MatchCounts:
    """Matching counts of one draw of n repeats, filled rows-first.

    Every trial works through the repeats that are not ended, in order.
    Its open count is the number of samples it has drawn since its last
    match, the part of its current count sampled so far.

    Parameters
    ----------
    n : int
        Repeats in the draw.
    n_trials : int
        Number of trials N.
    weights : ndarray of shape (N,) or None, optional
        The trial weights, given when the draw has a likelihood threshold:
        the counts then keep the repeats' running bounds.

    Attributes
    ----------
    n : int
        Repeats in the draw.
    K : ndarray of int64, shape (n, N)
        ``K[r, i]`` is trial i's matching count in repeat r, once
        completed, and 0 before.
    completed : ndarray of bool, shape (n, N)
        Whether ``K[r, i]`` is complete.
    repeat : ndarray of int64, shape (N,)
        The repeat each trial is sampling; n once the trial is done.
    open_count : ndarray of int64, shape (N,)
        Each trial's open count; 0 once the trial is done.
    ended : ndarray of bool, shape (n,)
        Whether each repeat was ended by the likelihood threshold.
    ended_var : ndarray of shape (n,)
        The variance estimate of each ended repeat, and 0 for the others.
    """

    def __init__(self, n, n_trials, weights=None):
        self.n = n
        self.K = np.zeros((n, n_trials), dtype=np.int64)
        self.completed = np.zeros((n, n_trials), dtype=bool)
        self.repeat = np.zeros(n_trials, dtype=np.int64)
        self.open_count = np.zeros(n_trials, dtype=np.int64)
        self.ended = np.zeros(n, dtype=bool)
        self.ended_var = np.zeros(n)
        self.weights = weights
        # The repeats that are not ended, in order, followed by n.
        self._active = np.arange(n + 1, dtype=np.int64)
        # The completed trials' part of each repeat's running bound, the
        # sum of w_i (digamma(K[r, i]) - digamma(1)); None without weights.
        self._closed = None if weights is None else np.zeros(n)

    def open_trials(self):
        """Indices of the trials that need more matches, in order."""
        return np.flatnonzero(self.repeat < self.n)

    def absorb(self, trials, hits):
        """Split one round's samples of the open trials at their matches.

        Parameters
        ----------
        trials : ndarray of int, shape (n_open,)
            The open trials, as returned by :meth:`open_trials`.
        hits : ndarray of bool, shape (n_open, m)
            Row j holds the outcomes of the m samples of ``trials[j]``, in
            sampling order.
        """
        m = hits.shape[1]
        n_hits = np.count_nonzero(hits, axis=1)
        # The matches in trial-major sampling order, and each one's rank
        # among its trial's matches in this round.
        row, col = np.nonzero(hits)
        first_hit = np.cumsum(n_hits) - n_hits
        rank = np.arange(row.size) - first_hit[row]
        # A count is the gap from the previous match of the same trial; the
        # first match of a round closes the count left open before it.
        gaps = np.empty_like(col)
        gaps[1:] = col[1:] - col[:-1]
        first = rank == 0
        gaps[first] = col[first] + 1 + self.open_count[trials[row[first]]]
        # A trial's matches close, in order, the repeats that are not ended
        # from its current one on; pos is the current repeat's position
        # among them. Matches beyond the last of them are surplus.
        pos = np.searchsorted(self._active, self.repeat[trials])
        needed = self._active.size - 1 - pos
        keep = rank < needed[row]
        trial = trials[row[keep]]
        repeat = self._active[pos[row[keep]] + rank[keep]]
        self.K[repeat, trial] = gaps[keep]
        self.completed[repeat, trial] = True
        if self._closed is not None:
            self._closed += np.bincount(
                repeat,
                weights=-self.weights[trial]
                * _estimates.ibs_loglik(gaps[keep]),
                minlength=self.n,
            )
        # Open counts: samples after the last match, or all m added to the
        # previous open count when there was no match.
        open_count = self.open_count[trials] + m
        has_hit = n_hits > 0
        last_col = col[(first_hit + n_hits - 1)[has_hit]]
        open_count[has_hit] = m - 1 - last_col
        self.repeat[trials] = self._active[pos + np.minimum(n_hits, needed)]
        open_count[self.repeat[trials] >= self.n] = 0
        self.open_count[trials] = open_count

    def bounds(self):
        """Running bounds on the repeats' negative log-likelihoods.

        Available when the counts have weights.

        Returns
        -------
        bounds : ndarray of shape (n,)
            For each repeat r that is not ended, B_r, the sum over the
            trials of ``w_i (digamma(k_i) - digamma(1))``, where k_i is
            ``K[r, i]`` for a trial that has completed r, ``c_i + 1`` for a
            trial sampling r with open count c_i, and 1 otherwise. The
            entries of ended repeats have no meaning.
        """
        sampling = np.flatnonzero(self.repeat < self.n)
        open_part = np.bincount(
            self.repeat[sampling],
            weights=-self.weights[sampling]
            * _estimates.ibs_loglik(self.open_count[sampling] + 1),
            minlength=self.n,
        )
        return self._closed + open_part

    def end_above(self, threshold):
        """End the incomplete repeats whose running bound exceeds threshold.

        The bound of a repeat that every trial has completed is its
        complete negative log-likelihood, and ending such a repeat changes
        no sampling; :meth:`end_complete_below` checks those repeats on
        their exact values instead.

        Parameters
        ----------
        threshold : float
            The likelihood threshold T.

        Returns
        -------
        ended : ndarray of int
            The repeats ended, in order.
        """
        # The repeats before the lowest one that a trial is sampling are
        # complete or ended; every repeat from it on that is not ended is
        # incomplete.
        lo = int(self.repeat.min())
        if lo == self.n:
            return np.empty(0, dtype=np.int64)
        over = self.bounds()[lo:] > threshold
        rows = lo + np.flatnonzero(over & ~self.ended[lo:])
        if rows.size == 0:
            return rows
        # The counts of the bound: completed counts, c + 1 for the trials
        # sampling a repeat, and 1 for the trials that have not reached it.
        sampling = np.flatnonzero(np.isin(self.repeat, rows))
        at = np.searchsorted(rows, self.repeat[sampling])
        bound_counts = np.where(self.completed[rows], self.K[rows], 1)
        bound_counts[at, sampling] = self.open_count[sampling] + 1
        _, var, _, _ = _estimates.repeat_estimates(bound_counts, self.weights)
        self.ended_var[rows] = var
        self.ended[rows] = True
        self._active = np.append(np.flatnonzero(~self.ended), self.n)
        # The trials sampling an ended repeat drop their open count and
        # move on to the next repeat that is not ended.
        self.repeat[sampling] = self._active[
            np.searchsorted(self._active, self.repeat[sampling])
        ]
        self.open_count[sampling] = 0
        return rows

    def complete_repeats(self):
        """The repeats, not ended, that every trial has completed."""
        return np.flatnonzero(~self.ended & self.completed.all(axis=1))

    def end_complete_below(self, threshold):
        """End the complete repeats whose value lies below ``-threshold``.

        A complete repeat's bound is its negative value; :meth:`end_above`
        leaves the complete repeats to this check, which gives an ended
        repeat the variance estimate of its complete counts.

        Parameters
        ----------
        threshold : float
            The likelihood threshold T.

        Returns
        -------
        kept : ndarray of int
            The complete repeats that are not ended, in order.
        values, var_estimates : ndarray
            Their values and variance estimates.
        trial_value_sums, trial_var_sums : ndarray of shape (N,)
            The per-trial sums over them.
        """
        kept = self.complete_repeats()
        v, s, tv, ts = _estimates.repeat_estimates(self.K[kept], self.weights)
        below = -v > threshold
        if np.any(below):
            self.ended_var[kept[below]] = s[below]
            self.ended[kept[below]] = True
            kept, v, s = kept[~below], v[~below], s[~below]
            _, _, tv, ts = _estimates.repeat_estimates(
                self.K[kept], self.weights
            )
        return kept, v, s, tv, ts

    def clipped_estimates(self, threshold):
        """Estimates of a finished draw under a likelihood threshold.

        The repeats that are not ended are complete. Those whose value lies
        below ``-threshold`` are ended here (:meth:`end_complete_below`).

        Parameters
        ----------
        threshold : float
            The likelihood threshold T.

        Returns
        -------
        values, var_estimates : ndarray of shape (n,)
            -T for the ended repeats and the complete value otherwise, and
            the variance estimates.
        trial_value_sums, trial_var_sums : ndarray of shape (N,)
            The per-trial sums over the repeats that are not ended.
        """
        kept, v, s, tv, ts = self.end_complete_below(threshold)
        values = np.full(self.n, -threshold)
        values[kept] = v
        var_estimates = self.ended_var.copy()
        var_estimates[kept] = s
        return values, var_estimates, tv, ts


def _limited_estimates(
    K, completed, ended, ended_var, weights, threshold, max_time, n_name="n"
):
    """Estimates of a draw that the time limit stopped.

    Of the n repeats, n_e were ended by the likelihood threshold. Trial i
    has m_i completed counts in the other repeats, whose ``ibs_loglik``
    average is a_i. The log-likelihood estimate is
    ``(n_e / n) (-T) + (1 - n_e / n) sum_i w_i a_i``, and its variance
    estimate is ``sum(ended_var[ended]) / n**2 + (1 - n_e / n)**2
    sum_i w_i**2 S_i / m_i**2``, where S_i is the sum of the trial's
    ``ibs_var`` over its completed counts. Without ended repeats this is
    the weighted sum of each trial's average over its completed counts,
    and when every repeat is ended it is -T.

    Parameters
    ----------
    K : ndarray of int, shape (n, N)
        The counts; only the completed ones are read.
    completed : ndarray of bool, shape (n, N)
        Whether each count is complete.
    ended : ndarray of bool, shape (n,)
        Whether the likelihood threshold ended each repeat.
    ended_var : ndarray of shape (n,)
        The variance estimates of the ended repeats.
    weights : ndarray of shape (N,)
        The trial weights.
    threshold : float or None
        The likelihood threshold T, None when the draw has none (and so no
        ended repeat).
    max_time : float
        The time limit, which the error message names.
    n_name : str, optional
        The name of the number of repeats in the error message.

    Returns
    -------
    loglik, loglik_var : float
        The log-likelihood estimate and its variance estimate.
    trial_value_sums, trial_var_sums : ndarray of shape (N,)
        The unweighted sums of each trial's ``ibs_loglik`` and ``ibs_var``
        over its completed counts in the repeats that are not ended.
    trial_counts : ndarray of int64, shape (N,)
        m_i, the number of those counts.

    Raises
    ------
    IBSSamplingError
        If a repeat is not ended and a trial has no completed count in
        those repeats.
    """
    n = ended.size
    n_ended = int(np.count_nonzero(ended))
    kept = ~ended
    # A count of 1 adds exactly 0 to both sums, so the counts that are not
    # completed can stand at 1.
    counts = np.where(completed[kept], K[kept], 1)
    _, _, tv, ts = _estimates.repeat_estimates(counts, weights)
    m = np.count_nonzero(completed[kept], axis=0).astype(np.int64)
    loglik_var = float(np.sum(ended_var[ended])) / n**2
    if n_ended == n:
        return -threshold, loglik_var, tv, ts, m
    missing = np.flatnonzero(m == 0)
    if missing.size:
        if n_ended:
            where = (
                f"the {n - n_ended} of the {n_name} = {n} repeats that the "
                "likelihood threshold did not end"
            )
        else:
            where = f"the {n_name} = {n} repeats"
        raise IBSSamplingError(
            f"The time limit max_time = {max_time:g} s stopped the "
            f"sampling with no completed count in {where} for "
            f"{_name_trials(missing)}. IBS has no estimate for a trial "
            "without one: raise max_time."
        )
    frac = (n - n_ended) / n
    loglik = frac * float(np.sum(weights * tv / m))
    if n_ended:
        loglik -= n_ended / n * threshold
    loglik_var += frac**2 * float(np.sum(weights**2 * ts / m**2))
    return loglik, loglik_var, tv, ts, m


def _samples_per_trial(level, n_open, settings):
    """``ibslike.m``'s samples of each open trial in one call.

    ``min(max_samples, max(1, round(level)))``, then at most
    ``ceil(max_mem / n_open)``, with MATLAB's ``round``, which rounds
    halves away from zero: ``math.floor(level + 0.5)`` for the level, which
    is at least 1.
    """
    m = min(settings.max_samples, max(1, math.floor(level + 0.5)))
    return min(m, -(-settings.max_mem // n_open))


def sample(
    settings, theta, n, rng, *, vectorized=True, start=None, first=None
):
    """Draw n repeats of IBS at ``theta``.

    Parameters
    ----------
    settings : _Settings
        The simulator, the data and the sampling settings.
    theta : ndarray of shape (D,)
        The parameter vector, passed to every simulator call of the draw.
    n : int
        Repeats in the draw, at least 1.
    rng : numpy.random.Generator
        The generator, passed to every simulator call of the draw.
    vectorized : bool, optional
        True, the default, samples on the accelerated schedule; False
        requests one sample per open trial per call. See the Notes.
    start : float or None, optional
        The ``time.perf_counter()`` time from which ``settings.max_time``
        counts; None, the default, counts it from the start of the draw.
    first : _FirstRound or None, optional
        A simulator call already made for one sample of every trial, from
        :func:`first_round`, which the draw takes as its first round. When
        the schedule's first round requests one sample of every trial, it
        is that round; otherwise it is an extra round before it, after
        which the level does not grow. Whether the draw uses its samples
        thus never depends on their outcomes, which keeps the repeats
        independent even when the call's duration, which can decide
        ``vectorized``, depends on them.

    Returns
    -------
    draw : _Draw
        The counts, the per-repeat values and variance estimates, the
        draw's estimate, the per-trial sums, the ended repeats and the cost
        of the draw.

    Raises
    ------
    IBSSamplingError
        If a trial draws more than ``max_samples_per_trial * n`` samples,
        or the time limit stops the draw before a trial has completed a
        count in a repeat that the threshold did not end.
    ValueError
        If ``n`` is not an integer >= 1, or the simulator returns an array
        whose shape is not one that the responses take.
    TypeError
        If ``n`` is not a number, or the simulated rows are of a kind that
        NumPy never finds equal to the responses.

    Notes
    -----
    **Sampling.** A draw of n repeats needs n matches of every trial, or
    fewer when the likelihood threshold ends repeats. In each round, the
    trials that still need matches are open, and one simulator call
    requests m samples of each, in trial-major order
    (``idx = np.repeat(open_trials, m)``). The samples of a trial are
    taken in order; each match closes the trial's current count and
    starts the next. Samples after a trial's last needed match are
    surplus: they are counted in the cost and discarded.

    **Samples per call.** On the accelerated schedule, a round with
    ``n_open`` open trials requests ``m = min(max_samples, max(1,
    round(level)))`` samples of each, and at most ``ceil(max_mem /
    n_open)``, as ``ibslike.m`` does, with MATLAB's ``round``, which
    rounds halves away from zero. The level starts at ``initial_samples``
    (n by default) and is multiplied by ``acceleration`` after every round
    (or, with ``acceleration_threshold``, after every fast round) but an
    extra first round given as ``first``, and bounded by ``max_samples``,
    which changes no m. This default schedule
    depends only on the outcomes, so a seed reproduces a run. With
    ``vectorized=False``, every round requests one sample of each open
    trial, as ``ibslike.m``'s loop path does for one repeat at a time:
    at most N rows per call, and no surplus.

    **Output.** With the counts K of shape (n, N), repeat r's value is
    ``sum_i w_i ibs_loglik(K[r, i])`` and its variance estimate
    ``sum_i w_i**2 ibs_var(K[r, i])``. The per-trial outputs are the
    unweighted column sums of ``ibs_loglik(K)`` and ``ibs_var(K)`` over
    the repeats that the likelihood threshold did not end, which are all n
    without a threshold. All four come from
    :func:`pyibs._estimates.repeat_estimates`. The draw's estimate is the
    mean of the values, and its variance estimate the sum of the variance
    estimates over n**2. A draw holds its n x N counts in memory.

    **Cost.** The draw's ``calls`` counts the simulator calls, its
    ``samples`` the simulated rows, surplus included, and its ``seconds``
    the time spent inside the simulator; ``first`` counts as one of its
    calls. The samples drawn for the repeats that the likelihood threshold
    ended count in ``samples``.

    **Checks.** After every simulator call, a draw raises ``ValueError``
    if the simulator returned an array of a shape that the responses do
    not take: (r,) or (r, 1) for r requested rows of responses of one
    column, and (r, C) for responses of C > 1 columns; ``TypeError`` if
    the simulated rows and the responses are of two different kinds among
    text, bytes, and numbers or booleans, which NumPy never finds equal;
    and :class:`IBSSamplingError` if a trial has drawn more than
    ``max_samples_per_trial * n`` samples in the draw.

    **Time limit.** With a finite ``max_time``, the time is checked after
    every simulator call of the draw. Once ``max_time`` seconds have
    passed since ``start``, while trials still need matches, the draw
    stops: its ``timed_out`` is True, and its estimate is that of
    :func:`_limited_estimates`. Each trial's completed counts in the
    repeats that the threshold did not end are averaged, and a repeat that
    the threshold ended counts -T, as a whole. The repeats that the time
    limit left unfinished have no value of their own. A trial with no
    completed count raises :class:`IBSSamplingError`.

    **Likelihood threshold** ([1], Appendix C.1). With a threshold T, the
    sampler keeps a running bound B_r on each repeat's negative
    log-likelihood: the sum over the trials of
    ``w_i (digamma(k_i) - digamma(1))``, where k_i is the count
    ``K[r, i]`` of a trial that has completed repeat r, ``c_i + 1`` for a
    trial sampling it with open count c_i (its count will be at least
    that), and 1 for a trial that has not reached it. As the weights are
    >= 0, B_r never decreases while the repeat is sampled, and it reaches
    the repeat's complete negative log-likelihood -Y_r when every trial
    has completed it. After every round, including the last, every repeat
    with ``B_r > T`` is ended: the trials sampling it drop their open
    count and move on to the next repeat that is not ended, and the trials
    that reach it later skip it. A repeat is therefore ended exactly when
    ``-Y_r > T``. Its value is -T, so every repeat's value is
    ``max(Y_r, -T)``, a function of the repeat's own draws: the repeats
    stay independent. The variance estimate of an ended repeat is its
    bound's counterpart, ``sum_i w_i**2 ibs_var(k_i)`` with the counts k_i
    at the time it was ended. The value -T does not depend on the
    sampling schedule, but this variance estimate does: its counts are
    those at the end of the round after which the bound was checked, and
    a round of many samples per trial carries them further past T than a
    round of one. The draw's ``n_thresholded`` counts the ended repeats.
    The usual choice of T is the chance-level bound of [1], the negative
    log-likelihood of a model that assigns uniform probability to the
    possible responses: ``sum_i w_i log(n_i)`` with n_i possible responses
    on trial i, or ``N log 2`` for N unweighted binary choices.

    The clipped values are biased upward: the expectation of
    ``max(Y_r, -T)`` exceeds the log-likelihood by the expectation of
    ``max(-T - Y_r, 0)``. [1] notes that this bias is exponentially small
    in N when the log-likelihood lies well above -T.

    The per-trial outputs cover only the repeats that were not ended, n
    less ``n_thresholded``. These are the repeats whose values are at
    least -T, not a random sample of the repeats, so the per-trial sums
    divided by their number of repeats are biased estimates of each
    trial's log p_i, and their sum over the trials differs from the mean
    of the repeat values.

    ``ibslike.m`` keeps the partial counts of an ended repeat in its
    estimate, whose value then depends on the sampling schedule, and
    compares the unweighted sum of the trials' terms with
    ``NegLogLikeThreshold``. Its vectorized path checks only the lowest
    repeat still being sampled, with the open count ``c_i`` for the trials
    still sampling it, as the bound of [1] does; its loop path samples one
    repeat at a time and checks it after every call, with ``c_i + 1`` for
    the trials still sampling it, one term more, as here. This sampler
    checks every repeat, bounds the
    weighted sum, which is on the scale of a repeat's value, and returns
    -T for an ended repeat, as [1] does, so that every value is exactly
    ``max(Y_r, -T)``.
    """
    n = _check_count(n, "n")
    if start is None:
        start = time.perf_counter()
    threshold = settings.neg_loglik_threshold
    weights = settings.trial_weights
    counts = _MatchCounts(
        n, settings.n_trials, None if threshold is None else weights
    )
    limit = (
        None
        if settings.max_samples_per_trial is None
        else settings.max_samples_per_trial * n
    )
    timed = math.isfinite(settings.max_time)
    # Samples each trial has drawn in this draw, surplus included.
    trial_samples = np.zeros(settings.n_trials, dtype=np.int64)
    # Levels at or above max_samples give the same m, so bounding the level
    # by max_samples changes no round and keeps it finite.
    initial = (
        n if settings.initial_samples is None else settings.initial_samples
    )
    level = float(min(initial, settings.max_samples))
    calls, samples, seconds = 0, 0, 0.0
    timed_out = False
    while True:
        trials = counts.open_trials()
        if trials.size == 0:
            break
        if timed and calls and time.perf_counter() - start > settings.max_time:
            timed_out = True
            break
        m = (
            _samples_per_trial(level, trials.size, settings)
            if vectorized
            else 1
        )
        # The call given as first is the first round, whatever its outcomes,
        # so that using its samples never depends on them. When the
        # schedule's first round requests more than one sample per trial,
        # it is an extra round, after which the level does not grow.
        grow = True
        if calls == 0 and first is not None:
            grow = m == 1
            m = 1
            hits, elapsed = first.hits, first.elapsed
        else:
            hits, elapsed = _simulate(settings, theta, rng, trials, m)
        counts.absorb(trials, hits)
        if threshold is not None:
            counts.end_above(threshold)
        calls += 1
        samples += trials.size * m
        seconds += elapsed
        trial_samples[trials] += m
        if limit is not None:
            over = trials[trial_samples[trials] > limit]
            if over.size:
                raise _cap_error(
                    settings, n, limit, over, trial_samples[over], counts
                )
        if grow and (
            settings.acceleration_threshold is None
            or elapsed < settings.acceleration_threshold
        ):
            level = min(level * settings.acceleration, settings.max_samples)
    if timed_out:
        # The complete repeats keep their values, and the threshold's rule
        # ends those below -T; the unfinished repeats have none.
        if threshold is None:
            kept = counts.complete_repeats()
            kept_v, kept_s, _, _ = _estimates.repeat_estimates(
                counts.K[kept], weights
            )
        else:
            kept, kept_v, kept_s, _, _ = counts.end_complete_below(threshold)
        v = np.full(n, np.nan)
        s = np.full(n, np.nan)
        v[kept], s[kept] = kept_v, kept_s
        if threshold is not None:
            v[counts.ended] = -threshold
            s[counts.ended] = counts.ended_var[counts.ended]
        loglik, loglik_var, tv, ts, trial_counts = _limited_estimates(
            counts.K,
            counts.completed,
            counts.ended,
            counts.ended_var,
            weights,
            threshold,
            settings.max_time,
            settings.names[1],
        )
    else:
        if threshold is None:
            v, s, tv, ts = _estimates.repeat_estimates(counts.K, weights)
        else:
            v, s, tv, ts = counts.clipped_estimates(threshold)
        loglik = float(np.sum(v)) / n
        loglik_var = float(np.sum(s)) / n**2
        trial_counts = np.full(
            settings.n_trials, n - np.count_nonzero(counts.ended)
        )
    return _Draw(
        K=counts.K,
        values=v,
        var_estimates=s,
        loglik=loglik,
        loglik_var=loglik_var,
        trial_value_sums=tv,
        trial_var_sums=ts,
        trial_counts=trial_counts,
        ended=counts.ended,
        n_thresholded=int(np.count_nonzero(counts.ended)),
        timed_out=timed_out,
        calls=calls,
        samples=samples,
        seconds=seconds,
    )


def _cap_error(settings, n, limit, over, drawn, counts):
    """The error of a draw in which trials exceeded the cap.

    Parameters
    ----------
    settings : _Settings
        The settings of the draw, whose ``names`` name the cap's setting
        and the number of repeats.
    n : int
        Repeats in the draw.
    limit : int
        The cap times n.
    over : ndarray of int
        The trials over the cap, in order.
    drawn : ndarray of int
        Their samples in the draw.
    counts : _MatchCounts
        The draw's counts after the call that crossed the cap.
    """
    cap, n_name = settings.names
    over, drawn = over.tolist(), drawn.tolist()
    if len(over) == 1:
        which = (
            f"trial {over[0]} drew {drawn[0]} samples, more than "
            f"{cap} * {n_name} = {limit}."
        )
    else:
        listed = ", ".join(
            f"trial {i} drew {k}" for i, k in zip(over[:5], drawn[:5])
        )
        if len(over) > 5:
            listed += f", and {len(over) - 5} more"
        which = (
            f"{len(over)} trials drew more than {cap} * {n_name} = {limit} "
            f"samples: {listed}."
        )
    n_open = counts.open_trials().size
    if n_open:
        state = (
            f"{n_open} of {settings.n_trials} trials still need matches, "
            "and IBS returns no estimate for an incomplete repeat."
        )
    else:
        state = (
            "The simulator call that crossed the cap completed the "
            "draw, but a draw over the cap returns no estimate."
        )
    return IBSSamplingError(
        f"In a draw of {n} repeat{'' if n == 1 else 's'}, {which} "
        f"{state} Check that the simulator can produce every observed "
        f"response at this parameter vector, or raise {cap}."
    )


def _simulate(settings, theta, rng, trials, m):
    """Call the simulator for m samples of each trial, trial-major.

    Raises ``ValueError`` if the simulated array has a shape that the
    responses do not take, (r,) or (r, 1) for r requested rows of
    responses of one column and (r, C) for responses of C > 1 columns,
    and ``TypeError`` if its kind can never equal theirs.

    Returns
    -------
    hits : ndarray of bool, shape (trials.size, m)
        Whether each sample matches its trial's response.
    elapsed : float
        Seconds spent inside the simulator.
    """
    idx = np.repeat(trials, m)
    r = idx.size
    responses = settings.responses
    observed = responses[idx]
    design_rows = idx if settings.design is None else settings.design[idx]
    start = time.perf_counter()
    simulated = settings.simulator(theta, design_rows, rng)
    elapsed = time.perf_counter() - start
    simulated = np.asarray(simulated)
    if responses.ndim == 1 or responses.shape[1] == 1:
        if simulated.shape not in ((r,), (r, 1)):
            raise ValueError(
                f"The simulator was asked for {r} response"
                f"{'' if r == 1 else 's'} of one column "
                f"and returned an array of shape {simulated.shape}, where "
                f"it must return shape ({r},) or ({r}, 1)."
            )
        simulated, observed = simulated.reshape(r), observed.reshape(r)
    elif simulated.shape != observed.shape:
        raise ValueError(
            f"The simulator was asked for {r} response rows of "
            f"{responses.shape[1]} columns and returned an array of shape "
            f"{simulated.shape}, where it must return shape {observed.shape}."
        )
    _check_kinds(simulated, observed)
    hits = simulated == observed
    if hits.ndim == 2:
        hits = np.all(hits, axis=1)
    return hits.reshape(trials.size, m), elapsed

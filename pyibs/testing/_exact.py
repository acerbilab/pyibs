"""Exact IBS repeats and moments from known matching probabilities.

Under IBS, the matching count K of a trial with matching probability p is
geometric on {1, 2, ...} with parameter p. Drawing K directly gives repeats
distributed exactly as those of an IBS run with a simulator, without
simulating responses ([1]). The exact moments give the expected value and
the variance of one complete repeat.

References
----------
.. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
   efficient log-likelihood estimation with inverse binomial sampling.
   PLOS Computational Biology 16(12): e1008483.
   https://doi.org/10.1371/journal.pcbi.1008483
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import spence

from pyibs import _estimates
from pyibs._sampler import _check_count

_CHUNK_ELEMENTS = 10**6
"""Memory bound on the matching counts drawn at once.

A block holds ``max(1, _CHUNK_ELEMENTS // N)`` rows of N counts, so it has
at most ``_CHUNK_ELEMENTS`` counts when N <= ``_CHUNK_ELEMENTS``, and one
row of N counts otherwise.
"""

_MIN_TRIAL_PROB = 1e-15
"""The smallest matching probability that :func:`exact_draw` accepts.

NumPy draws geometric counts as int64 and clamps them at ``2**63 - 1``.
Below about p = 1e-18 the clamp is reached with non-negligible
probability, which biases the counts; at p = 1e-15 the mean count, 1/p,
is about 2**50, far below the clamp.
"""

_INT64_MAX = np.iinfo(np.int64).max


def _check_trial_probs(p):
    """Return matching probabilities as a 1-D float array in (0, 1].

    A scalar is taken as a single trial.

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, is empty, or has a value outside (0, 1].
    """
    p = np.array(p, dtype=float)
    if p.ndim == 0:
        p = p.reshape(1)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(
            "Matching probabilities must be a non-empty 1-D array, got "
            f"shape {p.shape}."
        )
    if not np.all((p > 0) & (p <= 1)):
        raise ValueError(
            "Matching probabilities must lie in (0, 1]; a probability of 0 "
            "makes an IBS repeat never end."
        )
    return p


def exact_loglik(p, weights=None):
    """Exact log-likelihood, the expected value of one IBS repeat.

    Parameters
    ----------
    p : array_like of shape (N,)
        Matching probabilities in (0, 1]. A scalar is one trial.
    weights : None, float or array_like of shape (N,), optional
        Trial weights, validated by
        :func:`pyibs._estimates.trial_weights`.

    Returns
    -------
    loglik : float
        ``sum(w * log(p))``.
    """
    p = _check_trial_probs(p)
    w = _estimates.trial_weights(weights, p.size)
    return float(np.sum(w * np.log(p)))


def exact_var(p, weights=None):
    """Exact variance of one complete IBS repeat.

    Parameters
    ----------
    p : array_like of shape (N,)
        Matching probabilities in (0, 1]. A scalar is one trial.
    weights : None, float or array_like of shape (N,), optional
        Trial weights, validated by
        :func:`pyibs._estimates.trial_weights`.

    Returns
    -------
    var : float
        ``sum(w**2 * Li2(1 - p))`` ([1], Eq 15), with
        ``Li2(1 - p) = scipy.special.spence(p)``.
    """
    p = _check_trial_probs(p)
    w = _estimates.trial_weights(weights, p.size)
    return float(np.sum(w**2 * spence(p)))


def _exact_sum(K):
    """Return the sum of a nonnegative int64 array as an exact Python int.

    The high and low 32-bit halves of the counts are summed separately, so
    neither sum can overflow int64 for fewer than 2**31 counts, and the
    total may exceed ``2**63 - 1``.
    """
    high = int(np.sum(K >> 32))
    low = int(np.sum(K & 0xFFFFFFFF))
    return (high << 32) + low


@dataclass(frozen=True, eq=False)
class ExactDraw:
    """What :func:`exact_draw` returns for n repeats of N trials.

    Attributes
    ----------
    values : ndarray of shape (n,)
        Each repeat's log-likelihood estimate.
    var_estimates : ndarray of shape (n,)
        Each repeat's variance estimate.
    trial_value_sums, trial_var_sums : ndarray of shape (N,)
        The unweighted sums over the repeats of each trial's
        ``ibs_loglik(K)`` and ``ibs_var(K)``.
    samples : int
        The sum of the matching counts, the simulator draws that an IBS
        run would take, as an exact integer.
    """

    values: np.ndarray
    var_estimates: np.ndarray
    trial_value_sums: np.ndarray
    trial_var_sums: np.ndarray
    samples: int

    @property
    def n(self):
        """Number of repeats."""
        return self.values.size


def exact_draw(p, n, rng, weights=None):
    """Draw n IBS repeats from geometric matching counts.

    Parameters
    ----------
    p : array_like of shape (N,)
        Matching probabilities of the N trials, in [1e-15, 1]. Smaller
        probabilities are rejected: NumPy's int64 geometric counts would be
        clamped, and so biased, below about 1e-18.
    n : int
        Repeats, at least 1.
    rng : numpy.random.Generator
        The generator of the counts.
    weights : None, float or array_like of shape (N,), optional
        Trial weights (``ibslike.m``'s ``TrialWeights``), finite and >= 0,
        validated by :func:`pyibs._estimates.trial_weights`.

    Returns
    -------
    draw : ExactDraw
        The per-repeat values and variance estimates, the per-trial sums
        and the sum of the counts.

    Raises
    ------
    ValueError
        If ``p`` is not a non-empty 1-D array in (0, 1], has a probability
        below 1e-15, or the weights are invalid or do not match its length.
    OverflowError
        If a count reaches the int64 maximum, where NumPy clamps it.

    Notes
    -----
    Repeat r's value is ``sum_i w_i ibs_loglik(K[r, i])`` and its variance
    estimate ``sum_i w_i**2 ibs_var(K[r, i])``. The per-trial outputs are
    the unweighted column sums of ``ibs_loglik(K)`` and ``ibs_var(K)``.
    All four come from :func:`pyibs._estimates.repeat_estimates`.

    The counts are drawn in row blocks of ``max(1, _CHUNK_ELEMENTS // N)``
    repeats, in order; the generator fills each block element by element
    in C order, and :func:`pyibs._estimates.repeat_estimates` reduces each
    row on its own, so the values and variance estimates do not depend on
    the block size.

    The draws have no likelihood threshold: the variance estimate of a
    thresholded repeat depends on the order in which a simulator is
    sampled, which exact draws do not have.
    """
    p = _check_trial_probs(p)
    if np.min(p) < _MIN_TRIAL_PROB:
        raise ValueError(
            "Exact draws need matching probabilities of at least "
            f"{_MIN_TRIAL_PROB:g}, got {float(np.min(p))!r}. NumPy draws "
            "geometric counts as int64 and clamps them at 2**63 - 1 "
            "(about 9.2e18), which biases the counts of probabilities "
            "below about 1e-18."
        )
    w = _estimates.trial_weights(weights, p.size)
    n = _check_count(n, "n")
    n_trials = p.size
    rows = max(1, _CHUNK_ELEMENTS // n_trials)
    values = np.empty(n)
    var_estimates = np.empty(n)
    trial_value_sums = np.zeros(n_trials)
    trial_var_sums = np.zeros(n_trials)
    samples = 0
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        K = rng.geometric(p, size=(hi - lo, n_trials))
        if np.any(K == _INT64_MAX):
            raise OverflowError(
                "A geometric count reached the int64 maximum 2**63 - 1, "
                "where NumPy clamps it; the draw would be biased."
            )
        v, s, tv, ts = _estimates.repeat_estimates(K, w)
        values[lo:hi] = v
        var_estimates[lo:hi] = s
        trial_value_sums += tv
        trial_var_sums += ts
        samples += _exact_sum(K)
    return ExactDraw(
        values=values,
        var_estimates=var_estimates,
        trial_value_sums=trial_value_sums,
        trial_var_sums=trial_var_sums,
        samples=samples,
    )

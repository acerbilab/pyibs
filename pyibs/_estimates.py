"""Inverse binomial sampling (IBS) estimates from matching counts.

The count formulas turn a matching count ``K`` (the number of simulator
draws up to and including the first match of one trial) into that trial's
contribution to one repeat's log-likelihood estimate and variance estimate
([1], Eqs 14 and 16). With trial weights ``w``, one repeat's value is
``sum_i w_i ibs_loglik(K_i)`` and its variance estimate
``sum_i w_i**2 ibs_var(K_i)``, as in MATLAB ``ibslike.m``;
:func:`repeat_estimates` computes both for a matrix of counts. The sampler
reduces its counts through it, and so do the exact draws of the tests.

References
----------
.. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
   efficient log-likelihood estimation with inverse binomial sampling.
   PLOS Computational Biology 16(12): e1008483.
   https://doi.org/10.1371/journal.pcbi.1008483
"""

import numbers

import numpy as np
from scipy.special import digamma, polygamma


def ibs_loglik(K):
    """IBS log-likelihood estimate of one trial from its matching count.

    Parameters
    ----------
    K : int or array_like of int
        Matching counts, each at least 1.

    Returns
    -------
    loglik : float or ndarray
        ``digamma(1) - digamma(K)`` elementwise, which equals
        ``-sum(1/k for k in 1..K-1)``; exactly 0 for ``K = 1``.
    """
    # SciPy evaluates integer types narrower than 32 bits in float32.
    return digamma(1) - digamma(np.asarray(K, dtype=float))


def ibs_var(K):
    """IBS variance estimate of one trial from its matching count.

    Parameters
    ----------
    K : int or array_like of int
        Matching counts, each at least 1.

    Returns
    -------
    var : float or ndarray
        ``polygamma(1, 1) - polygamma(1, K)`` elementwise, which equals
        ``sum(1/k**2 for k in 1..K-1)``; exactly 0 for ``K = 1``.
    """
    return polygamma(1, 1) - polygamma(1, np.asarray(K, dtype=float))


def trial_weights(weights, n_trials):
    """Validate per-trial weights (``ibslike.m``'s ``TrialWeights``).

    Parameters
    ----------
    weights : None, float or array_like of shape (n_trials,)
        None gives unit weights, and a scalar is broadcast to all trials.
    n_trials : int
        Number of trials N, at least 1.

    Returns
    -------
    weights : ndarray of shape (n_trials,)
        A new float64 array.

    Raises
    ------
    TypeError
        If the weights are not real numbers: booleans and strings,
        numeric or not, are refused.
    ValueError
        If the shape is wrong, or a weight is not finite or is negative.
    """
    if (
        isinstance(n_trials, bool)
        or not isinstance(n_trials, numbers.Integral)
        or n_trials < 1
    ):
        raise ValueError(
            f"n_trials must be an integer >= 1, got {n_trials!r}."
        )
    n_trials = int(n_trials)
    if weights is None:
        return np.ones(n_trials)
    w = np.asarray(weights)
    if w.dtype.kind not in "iuf":
        raise TypeError(
            "Trial weights must be real numbers, not booleans or strings, "
            f"got an array of dtype {w.dtype}."
        )
    w = np.array(w, dtype=float)
    if w.ndim == 0:
        w = np.full(n_trials, float(w))
    elif w.shape != (n_trials,):
        raise ValueError(
            f"Trial weights must be a scalar or have shape ({n_trials},), "
            f"got shape {w.shape}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Trial weights must be finite.")
    if np.any(w < 0):
        raise ValueError("Trial weights must be >= 0.")
    return w


def repeat_estimates(K, weights):
    """Per-repeat IBS estimates and per-trial sums from matching counts.

    Parameters
    ----------
    K : array_like of int, shape (n, N)
        Matching counts, each at least 1; row r holds repeat r's count of
        each of the N trials.
    weights : ndarray of shape (N,)
        Trial weights, as returned by :func:`trial_weights`.

    Returns
    -------
    values : ndarray of shape (n,)
        ``sum_i w_i ibs_loglik(K[r, i])``, each repeat's log-likelihood
        estimate.
    var_estimates : ndarray of shape (n,)
        ``sum_i w_i**2 ibs_var(K[r, i])``, each repeat's variance estimate.
    trial_value_sums, trial_var_sums : ndarray of shape (N,)
        The unweighted column sums of ``ibs_loglik(K)`` and ``ibs_var(K)``.

    Raises
    ------
    ValueError
        If ``K`` is not 2-D, its columns do not match the weights, its
        dtype is not an integer type, or a count is below 1.

    Notes
    -----
    The count formulas are tabulated at the counts 1 to
    ``max(1, min(max(K), K.size // 2))``, and looked up for the counts in
    that range when there are at least as many of them as table entries;
    the other counts are evaluated directly. ``ibslike.m`` tabulates them
    at the counts 1 to ``max(K)`` and looks every count up.

    Every term equals :func:`ibs_loglik` or :func:`ibs_var` of its count
    bitwise, whether looked up or evaluated directly, and rows are reduced
    as ``np.sum(x * w, axis=1)``, never as a matrix product. A row's terms
    and its summation then do not depend on the other rows, so a caller
    that reduces its repeats in blocks gets bitwise the same values
    whatever the block size.
    """
    K = np.asarray(K)
    w = np.asarray(weights, dtype=float)
    if K.ndim != 2 or w.shape != (K.shape[1],):
        raise ValueError(
            "K must have shape (n, N) and the weights shape (N,), got "
            f"{K.shape} and {w.shape}."
        )
    if K.dtype.kind not in "iu":
        raise ValueError(
            f"Matching counts must be integers, got dtype {K.dtype}."
        )
    if K.size and K.min() < 1:
        raise ValueError("Matching counts must be at least 1.")
    loglik, var = _count_terms(K)
    return (
        np.sum(loglik * w, axis=1),
        np.sum(var * w**2, axis=1),
        np.sum(loglik, axis=0),
        np.sum(var, axis=0),
    )


def _count_terms(K):
    """``ibs_loglik(K)`` and ``ibs_var(K)``, looked up in tables of both.

    ``K`` is an integer array of counts, each at least 1. The tables hold
    the formulas at the counts 1 to ``max(1, min(max(K), K.size // 2))``.
    An entry costs about as much as evaluating the formulas at one count,
    so the tables are used only when at least as many counts fall within
    them as they have entries, and the other counts are evaluated
    directly; otherwise every count is.
    """
    if K.size == 0:
        return np.zeros(K.shape), np.zeros(K.shape)
    k_max = int(K.max())
    size = max(1, min(k_max, K.size // 2))
    large = None
    if k_max > size:
        large = K > size
        if K.size - np.count_nonzero(large) < size:
            return ibs_loglik(K), ibs_var(K)
    counts = np.arange(1, size + 1)
    index = (K if large is None else np.minimum(K, size)) - 1
    loglik = ibs_loglik(counts)[index]
    var = ibs_var(counts)[index]
    if large is not None:
        loglik[large] = ibs_loglik(K[large])
        var[large] = ibs_var(K[large])
    return loglik, var

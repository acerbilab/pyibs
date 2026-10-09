"""Simulators and helpers shared by the tests of the sampler."""

import math
from types import SimpleNamespace

import numpy as np

from pyibs import _sampler

# Match probabilities of the Bernoulli simulator's 20 trials.
P = np.linspace(0.05, 1, 20)


def bernoulli(theta, idx, rng):
    """Match trial i with probability P[i]; the responses are all ones."""
    return (rng.random(len(idx)) < P[idx]).astype(float)


def never_matches(theta, idx, rng):
    """Return 0 for every requested trial; the responses are all ones."""
    return np.zeros(len(idx))


class ScriptedSimulator:
    """Simulator returning predetermined responses per trial, in order.

    It ignores ``theta`` and ``rng``: the k-th sample requested of trial i
    is ``streams[i][k]``. With a design, the trial index is read from the
    first design column.

    Attributes
    ----------
    requests : list of ndarray
        The trial indices of every call, in order.
    """

    def __init__(self, streams, design=False):
        self.streams = [np.asarray(s) for s in streams]
        self.design = design
        self.pos = np.zeros(len(self.streams), dtype=int)
        self.requests = []

    def __call__(self, theta, design_rows, rng):
        rows = np.asarray(design_rows)
        idx = rows[:, 0].astype(int) if self.design else rows
        self.requests.append(idx.copy())
        out = []
        for i in idx:
            out.append(self.streams[i][self.pos[i]])
            self.pos[i] += 1
        return np.array(out)


def draw(settings, n, seed):
    """Return a draw of ``n`` repeats with ``settings``.

    The draw is made at ``np.zeros(1)`` with a generator seeded by
    ``seed``.
    """
    return _sampler.sample(
        settings, np.zeros(1), n, np.random.default_rng(seed)
    )


def cost(result):
    """The cost of a draw without its wall time, which no seed reproduces.

    Returns ``(n, calls, samples)``.
    """
    return result.n, result.calls, result.samples


def summary(result):
    """What the tests read from one draw's result.

    ``result`` is a draw of :func:`pyibs._sampler.sample`, or of
    :func:`pyibs.testing._exact.exact_draw`, which ends no repeats.

    Returns
    -------
    summary : types.SimpleNamespace
        ``n``, the repeats; ``n_thresholded``, those that the likelihood
        threshold ended; ``trial_repeats``, those that it did not end;
        ``mean``, the mean of the repeat values, and ``se``, its nominal
        standard error ``sqrt(sum(var_estimates) / n**2)``; ``trial_loglik``,
        the per-trial value sums divided by ``trial_repeats``, and
        ``trial_nominal_var``, the per-trial variance sums divided by
        ``trial_repeats**2``, both None when ``trial_repeats`` is 0.
    """
    n = result.n
    n_thresholded = getattr(result, "n_thresholded", 0)
    trial_repeats = n - n_thresholded
    if trial_repeats:
        trial_loglik = result.trial_value_sums / trial_repeats
        trial_nominal_var = result.trial_var_sums / trial_repeats**2
    else:
        trial_loglik = trial_nominal_var = None
    return SimpleNamespace(
        n=n,
        n_thresholded=n_thresholded,
        trial_repeats=trial_repeats,
        mean=float(np.sum(result.values)) / n,
        se=math.sqrt(float(np.sum(result.var_estimates)) / n**2),
        trial_loglik=trial_loglik,
        trial_nominal_var=trial_nominal_var,
    )

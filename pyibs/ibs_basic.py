"""A bare-bone implementation of inverse binomial sampling, for teaching.

:func:`ibs_basic` follows ``ibs_basic.m`` of MATLAB IBS
(https://github.com/acerbilab/ibs). :class:`pyibs.IBS` is the
implementation to use.
"""

import numpy as np

from pyibs import _sampler
from pyibs.ibs import _rng, _takes_rng


def ibs_basic(sample_from_model, theta, R, S=None, *, random_seed=None):
    """Estimate a log-likelihood by inverse binomial sampling, one trial at
    a time.

    A slow, bare-bone implementation of IBS ([1]), which should be used
    only for didactic purposes: for every trial in turn, it simulates
    responses one at a time until one matches the observed response, and
    adds the trial's IBS estimate. :class:`pyibs.IBS` is the implementation
    to use.

    Parameters
    ----------
    sample_from_model : callable
        The simulator, ``sample_from_model(theta, s)``, or
        ``sample_from_model(theta, s, rng=rng)`` when it has a parameter
        named ``rng``, which then receives the generator of the call. It
        returns one simulated response: a row of ``R``. ``s`` is the row
        ``S[i]`` of trial i, or the trial's 0-based index i when ``S`` is
        None.
    theta : array_like
        The parameter vector, passed to the simulator as given.
    R : array_like of shape (N,) or (N, C)
        The observed responses, one row per trial. A simulated response
        matches a trial's only when every column agrees.
    S : array_like of shape (N, ...), optional
        The design of each trial, one row per trial. None, the default,
        passes the trial index instead.
    random_seed : None, int, numpy.random.SeedSequence or \
numpy.random.Generator, optional
        The seed of the generator passed to the simulator, as
        ``random_seed`` of :class:`pyibs.IBS`.

    Returns
    -------
    L : float
        The IBS estimate of the log-likelihood (not its negative).

    Raises
    ------
    ValueError
        If ``R`` is not a non-empty array of shape (N,) or (N, C), or holds
        a NaN, which no simulated response equals; or if ``S`` does not
        have N rows.
    TypeError
        If the simulator returns a response of a kind that NumPy never
        finds equal to ``R``, such as text for numeric responses.

    References
    ----------
    .. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
       efficient log-likelihood estimation with inverse binomial sampling.
       PLOS Computational Biology 16(12): e1008483.
       https://doi.org/10.1371/journal.pcbi.1008483
    """
    R = _sampler._check_responses(np.atleast_1d(R), "R")
    N = R.shape[0]
    S = _sampler._check_design(S, N, "S")
    rng = _rng(random_seed)
    if _takes_rng(sample_from_model):

        def draw(s):
            return sample_from_model(theta, s, rng=rng)

    else:

        def draw(s):
            return sample_from_model(theta, s)

    def simulate(s):
        r = np.asarray(draw(s))
        # A response of a kind that NumPy never finds equal to R, such as
        # text for numbers, could never match.
        _sampler._check_kinds(r, R)
        return r

    L = np.zeros(N)
    for i in range(N):  # Loop over all trials (rows)
        s = i if S is None else S[i]
        K = 1
        while not np.all(simulate(s) == R[i]):
            K += 1  # Sample until the generated response is a match
        L[i] = 0.0 - np.sum(1 / np.arange(1, K))  # IBS estimator of trial i
    return float(np.sum(L))  # Summed log-likelihood

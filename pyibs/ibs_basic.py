"""A minimal implementation of inverse binomial sampling, for teaching.

:func:`ibs_basic` follows ``ibs_basic.m`` of MATLAB IBS
(https://github.com/acerbilab/ibs). :class:`pyibs.IBS` is the
implementation for fitting models.
"""

import numpy as np

from pyibs import _sampler
from pyibs.ibs import _rng, _takes_rng


def ibs_basic(sample_from_model, theta, R, S=None, *, random_seed=None):
    """Estimate a log-likelihood by inverse binomial sampling, one trial at
    a time.

    This teaching implementation of IBS ([1]_) processes trials in order.
    For each trial, it simulates one response at a time until a response
    matches the observation, then adds that trial's log-likelihood
    estimate to the total. Use :class:`pyibs.IBS` for fitting models: it
    samples every open trial in each simulator call, estimates the
    variance, and has a sample cap, a time limit and a likelihood
    threshold.

    Parameters
    ----------
    sample_from_model : callable
        Called as ``sample_from_model(theta, s)``. If it has a parameter
        named ``rng`` that accepts a keyword, the call also supplies its
        random generator as ``rng=rng``. Here ``s`` is ``S[i]``, or the
        0-based trial index i when ``S`` is None.

        Return one independent simulated response with shape (C,) or
        (1, C), where C is the number of response columns. A scalar is
        also accepted for a single-column response.
    theta : array_like
        The parameter vector, passed to the simulator as given.
    R : array_like of shape (N,) or (N, C)
        Observed responses, one row per trial; a scalar represents one
        trial. A simulated response matches only if every column agrees.
        For responses that mix numbers and text, use ``dtype=object`` for
        both observed and simulated arrays. Otherwise NumPy converts the
        numbers to text: simulated text against observed objects raises
        ``TypeError``, and observed text never matches simulated objects,
        so the sampling never ends.
    S : array_like of shape (N, ...), optional
        Experimental conditions or other simulator inputs, one row per
        trial. None (the default) passes the trial index instead.
    random_seed : None, int, numpy.random.SeedSequence or \
numpy.random.Generator, optional
        Seed for the generator passed to the simulator. Accepts the same
        values as ``random_seed`` of :class:`pyibs.IBS`.

    Returns
    -------
    L : float
        The IBS estimate of the log-likelihood (not its negative).

    Raises
    ------
    ValueError
        If ``R`` is empty, has more than two dimensions, or contains a
        NaN, an element not equal to itself, which no simulated response
        matches; if ``S`` does not have N rows; or if the simulator returns
        a response of another shape.
    TypeError
        If the simulator returns a response of a kind that cannot match
        ``R``, such as text for numeric responses.

    Notes
    -----
    There is no sample cap or time limit. Sampling continues indefinitely
    if the simulator cannot produce an observed response.

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

    object_kinds = _sampler._object_kinds(R)
    columns = 1 if R.ndim == 1 else R.shape[1]
    valid_shapes = {(columns,), (1, columns)}
    if columns == 1:
        valid_shapes.add(())

    def simulate(i, s):
        r = np.asarray(draw(s))
        if r.shape not in valid_shapes:
            raise ValueError(
                "sample_from_model must return one response with shape "
                f"({columns},) or (1, {columns})"
                + (
                    ", or a scalar for one response column"
                    if columns == 1
                    else ""
                )
                + f"; got shape {r.shape}."
            )
        r = r.reshape(columns)
        # A response of a kind that NumPy never finds equal to R, such as
        # text for numbers, could never match.
        _sampler._check_kinds(
            r,
            R[i : i + 1],
            None if object_kinds is None else object_kinds[i : i + 1],
        )
        return r

    L = np.zeros(N)
    for i in range(N):  # Loop over all trials (rows)
        s = i if S is None else S[i]
        K = 1
        while not np.all(simulate(i, s) == R[i]):
            K += 1  # Sample until the generated response is a match
        L[i] = 0.0 - np.sum(1 / np.arange(1, K))  # IBS estimator of trial i
    return float(np.sum(L))  # Summed log-likelihood

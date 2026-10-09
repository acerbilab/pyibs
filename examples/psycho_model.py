"""The orientation discrimination model of the IBS paper.

A psychometric function model of a simple orientation discrimination task
([1], "Orientation discrimination" in Results), after ``psycho_gen.m`` and
``psycho_nll.m`` of MATLAB IBS (https://github.com/acerbilab/ibs). On each
trial, an observer sees a stimulus of orientation S (in degrees) and reports
whether it is tilted rightwards (1) or leftwards (-1). The observer's
measurement is S plus Gaussian noise of SD ``sigma``; the observer reports
rightwards when the measurement is at least ``bias``, and on a fraction
``lapse`` of the trials responds at random.

The parameter vector is ``theta = (log(sigma), bias, lapse)``.
:func:`psycho_generator` simulates responses, and :func:`psycho_neg_logl`
gives the exact negative log-likelihood. The model is simple enough to have
a closed-form likelihood, which one should use whenever there is one; it
serves to check IBS against it.

References
----------
.. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
   efficient log-likelihood estimation with inverse binomial sampling.
   PLOS Computational Biology 16(12): e1008483.
   https://doi.org/10.1371/journal.pcbi.1008483
"""

import numpy as np
from scipy.special import ndtr


def psycho_generator(theta, S, rng):
    """Simulate responses of the orientation discrimination model.

    Parameters
    ----------
    theta : array_like of shape (3,)
        ``(log(sigma), bias, lapse)``: the log of the sensory noise's SD,
        the bias and the lapse rate.
    S : array_like
        The stimulus orientation of each trial, in degrees.
    rng : numpy.random.Generator
        The generator of the simulation's draws.

    Returns
    -------
    R : ndarray of the shape of ``S``
        The responses, 1 for rightwards and -1 for leftwards, as floats.
    """
    sigma, bias, lapse = np.exp(theta[0]), theta[1], theta[2]
    S = np.asarray(S, dtype=float)
    # Noisy measurement: the orientation plus Gaussian noise.
    X = S + sigma * rng.standard_normal(S.shape)
    # Decision rule: rightwards (1) if the measurement is at least the bias.
    R = np.where(X >= bias, 1.0, -1.0)
    # Lapses: on these trials, the response is given at random.
    lapse_idx = rng.random(S.shape) < lapse
    R[lapse_idx] = 2.0 * rng.integers(2, size=np.count_nonzero(lapse_idx)) - 1
    return R


def psycho_neg_logl(theta, S, R):
    """Negative log-likelihood of the orientation discrimination model.

    Parameters
    ----------
    theta : array_like of shape (3,)
        ``(log(sigma), bias, lapse)``, as for :func:`psycho_generator`.
    S : array_like
        The stimulus orientation of each trial, in degrees.
    R : array_like of the size of ``S``
        The responses, 1 for rightwards and -1 for leftwards, one per
        stimulus, in the order of ``S``.

    Returns
    -------
    neg_logl : float
        The exact negative log-likelihood of the responses.
    """
    sigma, bias, lapse = np.exp(theta[0]), theta[1], theta[2]
    S = np.asarray(S, dtype=float)
    # Responses of the shape of S, so that a column and a vector of the
    # same trials do not broadcast against each other.
    R = np.asarray(R).reshape(S.shape)
    # Probability of each observed response (closed form).
    z = (S - bias) / sigma
    p = lapse / 2 + (1 - lapse) * ((R == -1) * ndtr(-z) + (R == 1) * ndtr(z))
    return float(-np.sum(np.log(p)))

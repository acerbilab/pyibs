import numpy as np


def ibs_basic(sample_from_model, theta, R, S=None):
    """
    Calculates an unbiased estimate of the log-likelihood for the simulated model and data using inverse binomial sampling.
    This is a slow, bare bone implementation of IBS which should be used only for didactic purposes.

    Parameters
    ----------
    sample_from_model: callable
        Callable that simulates the model's responses. Takes as input a vector of parameters theta and a row of the stimulus matrix,
        and generates one row of the response matrix.
    theta: np.array
        The parameter vector.
    R: np.array
        The response matrix, each row corresponds to one observation and each column to a response feature.
    S: np.array, optional
        The stimulus matrix, each row corresponds to one observation, default = None.

    Returns
    ----------
    L: float
        The log-likelihood.
    """
    N = len(R)
    L = np.zeros(N)
    for i in range(N):
        K = 1
        while not (sample_from_model(theta, S[i]) == R[i]).all():
            K += 1  # Sample until the generated response is a match
        L[i] = -np.sum(1 / np.arange(1, K))  # IBS estimator for the i-th trial

    return np.sum(L)  # Return summed log-likelihood

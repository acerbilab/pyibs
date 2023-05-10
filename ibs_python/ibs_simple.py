import numpy as np


def ibs_basic(fun, theta, R, S=None):
    """Returns an unbiased estimate L of the log-likelihood for the
    simulated model and data, calculated using inverse binomial sampling.
    This is a slow, bare bone implementation of IBS which should be used only
    for didactic purposes.

    Input:
    fun: function
        A function handle to a function that simulates the model's responses. Takes as input
        a vector of parameters 'theta' and a row of 'S', and generates one row of the
        response matrix.
    theta: np.array
        used as input to fun to generate the response
    R: np.array
        response matrix, each row corrisponds to one observation and
        each column to a response feature
    S: np.array, optional
        stimulus matrix, each row corrisponds to one observation

    Returns:
    L : float
        The log-likelihood.
    """
    N = len(R)
    L = np.zeros(N)
    for i in range(N):
        K = 1
        while not (fun(theta, S[i]) == R[i]).all():
            K += 1  # Sample until the generated response is a match
        L[i] = -np.sum(1 / np.arange(1, K))  # IBS estimator for the i-th trial

    return np.sum(L)  # Return summed log-likelihood

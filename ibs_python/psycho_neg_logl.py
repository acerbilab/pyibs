import numpy as np
from scipy.stats import norm


def psycho_neg_logl(theta, S, R):
    """
    Computes negative log-likelihood for psychometric function model (simple orientation discrimination task).

    Parameters:
    ----------
    S: np.array
        The stimulus orientation in degrees for each trial.
    theta: np.array
        The parameter vector, with theta(0) as eta=log(sigma), the log of the sensory noise,
        theta(1) as the bias term, theta(2) as the lapse rate.
    R: np.array
        The responses for each trial, 1 for "rightwards" and -1 for "leftwards".

    Returns:
    ----------
    L: float
        The negative log-likelihood.
    """

    # Extract model parameters
    sigma = np.exp(theta[0])
    bias = theta[1]
    lapse = theta[2]

    # Likelihood per trial (analytical solution)
    p_vec = lapse / 2 + (1 - lapse) * (
        (R == -1) * norm.cdf(-(S - bias) / sigma)
        + (R == 1) * norm.cdf((S - bias) / sigma)
    )

    # Total negative log-likelihood
    L = -np.sum(np.log(p_vec))

    return L

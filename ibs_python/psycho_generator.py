import numpy as np
import numpy.random as npr


def psycho_generator(theta, S):
    """
    Generate responses for psychometric function model (simple orientation discrimination task).

    Inputs:
    S: np.array
        The stimulus orientation in degrees for each trial.
    theta: np.array with 3 elements
        The parameter vector, with theta(0) as eta=log(sigma), the log of the sensory noise,
        theta(1) as the bias term, theta(2) as the lapse rate.

    Returns:
    R: np.array
        The responses for each trial, 1 for "rightwards" and -1 for "leftwards".
    """
    # Extract model parameters.
    sigma = np.exp(theta[0])
    bias = theta[1]
    lapse = theta[2]

    # Add Gaussian noise to true orientations S to simulate noisy measurements.
    X = S + sigma * npr.randn(np.size(S))

    # The response is 1 for "rightwards" if the internal measurement is larger than the bias term,
    # -1 for "leftwards" otherwise.
    R = np.zeros(np.size(S))
    R[X >= bias] = 1
    R[X < bias] = -1

    # Choose trials in which subject lapses, response there is given at chance.
    lapse_idx = npr.rand(np.size(S)) < lapse
    # Random responses (equal probability of 1 or -1).
    lapse_val = npr.randint(2, size=np.sum(lapse_idx)) * 2 - 1
    R[lapse_idx] = lapse_val

    return R

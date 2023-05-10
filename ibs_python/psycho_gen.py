import numpy as np
import numpy.random as npr


def psycho_gen(theta, S):
    """
    Generate responses for psychometric function model
    (simple orientation discrimination task).

    Inputs:
    S: np.array
        stimulus orientation (in deg) for each trial
    theta: np.array with 3 elements
        model parameter vector, with theta(0) as eta=log(sigma), the log of the sensory noise,
        theta(1) as the bias term,
        theta(2) as the lapse rate

    Returns:
    R: np.array
        responses per trial, 1 for "rightwards" and -1 for "leftwards"
    """
    # Extract model parameters
    sigma = np.exp(theta[0])
    bias = theta[1]
    lapse = theta[2]

    # Noisy measurement
    X = S + sigma * npr.randn(np.size(S))

    # Decision rule
    R = np.zeros(np.size(S))
    R[X >= bias] = 1
    R[X < bias] = -1

    # Lapses
    lapse_idx = npr.rand(np.size(S)) < lapse
    lapse_val = npr.randint(2, size=np.sum(lapse_idx)) * 2 - 1
    R[lapse_idx] = lapse_val

    return R

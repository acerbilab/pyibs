"""The example model's fit of ``ibs_example.m``, shared by the integration
tests.

``ibs_example.m`` of MATLAB IBS fits its orientation discrimination data
set with BADS and VBMC, an IBS estimate as their target. Its data set, as
``pyibs/testing/test_examples.py`` draws it, its bounds and its plausible
bounds are here, with the exact maximum-likelihood point, which the closed
form of the likelihood gives.
"""

import math

import numpy as np
from scipy.optimize import minimize

from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

SEED = 20261009
N_TRIALS = 600
# ibs_example.m's generating parameters: log(sigma), bias and lapse.
THETA_TRUE = np.array([math.log(1.0), 0.2, 0.03])
# ibs_example.m's hard and plausible bounds.
LB = np.array([math.log(0.1), -2.0, 0.01])
UB = np.array([math.log(10.0), 2.0, 1.0])
PLB = np.array([math.log(0.2), -1.0, 0.02])
PUB = np.array([math.log(5.0), 1.0, 0.2])
# Repeats per estimate: an SD of about 1 near the maximum-likelihood point,
# the noise that PyBADS and PyVBMC handle best.
NUM_REPS = 100


def data():
    """``ibs_example.m``'s data set: 600 orientations and their responses."""
    rng = np.random.default_rng(SEED)
    S = 3 * rng.standard_normal((N_TRIALS, 1))
    R = psycho_generator(THETA_TRUE, S, rng)
    return S, R


def exact_ml(S, R, starts=10):
    """The maximum-likelihood point of the closed form, and its value.

    L-BFGS-B within the hard bounds, from the generating parameters, the
    centre of the plausible box and ``starts`` seeded points in it; the
    best of the runs.
    """
    rng = np.random.default_rng(SEED)
    x0s = [THETA_TRUE, (PLB + PUB) / 2]
    x0s += list(PLB + rng.random((starts, PLB.size)) * (PUB - PLB))
    best = None
    for x0 in x0s:
        res = minimize(
            psycho_neg_logl,
            x0,
            args=(S, R),
            method="L-BFGS-B",
            bounds=list(zip(LB, UB)),
        )
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)

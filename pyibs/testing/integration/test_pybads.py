"""PyBADS fits the example model with an IBS estimate as its noisy target.

Run on its own, with PyBADS 1.5 or later installed:
``pytest -m integration pyibs/testing/integration/test_pybads.py -s -v``.
"""

from importlib.metadata import version

import numpy as np
import pytest

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl
from pyibs.testing.integration import _psycho_fit as fit

pybads = pytest.importorskip("pybads")

pytestmark = pytest.mark.integration


def test_pybads_reaches_the_maximum_likelihood():
    S, R = fit.data()
    ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=fit.SEED)

    def target(theta):
        # PyBADS minimizes, and takes the pair (value, SD) as a tuple.
        return ibs(theta, num_reps=fit.NUM_REPS, additional_output="std")

    # A start drawn in the plausible box, as in ibs_example.m.
    rng = np.random.default_rng(fit.SEED)
    x0 = fit.PLB + rng.random(fit.PLB.size) * (fit.PUB - fit.PLB)
    bads = pybads.BADS(
        target,
        x0,
        fit.LB,
        fit.UB,
        fit.PLB,
        fit.PUB,
        options={"specify_target_noise": True, "random_seed": fit.SEED},
    )
    result = bads.optimize()
    theta_ml, nll_min = fit.exact_ml(S, R)
    nll = psycho_neg_logl(result["x"], S, R)
    print(
        f"PyBADS {version('pybads')}: x = {result['x']}, exact negative "
        f"log-likelihood {nll:.4f}; exact minimum {nll_min:.4f} at "
        f"{theta_ml}; {result['func_count']} evaluations"
    )
    assert nll_min - 1e-6 <= nll <= nll_min + 1

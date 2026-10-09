"""PyVBMC infers the example model's posterior with an IBS estimate as its
noisy target.

Run on its own, with PyVBMC 1.5 or later installed:
``pytest -m integration pyibs/testing/integration/test_pyvbmc.py -s -v``.
"""

import inspect
from importlib.metadata import version

import numpy as np
import pytest

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator
from pyibs.testing.integration import _psycho_fit as fit

pyvbmc = pytest.importorskip("pyvbmc")
# PyVBMC takes a seed from 1.5 on. The test checks the parameter rather
# than the version, which an install from an untagged checkout of PyVBMC
# reads as 1.0.5.devN.
if "seed" not in inspect.signature(pyvbmc.VBMC).parameters:
    pytest.skip("needs PyVBMC 1.5 or later", allow_module_level=True)

pytestmark = pytest.mark.integration


def test_pyvbmc_posterior_holds_the_maximum_likelihood():
    from pyvbmc.priors import Trapezoidal

    S, R = fit.data()
    ibs = IBS(
        psycho_generator, R, S, vectorized=True, random_seed=fit.SEED_IBS
    )

    def log_likelihood(theta):
        # With a separate prior, PyVBMC takes the log-likelihood and its SD
        # and adds the log prior itself.
        return ibs(
            theta,
            num_reps=fit.NUM_REPS,
            additional_output="std",
            return_positive=True,
        )

    # ibs_example.m's prior, flat on the plausible box and falling linearly
    # to 0 at the hard bounds, and its start, the centre of the box.
    prior = Trapezoidal(fit.LB, fit.PLB, fit.PUB, fit.UB)
    x0 = (fit.PLB + fit.PUB) / 2
    vbmc = pyvbmc.VBMC(
        log_likelihood,
        x0,
        fit.LB,
        fit.UB,
        fit.PLB,
        fit.PUB,
        options={"specify_target_noise": True},
        prior=prior,
        seed=fit.SEED_FIT,
    )
    vp, results = vbmc.optimize()
    mean, cov = vp.moments(cov_flag=True)
    mean = np.ravel(mean)
    sd = np.sqrt(np.diag(cov))
    theta_ml, _ = fit.exact_ml(S, R)
    print(
        f"PyVBMC {version('pyvbmc')}: ELBO {results['elbo']:.4f} +- "
        f"{results['elbo_sd']:.4f}, {results['func_count']} evaluations; "
        f"posterior mean {mean}, SD {sd}; exact maximum-likelihood point "
        f"{theta_ml}; distance in SDs {np.abs(mean - theta_ml) / sd}"
    )
    assert np.isfinite(results["elbo"])
    assert np.all(np.abs(mean - theta_ml) <= 3 * sd)

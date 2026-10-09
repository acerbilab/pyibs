"""Ports of the self-tests of MATLAB ``ibslike.m`` (``ibslike('test')``).

The simulator is ``ibslike.m``'s ``fun = @(x,dmat) rand(size(dmat)) < x``:
every requested sample is a Bernoulli draw with probability ``theta[0]``,
so a response ``True`` matches with probability p and ``False`` with
probability 1 - p. The tests are seeded, and ``ibslike.m``'s negative
log-likelihood is the negative of the estimates here.

``ibslike.m`` makes the checks of ``runtest1`` and ``runtest3`` on one set
of estimates, and a correct sampler fails either check by chance at about
one seed in a thousand (1.4e-3 and 6e-4, estimated from exact geometric
counts). A seed's estimates change whenever the sampler changes how it
consumes its random stream, so each of the two ports makes its check at
three independent seeds and passes when at least two of them pass. A chance
failure then takes two, with a probability of at most about 6e-6, while an
estimate biased by several SD fails at nearly every seed. A seed that fails
issues a warning, so that a lone failure is seen and investigated.
"""

import math
import warnings

import numpy as np

from pyibs import _sampler
from pyibs.testing import _helpers

SEED = 20260929
SEEDS = (SEED, SEED + 1, SEED + 2)


def bernoulli(theta, idx, rng):
    return rng.random(len(idx)) < theta[0]


def estimate(responses, p, n_repeats, rng, **options):
    """IBS estimate of the log-likelihood of p, and its SE."""
    settings = _sampler._Settings(bernoulli, responses, **options)
    s = _helpers.summary(
        _sampler.sample(settings, np.array([p]), n_repeats, rng)
    )
    return s.mean, s.se


def passes_at_two_of_three_seeds(check, name):
    """Whether ``check`` passes at two or more of the seeds in SEEDS.

    ``check(rng)`` returns whether it passed and a summary to print.
    """
    passed = 0
    for seed in SEEDS:
        ok, summary = check(np.random.default_rng(seed))
        print(f"{name}, seed {seed}: {'pass' if ok else 'FAIL'}; {summary}")
        if not ok:
            warnings.warn(f"{name} failed at seed {seed}: {summary}")
        passed += ok
    return passed >= 2


def runtest1(rng):
    """``runtest1``'s check of the estimates of log p at one seed."""
    n_repeats = 1000
    p_model = np.exp(np.linspace(np.log(1e-3), 0, 10))
    results = [estimate(np.array([True]), p, n_repeats, rng) for p in p_model]
    loglik, sd = np.array(results).T
    log_p = np.log(p_model)
    rmse = math.sqrt(np.mean((loglik - log_p) ** 2))
    # The true value lies within 4 SD of each estimate (> 99.99 %).
    ok = (
        np.all(loglik - 4 * sd <= log_p)
        and np.all(log_p <= loglik + 4 * sd)
        and rmse < 2 / math.sqrt(n_repeats)
    )
    # At p = 1 every count is 1, and the estimate is exactly 0 with SD 0.
    nonzero = sd > 0
    z = np.abs(loglik[nonzero] - log_p[nonzero]) / sd[nonzero]
    return ok, f"max |z| {z.max():.3f}, RMSE {rmse:.4f}"


def test_runtest1_bernoulli_log_p():
    """``runtest1``: IBS estimates of log p for one Bernoulli trial."""
    assert passes_at_two_of_three_seeds(runtest1, "runtest1")


def test_runtest2_binomial_z_scores():
    """``runtest2``: z-scores of binomial log-likelihood estimates.

    2000 experiments of 100 trials, each estimated with 10 repeats; the
    z-scores against the exact log-likelihood are close to standard normal.
    """
    n_trials, n_experiments, n_repeats = 100, 2000, 10
    rng = np.random.default_rng(SEED)
    p_true, p_model = 0.9 * rng.random(2) + 0.05
    z = np.empty(n_experiments)
    for k in range(n_experiments):
        responses = rng.random(n_trials) < p_true
        loglik, sd = estimate(responses, p_model, n_repeats, rng)
        n_ones = np.count_nonzero(responses)
        exact = math.log(p_model) * n_ones + math.log(1 - p_model) * (
            n_trials - n_ones
        )
        z[k] = (loglik - exact) / sd
    mean, std = z.mean(), z.std(ddof=1)
    print(
        f"runtest2: p_true={p_true:.3f}, p_model={p_model:.3f}; z-scores "
        f"mean {mean:.4f}, SD {std:.4f}"
    )
    assert abs(mean) < 0.15
    assert abs(std - 1) < 0.1


def runtest3(rng):
    """``runtest3``'s check of the thresholded estimates at one seed."""
    n_repeats = 100
    threshold = -math.log(0.01)
    p_model = np.exp(np.linspace(np.log(1e-3), np.log(0.1), 10))
    results = [
        estimate(
            np.array([True]),
            p,
            n_repeats,
            rng,
            acceleration=1,
            neg_loglik_threshold=threshold,
        )
        for p in p_model
    ]
    loglik, sd = np.array(results).T
    log_p = np.log(p_model)
    far = log_p > -0.75 * threshold
    below = log_p < -threshold
    target = np.log(np.maximum(p_model, math.exp(-threshold)))
    rmse = math.sqrt(np.mean((loglik[far] - target[far]) ** 2))
    ok = (
        # Well above the threshold, log p lies within 4 SD of the estimate.
        np.all(loglik[far] - 4 * sd[far] <= log_p[far])
        and np.all(log_p[far] <= loglik[far] + 4 * sd[far])
        # Below it, the clipped estimates lie at least one SD above log p.
        and np.all(loglik[below] - sd[below] >= log_p[below])
        and rmse < 4 / math.sqrt(n_repeats)
    )
    return ok, (
        f"max |z| well above -T "
        f"{np.max(np.abs(loglik[far] - log_p[far]) / sd[far]):.3f}, "
        f"min (estimate - SD - log p) below -T "
        f"{np.min(loglik[below] - sd[below] - log_p[below]):.3f}, "
        f"RMSE {rmse:.4f}"
    )


def test_runtest3_thresholded_log_p():
    """``runtest3``: log p of one Bernoulli trial under a threshold.

    With the likelihood threshold T = -log(0.01), estimates well above -T
    are close to log p, and estimates for log p below -T lie above it.
    """
    assert passes_at_two_of_three_seeds(runtest3, "runtest3")

"""Samples drawn with vectorized=True, relative to vectorized=False.

``vectorized=False`` requests one sample per open trial and simulator call,
so it draws exactly the samples that the matches need. ``vectorized=True``
requests ``num_reps`` samples of each open trial at the first call, and
1.5 times as many at each later call, so a trial's last call can draw
samples beyond its last match. The ratio of the samples drawn under the two
schedules is that surplus, which a simulator whose cost is per sample pays
in time.

It prints the ratio for the example model of the quick start (600 trials)
and for data sets of 200 trials of one matching probability p, at
``num_reps`` 10 and 100, each the mean of 5 calls of one seeded ``IBS``
object. Run from the repository root as
``$PY dev/scripts/vectorized_surplus.py``, with ``$PY`` the venv's
interpreter (``.venv/Scripts/python.exe`` on Windows, ``.venv/bin/python``
elsewhere); it takes a few seconds.
"""

import warnings

import numpy as np

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator

# A call whose every count is 1 warns of a variance estimate of 0, which
# the trials of p = 0.99 can produce; the sample counts are unaffected.
warnings.simplefilter("ignore")

THETA = np.array([0.0, 0.2, 0.03])
CALLS = 5


def mean_cost(simulator, responses, design, vectorized, num_reps, seed):
    """Mean samples and simulator calls of one call of a seeded IBS."""
    ibs = IBS(
        simulator, responses, design, vectorized=vectorized, random_seed=seed
    )
    samples, calls = 0.0, 0.0
    for _ in range(CALLS):
        res = ibs(THETA, num_reps=num_reps, additional_output="full")
        samples += res.num_samples_per_trial * len(responses)
        calls += res.fun_count
    return samples / CALLS, calls / CALLS


def bernoulli(p):
    """A simulator that matches every trial with probability p."""

    def simulate(theta, rows, rng):
        return (rng.random(len(rows)) < p).astype(float)

    return simulate


def main():
    rng = np.random.default_rng(20261009)
    S = 3 * rng.standard_normal(600)
    R = psycho_generator(THETA, S, rng)

    print("Example model, 600 trials", flush=True)
    for num_reps in (1, 10, 100):
        f, cf = mean_cost(psycho_generator, R, S, False, num_reps, 1)
        t, ct = mean_cost(psycho_generator, R, S, True, num_reps, 1)
        print(
            f"  num_reps={num_reps:3d}: False {f:9.0f} samples, {cf:6.0f} "
            f"calls; True {t:9.0f} samples, {ct:4.0f} calls; "
            f"ratio {t / f:.2f}",
            flush=True,
        )

    print("200 trials of matching probability p", flush=True)
    ones = np.ones(200)
    for p in (0.99, 0.9, 0.5, 0.1, 0.02):
        ratios = []
        for num_reps in (10, 100):
            f, _ = mean_cost(bernoulli(p), ones, None, False, num_reps, 2)
            t, _ = mean_cost(bernoulli(p), ones, None, True, num_reps, 2)
            ratios.append(f"num_reps={num_reps}: ratio {t / f:.2f}")
        print(f"  p={p:<5}: " + "; ".join(ratios), flush=True)


if __name__ == "__main__":
    main()

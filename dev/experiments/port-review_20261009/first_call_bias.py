"""The first call of an IBS object with vectorized=None, on a fake clock.

One trial whose response the simulator matches with probability 0.5, and a
simulator that, in a call for one row (the timing call of vectorized=None
with one trial), advances the clock by 0.2 s when it matches and leaves it
when it misses. vectorized_threshold is 0.1 s, so the timing call decides
False exactly when it matches. Each estimate, with num_reps=2, is the first
call of a new IBS object; their mean is compared with the exact negative
log-likelihood, log 2. If the timing call's sample is kept exactly when it
matched, the expected estimate is 0.75 log 2.

Run from the repository root:
    .venv/bin/python -u dev/experiments/port-review_20261009/first_call_bias.py [M]
"""

import math
import sys

import numpy as np

import pyibs._sampler
import pyibs.ibs
from pyibs import IBS


class FakeTime:
    """A clock that only the simulator advances."""

    now = 0.0

    def perf_counter(self):
        return self.now


clock = FakeTime()
pyibs._sampler.time = clock
pyibs.ibs.time = clock


def simulator(params, rows, rng):
    out = rng.random(len(rows)) < 0.5
    if len(rows) == 1:
        clock.now += 0.2 * np.count_nonzero(out)
    return out


M = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(12345)
est = np.array(
    [
        IBS(simulator, np.ones(1, bool), random_seed=rng)(None, num_reps=2)
        for _ in range(M)
    ]
)
exact = math.log(2)
mean, se = est.mean(), est.std(ddof=1) / math.sqrt(M)
print(f"PyIBS {pyibs.__version__}, NumPy {np.__version__}")
print(
    f"M = {M}: mean estimate {mean:.4f} +- {se:.4f} (SE); exact {exact:.4f}, "
    f"z = {(mean - exact) / se:.1f}; expected if the timing call is kept "
    f"exactly when it matched {0.75 * exact:.4f}"
)

"""Replications of the share of zero variance estimates at num_reps=10.

In the validation's full run, the cell ``bernoulli_p0.999_vF_n10`` drew 811
estimates with a variance estimate of 0 out of 2,000, where 735.4 are
expected (z = 3.51, within the gate of 4.5). This script draws the cells of
num_reps=10 of the two models whose trials all match with probability
0.999, every setting of ``vectorized``, again at other run seeds, and
prints each replication's count with its z-score against the exact
probability ``0.999**1000``.

Run from the repository root:

    .venv/bin/python -u dev/experiments/validation_20261009/zero_share_replication.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import validate  # noqa: E402

REPLICATIONS = 10
ESTIMATES = 2000

cells = [
    c
    for c in validate.all_cells()
    if c.model in validate.CALIBRATION_REPORTED_ONLY
    and c.num_reps == 10
    and not c.threshold
]
zs = []
for r in range(1, REPLICATIONS + 1):
    validate.RUN_SEED = 20261010 + 1000 * r
    for cell in cells:
        model = validate.models_by_name()[cell.model]
        _, raw, _ = validate.run_cell(cell, ESTIMATES)
        p0 = float(np.prod(model.p) ** cell.num_reps)
        k = int(np.sum(raw["var"] == 0))
        z = (k - ESTIMATES * p0) / math.sqrt(ESTIMATES * p0 * (1 - p0))
        zs.append(z)
        print(
            f"run_seed {validate.RUN_SEED} {cell.name:<26} zero {k:>4} "
            f"expected {ESTIMATES * p0:.1f} z {z:+.2f}",
            flush=True,
        )
zs = np.array(zs)
print(
    f"{zs.size} replications: mean z {zs.mean():+.3f}, sd {zs.std(ddof=1):.3f}"
    f", max |z| {np.abs(zs).max():.2f}"
)

"""The figure of "How does it work?" in docsrc/source/index.rst.

For a trial whose observed response the simulator produces with
probability p, IBS takes K ~ Geometric(p) samples, 1/p on average, and the
variance of its estimate of log p is Li_2(1 - p) = scipy.special.spence(p),
which tends to pi^2/6 as p -> 0 (van Opheusden, Acerbi & Ma, 2020,
Sections 4.2 and 4.3).

Run from the repository root as
``.venv/bin/python dev/scripts/ibs_cost_variance.py OUT``, where OUT is
``docsrc/source/_static/ibs-cost-and-variance.png``.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spence

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

p = np.logspace(-4, 0, 400)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.titlesize": 10.5,
        "axes.titlecolor": INK,
    }
)
fig, (ax_cost, ax_var) = plt.subplots(
    1, 2, figsize=(8.4, 3.2), dpi=150, facecolor=SURFACE
)
for ax in (ax_cost, ax_var):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("p, probability of the observed response")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

ax_cost.plot(p, 1 / p, color=SERIES, linewidth=2)
ax_cost.set_yscale("log")
ax_cost.set_ylim(0.5, 2e4)
ax_cost.set_title("Expected number of samples, 1/p", loc="left")

bound = np.pi**2 / 6
ax_var.axhline(bound, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
ax_var.text(
    1.5e-4, bound + 0.05, "bound: π²/6 ≈ 1.64", color=INK_2, va="bottom"
)
ax_var.plot(p, spence(p), color=SERIES, linewidth=2)
ax_var.set_ylim(0, 2)
ax_var.set_title(
    "Variance of the IBS estimate of log p, Li₂(1 − p)", loc="left"
)

fig.tight_layout(w_pad=3)
fig.savefig(sys.argv[1], facecolor=SURFACE)

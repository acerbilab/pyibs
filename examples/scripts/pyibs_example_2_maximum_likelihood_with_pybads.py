import math

import numpy as np
from pybads import BADS
from scipy.optimize import minimize

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

# Independent seeds for the data, the IBS estimates, the start and PyBADS
seed_data, seed_ibs, seed_start, seed_bads = np.random.SeedSequence(
    2026
).spawn(4)

theta_true = np.array([math.log(1.0), 0.2, 0.03])  # log(sigma), bias, lapse
n_trials = 600

rng = np.random.default_rng(seed_data)
S = 3 * rng.standard_normal((n_trials, 1))  # Orientation of each trial
R = psycho_generator(theta_true, S, rng)  # Response of each trial


LB = np.array([math.log(0.1), -2.0, 0.01])  # Lower bounds
UB = np.array([math.log(10.0), 2.0, 1.0])  # Upper bounds
PLB = np.array([math.log(0.2), -1.0, 0.02])  # Plausible lower bounds
PUB = np.array([math.log(5.0), 1.0, 0.2])  # Plausible upper bounds


exact_fit = minimize(
    psycho_neg_logl,
    (PLB + PUB) / 2,
    args=(S, R),
    method="L-BFGS-B",
    bounds=list(zip(LB, UB)),
)
theta_ml, neg_logl_min = exact_fit.x, exact_fit.fun
print(f"Exact maximum-likelihood point: {theta_ml.round(4)}")
print(f"Exact minimum of the negative log-likelihood: {neg_logl_min:.2f}")


ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=seed_ibs)


def target(theta):
    """IBS estimate of the negative log-likelihood, and its SD."""
    return ibs(theta, num_reps=100, additional_output="std")


x0 = PLB + np.random.default_rng(seed_start).random(3) * (PUB - PLB)
value, sd = target(x0)
print(f"Starting point: {x0.round(4)}")
print(f"Target at the start: {value:.2f} +/- {sd:.2f}")
print(f"Exact value:         {psycho_neg_logl(x0, S, R):.2f}")


bads = BADS(
    target,
    x0,
    LB,
    UB,
    PLB,
    PUB,
    options={"specify_target_noise": True, "random_seed": seed_bads},
)
optimize_result = bads.optimize()


theta_bads = optimize_result["x"]
fval, fsd = optimize_result["fval"], optimize_result["fsd"]
print(f"PyBADS estimate at its solution: {fval:.2f} +/- {fsd:.2f}")
print(f"Target evaluations: {optimize_result['func_count']}")
print()
print(f"{'':17}{'eta':>8}{'bias':>8}{'lapse':>8}")
for name, theta in [
    ("Generating", theta_true),
    ("Exact ML point", theta_ml),
    ("PyBADS solution", theta_bads),
]:
    print(f"{name:17}" + "".join(f"{x:8.4f}" for x in theta))


neg_logl, neg_logl_sd = ibs(theta_bads, num_reps=1000, additional_output="std")
neg_logl_exact = psycho_neg_logl(theta_bads, S, R)
print(
    "IBS estimate at the solution (1,000 repeats): "
    f"{neg_logl:.2f} +/- {neg_logl_sd:.2f}"
)
print(f"Exact value at the solution:  {neg_logl_exact:.2f}")
print(f"Exact minimum:                {neg_logl_min:.2f}")
print(f"Difference:                   {neg_logl_exact - neg_logl_min:.2f}")

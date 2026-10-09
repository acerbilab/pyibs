import math

import numpy as np
from pyvbmc import VBMC
from pyvbmc.priors import Trapezoidal
from scipy.optimize import minimize
from scipy.special import logsumexp

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

# Independent seeds for the data, the IBS estimates and PyVBMC
seed_data, seed_ibs, seed_vbmc = np.random.SeedSequence(2026).spawn(3)

theta_true = np.array([math.log(1.0), 0.2, 0.03])  # log(sigma), bias, lapse
n_trials = 600

rng = np.random.default_rng(seed_data)
S = 3 * rng.standard_normal((n_trials, 1))  # Orientation of each trial
R = psycho_generator(theta_true, S, rng)  # Response of each trial


LB = np.array([math.log(0.1), -2.0, 0.01])  # Lower bounds
UB = np.array([math.log(10.0), 2.0, 1.0])  # Upper bounds
PLB = np.array([math.log(0.2), -1.0, 0.02])  # Plausible lower bounds
PUB = np.array([math.log(5.0), 1.0, 0.2])  # Plausible upper bounds

prior = Trapezoidal(LB, PLB, PUB, UB)


ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=seed_ibs)


def log_likelihood(theta):
    """IBS estimate of the log-likelihood, and its SD."""
    return ibs(
        theta, num_reps=100, additional_output="std", return_positive=True
    )


x0 = (PLB + PUB) / 2
vbmc = VBMC(
    log_likelihood,
    x0,
    LB,
    UB,
    PLB,
    PUB,
    options={"specify_target_noise": True},
    prior=prior,
    seed=seed_vbmc,
)
vp, results = vbmc.optimize()


post_mean, post_cov = vp.moments(cov_flag=True)
post_mean = post_mean.ravel()
post_sd = np.sqrt(np.diag(post_cov))
elbo, elbo_sd = results["elbo"], results["elbo_sd"]

print(f"ELBO: {elbo:.2f} +/- {elbo_sd:.2f}")
print(f"Target evaluations: {results['func_count']}")
print()
print(f"{'':16}{'eta':>8}{'bias':>8}{'lapse':>8}")
for name, values in [("Posterior mean", post_mean), ("Posterior SD", post_sd)]:
    print(f"{name:16}" + "".join(f"{x:8.4f}" for x in values))


exact_fit = minimize(
    psycho_neg_logl,
    (PLB + PUB) / 2,
    args=(S, R),
    method="L-BFGS-B",
    bounds=list(zip(LB, UB)),
)
theta_ml = exact_fit.x

vp.plot(
    plot_style={
        "corner": {
            "labels": [
                r"$\eta$ (log noise)",
                r"$\mu$ (bias)",
                r"$\gamma$ (lapse)",
            ],
            "truths": theta_ml,
        }
    }
)


n_grid = 31  # Points per parameter
axes = [
    np.linspace(max(lb, m - 6 * s), min(ub, m + 6 * s), n_grid)
    for lb, ub, m, s in zip(LB, UB, post_mean, post_sd)
]
grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

# Log of the likelihood times the prior at each grid point
log_joint = prior.log_pdf(grid).ravel() - np.array(
    [psycho_neg_logl(theta, S, R) for theta in grid]
)
cell_volume = np.prod([a[1] - a[0] for a in axes])
log_evidence = logsumexp(log_joint) + math.log(cell_volume)
weights = np.exp(log_joint - logsumexp(log_joint))
exact_mean = weights @ grid
exact_sd = np.sqrt(weights @ (grid - exact_mean) ** 2)

print(f"ELBO (PyVBMC):       {elbo:.2f} +/- {elbo_sd:.2f}")
print(f"Exact log evidence: {log_evidence:.2f}")
print()
print(f"{'':22}{'eta':>8}{'bias':>8}{'lapse':>8}")
for name, values in [
    ("Exact ML point", theta_ml),
    ("Posterior mean", post_mean),
    ("Exact posterior mean", exact_mean),
    ("Posterior SD", post_sd),
    ("Exact posterior SD", exact_sd),
]:
    print(f"{name:22}" + "".join(f"{x:8.4f}" for x in values))

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


def grid_reference(n_grid, width):
    """Numerical log evidence and posterior moments on a bounded grid."""
    axes = [
        np.linspace(max(lb, m - width * s), min(ub, m + width * s), n_grid)
        for lb, ub, m, s in zip(LB, UB, post_mean, post_sd)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # Closed-form log likelihood plus the normalized log prior
    log_joint = prior.log_pdf(grid).ravel() - np.array(
        [psycho_neg_logl(theta, S, R) for theta in grid]
    )
    cell_volume = np.prod([a[1] - a[0] for a in axes])
    log_sum = logsumexp(log_joint)
    log_evidence = log_sum + math.log(cell_volume)
    weights = np.exp(log_joint - log_sum)
    mean = weights @ grid
    sd = np.sqrt(weights @ (grid - mean) ** 2)
    return log_evidence, mean, sd


configurations = [(31, 6), (61, 6), (61, 9)]
references = [grid_reference(n, width) for n, width in configurations]
log_evidence, reference_mean, reference_sd = references[1]

print("Grid check: points per parameter, half-width in posterior SDs, log Z")
for (n, width), (log_z, mean, sd) in zip(configurations, references):
    delta_log_z = abs(log_z - log_evidence)
    delta_mean = np.max(abs(mean - reference_mean) / reference_sd)
    delta_sd = np.max(abs(sd - reference_sd) / reference_sd)
    print(
        f"{n:3d} points, {width} SDs: {log_z:.5f}; "
        f"changes {delta_log_z:.4f} in log Z, "
        f"{delta_mean:.4f} SDs in mean, {delta_sd:.2%} in SD"
    )
    assert delta_log_z < 0.02, "Refine or widen the evidence grid."
    assert delta_mean < 0.02, "Refine or widen the posterior grid."
    assert delta_sd < 0.005, "Refine or widen the posterior grid."

print()
print(f"ELBO (PyVBMC):          {elbo:.2f} +/- {elbo_sd:.2f}")
print(f"Numerical log evidence: {log_evidence:.2f}")
print()
print(f"{'':22}{'eta':>8}{'bias':>8}{'lapse':>8}")
for name, values in [
    ("Reference ML point", theta_ml),
    ("PyVBMC mean", post_mean),
    ("Grid mean", reference_mean),
    ("PyVBMC SD", post_sd),
    ("Grid SD", reference_sd),
]:
    print(f"{name:22}" + "".join(f"{x:8.4f}" for x in values))

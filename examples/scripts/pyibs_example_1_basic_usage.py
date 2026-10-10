import sys

if "google.colab" in sys.modules:  # Colab lacks PyIBS: install it
    get_ipython().run_line_magic("pip", 'install "pyibs>=1.5"')


import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pyibs import IBS, ibs_basic
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

# Independent seeds for the data and for the IBS estimates
seed_data, seed_ibs, seed_basic = np.random.SeedSequence(2026).spawn(3)

theta_true = np.array([math.log(1.0), 0.2, 0.03])  # log(sigma), bias, lapse
n_trials = 600

rng = np.random.default_rng(seed_data)
S = 3 * rng.standard_normal((n_trials, 1))  # Orientation of each trial
R = psycho_generator(theta_true, S, rng)  # Response of each trial

print("First orientations:", S[:5, 0].round(2))
print("First responses:   ", R[:5, 0])
print(f"Rightwards responses: {np.mean(R == 1):.1%}")


ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=seed_ibs)

neg_logl = ibs(theta_true)
print(f"Negative log-likelihood estimate: {neg_logl:.2f}")


neg_logl, neg_logl_sd = ibs(theta_true, additional_output="std")
print(
    f"Negative log-likelihood estimate: {neg_logl:.2f} +/- {neg_logl_sd:.2f}"
)


result = ibs(theta_true, additional_output="full")
print(result)


exact = psycho_neg_logl(theta_true, S, R)
print(f"IBS estimate: {neg_logl:.2f} +/- {neg_logl_sd:.2f}")
print(f"Exact value:  {exact:.2f}")
print(f"Difference:   {(neg_logl - exact) / neg_logl_sd:.2f} SDs")


n_estimates = 2000
estimates = np.array(
    [ibs(theta_true, additional_output="std") for _ in range(n_estimates)]
)
values, sds = estimates[:, 0], estimates[:, 1]
z = (values - exact) / sds

standard_error = values.std(ddof=1) / math.sqrt(n_estimates)
reported_sd = math.sqrt(np.mean(sds**2))  # Root mean square of the SDs
print(f"Exact value:            {exact:.2f}")
print(f"Mean of the estimates:  {values.mean():.2f} +/- {standard_error:.2f}")
print(f"SD of the estimates:    {values.std(ddof=1):.2f}")
print(f"RMS reported SD:        {reported_sd:.2f}")
print(f"z-scores: mean {z.mean():.3f}, SD {z.std(ddof=1):.3f}")


fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(
    z,
    bins=40,
    density=True,
    color="#2a78d6",
    edgecolor="white",
    linewidth=0.5,
    label="IBS estimates",
)
x = np.linspace(-4, 4, 201)
ax.plot(x, norm.pdf(x), color="#0b0b0b", linewidth=2, label="Standard normal")
ax.set_xlabel("z-score, (estimate - exact) / SD")
ax.set_ylabel("Density")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.show()


print("num_reps     SD   SD * sqrt(num_reps)   samples per trial   calls")
for num_reps in [3, 10, 30, 100, 300, 1000]:
    result = ibs(theta_true, num_reps=num_reps, additional_output="full")
    print(
        f"{num_reps:8d} {result.neg_logl_std:6.2f} "
        f"{result.neg_logl_std * math.sqrt(num_reps):20.2f} "
        f"{result.num_samples_per_trial:19.1f} {result.fun_count:7d}"
    )


loglik_basic = ibs_basic(
    psycho_generator, theta_true, R, S, random_seed=seed_basic
)
print(f"ibs_basic estimate (one repeat): {loglik_basic:.2f}")
print(f"Exact log-likelihood:            {-exact:.2f}")

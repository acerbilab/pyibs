# Validation of PyIBS 1.5: estimates, PyBADS and PyVBMC, timing

Date: 2026-10-09. Plan: [`dev/plans/pyibs-1.5.md`](../plans/pyibs-1.5.md),
Phase 3. Evidence:
[`dev/experiments/validation_20261009/`](../experiments/validation_20261009/).

## Question

Are PyIBS 1.5's estimates of the negative log-likelihood unbiased, and its
variance estimates calibrated, across models and settings? Do PyBADS 1.5
and PyVBMC 1.5 run with an IBS estimate as their noisy target? How does
the speed of PyIBS 1.5 compare with that of PyIBS 0.1.0?

## Method

PyIBS at `c76b5ac` (branch `dev-next`), on Linux with Python 3.12.3,
NumPy 2.5.3 and SciPy 1.18.1; the evidence directory gives the
provenance of each output.

**Validation** (`dev/scripts/validate.py`). Sixteen models, each with a
known matching probability p_i for every trial, so that the exact
log-likelihood is `sum_i w_i log p_i`:

- Bernoulli: 100 trials at each p in {0.001, 0.01, 0.1, 0.5, 0.9, 0.999},
  the responses drawn at that p. The seeded draws hold 0, 1, 11, 43, 89
  and 100 ones: the models of p = 0.001 and p = 0.999 hold no rare
  response, so all their trials match with probability 0.999.
- Weights: the Bernoulli model at p = 0.5 with integer weights from 1 to
  3, and with fractional weights from 0.2 to 2 (drawn: 0.22 to 2.00).
- Categorical: 200 trials with 2, 4 and 8 outcomes, the outcome
  probabilities drawn once from a flat Dirichlet; the smallest matching
  probability is 0.39, 0.017 and 0.0095.
- Text: the categorical model with 4 outcomes, its responses as the
  strings `"left"`, `"right"`, `"up"` and `"down"`.
- Two response columns: 100 trials, a binary column whose probability
  runs from 0.1 to 0.9 over the trials and an independent column of 3
  outcomes; a sample matches only when both columns do.
- The example model, `pyibs.examples.psycho_model`, on `ibs_example.m`'s
  data set of 600 trials, at its generating parameters and the two other
  vectors of `pyibs/testing/test_examples.py`.

A cell is a model with `vectorized` True, False or None, `num_reps` 1, 10
or 100, and the likelihood threshold off or, for the Bernoulli, weighted
and categorical models, at the chance level `sum_i w_i log k_i`: 243
cells of 2,000 estimates, each estimate from a new `IBS` object with its
own seed, so that the cells of `vectorized=None` sample the object's
deciding call. Per cell, the script reports the bias in standard errors
of the mean; with the threshold, against the expected value of a
thresholded estimate, `-mean(max(Y, -T))` over 10**5 exact repeats Y
(`pyibs/testing/_exact.py`), combining both standard errors. It reports
the mean squared z-score `mean(err**2 / var_estimate)` in standard errors
from 1, beside the same statistic of exact IBS estimates drawn from
geometric counts; the 95% coverage; the ratio of the estimates' SD to the
exact SD; the samples per trial against `num_reps * mean_i(1 / p_i)`; and
the share of estimates whose variance estimate is 0 against its exact
probability `prod_i p_i**num_reps`. A cell passes when its bias is within
4.5 standard errors and, at `num_reps` of 10 or more without the
threshold, its mean squared z-score is within 4.5 standard errors of 1.

A smoke pass of 100 estimates per cell projected the full run at 1.5 h on
4 workers, and found six cells failing the calibration gate: those of
`num_reps=10` of the two models whose trials all match with probability
0.999. There, every count of an estimate is 1 with probability 0.999**1000
= 0.37, which makes the variance estimate 0 and the z-score infinite, and
exact IBS draws give the same; at `num_reps=100`, the exact draws' mean
squared z-score is 1.28 ± 0.04, which would fail the gate as well. The PI
ruled (2026-10-09) that the calibration of these two models is reported
and not gated, and that their share of zero variance estimates is gated
against its exact probability instead; their bias stays gated. The PI
approved the full run of every cell at 2,000 estimates.

Separately, a model of 50 trials whose simulator always returns the
observed response, so that every p_i = 1, is called with every setting of
`vectorized` and `num_reps`, with and without a threshold (18 calls).

**PyBADS and PyVBMC** (`pyibs/testing/integration/`). Both fit
`ibs_example.m`'s data set of the example model, within its bounds, with
an IBS estimate of 100 repeats as the noisy target (`vectorized=True`,
seeded): an SD of about 1 near the maximum-likelihood point, the noise
that PyBADS's FAQ and the IBS wiki advise. PyBADS 1.5.1 minimizes it from
a start drawn in the plausible box; the test checks that the exact
negative log-likelihood at the returned point is within 1 of the exact
minimum, found by L-BFGS-B on the closed form from twelve starts. PyVBMC
1.5 (the release branch at `89007a4`) infers the posterior under
`ibs_example.m`'s trapezoidal prior from the centre of the plausible box;
the test checks that the ELBO is finite and that the posterior mean lies
within three posterior SDs of the exact maximum-likelihood point in each
coordinate.

**Timing** (`dev/scripts/timing.py`). The wall time of one estimate of
the example model at N = 100, 1,000 and 10,000 trials (the orientations
drawn as in `ibs_example.m`, the responses at its generating parameters),
`num_reps` 10 and 100, with the fast simulator and with one that sleeps
0.2 s per call, for PyIBS 1.5 and for PyIBS 0.1.0 from PyPI in a venv of
its own. Each version runs at its defaults, with a new `IBS` object per
estimate, so that `vectorized=None` decides at each, and 0.1.0 with
`max_iter=10**5`, so that both sample every repeat to completion. Both call
the same two-argument simulator, which counts its calls and rows. A fast
cell reports the median of at least 5 estimates and 2 s, up to 50; a slow
cell, one estimate. The fast cells ran alone; the slow ones, which sleep
for nearly all of their time, ran together, beside the validation, as the
PI approved after a smoke pass projected 4.4 h for them one at a time.

## Main results

- **Bias.** Every one of the 243 cells passes: the largest bias is -2.95
  standard errors (`bernoulli_p0.01_vF_n10`), and over the cells the
  z-scores have mean 0.13 and SD 1.01. Over the 99 cells with the
  threshold, their mean is 0.28; each model's nine cells with the
  threshold share one exact reference, whose error moves them together. In
  the two models whose trials all match with probability 0.999, where the
  threshold never acts, the reference lies 1.5 and 1.4 of its standard
  errors below the exact value, and their cells' mean z-score is +0.9.
- **Calibration.** The 84 gated cells all pass: the largest deviation of
  the mean squared z-score from 1 is 3.33 standard errors
  (`bernoulli_p0.01_vN_n10`, 1.127), and over the cells the z-scores have
  mean -0.06 and SD 1.07. Against the exact IBS references the
  differences, in combined standard errors, have mean -0.05 and SD 1.13.
  The 95% coverage of the gated cells lies between 0.936 and 0.960 (mean
  0.950), and the SD of the estimates is 0.950 to 1.056 times the exact SD
  in the 144 cells without the threshold (mean 0.999).
- **The two models whose trials all match with probability 0.999.** In
  all 36 cells, the share of zero variance estimates agrees with its exact
  probability (0.905, 0.368 and 4.5e-5 at `num_reps` 1, 10 and 100). The
  largest deviation, 3.51 standard errors (811 against 735.4 at
  `bernoulli_p0.999_vF_n10`), is chance: drawn again at ten other run
  seeds, the six cells of `num_reps=10` give z-scores of mean -0.08 and SD
  0.91, at most 2.06 in absolute value, and that cell's own from -2.06 to
  0.86 (`zero_share_replication.txt`). At `num_reps=100` the mean squared
  z-scores are 1.14 to 1.37, against 1.28 ± 0.04 for exact IBS.
- **Samples.** At `vectorized=False`, without the threshold, the samples
  per trial agree with `num_reps * mean_i(1 / p_i)` in all 48 cells:
  z-scores of mean -0.01 and SD 1.03, at most 2.61 in absolute value
  (ratios 0.995 to 1.018). The accelerated schedules draw 1.01 to 1.56
  times as many, the surplus included.
- **Threshold.** The chance level acts where it lies at or near the exact
  value, in `bernoulli_p0.5`, the two weighted models and
  `categorical_k2`, where a repeat ends with probability 0.41 to 0.50.
  There the share of estimates with exit flag 1 agrees with
  `1 - (1 - q)**num_reps`, q from the exact repeats (at most 2.34 standard
  errors), and the estimates agree with the expected value of a
  thresholded estimate, which lies 3.0 to 6.6 below the exact negative
  log-likelihood.
- **vectorized=None** decided True in every estimate of its cells: one
  simulation of all trials takes far less than 0.1 s with these
  simulators.
- **Every trial at p = 1.** The value and the variance estimate are
  exactly 0 in all 18 calls, each with the warning on a zero variance.
- **PyBADS 1.5.1** returns `(0.0844, 0.1987, 0.0298)` after 348
  evaluations, whose exact negative log-likelihood, 172.103, is 0.041
  above the exact minimum, 172.062 at `(0.0884, 0.2255, 0.0306)`. The test
  takes about 40 s.
- **PyVBMC 1.5** stops after 115 evaluations with an ELBO of -179.99 ±
  0.16. Its posterior mean `(0.0494, 0.2269, 0.0414)`, with SDs `(0.122,
  0.098, 0.017)`, lies 0.32, 0.01 and 0.64 SDs from the exact
  maximum-likelihood point. The test takes about 2 min alone.
- **Timing.** PyIBS 1.5 is faster than 0.1.0 in every cell (seconds per
  estimate, the median of the estimates given; calls and simulated rows
  per trial are means):

  | Simulator | N | `num_reps` | 1.5 (s) | 0.1.0 (s) | 0.1.0 / 1.5 | Calls, 1.5 | Calls, 0.1.0 | Rows per trial, 1.5 | Rows per trial, 0.1.0 | Estimates |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | fast | 100 | 10 | 0.0013 | 0.0027 | 2.06 | 9 | 9 | 28 | 30 | 50, 50 |
  | fast | 100 | 100 | 0.0035 | 0.0047 | 1.36 | 9 | 9 | 326 | 359 | 50, 50 |
  | fast | 1,000 | 10 | 0.0038 | 0.0059 | 1.59 | 11 | 11 | 27 | 29 | 50, 50 |
  | fast | 1,000 | 100 | 0.029 | 0.040 | 1.40 | 10 | 10 | 314 | 347 | 50, 50 |
  | fast | 10,000 | 10 | 0.025 | 0.038 | 1.53 | 12 | 12 | 27 | 29 | 50, 50 |
  | fast | 10,000 | 100 | 0.25 | 0.33 | 1.36 | 11 | 11 | 309 | 312 | 8, 6 |
  | slow | 100 | 10 | 99 | 130 | 1.32 | 491 | 649 | 23 | 24 | 1, 1 |
  | slow | 100 | 100 | 931 | 1,109 | 1.19 | 4,640 | 5,531 | 218 | 218 | 1, 1 |
  | slow | 1,000 | 10 | 191 | 299 | 1.56 | 954 | 1,490 | 20 | 21 | 1, 1 |
  | slow | 1,000 | 100 | 1,332 | 3,211 | 2.41 | 6,640 | 16,021 | 206 | 206 | 1, 1 |
  | slow | 10,000 | 10 | 225 | 573 | 2.55 | 1,119 | 2,860 | 21 | 21 | 1, 1 |
  | slow | 10,000 | 100 | 1,619 | 6,224 | 3.84 | 8,065 | 31,048 | 205 | 207 | 1, 1 |

  With the fast simulator, both versions take the accelerated schedule
  and make as many calls, and 0.1.0 takes 1.4 to 2.1 times as long. With
  the slow one, both take one sample per open trial and call, and the
  time is 0.2 s per call to within 0.4%, for either version.

## Interpretation

PyIBS 1.5's estimates are unbiased in every model and setting tested,
the thresholded ones equal to the expected value of a thresholded
estimate, and its variance estimates are calibrated wherever exact IBS's
are: where a model's counts are almost all 1, the variance estimates of
exact IBS and of PyIBS alike are often 0, and their mean squared z-score
departs from 1 alike, a property of the estimator that the share of zero
variance estimates confirms PyIBS to share. Neither the schedule
(`vectorized`), the weights, text or two-column responses, nor the
example model changes this, and a model whose trials always match gives
exactly 0.

PyBADS 1.5 and PyVBMC 1.5 take PyIBS's `"std"` output as their noisy
target unchanged: PyBADS reaches the exact maximum likelihood to 0.04,
and PyVBMC's posterior holds the exact maximum-likelihood point well
within its spread.

PyIBS 1.5 is faster than 0.1.0 everywhere tested. With a fast simulator
the gain, 1.4 to 2.1 times, is in the sampler's own work, on the same
number of calls. With a slow simulator the time is the number of calls,
and 1.5 needs fewer: it samples all repeats of a trial as one stream, so
that a call takes one sample of each trial that still needs a match in
any repeat, where 0.1.0 runs the repeats one after the other and waits in
each for its slowest trial. The gain grows with N and `num_reps`, to 3.8
times at 10,000 trials and 100 repeats. In absolute terms, an estimate of
100 repeats with a simulator of 0.2 s per call still takes 15 to 27
minutes. The accelerated schedule makes 9 to 12 calls in these cells, so
a simulator whose time goes to the call rather than to the rows it
simulates gains from `vectorized=True`, given explicitly: `vectorized=None`
decides False for any simulator whose call takes 0.1 s or more.

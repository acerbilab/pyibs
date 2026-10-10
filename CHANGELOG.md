# Changelog

All notable changes to PyIBS are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since PyIBS 0.1.0. The sampling follows MATLAB `ibslike.m` 0.96,
the reference implementation: the [catalogue of deliberate
differences](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md)
says where PyIBS differs from it, and why.

### Upgrading from 0.1.0

- PyIBS needs Python 3.10 or later, NumPy 2.0 or later and SciPy 1.13 or
  later, and installs no other package ("Requirements" under Changed).
- Results differ from 0.1.0, also after `np.random.seed`.
- A trial that draws more than `max_iter * num_reps` samples in a call
  raises `IBSSamplingError`, where 0.1.0 returned a biased estimate, or
  NaN, with exit flag 3, which no longer exists. `max_iter` counts the
  samples of one trial per repeat, where 0.1.0 counted rounds of simulator
  calls; its default is 10**5 (0.1.0's, written `10 ^ 5`, was 15).
- `max_mem` defaults to `max(min(N, 10**4), 10) * 100` samples per
  simulator call, instead of 1e6.
- The samples per simulator call grow after every call by default.
  With an explicit `vectorized` setting and no time limit, a seed
  reproduces a run. `acceleration_threshold=0.1` restores 0.1.0's timing
  rule, which grows the requests only after calls faster than 0.1 s.
- A repeat stopped by `neg_logl_threshold` contributes exactly
  `neg_logl_threshold` to the negative log-likelihood before averaging
  ("Likelihood threshold" under Changed).
- When `max_time` stops the sampling, each trial's value averages its
  completed repeats, and a trial with none raises `IBSSamplingError` ("Time
  limit" under Changed).
- `vectorized=None` is decided once per `IBS` object, at its first call
  with `num_reps > 1`, where 0.1.0 timed a simulation at every call. A
  call with `num_reps=1` requests one sample per trial and simulator call,
  with a warning when `vectorized=True` was given.
- A NaN in `response_matrix` raises `ValueError` when `IBS` is created,
  where 0.1.0 sampled its trial until the iteration limit, with exit flag 3.
- `IBS` checks its settings at construction and checks `num_reps`,
  `trial_weights`, `additional_output` and `return_positive` at each call.
  It raises `ValueError` or `TypeError`
  for values that 0.1.0 accepted ("Checks of the settings" under Changed).
  The settings are read-only attributes: create a new `IBS` object to
  change one.
- The estimates are Python floats, also in the tuples of
  `additional_output="var"` and `"std"`. The warning on reaching `max_time`
  is issued through Python's `warnings` module, where 0.1.0 printed it, and
  reaching the likelihood threshold prints nothing: the exit flag reports
  it.
- `num_samples_per_trial` counts every response that the simulator
  returned, where 0.1.0's vectorized sampling counted one per trial and
  call, and `fun_count` counts every simulator call.
- The example model is the module `pyibs.examples.psycho_model`, whose
  `psycho_generator(theta, S, rng)` takes a `numpy.random.Generator`; the
  modules `pyibs.psycho_generator` and `pyibs.psycho_neg_logl` are removed.
- `pyibs.ibs_basic` is the function `ibs_basic`, which `pyibs` exports:
  `from pyibs.ibs_basic import ibs_basic` works as in 0.1.0, but
  `import pyibs.ibs_basic as m` gives the function rather than its module.
- `ibs_basic` validates observed data and simulated response types and
  shapes, raising an error for inputs that could previously produce a
  false match or sample indefinitely ("ibs_basic" under Fixed).

### Added

- **Reproducible runs.** `IBS(..., random_seed=...)` seeds the object's
  random generator and passes it to a simulator with a parameter named
  `rng`, following PyBADS's `random_seed` interface; `ibs_basic` accepts
  the same argument. Two objects with the same seed reproduce the same
  sequence of estimates when `vectorized` is explicit and `max_time` and
  `acceleration_threshold` retain their defaults.
- **Per-trial estimates.** `additional_output="full"` returns each trial's
  negative log-likelihood estimate and its variance estimate,
  `neg_logl_trials` and `neg_logl_var_trials`; they are NaN when the
  likelihood threshold ended a repeat.
- **Optional design.** `design_matrix` defaults to None, which passes the
  trial indices to the simulator.
- **A warning on a zero variance.** A call that returns a variance
  estimate of 0, as it is when every trial matches at its first sample,
  warns that PyBADS and PyVBMC refuse an SD of 0, and links the FAQ.
- **Documentation, examples and a coding-agent skill.** PyIBS has a
  [documentation site](https://acerbilab.github.io/pyibs/) with a page of
  frequently asked questions, and three example notebooks, installed in
  `pyibs/examples`, which replace 0.1.0's: basic use and calibration,
  maximum-likelihood estimation with PyBADS, and the posterior and the
  model evidence with PyVBMC. `skills/pyibs/SKILL.md` in the repository
  points a coding agent to the documentation relevant to its task.
- **Update check.** `pyibs.check_for_updates()` asks PyPI whether a newer
  version of PyIBS exists and gives the command that installs it. It is
  PyIBS's only network access, made only when the function is called.
- **Tests and version.** The tests ship with the package and run with
  `pytest --pyargs pyibs`, with the `test` extra installed;
  `pyibs.__version__` gives the installed version.

### Changed

- **Sampling.** PyIBS is rebuilt on one tested sampler, which runs two
  schedules: `vectorized=True` requests several samples per trial and
  simulator call, a number that grows from call to call, and
  `vectorized=False` one sample per trial and call.
- **Likelihood threshold.** Both sampling schedules check each repeat
  against `neg_logl_threshold`; a stopped repeat counts exactly that
  threshold in the weighted negative log-likelihood (its negative in the
  log-likelihood that `return_positive=True` returns), before the average
  over the repeats. This follows Appendix C.1 of the IBS paper and
  produces exit flag 1 when a repeat is stopped.
- **Time limit.** When `max_time` stops the sampling, each trial's value
  averages its completed repeats that were not thresholded; thresholded
  repeats contribute to the total as described above. The call warns with
  exit flag 2, or raises `IBSSamplingError` if a trial has no completed repeat.
- **The simulator's output.** For r requested rows, a simulator returns
  an array of shape (r,) or (r, 1) for single-column observations, or
  (r, C) for C > 1 columns. Other shapes raise `ValueError`; incompatible
  response types, such as text for numeric observations, raise `TypeError`.
- **Checks of the settings.** `IBS` raises `ValueError` or `TypeError`,
  naming the setting, for a value out of range or of the wrong type, such
  as a negative `acceleration`, a boolean for a count, or a
  non-whole `num_reps` (0.1.0 truncated 2.5 to 2); the counts take
  whole-number floats such as `1e5`. Trial weights must be real numbers,
  not booleans or strings; `return_positive` must be a Python or NumPy
  boolean, and an unknown `additional_output` raises `ValueError` (0.1.0
  returned None).
- **Requirements.** PyIBS needs Python 3.10, NumPy 2.0 and SciPy 1.13 or
  later, the versions that PyBADS 1.5 needs, and no other package.
- **Speed.** An estimate takes less time: in [timings of the example
  model](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
  0.1.0 took 1.3 to 2.1 times as long with a fast simulator, and up to 3.8
  times as long with one dominated by a fixed cost of 0.2 s per call,
  which PyIBS 1.5 calls fewer times by sampling all the repeats of a trial
  together.

### Fixed

#### Sampling

- `max_iter` defaulted to 15, so that a repeat in which a trial had not
  matched after 15 rounds of simulator calls (15 * `num_reps` with
  `vectorized=True`) was left out of that trial's estimate, which was
  biased, or NaN; with `vectorized=False`, `max_iter=1e5` raised
  `TypeError`.
- A `response_matrix` of several columns raised an error.
- With the default settings, no seed reproduced a run: the samples
  requested depended on the timing of the simulator calls, and the
  simulator could only draw from NumPy's global state.
- With `vectorized=None`, a call discarded the responses of the simulation
  that times the simulator whenever it went on to request several samples
  per trial and simulator call, which biased the estimate when the
  simulator's running time depended on the responses it simulated. Those
  responses are now always used.

#### ibs_basic

- `ibs_basic` works without a design (`S=None`, its default), when the
  simulator receives the trial index.
- Invalid or NaN observations and wrongly shaped simulated responses raise
  `ValueError`; incompatible response types raise `TypeError`. Simulators
  must return shape (C,) or (1, C), with a scalar also accepted for C = 1;
  previously, malformed responses could count as matches or sample forever.

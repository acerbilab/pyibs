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
  simulator call, as in `ibslike.m`, instead of 1e6.
- The samples per simulator call grow after every call by default, so that
  a seed reproduces a run; `acceleration_threshold=0.1` restores 0.1.0's
  rule, which grows them only after calls faster than 0.1 s.
- Under `neg_logl_threshold`, a repeat counts exactly `-neg_logl_threshold`
  once its negative log-likelihood exceeds the threshold ("Likelihood
  threshold" under Changed).
- When `max_time` stops the sampling, each trial's value averages its
  completed repeats, and a trial with none raises `IBSSamplingError` ("Time
  limit" under Changed).
- `vectorized=None` is decided once per `IBS` object, at its first call
  with `num_reps > 1`, where 0.1.0 timed a simulation at every call.
- A NaN in `response_matrix` raises `ValueError` when `IBS` is created,
  where 0.1.0 sampled its trial until the iteration limit, with exit flag 3.
- `IBS` checks its settings, and `num_reps`, `trial_weights` and
  `additional_output` at each call, and raises `ValueError` or `TypeError`
  for values that 0.1.0 accepted ("Checks of the settings" under Changed).
  The settings are read-only attributes: create a new `IBS` object to
  change one.
- The estimates are Python floats, and `additional_output="var"` and
  `"std"` return a tuple of two. Warnings are issued through Python's
  `warnings` module instead of printed.
- `num_samples_per_trial` counts every response that the simulator
  returned, where 0.1.0's vectorized sampling counted one per trial and
  call, and `fun_count` counts every simulator call.
- The example model is the module `pyibs.examples.psycho_model`, whose
  `psycho_generator(theta, S, rng)` takes a `numpy.random.Generator`; the
  modules `pyibs.psycho_generator` and `pyibs.psycho_neg_logl` are removed.

### Added

- **Reproducible runs.** `IBS(..., random_seed=...)` creates the
  generator of the object's calls, `rng`, from a seed, as PyBADS's
  `random_seed` does, and a simulator that has a parameter named `rng`
  receives it: two objects with the same seed then give the same
  estimates.
- **Per-trial estimates.** `additional_output="full"` returns each trial's
  negative log-likelihood estimate and its variance estimate,
  `neg_logl_trials` and `neg_logl_var_trials`, as `ibslike.m` does.
- **Optional design.** `design_matrix` defaults to None, which passes the
  trial indices to the simulator.
- **A warning on a zero variance.** A call that returns a variance
  estimate of 0, as it is when every trial matches at its first sample,
  warns that PyBADS and PyVBMC refuse an SD of 0, and links the FAQ.
- **Tests and version.** The tests ship with the package and run with
  `pytest --pyargs pyibs`, with the `test` extra installed;
  `pyibs.__version__` gives the installed version.

### Changed

- **Sampling.** PyIBS is rebuilt on a tested port of the `ibslike.m`
  sampler, whose two schedules one sampler runs: `vectorized=True`
  requests several samples per trial and simulator call, a number that
  grows from call to call, and `vectorized=False` one sample per trial and
  call. The samples per call follow `ibslike.m`'s formula.
- **Likelihood threshold.** Every repeat is checked against
  `neg_logl_threshold`, on the scale of the weighted negative
  log-likelihood, and a repeat that exceeds it counts exactly
  `-neg_logl_threshold`, as in the IBS paper (Appendix C.1), whatever the
  sampling schedule. The result has exit flag 1 when a repeat was ended.
- **Time limit.** When `max_time` stops the sampling, each trial's value
  averages its completed repeats, a repeat that the likelihood threshold
  ended counts `-neg_logl_threshold`, and the call warns, with exit flag 2;
  a trial with no completed repeat raises `IBSSamplingError`.
- **Responses of one column.** A simulator can return the responses of
  one column, `response_matrix` of shape (N,) or (N, 1), with shape (r,)
  or (r, 1) for r requested trials, as a model ported from MATLAB does;
  for C > 1 columns it returns shape (r, C).
- **Checks of the settings.** `IBS` raises `ValueError` or `TypeError`,
  naming the setting, for a value out of range or of the wrong type, such
  as a negative `acceleration`, a boolean for a count, or a
  non-whole `num_reps` (0.1.0 truncated 2.5 to 2); the counts take
  whole-number floats such as `1e5`. Trial weights must be real numbers,
  not booleans or strings, and an unknown `additional_output` raises
  `ValueError` (0.1.0 returned None).
- **Requirements.** PyIBS needs Python 3.10, NumPy 2.0 and SciPy 1.13 or
  later, the versions that PyBADS 1.5 needs, and no other package.

### Fixed

#### Sampling

- `max_iter` defaulted to 15, so that a repeat in which a trial had not
  matched after 15 rounds was left out of that trial's estimate, which was
  biased, or NaN; `max_iter=1e5` raised `TypeError`.
- A `response_matrix` of several columns raised an error.
- `fun_count` missed simulator calls, the timing call of `vectorized=None`
  among them.
- No seed reproduced a run: the samples requested depended on the timing
  of the simulator calls, and the simulator could only draw from NumPy's
  global state.

#### ibs_basic

- `ibs_basic` works without a design (`S=None`, its default), when the
  simulator receives the trial index; it takes `random_seed`, and passes
  the generator to a simulator that has a parameter named `rng`.

### Removed

- **Exit flag 3.** The cap on the samples raises `IBSSamplingError`.
- **Example modules.** `pyibs.psycho_generator` and `pyibs.psycho_neg_logl`
  are removed; the example model is `pyibs.examples.psycho_model`.

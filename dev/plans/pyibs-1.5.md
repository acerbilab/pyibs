# Plan: PyIBS 1.5

Created: 2026-10-09
Status: APPROVED (2026-10-09); the open questions are settled as D17 to D19

## Summary

Bring PyIBS to the level of PyBADS 1.5 and PyVBMC 1.5 and release it as
1.5.0, a version number that marks it as part of the same generation as
those two. PyIBS 1.5 implements inverse binomial sampling as the MATLAB
function `ibslike.m` does, the reference implementation, with a test
suite, a calling convention that PyBADS 1.5 and PyVBMC 1.5 take directly as
a noisy target, documentation, examples, an agent skill and the release
tooling of the other two. Its sampling engine is the lab's tested Python
port of the `ibslike.m` sampler, which Phase 0 brings into this
repository. From Phase 1 on, every phase but the film runs from a clone
of this repository and its sibling repositories, on any machine.

## Context

[1] is the IBS paper: B. van Opheusden, L. Acerbi and W. J. Ma (2020),
"Unbiased and efficient log-likelihood estimation with inverse binomial
sampling", PLOS Computational Biology 16(12): e1008483,
https://doi.org/10.1371/journal.pcbi.1008483. Its Markdown transcription is
in `../pubs-llms/publications/vanopheusden2020unbiased_{main,appendix,backmatter}.md`.

### PyIBS 0.1.0

The tree on `main` is PyIBS 0.1.0, as published on PyPI and conda-forge:
`pyibs/ibs.py` (the class `IBS` and the result `EstimateResult`),
`pyibs/ibs_basic.py` (a didactic loop), an example model
(`pyibs/psycho_generator.py`, `pyibs/psycho_neg_logl.py`) and three
notebooks inside the package directory. It has no tests, no documentation
site, no CI and no changelog. Its defects, which 1.5 fixes and its
changelog lists under "Upgrading from 0.1.0":

1. The default `max_iter=10 ^ 5` is 15 (`^` is XOR in Python). The loop
   path, which `vectorized=None` takes when one simulation of all trials
   lasts at least 0.1 s, ends each repeat after 15 rounds. A repeat in
   which a trial has not matched by then is left out of that trial's
   average (`num_reps_per_trail = np.sum(K > 0, axis=1)`), so the returned
   negative log-likelihood of a trial with a low probability of matching
   is biased downwards, with a printed warning, and is NaN when no repeat
   matched. The vectorized path caps the rounds at `15 * num_reps`. The
   value that the README documents, `max_iter=1e5`, makes the loop path
   raise `TypeError` (`range` of a float).
2. A response matrix with several columns raises an error in both paths:
   the comparison `simulated_data == self.response_matrix[T]` gives a 2-D
   array that the code assigns or reshapes as 1-D.
3. The loop path counts no simulator call when `design_matrix` is None,
   and the vectorized path never counts the timing call that
   `vectorized=None` makes, which `ibslike.m` counts.
4. No generator reaches the simulator: a run reproduces only through
   `np.random.seed`, and not at all while the acceleration or the choice
   of path depends on timing, as they do by default.
5. Warnings are printed rather than issued through `warnings`.
6. An invalid `additional_output` returns None instead of raising.
7. `ibs_basic` indexes the design (`S[i]`) even when it is None, its
   default.
8. The runtime dependencies include test and plotting packages (cma,
   corner, dill, gpyreg, imageio, matplotlib, plotly, pytest, pytest-mock,
   pytest-rerunfailures), although the package imports only NumPy and
   SciPy.
9. `pyproject.toml` declares the static version 0.1.0 while `setup.py`
   asks setuptools_scm for one, and the repository has no tags.

### The MATLAB reference

`ibslike.m` of https://github.com/acerbilab/ibs (header: Version 0.96,
release date Jan 21, 2021). Options and defaults: `Nreps` 10,
`NegLogLikeThreshold` Inf, `Vectorized` `'auto'`, `Acceleration` 1.5,
`NsamplesPerCall` 0 (meaning the number of repeats), `MaxIter` 1e5 ("per
trial and estimate"), `ReturnPositive` false, `ReturnStd` false, `MaxTime`
Inf, `TrialWeights` empty. Hard-coded: `MaxSamples` 1e4 samples per trial
and call, `AccelerationThreshold` 0.1 s, `VectorizedThreshold` 0.1 s, and
`MaxMem = max(min(N, 1e4), 10) * 100` (assigned after an earlier 1e6,
which it replaces). Outputs: the negative log-likelihood; its variance, or
its SD with `ReturnStd`; an exit flag (0 unbiased, 1 likelihood threshold
reached, 2 `MaxTime` reached); and a structure with `funcCount`,
`NsamplesPerTrial`, `nlogL_trials` and `nlogLvar_trials`. Reaching
`MaxIter` is an error (`ibslike:ConvergenceFail`) in both sampling paths.
`ibslike('test')` runs `runtest1` to `runtest3`. The repository also holds
`ibs_basic.m`, the tutorial `ibs_example.m` and the example model
`psycho_gen.m`, `psycho_nll.m`. Its README sends practical questions to the
FAQ of its wiki (https://github.com/acerbilab/ibs/wiki), which has a
section on `ibslike.m`.

### The engine after Phase 0

Phase 0 brings in the engine and its tests. Afterwards the package holds,
next to the 0.1.0 modules that Phase 1 replaces:

- `pyibs/_estimates.py`: the per-count formulas `ibs_loglik(K) = psi(1) -
  psi(K)` and `ibs_var(K) = psi_1(1) - psi_1(K)`; `trial_weights`, which
  checks per-trial weights (finite, at least 0, a scalar or one per trial);
  and `repeat_estimates`, which turns a matrix of counts into per-repeat
  values, variance estimates and per-trial sums, looking the formulas up in
  tables bitwise equal to them. Two properties keep its results bitwise
  equal whether it reduces the counts in one block or in several, and
  with the likelihood threshold or without, and tests rely on them: the
  sum over trials is `np.sum(... * w, axis=1)`, never `@`, and every
  looked-up term equals its formula bitwise.
- `pyibs/_sampler.py`: the sampler, `ibslike.m`'s vectorized sampling with
  per-repeat output. In a draw of n repeats, every trial's samples form one
  stream, split at its matches into its n counts. Each round asks the
  simulator, in one call, for m samples of every trial still open, in
  trial-major order, with `m = max(1, min(floor(level),
  max_samples_per_call // n_open))`, `max_samples_per_call = 10**6`; the
  level starts at n and is multiplied by the acceleration after every
  call, or only after calls faster than `acceleration_threshold` when one
  is given. It raises `IBSSamplingError` once a trial has drawn more than
  `max_samples_per_trial * n` samples in the draw, surplus included
  (default `10**5`). Its likelihood threshold follows [1], Appendix C.1:
  every repeat is checked against a weighted running bound, and an ended
  repeat is worth exactly -T. It counts the calls, every simulated row and
  the seconds spent in the simulator; matches a response only when every
  column agrees; and raises `TypeError` when the simulated and observed
  responses are of kinds that NumPy never finds equal.
- `pyibs/testing/_exact.py`: exact IBS draws from geometric matching counts
  at given trial probabilities, and the exact log-likelihood and per-repeat
  variance.
- Tests under `pyibs/testing/`: `test_estimates.py`, `test_sampler.py`,
  `test_threshold.py`, `test_exact.py` and `test_ibslike_ports.py`, the
  ports of `runtest1` and `runtest3` (each checked at three seeds and
  passing at two) and of `runtest2` (one seed, `ibslike.m`'s criteria).

### PyBADS 1.5 and PyVBMC 1.5

PyBADS 1.5.1 is on PyPI (2026-10-06). PyVBMC 1.5.0 is prepared on the
branch `feat-release-1.5-preparation` of `acerbilab/pyvbmc`, whose
changelog dates it 2026-10-13; until then PyPI has 1.0.4. PyVBMC 1.5
requires SciPy 1.15 or newer. `AGENTS.md` ("PyBADS and PyVBMC") states what
their noisy-target interface requires of a target.

### Release access

Publishing 1.5.0 needs owner access to the `pyibs` project on PyPI and
maintainer access to `conda-forge/pyibs-feedstock`. The lab obtained both
from the original maintainer on 2026-10-09.

### Parity with `ibslike.m`

| Topic | `ibslike.m` 0.96 | Engine after Phase 0 | PyIBS 1.5 |
| :--- | :--- | :--- | :--- |
| Sampling paths | Vectorized (accelerated) and loop. `'auto'` times one simulation of all trials and takes the loop path at 0.1 s or more, or when `Nreps == 1`; `true` with `Nreps == 1` falls back to the loop path with a warning | One sampler | One sampler; `vectorized` selects its schedule, and `None` is decided once per `IBS` object (D7, D19) |
| Acceleration | Grows the samples per call while a call lasts less than 0.1 s | Grows after every call; the time rule is opt-in | As the engine (D4, deliberate difference) |
| Samples per call | `min(1e4, max(1, round(level)))`, with MATLAB's `round` (halves away from zero), then at most `ceil(MaxMem / n_open)` | `max(1, min(floor(level), 10**6 // n_open))` | `ibslike.m`'s formula and defaults (D6) |
| Likelihood threshold | Vectorized path: checks the lowest repeat still sampled, keeps the partial counts of an ended repeat, compares the unweighted sum. Loop path: checks each repeat, keeps partial counts | Every repeat against a weighted bound; an ended repeat is worth exactly -T | As the engine (D5, deliberate difference) |
| Cap | `MaxIter` rounds (times `Nreps` in the vectorized path); error | More than `max_samples_per_trial * n` samples of one trial in a draw of n repeats; error | As the engine, with `max_iter` as `max_samples_per_trial`, default `10**5` (D3, D6, deliberate difference) |
| Time limit | `MaxTime`, exit flag 2. Vectorized path: a trial's value averages its repeats with a positive count, including the partial count of the repeat it was sampling; NaN for every trial when the limit passed before its first round. Loop path: averages its completed repeats, NaN when it has none, and under a threshold also the partial count c + 1 of the repeat it stopped | None | Each trial's value averages its completed repeats; a trial with none raises; exit flag 2 and a warning; with the likelihood threshold, D24 (D3, D24, deliberate differences) |
| Outputs | Value; variance, or SD with `ReturnStd`; exit flag; `funcCount`, `NsamplesPerTrial`, per-trial values and variances | Per-repeat values and variances, per-trial sums, calls, samples, seconds | 0.1.0's `additional_output` forms, as Python floats; `"full"` adds the per-trial arrays, NaN when the threshold ended a repeat (D2, D10, D21, deliberate difference) |
| Sample count | The vectorized path adds the number of open trials per call, whatever the samples requested of each | Every simulated row | As the engine (deliberate difference) |
| Simulator | `fun(params, dmat, varargin{:})`, global random state | Called with the generator of the draw | `sample_from_model(params, design_rows)`, with `rng=` when its signature has a parameter named `rng` (D8, D18) |
| Matching | Every column must agree (`all(respMat(T,:) == simdata, 2)`); only the number of returned rows is checked | Every column must agree, and the output has the shape of the requested responses; refuses kinds that NumPy never finds equal | As the engine, except that one-column responses take an output of shape (r,) or (r, 1) for r rows requested, and a NaN response raises when `IBS` is created (D22, D23, deliberate differences) |
| Self-tests | `runtest1` to `runtest3` | Ports of all three | Ports of all three, through `IBS` |

## Scope

- **In scope**: the phases below. That is, the engine brought in with its
  tests; the package rebuilt on it with `ibslike.m`'s options and outputs
  and 0.1.0's calling convention; a review against `ibslike.m` with a
  catalogue of the deliberate differences; a statistical validation, runs
  with PyBADS and PyVBMC, and timings; packaging, changelog, CI and release
  workflows; README, documentation site, FAQ, examples, API pages, the
  update check and the agent skill; the 1.5.0 release on GitHub, PyPI and
  conda-forge; the links that other repositories need for it; and the
  film.
- **Out of scope**: any feature beyond `ibslike.m` that no decision below
  adds. Changes to PyBADS, PyVBMC, gpyreg and MATLAB IBS, other than the
  links of Phase 6.

## Conventions for every phase

- Work in this repository on the branch `dev-next`. `$PY` below is the
  venv's interpreter: `.venv/Scripts/python.exe` on Windows,
  `.venv/bin/python` elsewhere. A fresh clone first creates the venv as
  `AGENTS.md` ("Setup and commands") states, `dev/scripts/runs/` with
  `mkdir -p dev/scripts/runs`, and the sibling checkouts that the phase
  names, as `AGENTS.md` ("Sibling repositories") lists them.
- Phase 0 runs on the PI's machine. From Phase 1 on, no step reads
  `dev/private/` or any path that a clone of this repository and the
  public siblings lacks, except the film (Phase 7).
- Commit at the end of each phase, or of each group of steps a phase
  names, once its verification passes; conventional commits, as
  `AGENTS.md` states. Nothing is pushed, and no pull request, tag or
  release is made, without the PI's instruction.
- On the PI's workstation, one heavy process at a time (`AGENTS.md`,
  "Setup and commands"); a cloud session may run them in parallel. Long
  runs are unbuffered and logged as `dev/scripts/runs/<name>_$(date
  +%s).log`.
- Every random draw goes through an explicit `numpy.random.Generator`;
  every random test is seeded; statistical tolerances are stated in
  standard errors, 4.5 by default; a failing statistical test is
  investigated, never reseeded.
- When a check contradicts an assumption that a phase's steps rest on,
  stop and report the mismatch to the PI rather than improvise around it.
- After each phase, review its diff with `/doublecheck`, or, where that
  skill is unavailable, with read-only Opus sub-agents that did not do the
  work, briefed with the phase's goal and steps; fix what must be fixed,
  set the phase's Status, and record the phase in the Worklog below:
  the date, the commits, the verification's outcome and any deviation
  from the steps.

## Phases

### Phase 0: tooling, references and the engine

**Status**: done
**Executor**: Opus (orchestrator), on the PI's machine
**Needs**: `dev/private/extraction.md` and what it names; `../pybads`;
`../pyvbmc` at its branch `feat-release-1.5-preparation` (the CI matrix).
**Goal**: the repository's tooling at the level of PyBADS's, the MATLAB
reference at hand, and the engine with its tests in the package, pushed so
that every later phase can start from a clone.

**Steps**:
1. [x] Commit the plan and the files that come with it, as they stand:
   `git add AGENTS.md CLAUDE.md dev .gitignore`, then check with
   `git status --short` that nothing under `dev/private/` is staged, and
   commit as `docs: plan for PyIBS 1.5, AGENTS.md and CLAUDE.md`.
2. [x] Line endings: copy `../pybads/.gitattributes`, `git add .gitattributes`,
   `git add --renormalize .`, and commit the result alone as
   `chore: normalize line endings`.
3. [x] Clone the MATLAB reference and its wiki:
   `git clone https://github.com/acerbilab/ibs ../ibs` and
   `git clone https://github.com/acerbilab/ibs.wiki.git ../ibs.wiki`.
   Record `git -C ../ibs rev-parse HEAD`, `git -C ../ibs.wiki rev-parse
   HEAD` and the `Version` and `Release date` lines of `../ibs/ibslike.m`
   in the Worklog. Expected: Version 0.96. If the header shows another
   version, or `ibslike.m` differs from the description under Context,
   stop and report.
4. [x] Packaging, after `../pybads/pyproject.toml`, `MANIFEST.in` and
   `.gitignore`:
   - `pyproject.toml`: `name = "PyIBS"`, `dynamic = ["version"]`, the
     description, `readme`, `license = "BSD-3-Clause"`,
     `license-files = ["LICENSE"]`;
     `dependencies = ["numpy >= 2.0.0", "scipy >= 1.13.0"]` and
     `requires-python = ">=3.10"` (D9); `[project.urls]` as PyBADS's, with
     `https://acerbilab.github.io/pyibs/` for the documentation and
     `acerbilab/pyibs` for the rest; extras `test = ["pytest >= 6.2.5"]`
     and `dev` = the `test` packages, `pytest-cov`, `pre-commit`, `build`,
     and PyBADS's documentation packages (`sphinx`,
     `sphinx-book-theme >= 1.0.0`, `numpydoc`, `myst_nb`);
     `[tool.setuptools]` with `include-package-data = true` and
     `packages = ["pyibs", "pyibs.testing"]`; `[build-system]` with
     `setuptools >= 77`; `[tool.setuptools_scm]` with
     `write_to = "pyibs/_version.py"`; `[tool.pytest.ini_options]` with
     `testpaths = ["pyibs/testing"]`,
     `markers = ["integration: runs PyBADS or PyVBMC"]` and
     `addopts = "-m 'not integration'"`; black, isort and pycln as they
     are. `setup.py` stays the shim it is.
   - `MANIFEST.in` from PyBADS's, with its `prune dev` and `prune docsrc`.
   - `.gitignore`: `pyibs/_version.py`, `docs/`, `docsrc/_build/`,
     `docsrc/source/_examples/` and `.venv-*/`.
   Commit as `build: packaging for PyIBS 1.5`.
5. [x] Pre-commit and the venv: copy `../pybads/.pre-commit-config.yaml` (its
   hook versions), keeping the `exclude` patterns that apply here. Create
   the venv, `uv venv --python 3.12 .venv`, `uv pip install -e ".[dev]"`,
   `$PY -m pre_commit install`. Run `$PY -m pre_commit run --all-files`
   and commit the reformatting alone as `style: format the tree with the
   pre-commit hooks`; list that commit's hash in a new
   `.git-blame-ignore-revs`, as PyBADS does, in a second commit.
6. [x] `CHANGELOG.md` with PyBADS's header (Keep a Changelog) and an empty
   `## [Unreleased]`. Commit.
7. [x] The engine: follow `dev/private/extraction.md`, which yields
   `pyibs/_estimates.py`, `pyibs/_sampler.py`, `pyibs/testing/_exact.py`,
   `pyibs/testing/_helpers.py` and the tests listed under Context. If that
   file is absent, stop. Commit as `feat: IBS engine and its tests`.
8. [x] Check the installation and the build:
   `$PY -c "import importlib.metadata as m; print(m.version('pyibs'))"`
   prints a development version from setuptools_scm;
   `$PY -m build` builds an sdist and a wheel; the wheel's `METADATA`
   requires only NumPy and SciPy outside the extras
   (`unzip -p dist/*.whl '*/METADATA' | grep Requires-Dist`); and the sdist
   holds no file of `dev/` (`tar -tzf dist/*.tar.gz | grep /dev/` prints
   nothing).
9. [x] CI: copy `../pybads/.github/workflows/test-matrix.yml`,
   `merge-tests.yml` and `tests.yml`, adapted: no gpyreg pin and no drift
   run (PyIBS does not depend on gpyreg); paths `pyibs/`,
   `pyproject.toml` and `setup.py`; the matrix Ubuntu, Windows and macOS ×
   Python 3.10 to 3.14, as on PyVBMC's release branch. Check that the
   files parse:
   `$PY -c "import sys, yaml; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml`
   (PyYAML comes with pre-commit). Commit as `ci: test workflows`.
10. [x] `AGENTS.md`: "Setup and commands" gains the version from git tags
    through setuptools_scm and the generated `pyibs/_version.py`, the test
    command (`$PY -m pytest`), the CI workflows and what triggers each,
    the pre-commit hooks as the only enforcement of formatting, and
    `.git-blame-ignore-revs`; the install line drops its separate
    `pre-commit`, which the `dev` extra now holds. A section "What spans
    files" states the two bitwise properties of `repeat_estimates`. Under
    "The project", the sentence on the 0.1.0 code becomes: the package
    holds the engine (`pyibs/_estimates.py`, `pyibs/_sampler.py`) and its
    tests next to the 0.1.0 interface, which Phase 1 replaces. Commit.
11. [x] `/doublecheck` on the phase. Then, on the PI's instruction,
    `git push -u origin dev-next`, and check that the smoke run of
    `tests.yml` passes (`gh run list --branch dev-next`).

**Verification**:
- [x] `$PY -m pytest` passes; the Worklog records the number of tests and
      the runtime.
- [x] The check of `dev/private/extraction.md` passes, and
      `git ls-files dev/private` prints nothing.
- [x] `$PY -m pre_commit run --all-files` passes.
- [x] The installation, build and YAML checks of steps 8 and 9 pass.
- [x] After the push, the smoke run of `tests.yml` passes.

### Phase 1: the interface and parity with `ibslike.m`

**Status**: done
**Executor**: Opus (orchestrator); the tests of step 6 may go to an Opus
sub-agent once steps 2 to 5 are committed.
**Needs**: `../ibs`, `../pybads`.
**Goal**: the package rebuilt on the engine, with the interface of D2 and
the behaviour of the parity table's last column.

The layout at the end of the phase:

```
pyibs/
  __init__.py      IBS, EstimateResult, IBSSamplingError, ibs_basic, __version__
  ibs.py           class IBS (the interface) and EstimateResult
  _sampler.py      the sampler
  _estimates.py    the per-count formulas and their reduction
  ibs_basic.py     the didactic loop
  README.md        the catalogue of deliberate differences from ibslike.m
  testing/         the tests, with _exact.py and _helpers.py
examples/
  psycho_model.py  the orientation discrimination model, installed as
                   pyibs.examples.psycho_model
```

**Steps**:
1. [x] Record `git -C ../ibs rev-parse HEAD` in the Worklog; if it differs
   from Phase 0's, say what changed in `ibslike.m` and `ibs_basic.m`.
2. [x] `pyibs/_sampler.py`, per the parity table, reading `../ibs/ibslike.m`
   for each item:
   - samples per call (D6): `m = min(max_samples, max(1,
     math.floor(level + 0.5)))`, then `m = min(m, ceil(max_mem /
     n_open))`, with `max_samples = 10**4` and `max_mem = max(min(N,
     10**4), 10) * 100` unless given. `math.floor(level + 0.5)` is
     MATLAB's `round` for a positive level; `np.round` and `round` round
     halves to even, and differ at levels such as 22.5. The level is
     bounded by `max_samples` after each multiplication, which changes no
     `m` and keeps it finite (an unbounded level overflows to inf after
     enough calls, and `math.floor(inf)` raises).
   - `vectorized` (D7, D19): `False` requests one sample per open trial
     per call, without acceleration; `True` the accelerated schedule.
     `None` is decided at the object's first call with `num_reps > 1` by
     timing one simulation of all trials against `vectorized_threshold`,
     as `ibslike.m` does (`False` at the threshold or above), and the
     decision is kept for the object's later calls, readable as an
     attribute. A call with `num_reps == 1` requests one sample per open
     trial per call whatever the setting, as `ibslike.m` does; `True` with
     `num_reps == 1` falls back to `False` with a warning, as in
     `ibslike.m`.
   - `max_time` (D3): checked after every call; once exceeded, sampling
     stops, each trial's value averages its completed repeats, a trial
     with none raises `IBSSamplingError`, and the exit flag is 2. Under the
     likelihood threshold, the repeats it ended count -T each and the
     others are averaged per trial, as D24 states.
   - The shape check of `_simulate` (D22): when the responses have one
     column, an output of shape (r,) or (r, 1) for r rows requested is
     compared with that column; otherwise the output's shape is that of
     the requested responses. The case of `test_wrong_simulator_output_raises`
     (`test_sampler.py`) that gives responses of shape (3,) an output of
     shape (r, 1) moves to a test that it matches, and the docstrings that
     state the shape rule (`_Settings`, `sample`, `_simulate`) follow.
   - The sampler's tests whose expectations depend on the schedule (exact
     calls, samples per call) are updated, and the Worklog lists each;
     the statistical tests, the ports of `runtest1` to `runtest3` among
     them, stay as they are: one that fails after the change of schedule
     is investigated, never reseeded (the port of `runtest2` checks
     `ibslike.m`'s fixed tolerances at one seed).
3. [x] `pyibs/ibs.py` replaces 0.1.0's, with the class `IBS` (D2, D3, D8, D10,
   D16, D21, D22, D23):
   - `IBS(sample_from_model, response_matrix, design_matrix=None,
     vectorized=None, acceleration=1.5, num_samples_per_call=0,
     max_iter=10**5, max_time=np.inf, max_samples=10**4,
     acceleration_threshold=None, vectorized_threshold=0.1, max_mem=None,
     neg_logl_threshold=np.inf, *, random_seed=None)`: 0.1.0's parameters
     in 0.1.0's order, `design_matrix` defaulting to None, and
     `random_seed` keyword-only. `num_samples_per_call=0` means the number
     of repeats; `max_iter` is the engine's `max_samples_per_trial` (D6);
     `max_mem=None` is `ibslike.m`'s formula. The count settings take
     integers and whole-number floats (D16). The constructor validates
     every setting, raising `ValueError` or `TypeError` with a message that
     names the setting and what it takes. Responses holding a NaN, an
     element not equal to itself, raise `ValueError`, whose message names
     those trials as the cap's error does and suggests recoding the
     response or removing the trial; the design is not checked (D23).
   - `random_seed` takes what `options['random_seed']` takes in PyBADS
     (the `rng` attribute in `../pybads/pybads/bads/bads.py`): None
     derives the generator from NumPy's global state, an integer or a
     `SeedSequence` seeds a new one, a `Generator` is used as given. The
     object keeps it as `self.rng`, and every call draws from it. The
     simulator is called with `rng=self.rng` when its signature has a
     parameter named `rng`, and otherwise as
     `sample_from_model(params, design_rows)` (D18); a callable whose
     signature `inspect.signature` cannot read counts as having none.
   - `__call__(params, num_reps=10, trial_weights=None,
     additional_output=None, return_positive=False)` returns the negative
     log-likelihood (the log-likelihood with `return_positive=True`) as a
     Python `float`; with `"var"` or `"std"`, a `tuple` of two Python
     floats; with `"full"`, an `EstimateResult` holding 0.1.0's fields
     (`neg_logl`, `neg_logl_var`, `neg_logl_std`, `exit_flag`, `message`,
     `elapsed_time`, `num_samples_per_trial`, `fun_count`) and the
     per-trial arrays `neg_logl_trials`, the engine's
     `-trial_value_sums / num_reps`, and `neg_logl_var_trials`,
     `trial_var_sums / num_reps**2`, both unweighted as in `ibslike.m`
     (lines 213, 220-221) and NaN when the likelihood threshold ended any
     repeat (D21); all are shown by its `__repr__`. `"none"` is accepted
     as None, as in 0.1.0; any other value raises `ValueError`. As in
     `ibslike.m`, `return_positive` changes the sign of the total only.
     `trial_weights` refuse booleans and numeric strings, as the settings
     do.
   - Exit flags 0, 1 and 2, each with a message (0.1.0's until Phase 4
     reworded them after `ibslike.m`'s descriptions); reaching
     `max_time` also issues a `UserWarning` (D3). The cap raises
     `IBSSamplingError`, a `RuntimeError` naming the trials over the cap
     by their 0-based indices and the cap as `max_iter * num_reps`, where
     the engine's message says `max_samples_per_trial * n`.
   - A zero variance is returned as computed, with the `UserWarning` of
     D10; its link points to the FAQ answer that Phase 4 writes, under the
     published documentation's address.
4. [x] `pyibs/ibs_basic.py`: read `../ibs/ibs_basic.m`; keep the function's
   signature, make a missing design work (the simulator then receives the
   trial index, as `IBS` does), compare every column, pass the generator as
   `IBS` does, and raise on a NaN response as `IBS` does (D23), where
   `ibs_basic.m` loops forever. `pyibs/__init__.py` exports the public
   names and `__version__` (from `importlib.metadata`).
5. [x] The example model: `examples/psycho_model.py`, after
   `../ibs/psycho_gen.m` and `psycho_nll.m`, with the simulator and the
   closed-form log-likelihood. In `pyproject.toml`, `packages` gains
   `pyibs.examples`, with `package-dir = {"pyibs.examples" = "examples"}`
   and `[tool.setuptools.package-data]` `"pyibs.examples" = ["*.ipynb"]`,
   as in PyBADS. Remove `pyibs/psycho_generator.py`,
   `pyibs/psycho_neg_logl.py` and the three notebooks from `pyibs/`
   (D13). Commit steps 2 to 5, with the changelog entries of their
   changes (`AGENTS.md`, "Changelog").
6. [x] Tests:
   - `test_ibs.py`: each output form and its types (`type(res) is tuple`,
     Python floats); `return_positive`; scalar and per-trial weights;
     responses with several columns, text responses and a design of None;
     responses of shape (N,) and (N, 1), each with outputs of shape (r,)
     and (r, 1), and the outputs that raise (one column for responses of
     two, two columns for responses of one); NaN responses that raise
     (float, complex, `NaT`, in one column of several, in an object array)
     and text responses that do not; trial weights that are booleans or
     numeric strings; the exit flags; the cap's error and its names; the
     per-trial arrays, NaN when the threshold ended a repeat and otherwise
     adding up, weighted, to the negative log-likelihood, and the
     variances, with the weights squared, to its variance
     (`np.testing.assert_allclose`, since they agree to rounding), whatever
     `return_positive`; the warning on a zero variance (trials that always
     match); `max_time` with a simulator that sleeps, with margins wide
     enough for slow CI runners; the reduction of D24 on given counts and
     ended repeats, without timing (its value, its variance, its two
     limits, the exit flag and a trial with no completed count); validation
     errors, and whole-number floats accepted for the counts;
     reproducibility (two objects with one seed and a simulator that takes
     the generator give equal estimates; a simulator without a generator
     parameter works); and agreement in distribution of the `vectorized`
     settings.
   - `test_ibs_basic.py`, a NaN response among its cases, and
     `test_examples.py`: the IBS estimate of the example model agrees with
     its closed form within 4.5 standard errors at three seeded parameter
     vectors.
   - `test_ibslike_ports.py` calls the public `IBS`.
7. [x] `pyibs/README.md`: the catalogue of deliberate differences from
   `ibslike.m` 0.96, after `../pybads/pybads/bads/README.md`, one entry per
   deliberate difference of the parity table and of steps 2 to 4, each
   with its reason. Every statement about what `ibslike.m` does is checked
   against `../ibs/ibslike.m` and cites its lines; descriptions of
   `ibslike.m` in the engine's docstrings are not copied unchecked.
8. [x] `CHANGELOG.md`, under `Unreleased`: the "Upgrading from 0.1.0" list
   that opens the section, covering the defects listed under Context and
   the changed behaviour: the cap raises; `max_iter` counts samples;
   `max_mem` defaults to `ibslike.m`'s formula instead of 1e6;
   acceleration is deterministic by default; the threshold's semantics;
   a NaN response raises when `IBS` is created (0.1.0 sampled it until
   its iteration limit, exit flag 3); the example modules leave the
   package; Python 3.10 or newer. Check that every change of the phase
   has its entry.
9. [x] `AGENTS.md`: an "Architecture" section (the modules and what each
   owns); under "What spans files", the catalogue as the list of
   deliberate differences, which a change that adds or removes one
   updates; in the convention "Changelog", the catalogue beside the
   records under `dev/` as the place of an entry's reasons; a section
   "Tests and their traps" (the `max_time` test's timing margins); and the
   sentence under "The project" that says the tree holds the 0.1.0 code
   goes. Commit.

**Verification**:
- [x] `$PY -m pytest` passes, the ports of `runtest1` to `runtest3`
      included; the Worklog records the number of tests and the runtime.
- [x] `$PY -m pre_commit run --all-files` passes.
- [x] Every row of the parity table is implemented as its last column
      says, and the catalogue lists every deliberate difference.

### Phase 2: review against `ibslike.m`

**Status**: done
**Executor**: Opus (orchestrator), with two Opus sub-agents that only read
and reason (no test runs or other heavy processes).
**Needs**: `../ibs`, `../pubs-llms`.
**Goal**: every difference between PyIBS and `ibslike.m` found, and either
listed in the catalogue with its reason or fixed.

**Steps**:
1. [x] Reviewer A compares `../ibs/ibslike.m` and `../ibs/ibs_basic.m` (the
   commit recorded in Phase 1) with `pyibs/` line by line: options and
   defaults, validation, the sampling schedule, the threshold, the cap,
   the time limit, outputs, exit flags, errors and the self-tests. It
   reports each behavioural difference with file and line on both sides,
   and checks every statement about `ibslike.m` in the docstrings and the
   catalogue against its source.
2. [x] Reviewer B checks the package against [1] and for internal
   correctness: the estimator and its variance estimate, the weights, the
   independence of the repeats under the sampler's schedule, the
   threshold of Appendix C.1, the cost counts, and edge cases (one trial,
   `num_reps=1`, trials that always match, a zero variance, text, bytes
   and object responses, a design of None).
3. [x] Consolidate both reports into the ledger
   `dev/results/<YYYY-MM-DD>-port-review.md`: each finding with its
   verdict (deliberate difference, defect, or no issue) and its fix,
   catalogue entry or `dev/TODO.md` item. The PI rules on the verdicts.
4. [x] Fix the defects, each with a test that fails before the fix, and
   update the catalogue and the changelog. Index the ledger in
   `dev/README.md`. Commit.

**Verification**:
- [x] Every finding of the ledger has a verdict and an outcome.
- [x] The suite and the pre-commit hooks pass.

### Phase 3: statistical validation, PyBADS and PyVBMC, timing

**Status**: done
**Executor**: Opus (orchestrator), running one heavy process at a time on
the PI's workstation.
**Needs**: PyBADS and PyVBMC as step 2 installs them.
**Goal**: evidence that PyIBS 1.5's estimates are unbiased and calibrated
across models and settings, that PyBADS 1.5 and PyVBMC 1.5 run with it as
their target, and how its speed compares with 0.1.0's.

**Steps**:
1. [x] `dev/scripts/validate.py`, over these models, each with an exact
   log-likelihood:
   - Bernoulli: 100 trials at each p in {0.001, 0.01, 0.1, 0.5, 0.9,
     0.999}, the responses drawn at that p;
   - categorical: 200 trials with 2, 4 and 8 outcomes, the outcome
     probabilities drawn once from a seeded Dirichlet;
   - two response columns: 100 trials, independent columns;
   - weights: the Bernoulli model at p = 0.5 with integer weights from 1
     to 3 and with fractional weights from 0.2 to 2;
   - text responses: the categorical model with 4 outcomes, as strings;
   - the example model, `pyibs.examples.psycho_model`, with 600 trials at
     three parameter vectors.
   Settings: `vectorized` True, False and None; `num_reps` 1, 10 and 100;
   the threshold off, and at the chance level `sum_i w_i log k_i` for the
   Bernoulli and categorical models. The script first runs a smoke pass
   of 100 estimates per cell, timing each cell, and prints the projected
   runtime of 2,000 estimates per cell; the PI approves the full run, or
   drops or shrinks cells, before it starts. Per cell it reports, over
   seeded estimates e - exact: the mean error in standard errors; the
   mean squared z-score; the 95% coverage; and, at `vectorized=False`
   (which draws no surplus), the samples per trial against their
   expectation `num_reps * mean_i(1 / p_i)` (the accelerated schedules
   report the ratio, surplus included). With the threshold, the mean is
   compared with the mean of `max(Y, -T)` over 10**5 exact draws from
   `pyibs/testing/_exact.py`, combining both standard errors. A cell
   passes when the bias is within 4.5 standard errors of zero and, at
   `num_reps` of 10 or more without the threshold, the mean squared
   z-score within 4.5 standard errors of 1. The calibration and the
   samples are reported, not gated, at `num_reps=1` and in the threshold
   cells: the variance estimate of a repeat that the threshold ends
   describes its counts when it was ended, not the variance of
   `max(Y, -T)`, and ended repeats draw fewer samples; elsewhere at
   `vectorized=False`, the samples are within 4.5 standard errors of
   their expectation. In the models whose data hold no rare response, so
   that all their trials match with probability 0.999, the calibration is
   reported and not gated either, since exact IBS fails its gate there
   too, and every cell is gated instead on the number of estimates whose
   variance estimate is 0, binomial with its exact probability (PI,
   2026-10-09, after the smoke pass).
   Separately, a model whose trials all have p = 1 returns a value and a
   variance of exactly 0. Report every failing cell to the PI before
   going on.
2. [x] Integration tests, `pyibs/testing/integration/test_pybads.py` and
   `test_pyvbmc.py`, with the marker `integration`, and each skipping
   through `pytest.importorskip` when its package is absent (the wheel
   ships the tests, and `pytest --pyargs pyibs` ignores `addopts`);
   `packages` gains `pyibs.testing.integration`. Install PyBADS with
   `uv pip install "pybads>=1.5.1"`, and PyVBMC with
   `uv pip install "pyvbmc>=1.5.0"` once released, before that with
   `uv pip install "pyvbmc @ git+https://github.com/acerbilab/pyvbmc@feat-release-1.5-preparation"`.
   Each test fits the example model with IBS as the noisy target and
   checks: for PyBADS, that the exact negative log-likelihood at the
   returned point is within 1 of the exact minimum, found by optimizing
   the closed form; for PyVBMC, that the run completes with a finite
   ELBO and that the variational posterior's mean lies within three of
   its SDs of the exact maximum-likelihood point in each coordinate. Run
   each file alone, unbuffered, logged under `dev/scripts/runs/`:
   `$PY -u -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v > dev/scripts/runs/it_pybads_$(date +%s).log 2>&1`.
   `AGENTS.md` gains the procedure.
3. [x] `dev/scripts/timing.py`: the wall time of one estimate at N = 100,
   1,000 and 10,000 trials and `num_reps` 10 and 100, with a fast
   simulator and one that sleeps 0.2 s per call, for PyIBS 1.5 and for
   0.1.0, installed from PyPI in a separate venv (`uv venv --python 3.12
   .venv-0.1`, `uv pip install --python .venv-0.1 pyibs==0.1.0`). 0.1.0
   runs with `max_iter=10**5` (an integer: its default is 15, and the
   float `1e5` stops its loop path; Context, defect 1), so that both
   versions sample every repeat to completion. As in step 1, a smoke pass
   projects the runtime first, since 0.1.0's loop path with the slow
   simulator can take minutes per estimate, and the PI approves the full
   run. MATLAB is not part of the comparison.
4. [x] The record: `dev/experiments/validation_<YYYYMMDD>/` with its
   `README.md` and provenance (`dev/README.md`), and the summary
   `dev/results/<YYYY-MM-DD>-validation.md`, both indexed in
   `dev/README.md`. Commit.

**Verification**:
- [x] Every cell of step 1 passes, or the PI has ruled on its failure.
- [x] Both integration files pass.
- [x] The record holds its provenance.

### Phase 4: documentation, examples, update check and skill

**Status**: done
**Executor**: Opus (orchestrator); the documentation site, the FAQ and the
notebooks may each go to an Opus sub-agent, one at a time for anything
that runs code on the PI's workstation.
**Needs**: `../pybads`, `../ibs`, `../ibs.wiki`, `../pubs-llms`; the
packages that the notebooks import, installed with
`uv pip install nbconvert ipykernel matplotlib "pybads>=1.5.1"` and PyVBMC
as in Phase 3, step 2.
**Goal**: the user-facing material of PyBADS 1.5, scaled to PyIBS.

**Positioning.** The README's "When should I use PyIBS?", `index.rst`, the
FAQ, the skill and the film say when IBS is the right tool as someone who
knows the alternatives would, every claim cited:
- IBS gives unbiased estimates of the log-likelihood, with a calibrated
  estimate of their variance, for a model that can be simulated, on data
  with discrete responses, each trial conditioned on its own context. It
  is the bridge from a simulator to likelihood-based methods:
  maximum-likelihood or maximum-a-posteriori estimation with PyBADS,
  posteriors and model evidence with PyVBMC, and model comparison.
- Amortized simulation-based inference (neural posterior or likelihood
  estimation) is often the better choice when one model is fitted to many
  datasets and its simulations are cheap; the text says so plainly.
- IBS remains the method of choice where the trials' contexts are many
  and richly structured, which makes amortization hard: an amortized
  estimator has to learn the model's behaviour across every context it
  may meet, while IBS only simulates the model in the contexts of the
  data. Its one example is a model of how people play a board game, which
  chooses each move from the current position on the board, a position
  that may occur only once in the data ([1], Section 5.4; B. van
  Opheusden et al., 2023, "Expertise increases planning depth in human
  gameplay", Nature 618: 1000–1005). And it serves where per-dataset
  guarantees matter: amortized estimates can fail on a given dataset, and
  need diagnostics and a fallback (C. Li et al., 2026, "Amortized Bayesian
  Workflow", Transactions on Machine Learning Research,
  https://openreview.net/forum?id=osV7adJlKD), while IBS's estimates are
  unbiased on every dataset, without training (PI, 2026-10-09).
- Its costs: about 1/p_i samples for trial i, so improbable responses are
  expensive (the likelihood threshold bounds the cost at poor
  parameters); responses must be discrete, or binned.
The Nature reference is verified (authors, volume, pages, DOI) before it
is cited; the transcriptions in `../pubs-llms/publications/`
(`vanopheusden2020unbiased_*`, `li2026amortized_*`) are the sources for
the other two.

**Steps**:
1. [x] The update check (D12): `pyibs/_update_check.py` with
   `check_for_updates()`, exported by `pyibs/__init__.py`, copied and
   adapted from `../pybads/pybads/_update_check.py`, with its tests
   (which do not reach the network). It keeps PyBADS's rule that its
   networking modules are imported inside the function.
   `../pybads/dev/plans/version-check.md` is the design; its old-release
   reminder and `RELEASE_DATE` are not carried over. Its changelog entry
   comes with it.
2. [x] `examples/`: notebooks 1, basic use and calibration, after
   `../ibs/ibs_example.m`; 2, maximum-likelihood estimation with PyBADS;
   3, posterior and evidence with PyVBMC; and
   `examples/scripts/Makefile` after PyBADS's. Rerun the notebooks with
   the venv's interpreter first on `PATH`:
   `PATH="$PWD/.venv/Scripts:$PATH" make -C examples/scripts run`
   (`.venv/bin` elsewhere), and commit their outputs.
3. [x] `README.md`, after `../pybads/README.md`'s sections: What is it?,
   What's new in PyIBS 1.5, Documentation, When should I use PyIBS?,
   Installation, Quick start (with the calls for PyBADS and PyVBMC), Next
   steps, How does it work?, Troubleshooting and contact, References and
   citation (with BibTeX), License, Acknowledgments. "When should I use
   PyIBS?" follows the Positioning above. Links follow `AGENTS.md`, "Links
   to the lab". With the README rewritten, no code or text of 0.1.0
   remains, and `LICENSE` (BSD 3-Clause) names the lab as its copyright
   holder, as PyBADS's does: `Copyright (c) 2026, acerbilab`. If any part
   of 0.1.0 is kept after all, its copyright line stays beside the lab's,
   as the licence requires.
4. [x] `docsrc/` after `../pybads/docsrc/`: `Makefile`, `make.bat`,
   `.nojekyll` (which the `github` target copies into `docs/`), and under
   `docsrc/source/`: `conf.py`, `index.rst`, `installation.rst`,
   `quickstart.rst`, `documentation.rst`, hand-written API pages under
   `api/` (`IBS`, `EstimateResult`, `IBSSamplingError`, `ibs_basic`,
   `check_for_updates`), `examples.rst`, `faq.md`, `development.rst`,
   `about_us.rst`, `_static/` and `css/`.
5. [x] `docsrc/source/faq.md`: port the FAQ of `../ibs.wiki` (the commit of
   Phase 0), translated to PyIBS's names, with its section on `ibslike.m`
   rewritten for `IBS`. Add answers on: when to use IBS rather than
   amortized simulation-based inference, per the Positioning above; the
   settings for PyBADS and PyVBMC
   (the negative log-likelihood and the log-likelihood, the SD as a
   tuple's second element); a zero SD, and why PyBADS and PyVBMC refuse
   it; choosing `num_reps` and the threshold; reproducibility with
   `random_seed`; and the differences from MATLAB, which link the
   catalogue. A label linked from elsewhere is listed in `AGENTS.md`, as
   PyBADS's are.
6. [x] `skills/pyibs/SKILL.md` after `../pybads/skills/pybads/SKILL.md`; what
   it tells an agent about when PyIBS fits a problem follows the
   Positioning above.
7. [x] Build the documentation:
   `PATH="$PWD/.venv/Scripts:$PATH" make -C docsrc github`
   (`.venv/bin` elsewhere).
8. [x] `AGENTS.md`: the documentation build, the FAQ's linked labels, the
   examples and their rerun, and the network rule of the update check, as
   in `../pybads/AGENTS.md`. Commit.

**Verification**:
- [x] The documentation builds without warnings.
- [x] `make -C examples/scripts run` reruns every notebook without error.
- [x] The suite and the pre-commit hooks pass.

### Phase 5: release

**Status**: pending
**Executor**: Opus (orchestrator); each outward step on the PI's
instruction.
**Needs**: `../pybads`; the GitHub CLI `gh`, signed in to an account that
can fork repositories.
**Goal**: PyIBS 1.5.0 on GitHub, PyPI and conda-forge, by the procedure of
`../pybads/AGENTS.md`, "Setup and commands".
**Timing**: steps 1 to 3 change nothing outside the repository and may run
before the release (PI, 2026-10-09), once the PI says to start: the
workflows of step 1 run only on `main`, on a published release or on
dispatch; the gate of step 2 is run again on the head that goes into the
pull request, its integration tests and notebooks with PyVBMC 1.5.0 from
PyPI; step 3 takes the expected release date, corrected if the release
moves, and from then on a change for 1.5.0 goes into the section
`[1.5.0]`, which `AGENTS.md` ("Changelog") then says. PyVBMC 1.5.0 is on
PyPI, and its documentation published, before step 4: the README and the
installation pages install `pyvbmc>=1.5` and link that documentation.

**Steps**:
1. Copy `../pybads/.github/workflows/build.yml`, `release.yml` (trusted
   publishing through the `pypi` environment, which admits only `v*`
   tags) and `docs.yml`, adapted to PyIBS. Add the release procedure to
   `AGENTS.md`. Commit.
2. The release gate: the full suite, the pre-commit hooks, the
   integration tests, the documentation build, and the notebooks rerun.
   Then `/doublecheck` on the whole of `dev-next` against `main`.
3. `CHANGELOG.md`: `Unreleased` becomes `[1.5.0] - <date>` under a new,
   empty `Unreleased`. Commit.
4. The PI creates the branch `gh-pages` on `origin`, which `docs.yml`
   checks out, holding only an empty `.nojekyll` at its root, as PyBADS's
   and PyVBMC's do (`docs.yml` copies `docs/*`, which skips dotfiles, and
   without `.nojekyll` GitHub Pages drops `_static/` and serves the site
   unstyled): `git switch --orphan gh-pages`, `touch .nojekyll`,
   `git add .nojekyll`, `git commit -m "docs: gh-pages"`,
   `git push origin gh-pages`, `git switch dev-next`. The PI sets GitHub
   Pages to serve it, pushes `dev-next`, and opens the pull request into
   `main`. CI passes; the PI squash-merges it, and sets the branch
   protection of `main` to checks that report on every pull request: the
   test jobs report only when `merge-tests.yml` runs them (`AGENTS.md`,
   "Setup and commands"), so requiring one holds every pull request that
   skips them.
5. The PI adds the trusted publisher on PyPI (repository
   `acerbilab/pyibs`, workflow `release.yml`, environment `pypi`) and
   creates that environment on GitHub.
6. Tag `v1.5.0` on `main` and publish the GitHub release, its notes the
   changelog's section with each paragraph joined onto one line.
   `release.yml` uploads to PyPI. Check that
   `curl -s https://pypi.org/pypi/pyibs/json` reports version 1.5.0, and
   that a fresh venv outside the repository passes the installed tests:
   from the parent directory, `uv venv pyibs-release-check`,
   `uv pip install --python pyibs-release-check "pyibs[test]==1.5.0"`,
   then `pyibs-release-check/Scripts/python.exe -m pytest --pyargs pyibs`
   (`pyibs-release-check/bin/python` elsewhere). If the workflow fails,
   rerun it; a release that installs broken is yanked on PyPI and fixed in
   1.5.1.
7. The conda-forge recipe: from the parent directory,
   `gh repo fork conda-forge/pyibs-feedstock --clone`, which creates
   `../pyibs-feedstock`; in its `recipe/meta.yaml`, the version,
   the sdist's sha256 from PyPI, the requirements (Python 3.10 or newer,
   NumPy 2.0, SciPy 1.13, none of 0.1.0's others) and the test
   `pytest --pyargs pyibs` with `pytest` among the test requirements, in a
   pull request opened before the version bot's or pushed to it before it
   merges. Its CI passes, and it is merged. Check that
   `curl -s https://api.anaconda.org/package/conda-forge/pyibs` reports
   1.5.0 as the latest version.
8. `dev-next` is reset onto `main` (`AGENTS.md`, "Conventions").

**Verification**:
- [ ] PyPI and conda-forge serve 1.5.0, and the PyPI install passes its
      tests in a fresh environment.
- [ ] The documentation is served at https://acerbilab.github.io/pyibs/.

### Phase 6: other repositories

**Status**: pending
**Executor**: Opus (orchestrator), in each repository under its own
`AGENTS.md`, on the PI's instruction.
**Needs**: `../model-fitting`, `../pybads`, `../pyvbmc`.
**Goal**: the lab's pages and the sibling packages point at PyIBS 1.5.

**Steps**:
1. `../model-fitting`, `site/index.html`, the PyIBS card: a Docs link and a
   "New in 1.5" line, as on the PyBADS card, and an item in the news list.
2. `../pybads` and `../pyvbmc`: where `README.md`, `docsrc/source/index.rst`,
   `docsrc/source/faq.md` and `skills/*/SKILL.md` link PyIBS on GitHub, or
   MATLAB IBS where they address Python users, link the PyIBS
   documentation, in a pull request of each.

**Verification**:
- [ ] Each change is merged in its repository, and its links resolve.

### Phase 7: the film

**Status**: pending
**Executor**: Opus (orchestrator), on the PI's machine.
**Needs**: `dev/private/film.md` and what it names. The phase comes last,
after the release (D17).
**Goal**: a short film of PyIBS, like those of PyBADS and PyVBMC, linked
from the README, the documentation and the model-fitting page.

**Steps**:
1. Follow `dev/private/film.md`. The PyBADS film's sources, `dev/film/` on
   PyBADS's `feat-film` branch, are the model of a production.
2. The script and the storyboard source every claim from [1], from the
   validation record of Phase 3 or from the references of Phase 4's
   Positioning, which they follow; the PI approves both before anything is
   recorded.
3. Production never runs alongside a heavy process of another phase.
   Once published, the film is linked from `README.md` ("What is it?" and
   "How does it work?"), `docsrc/source/index.rst` and the model-fitting
   page. Commit the links.

**Verification**:
- [ ] The PI approves the master, and the links resolve.

## Documentation

- `AGENTS.md` grows in Phases 0, 1, 3, 4 and 5, as their steps say;
  `CLAUDE.md` imports it.
- `CHANGELOG.md` (Phase 0) owns the release notes; the GitHub release takes
  its section.
- `pyibs/README.md` (Phase 1) owns the catalogue of deliberate differences
  from `ibslike.m`, as `pybads/bads/README.md` does for PyBADS; the
  changelog and the FAQ link it.
- `dev/results/` holds the ledger of the review (Phase 2) and the
  validation's summary (Phase 3), `dev/experiments/` their evidence;
  `dev/README.md` indexes them.
- `README.md` is rewritten (Phase 4); the documentation site, the FAQ, the
  examples and the skill are new (Phase 4).

## Decisions

- **D1. The engine is the lab's tested Python port of the `ibslike.m`
  sampler, brought in by Phase 0** — it ports `ibslike.m`'s vectorized
  sampling with tests, among them ports of `ibslike.m`'s self-tests, and
  it has none of 0.1.0's defects. Rejected: repairing 0.1.0 in place (two
  sampling paths that duplicate logic and carry the defects, and no tests
  to guard a repair); a fresh port of `ibslike.m` (it would duplicate
  tested work).
- **D2. 0.1.0's calling convention stays**: `IBS(sample_from_model,
  response_matrix, design_matrix, ...)` with 0.1.0's parameter names and
  order, and `ibs(params, num_reps, trial_weights, additional_output,
  return_positive)` with its output forms. Rejected: new names for 1.5
  (0.1.0 is published on PyPI and conda-forge, and its README's recipe for
  PyVBMC uses this convention); a MATLAB-style function `ibslike(fun,
  params, ...)` (an object holds the data, the settings and the generator
  across the many calls of an optimization or of inference).
- **D3. The cap raises; the time limit returns a flagged estimate from
  the completed repeats** — `ibslike.m` raises at its cap
  (`ibslike:ConvergenceFail`) and flags the time limit with exit flag 2;
  PyIBS adds a warning for the time limit, since the flag is invisible in
  the `"std"` output that PyBADS and PyVBMC use, and raises for a trial
  with no completed repeat. Rejected: an estimate at the cap, as 0.1.0's
  exit flag 3 gave (a truncated count biases it, and inside an
  optimization nothing reads the flag); `ibslike.m`'s handling of the time
  limit (its vectorized path averages in the partial count of an
  unfinished repeat, whose value depends on the schedule, or returns NaN
  for every trial when the limit passed before its first round, and its
  loop path, without a threshold, returns NaN for a trial with no
  completed repeat).
- **D4. Acceleration grows after every call by default, and the time
  rule of `ibslike.m` is opt-in** — so that a seed reproduces a run, as in
  PyBADS and PyVBMC. The values of complete repeats do not depend on the
  schedule; it changes the cost, the variance estimate of a repeat that
  the threshold ends, whether a call reaches the cap, which counts the
  samples drawn after a trial's last match, and, under the time limit,
  which repeats complete. Rejected: `ibslike.m`'s default, which makes
  the samples requested depend on the wall-clock time.
- **D5. The likelihood threshold follows [1], Appendix C.1, as the
  engine implements it** — every repeat is checked against a weighted
  bound, and an ended repeat is worth exactly -T, whatever the schedule.
  Rejected: `ibslike.m`'s (its value depends on the schedule, its
  vectorized path checks only the lowest repeat, it compares the
  unweighted sum, and its own comment calls the threshold incompatible
  with vectorized sampling).
- **D6. `max_iter` caps the samples of one trial in a call of `num_reps`
  repeats at `max_iter * num_reps`, surplus included, and the samples per
  call follow `ibslike.m`'s formula** — the cap's meaning is `ibslike.m`'s
  "per trial and estimate", scaled by the repeats as its vectorized path
  scales `MaxIter`, and the formula keeps the size of a call as in MATLAB.
  Rejected: counting rounds, as `ibslike.m`'s code and 0.1.0 do (a round
  can hold many samples); a cap per repeat (bookkeeping per repeat for a
  check that guards against responses the simulator cannot produce).
- **D7. One sampler, whose schedule `vectorized` selects** — `False`
  requests one sample per open trial per call, without acceleration: at
  most N rows per call and no surplus, as in the loop path, although the
  number of calls can differ. Rejected: two code paths (duplicated logic,
  where 0.1.0's two paths diverged).
- **D8. The generator: `random_seed` with PyBADS's semantics, kept as
  `self.rng`** — one convention across the three packages. A run with
  `BADS(..., options={"random_seed": s})` and an `IBS(...,
  random_seed=s)` target reproduces when the simulator takes the
  generator (D18), `vectorized` is set explicitly or decided by the same
  first-call timing (D19), and no timing decides the sampling:
  `acceleration_threshold` is None and `max_time` infinite, their
  defaults. Rejected: a generator per call (the
  callers call the target with the parameters alone).
- **D9. Requirements: Python 3.10, NumPy 2.0 and SciPy 1.13 or newer** —
  PyBADS 1.5's floors, so the three install together. Rejected: 0.1.0's
  list (Context, defect 8).
- **D10. Outputs are Python floats, and a zero variance is returned as
  computed, with a warning** — PyBADS accepts only a tuple, and zero is
  the correct estimate when every trial matched at its first sample. A
  call that returns a zero variance or SD issues a `UserWarning` with a
  fixed text, which Python's default filter shows once per session: it
  says that PyBADS and PyVBMC refuse a zero SD, and links the FAQ answer
  on it. Rejected: a floor (it would misstate the precision); an error
  (standalone use has no need of one); silence (the user meets the
  refusal inside PyBADS or PyVBMC, without the cause).
- **D11. One review wave with two read-only reviewers** — about 700 lines
  of MATLAB against about 1,000 of Python. Rejected: the multi-wave
  reviews of PyBADS and PyVBMC (sized for packages ten to a hundred times
  larger).
- **D12. PyIBS prints nothing of its own accord: no runtime tips and no
  old-release reminder, with `check_for_updates()` kept as a function
  that the user calls** — PyIBS has no run or display of its own: an
  `IBS` object is created and called inside a script or another tool's
  run, often many times over, so any message would stand alone and read
  as noise. `check_for_updates()`, which prints only when called and is
  the package's only network access, keeps the three packages uniform
  for users and for the agent skill. Rejected: tips or a reminder when an
  `IBS` object is created (unprompted output from an otherwise silent
  library, repeated in loops and batch jobs); no update check at all (the
  skill would need another way to compare versions).
- **D13. The example model and the notebooks leave the package directory
  for `examples/`, installed as `pyibs.examples`** — as in PyBADS; the
  modules `pyibs.psycho_generator` and `pyibs.psycho_neg_logl` stop being
  importable, a line under "Upgrading from 0.1.0". Rejected: keeping them
  in the package (example code in the API).
- **D14. The exact draws and exact moments are test helpers, not public
  API** — `ibslike.m` has no counterpart. Rejected: a public module (a
  feature beyond MATLAB).
- **D15. Only Phase 0 reads anything beyond this repository, its sibling
  repositories and PyPI** — every later phase runs from a clone, on any
  machine or in a cloud session; the film, made on the PI's machine, is
  the exception. Rejected: reading the engine's source in later phases
  (every phase would be tied to one machine).
- **D16. The count settings, `num_reps` among them, take integers and
  whole-number floats** — 0.1.0's signature and README give `1e4` and
  `1e5`, and 0.1.0 converted `num_reps` with `int()`, so scripts pass
  floats. Rejected: integers only (stops those scripts); 0.1.0's
  truncation of a fractional `num_reps` (a value such as 2.5 is an error,
  not 2).

- **D17. The film comes last, after the release** (PI, 2026-10-09) — the
  release does not wait for it, and its end card can name the released
  version. Rejected: making it alongside the earlier phases.
- **D18. The simulator receives the generator as `rng=` when its
  signature has a parameter named `rng`** (PI, 2026-10-09) — 0.1.0's
  two-argument simulators keep working, and a new simulator opts in by
  naming the parameter. Rejected: an explicit setting such as
  `simulator_takes_rng=True` (one more option to get right); always three
  positional arguments (stops every 0.1.0 simulator).
- **D19. `vectorized=None` is decided once per `IBS` object, at its first
  call with `num_reps > 1`, and kept** (PI, 2026-10-09) — by timing one
  simulation of all trials, as `ibslike.m` does, so the decision follows
  MATLAB's rule while later calls stay on one schedule; `True` or `False`
  given explicitly makes a run fully reproducible. The timing call's
  samples are always the call's first round, an extra one when the
  schedule's first round requests more than one sample per trial (PI,
  2026-10-09, after the port review's F-1): `ibslike.m` uses them only when
  its first round requests one sample per trial, which depends on how long
  the call took, and so biases the call when the simulator's running time
  depends on its outcomes. A call with
  `num_reps = 1` samples one sample per trial and call whatever the
  setting, as in `ibslike.m`, and so decides nothing. Rejected:
  `ibslike.m`'s decision at every call (results depend on the timing
  whenever the decision flips); the decision at the object's first call
  whatever its `num_reps` (a first call with `num_reps = 1` would hold a
  fast model to the one-sample schedule for every later call).
- **D20. CI tests the minimum versions** (PI, 2026-10-09) — a job of
  `test-matrix.yml` runs the suite on Python 3.10 with the lowest NumPy,
  SciPy and pytest that `pyproject.toml` allows (D9), which uv resolves
  from `pyproject.toml` itself, whenever `tests.yml` or `merge-tests.yml`
  runs the tests. The matrix
  installs the newest release for each Python, so without the job the
  oldest versions tested would be those of the newest release for Python
  3.10 (NumPy 2.2.6 and SciPy 1.15.3 on 2026-10-09). Rejected: the NumPy,
  SciPy and pytest versions pinned in the workflow (a second place to
  change with D9, as Python 3.10 already is); a check at the
  release gate only (code of Phases 1 to 4 could pass the matrix and fail
  at the minimum versions until then).
- **D21. The per-trial arrays of `"full"` are NaN when the likelihood
  threshold ended any repeat** (PI, 2026-10-09) — a repeat the threshold
  ended is worth exactly -T as a whole (D5), with no share in any trial.
  When no repeat ended, the threshold did not act on the draw: the arrays
  are those that the same draw gives without a threshold, and they add up,
  weighted, to the negative log-likelihood, as in `ibslike.m` (lines
  220-228). Under a threshold they are biased even so, since they are
  finite only on draws whose repeats all stayed above -T, which favours
  small values of `neg_logl_trials`; a user who wants per-trial values sets
  no threshold. Rejected: the average over the repeats not ended (biased in
  the same way, and adding up to nothing that the call returns);
  `ibslike.m`'s partial counts of the ended repeats (at odds with D5).
- **D22. Responses of one column, of shape (N,) or (N, 1), accept a
  simulator output of shape (r,) or (r, 1) for r rows requested** (PI,
  2026-10-09) — so that a model ported from MATLAB, where both are column
  vectors, runs unchanged; responses of C > 1 columns take an output of
  shape (r, C) only. The rule lives in the engine's shape check (Phase 1,
  step 2), the one place where `IBS` checks the output's shape. `ibslike.m` checks
  only the number of rows and compares with
  `all(respMat(T,:) == simdata, 2)` (lines 303-316, 444-450), which
  MATLAB's implicit expansion broadcasts, so it accepts more shapes: a
  deliberate difference. Rejected: the engine's exact match of shapes (a
  `ValueError` for such a port); MATLAB's broadcasting (an output of one
  column compared with every column of the responses); 0.1.0's comparison
  (an (N, 1) response against an (r,) output broadcasts to an (r, r)
  array). `ibs_basic`, which asks for one response at a time, takes an
  output of shape (C,) or (1, C) for responses of C columns, or a scalar
  when C = 1, and raises `ValueError` for any other (PI, 2026-10-10):
  `ibs_basic.m` compares with `any(fun(theta,S(i,:)) ~= R(i,:))` (line 33),
  which broadcasts a column against the row, and ends its loop on an empty
  output as on a match.
- **D23. A NaN response raises `ValueError` when `IBS` is created** (PI,
  2026-10-09) — a NaN never matches. Without a likelihood threshold its
  trial samples until the cap and fails there after `max_iter * num_reps`
  samples; with one that the trial's growing term reaches before the cap,
  and a positive weight on that trial, every repeat ends below -T and the
  estimate is -T at every parameter vector, with no error.
  A NaN is an element not equal to itself, which finds float and complex
  NaN, `NaT` and a NaN in an object array, and never flags text. The design
  is not checked: `ibslike.m`'s own examples call their simulator with a
  design of NaN (lines 57, 511, 587, 630), and a port that passes such a
  design to `IBS` keeps working. `ibs_basic` raises in the same way.
  Rejected: leaving it to the cap or the threshold (the whole cap's cost, or
  a silent -T, with nothing that names the cause).

- **D24. When the time limit stops a draw in which the likelihood threshold
  ended repeats, the ended repeats count -T each and the others are
  averaged per trial** (PI, 2026-10-09) — of the n repeats, n_e ended;
  trial i has m_i completed counts in the other repeats, whose `ibs_loglik`
  average is ā_i. The log-likelihood estimate is (n_e / n)(-T) +
  (1 - n_e / n) Σ_i w_i ā_i, and its variance estimate is (Σ of the ended
  repeats' variance estimates) / n² + (1 - n_e / n)² Σ_i w_i² (Σ of the
  trial's `ibs_var` over its completed counts) / m_i². The exit flag is 2,
  with D3's warning, which also gives n_e; the per-trial arrays are NaN
  (D21); a trial with m_i = 0 raises, as in D3. Each rule acts on the
  repeats it governs: D5 values an ended repeat as a whole, and D3
  summarizes the repeats left unfinished by trial, so the estimate is D3's
  when no repeat ended and -T when every repeat did, and the bias it adds
  to the threshold's is the time limit's, which the flag and the warning
  report. Rejected: the complete repeats alone, with their values
  max(Y_r, -T) (the counts that faster trials completed beyond them are
  lost, and with few complete repeats the call raises, which D3 exists to
  avoid); refusing the combination (`ibslike.m` allows it, and the user
  would lose the time limit under a threshold); -T whenever a repeat ended
  (one ended repeat does not put the parameter vector below the threshold);
  `ibslike.m`'s per-trial average over every count, the partial counts of
  ended repeats included (at odds with D5).
- **D25. `return_positive` takes only Python and NumPy booleans** (PI,
  2026-10-10) — it is a boolean option, and a value read for its truth, as
  0.1.0 and `ibslike.m` (line 226) read it, turns the string `"False"` into
  the log-likelihood without an error. A script that passes 0 or 1 stops
  with a `TypeError` that names the argument, which the changelog's
  "Upgrading from 0.1.0" lists. Rejected: any value read for its truth
  (0.1.0 and `ibslike.m`); the integers 0 and 1 as well (a second spelling
  of a boolean option, which `vectorized` does not take either).

## Open Questions

None. The PI settled the plan's questions on 2026-10-09: D17 to D19, D24,
and timings against 0.1.0 only, with no MATLAB installation in the
comparison (Phase 3, step 3).

## Worklog

Entries are added per phase as `### Phase N — YYYY-MM-DD`.

### Phase 0 — 2026-10-09

- Commits: `e2ccd24` (step 1), `8f14f63` (2), `23d43ec` (4), `2fe31bc`
  and `e8b99d3` (5), `dba7a46` (5, `.git-blame-ignore-revs`), `adf4201`
  (6), `79d2c45` (7), `bb8e821` (9), `6b725f1` (10); after the review,
  `1a76aa4` and `7cc3e8f`.
- References (step 3): `../ibs` at
  `2229c00c4a19eb9f236f9f257100dab9e87b6f92`, `../ibs.wiki` at
  `15d62f2f55dc627cd5e51677c94d995c08bf28e8`; `ibslike.m` reads
  "Version: 0.96" and "Release date: Jan 21, 2021", and its options,
  hard-coded values, outputs and self-tests match the Context.
- Engine (step 7): extracted as `dev/private/extraction.md` states, which
  records the details.
- Verification: `$PY -m pytest`, 173 tests passed in about 5 s (11 s on
  a cold first run); the check of the private brief passes and
  `git ls-files dev/private` prints nothing; the pre-commit hooks pass;
  the installed version reads `0.1.dev75+g23d43ec76.d20261009`; `$PY -m
  build` builds an sdist and a wheel, whose `METADATA` requires only
  `numpy>=2.0.0` and `scipy>=1.13.0` outside the extras, and the sdist
  holds no file of `dev/`; the three workflows parse. After the push,
  with `b99e39e` at the head of `dev-next`, the smoke run of `tests.yml`
  passed
  ([run 37906670683](https://github.com/acerbilab/pyibs/actions/runs/37906670683)):
  Ubuntu with Python 3.14.8, NumPy 2.5.3 and SciPy 1.18.1, 173 tests
  passed in 4.1 s, the package installed as `0.1.dev85+gb99e39e37`.
  Beyond step 11, the full matrix of `tests.yml`, dispatched with
  `0a84c10` at the head, passed in all 15 jobs, Ubuntu, Windows and macOS
  × Python 3.10 to 3.14
  ([run 37908926522](https://github.com/acerbilab/pyibs/actions/runs/37908926522));
  on Python 3.10 it installs NumPy 2.2.6 and SciPy 1.15.3, the oldest
  versions the matrix tests.
- Deviations: the hooks' new versions are a commit of their own
  (`2fe31bc`), so that the formatting commit holds only the reformatting
  (black 23.3 removed four blank lines of `pyibs/ibs.py`); the packaging
  commit adds an empty `pyibs/testing/__init__.py`, so that the
  `packages` it names exist; the renormalization of step 2 changed no
  file, since every blob was LF already; the hook `exclude` patterns
  `\.patch$` and `docs/tutorials` are dropped as matching nothing here;
  `.gitignore`'s `docs/_build/` gives way to `docs/`. Beyond step 10,
  `AGENTS.md` states that `packages` lists every directory and that the
  tests ship in the wheel. The Context's "Release access" records the
  PI's access, obtained on 2026-10-09.
- Review (`/doublecheck`, three read-only Opus reviewers: the extraction,
  the tooling and documents, the engine's code): no finding to be fixed
  before the push. Fixed: the statement of `repeat_estimates`'s bitwise
  properties in `AGENTS.md`; the engine's docstrings on `ibslike.m`'s
  threshold (its loop path checks each repeat with `c_i + 1` for its open
  trials) and tables, on the kinds of responses that raise, and on the
  cap's error; a test of the level's bound under a large acceleration.
- For Phase 1, from the review: the cap's error names
  `max_samples_per_trial * n` and 0-based trial indices, which `IBS`
  exposes as `max_iter` and `num_reps`; when the threshold ends every
  repeat, the per-trial sums cover no repeat, and once it ends any, the
  total is not the weighted sum of the per-trial values (as it is in
  `ibslike.m`), so step 3 needs a rule for the `"full"` per-trial arrays;
  responses of shape (N, 1) with a simulator returning shape (n,) raise
  `ValueError`, where `ibslike.m` compares row by row; trial weights
  accept booleans and numeric strings, which the other settings refuse;
  a NaN response never matches and samples until the cap; the `runtest2`
  port checks `ibslike.m`'s fixed tolerances at one seed, so a schedule
  that consumes the generator differently (D6) can fail it by chance.
  The PI's rules: D21 for the per-trial arrays, D22 for responses of one
  column, D23 for a NaN response; Phase 1's steps 2 and 3 take the other
  three.
- After the phase: `0a84c10` and `1bdda9a` record the runs above; `5f906d8`
  adds the job at the minimum versions (D20), whose first run passed (run
  37910206877: Python 3.10.22, NumPy 2.0.0, SciPy 1.13.0, pytest 6.2.5, 173
  tests); `cf21331` drops the release's wait on "Release access", which the
  lab holds, removes `.git-blame-ignore-revs`, since `e8b99d3` only deletes
  lines and so leaves none for `git blame` to hide (PyBADS's entry,
  `69be885`, hides nothing either: at `ff415ca0` it is in no branch's
  history), copies PyBADS's `.github/dependabot.yml` and `.coveragerc`,
  adapts the convention "Changelog" of its `AGENTS.md` and records D21;
  `1a8c371` makes `merge-tests.yml` run the tests on a pull request that
  changes a test workflow, as a Dependabot update of an action does;
  `8e1ecb5` and `35c4d94` record D22 and D23. A review of these commits
  (`/doublecheck`, three read-only Opus reviewers) found D21's stated
  reasons wrong (under a threshold, finite arrays are not unbiased) and
  gaps in Phase 1's steps; the commit after `35c4d94` takes its findings:
  D21's reasons, the parity table's rows, D22's comparison with
  `ibslike.m`, D23's case under a threshold and its definition of NaN, the
  per-trial variances, the Phase 1 steps for the review's remaining inputs,
  the changelog entries in each commit, and the open question on the time
  limit with the threshold, which the PI settled as D24. A checkout that
  followed the removed advice of `AGENTS.md`,
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`, needs
  `git config --unset blame.ignoreRevsFile`: without the file, `git blame`
  stops with "fatal: could not open object name list".

### Phase 1 — 2026-10-09

- References (step 1): `../ibs` at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`,
  Phase 0's commit, so `ibslike.m` and `ibs_basic.m` are unchanged;
  `../pybads` at `ff415ca0316ba3d58b5c48337978ddfe81851cc2`.
- Commits: `65add3a` (steps 2 to 5), `781b7e7` (steps 6 to 9), and the
  commit after it (the review's fixes and this entry).
- Tests that the schedule or the checks changed (step 2): in
  `test_sampler.py`, `test_scripted_counts[2-1.5]` (the third call
  requests 5 samples per trial, MATLAB's round of 4.5, where it requested
  4); `test_counts_match_reference_sampler`, and
  `test_matches_reference_sampler` of `test_threshold.py`, whose reference
  samplers take `ibslike.m`'s
  formula from `_helpers.ibslike_samples`, with `max_samples` and `max_mem`
  in place of `max_samples_per_call`, and two more cases;
  `test_level_is_bounded_by_max_samples` (renamed from
  `..._by_max_samples_per_call`) and
  `test_level_stays_finite_under_a_large_acceleration` (`max_samples` in
  place of `max_samples_per_call`); `test_wrong_simulator_output_raises`,
  whose case of an (r, 1) output for responses of shape (N,) moved to
  `test_one_column_responses_take_both_output_shapes` (D22); and
  `test_invalid_settings_raise`, `test_sample_rejects_bad_n`,
  `test_exact.py::test_rejects_bad_n` and
  `test_threshold.py::test_invalid_threshold_raises`, since a boolean or a
  value of another type raises `TypeError`. The statistical tests, the
  ports among them, pass unchanged; the ports, which call `IBS` with
  `vectorized=True`, draw bitwise what the engine-based ports drew.
- Verification: `$PY -m pytest`, 453 tests passed in about 8 s; the
  pre-commit hooks pass; the review below found every row of the parity
  table implemented as its last column says, and every deliberate
  difference in the catalogue (KD-1 to KD-19).
  After the push, with `ac577c4` at the head of `dev-next`, the smoke run
  of `tests.yml` passed
  ([run 37927106106](https://github.com/acerbilab/pyibs/actions/runs/37927106106)),
  and so did the full matrix, dispatched, in all 16 jobs, Ubuntu, Windows
  and macOS × Python 3.10 to 3.14 and the minimum versions, where 453
  tests passed in 4.9 s
  ([run 37927122127](https://github.com/acerbilab/pyibs/actions/runs/37927122127)).
- Deviations: `vectorized=None` is decided at the object's first call with
  `num_reps > 1`, where step 2 and D19 said "at its first call"; the PI
  ruled for it on 2026-10-09, and step 2 and D19 now say so, D19 with the
  reason. Where the steps are silent: the timing
  call of `vectorized=None` is the sampling's first round when that round
  requests one sample of every trial, as in `ibslike.m`, and otherwise its
  sample counts toward the cap; the time limit is checked after every
  simulator call of the sampling, not before its first; the settings of
  `IBS` are read-only attributes; `IBS` pickles when its simulator does.
  The tests of step 6 were written by an Opus sub-agent. Step 7 found the
  engine's docstring wrong on `ibslike.m`'s vectorized threshold, which
  bounds the lowest open repeat with the open count c that its count
  matrix holds (a transliteration of lines 319-373 confirmed it), and
  corrected it. `781b7e7`, a commit of tests and documents, also changes
  code: the cap's count of a discarded timing call, the picklable
  simulator wrapper and docstrings. The parity table's row "Time limit",
  D3 and D23 overstated what `ibslike.m` and PyIBS do, and are corrected in
  place: under a threshold, the loop path also averages the partial count
  of the repeat it stopped, and a NaN response ends every repeat at the
  threshold only when its term reaches T before the cap.
- Review (`/doublecheck`, three read-only Opus reviewers: the code, the
  statements about `ibslike.m`, the tests and records): nothing in the code
  to undo. Fixed: the catalogue's KD-13 (the paper's bound counts c,
  PyIBS's c + 1), KD-14 (the loop path under a threshold) and KD-17, with
  a paragraph of notation, missing citations and the data checks in KD-6;
  this entry's list of tests; the changelog's statements about 0.1.0, its
  duplicates of the "Upgrading" lines, the conditions of reproducibility,
  the calls with `num_reps=1` and the module `pyibs.ibs_basic`, which the
  function now shadows; `AGENTS.md` on the FAQ label and the checks of
  the settings; `psycho_neg_logl` with responses of another shape than the
  stimuli; the NaN message, which named a cap that `ibs_basic` lacks; the
  bias direction in `EstimateResult`'s docstring; object arrays of real
  numbers as trial weights; arrays left writable by pickling; `pyibs/README.md`
  as package data; two tests that failed at about 0.2 % of seeds; and tests
  of the paths that none covered.
- For Phase 2: `IBS` checks the responses, the design, `max_iter` and
  `num_samples_per_call` before `_Settings` checks them again; `ibs_basic`
  loops forever on responses of a kind that never matches, as
  `ibs_basic.m` does; inside the engine, `max_samples` (per trial and
  call) sits next to `max_samples_per_trial` (per trial and repeat). Two
  residual risks of the review, for reviewer B: when a simulator's
  runtime depends on its outcomes, whether the object's first call keeps
  the timing call's outcomes (decided False) or discards them (decided
  True, with more than one sample per trial in the first round) depends on
  those outcomes, which may bias that one call slightly, as it biases
  every call of `ibslike.m`; and, by MATLAB's indexing rules, with a
  single trial `ibslike.m`'s `nlogLvar_trials` (line 213) and its loop
  path's `nlogL_trials` (line 485) may come out with `Nreps` entries,
  unchecked without MATLAB, where PyIBS returns shape (N,).
- For Phase 4: 0.1.0's wheel shipped three notebooks in `pyibs/`, which
  the new examples replace, and its README describes 0.1.0; the FAQ's
  answer on a zero SD carries the label of `_FAQ_ZERO_SD`
  (`AGENTS.md`, "What spans files").

### Phase 2 — 2026-10-09

- References: `../ibs` at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`,
  Phase 1's commit; `../pubs-llms` at
  `a25f58be92111a4c1bb3c3dd3d496605e23b1e98`; GNU Octave 8.4.0.
- Commits: `cf8d732` (the phase's status), `6df6409` (`ibslike.m` under
  Octave), `b6fe6fb` (the checks of the review), `9221a1e` (the ledger and
  its evidence, step 3), `27e321c` (F-1), `cf0ed60` (F-2), `16790c6` (the
  catalogue and documentation, step 4), `3c63ec1` (the records); after the
  review of the phase, `28a0891`, `463880f`, `09bda5a`, `2b2e299` and
  `14b0958`, and the commit after it (the evidence and this entry).
- The review (steps 1 to 3): two read-only Opus reviewers, A against
  `ibslike.m` and `ibs_basic.m` and B against [1] and for internal
  correctness. The ledger,
  [`dev/results/2026-10-09-port-review.md`](../results/2026-10-09-port-review.md),
  holds twelve findings and four items of no issue. Two are defects: F-1,
  an object's first call with `vectorized=None` biased when the
  simulator's running time depends on its outcomes, since the timing
  call's samples were used only when the first round requested one sample
  per trial, at the default settings when the decision was False; and
  F-2, `ibs_basic` looping forever on simulated responses of a kind that
  NumPy never finds equal to the observed ones. The PI's rulings: for F-1,
  the timing call is always the first round (D19 states it); for F-2, the
  kind check of `IBS`; for F-4, `max_iter` stays finite; the others as
  proposed. The points that Phase 1 left are F-1 to F-3, N-1 and N-2; F-3,
  the shapes of `ibslike.m`'s per-trial arrays with one trial, Octave
  settled.
- Tests (step 4): F-1 adds
  `test_first_call_is_unbiased_when_the_timing_tracks_the_outcomes`
  (`test_ibs.py`, about 15 standard errors off before the fix) and
  replaces `test_first_round_is_ignored_on_more_samples_per_trial` by
  `test_first_round_precedes_more_samples_per_trial` (`test_sampler.py`),
  whose counts include the timing call's sample. It changes
  `test_max_time_counts_the_timing_call` (the timing call's miss is now the
  first sample of a count of 3, where the count was 2), the parameters of
  `test_vectorized_none_counts_the_timing_call`, the cases of
  `test_first_round_reproduces_the_draw_without_it`, which gain `max_mem`,
  and the name of `test_cap_counts_the_timing_call`, whose simulator calls
  are unchanged and whose first count is 20 where it was 19; and it drops
  the assertions on `used_first`. F-2 adds
  `test_responses_that_never_match_raise` (`test_ibs_basic.py`), F-3 an
  assertion on the shape of `neg_logl_var_trials`, F-4 the case
  `num_samples_per_call=math.inf` of `test_settings_out_of_range_raise`.
  The check of object responses adds cases to
  `test_responses_of_another_kind_raise` and
  `test_responses_of_a_comparable_kind_match` (`test_sampler.py`), and
  `test_responses_that_mix_numbers_and_text` (`test_ibs_basic.py`). The
  ports of the self-tests run with `vectorized=True`, which F-1 does not
  reach, and draw as before.
- Verification: `$PY -m pytest`, 468 tests passed in about 8 s; the
  pre-commit hooks pass. After the push, with `911dc5c` at the head of
  `dev-next`, the smoke run of `tests.yml` passed
  ([run 37937327184](https://github.com/acerbilab/pyibs/actions/runs/37937327184)).
- Deviations: GNU Octave was installed at the PI's request, and
  `ibslike.m`, unmodified, runs under it with two directories of shims;
  `AGENTS.md` states how ("Sibling repositories"), and the shims are in
  `dev/scripts/octave/`. The reviewers and the orchestrator ran small
  checks of `ibslike.m` under it where the steps planned reading only, and
  Reviewer B ran one check of PyIBS on a fake clock, about 1 s, for F-1;
  the record cites the checks from
  `dev/experiments/port-review_20261009/`. At the PI's request, the public
  docstrings of `IBS` and the changelog no longer justify PyIBS's behaviour
  by `ibslike.m`'s (`28a0891`), the catalogue holding the comparison, and
  `2b2e299` checks the kinds of the elements of object responses, beyond
  F-12's documentation.
- Review (`/doublecheck`, three read-only Opus reviewers: the code and
  tests, the statements about `ibslike.m`, the records): no defect in the
  code. Fixed: the changelog entry of F-1, which PyIBS 0.1.0 shared; a
  sampler test that did not see the timing call's sample; the docstrings
  of `num_samples_per_call` and `initial_samples`; KD-6 on a per-trial
  `Nreps` vector, which MATLAB's `&&` stops at line 193 and which fails at
  line 361 or 372 once fewer than all trials are open; KD-11 on `MaxIter =
  Inf`; KD-18's data checks; KD-19 and F-11, where `runtest3` sets the
  acceleration to 1; F-2, whose example loops in `ibs_basic.m` too; the
  ledger's commits, the runs behind its findings, F-1's condition, F-4's
  test, F-5's message of exit flag 0, which is 0.1.0's, and an
  interpretation; the evidence's commands, which name the commit of each
  output; the Octave paragraph of `AGENTS.md`; and this entry. A
  self-review before it rewrote the docstrings, comments and catalogue
  entries of the phase that read badly (`463880f`).
- For Phase 3: under `vectorized=None`, an `IBS` object decides at its
  first call with `num_reps > 1` and keeps the decision (D19), so the
  validation's cells of `None` create an object per estimate if they are to
  sample the deciding call, whose first round is the timing call (F-1);
  with one object, every estimate after the first samples as `True` or
  `False` does.

### Phase 3 — 2026-10-09

- References: `../ibs` at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`,
  Phase 2's commit, for `ibs_example.m`'s recipe of its data set, its
  bounds, starts and prior; `../ibs.wiki` at `15d62f2`, for its advice on
  the noise of the estimate. PyBADS 1.5.1 from PyPI with gpyreg 1.4.0;
  PyVBMC from `feat-release-1.5-preparation` at `89007a4`, whose installed
  version reads `1.0.5.dev1379+g89007a4eb`; PyIBS 0.1.0 from PyPI in
  `.venv-0.1`.
- Commits: `c73ca08` (step 2), `c76b5ac` (the scripts of steps 1 and 3,
  and the phase's status), `5e1876b` (the steps' marks), `4e4f0d9` (the
  replication of step 1), `46f62c8` (the record of step 4); after the
  review of the phase, `55af9e2` (the code) and the commit after it (the
  record and this entry).
- The record:
  [`dev/results/2026-10-09-validation.md`](../results/2026-10-09-validation.md),
  with its evidence in
  [`dev/experiments/validation_20261009/`](../experiments/validation_20261009/).
  Every cell of step 1 passes its gates, under the PI's ruling below; both
  integration files pass; PyIBS 1.5 is faster than 0.1.0 in every timing
  cell, 1.3 to 2.1 times with the fast simulator and 1.2 to 3.8 times with
  the one dominated by a fixed cost of 0.2 s per call. The record times the
  sampler's own work, and computes the simulator's time from the calls and
  responses that each version makes on each schedule, for a cost per call
  and a cost per response (PI, after the review): the cells that sleep 0.2 s
  per call, which step 3 asked for, measure only a cost per call, and their
  times are 0.2 s times their calls to within 0.4%. For a cost per
  response, both versions simulate about as many responses.
- The smoke passes and the PI's rulings (2026-10-09). The validation's
  smoke pass, 100 estimates per cell, projected its full run at 1.5 h on
  the container's 4 workers, and found six cells failing the calibration
  gate: `num_reps=10` in `bernoulli_p0.001` and `bernoulli_p0.999`, whose
  seeded data hold no rare response, so that all their trials match with
  probability 0.999 and 37% of the variance estimates are 0, for exact IBS
  draws as for PyIBS. The PI approved all 243 cells at 2,000 estimates, and
  ruled that the calibration of those two models is reported and not
  gated, all their cells being gated instead on the number of zero
  variance estimates against its exact probability; step 1 states the
  ruling. The timing's smoke pass projected 4.4 h, with the slow cells
  run one at a time; the PI approved running the slow cells together after
  the fast ones, each sleeping for nearly all of its time, beside the
  validation.
- Verification: the validation has no failing cell and passes the zero
  check (`validation.json`); both integration files pass; the record names
  the commit, the clean tree, the versions and the platform of every
  output. `$PY -m pytest`, 468 tests passed and 2 deselected; the
  integration files skip when their package is absent or too old (checked
  with stand-ins for PyBADS 1.0.4 and a PyVBMC without `seed`); collected
  from outside the checkout, they raise no warning on their marker; the
  wheel holds `pyibs/testing/integration`; the pre-commit hooks pass.
  After the push, with `d279d9e` at the head of `dev-next`, the smoke run
  of `tests.yml` passed
  ([run 37956627875](https://github.com/acerbilab/pyibs/actions/runs/37956627875)),
  both its jobs: on Ubuntu with Python 3.14, 468 tests passed and the two
  integration tests skipped, their packages being absent.
- Deviations: the phase ran in a cloud container (Linux, 4 CPUs), not on
  the PI's workstation, and the validation and the timing ran together as
  the PI approved. Where the steps are silent: the threshold cells cover
  the two weighted models too, the Bernoulli model at p = 0.5, whose
  chance level `sum_i w_i log 2` uses the weights, and not the text,
  two-column and example models; the two-column model's first column has a
  probability that varies over the trials, its second 3 outcomes; the
  integration tests take 100 repeats, an SD of about 1 near the optimum,
  which PyBADS's FAQ and the IBS wiki advise. Step 1 leaves open whether
  the samples are gated where they are not "reported, not gated": they
  are gated at `vectorized=False` without the threshold and at `num_reps`
  of 10 or more, which all 32 such cells pass. Beyond the steps:
  `validate.py` reports, beside each cell's mean squared z-score, that of
  exact IBS estimates drawn from geometric counts, the ratio of the
  estimates' SD to the exact SD, the number of zero variance estimates
  against its exact probability and, under the threshold, the share of
  exit flag 1 against its exact probability; it takes the calibration of
  the threshold cells about the expected value of a thresholded
  estimate; the cell `bernoulli_p0.999_vF_n10`, whose number of zero
  variance estimates was 3.51 standard errors high, was drawn again at ten
  other run seeds (`zero_share_replication.py`), which showed it to be
  chance; the changelog gains an entry for the speed-up, with a link to the
  record; the evidence's outputs are `.txt` files, since `.gitignore`
  ignores `*.log`.
- Review (`/doublecheck`, three read-only Opus reviewers: the code, the
  statistical method, the records). Fixed in `55af9e2`: the PyVBMC test
  failed, rather than skipped, with PyVBMC 1.0.4, the release on PyPI and
  conda-forge, whose `VBMC` takes no `seed`, and the PyBADS test had no
  minimum version; the integration tests drew their start, their IBS
  estimates and the fit from the data's seed, whose stream the IBS draws
  then repeated; their marker was unknown to `pytest --pyargs pyibs`
  outside the checkout; the gate on the number of zero variance estimates
  rested on a normal approximation that fails where fewer than one is
  expected (a spurious failure in about 4.5% of full runs); the
  calibration of the threshold cells was centred on the exact value, so
  that it measured the threshold's bias; the zero check did not reach the
  exit status; the samples' z-score was printed where it has no
  expectation; `timing.py` gave the upper middle value as the median of an
  even number of estimates. The estimates being seeded, `validate.py
  --restat` recomputed the full run's statistics at `55af9e2` from its
  saved estimates rather than draw them again: every verdict stands, and
  the gates added since pass. The integration tests ran again at
  `55af9e2`. Fixed in the record: three values of the timing table; the
  decision of `vectorized=None`, which `num_reps=1` does not make; the
  growth of the gain with the 0.2 s per call, which at N = 100 does not
  grow with `num_reps`; claims of unbiasedness
  and calibration, now stated as what the evidence resolves, with the
  `num_reps=1` cells and the threshold cells' variance estimates, which
  overstate the variance about 2 to 3 times where the threshold acts; the
  data set, drawn as `ibs_example.m` draws its own; the README's commands,
  environments, seeds and the smoke passes' outputs, which are not kept.
  Not taken (PI): a reviewer's point that 0.1.0 at its default `max_iter`
  can take less time than 1.5 with the 0.2 s per call; that default is the
  defect `10 ^ 5` = 15, so the record compares only with
  `max_iter=10**5`.
- For Phase 4: the FAQ and the examples can cite the record on the noise
  of an estimate (100 repeats give an SD of about 1.1 on the example's 600
  trials at its generating parameters), on the variance estimate under a
  likelihood threshold, which overstates the variance, and on a simulator
  dominated by a fixed cost per call: there `vectorized=None` decides
  False, and an estimate of 100 repeats with 0.2 s per call takes 15.5 to
  27 minutes, where `vectorized=True`, for its 9 to 12 calls, takes about
  2 s. For a simulator whose every response costs time, the decision False
  is the right one. The
  changelog's link to the record resolves once the record is on `main`.
  `pyibs/testing/test_examples.py` seeds its `IBS` objects with the seed of
  its data, as the integration tests did. For Phase 5: once PyVBMC 1.5.0 is
  on PyPI, `AGENTS.md` ("PyBADS and PyVBMC") installs it from there.

### Phase 4 — 2026-10-09

- References: `../ibs` at `2229c00`, `../ibs.wiki` at `15d62f2`, Phase 0's
  commits; `../pybads` at `b64b4230`; `../pubs-llms` at `a25f58b`. PyBADS
  1.5.1 from PyPI with gpyreg 1.4.0; PyVBMC from
  `feat-release-1.5-preparation` at `89007a4`, as in Phase 3. The Nature
  reference of the Positioning (van Opheusden et al., 2023, Nature 618:
  1000–1005, issue 7967, doi 10.1038/s41586-023-06124-2) agrees with the
  reference list of PyBADS's JOSS paper in `../pubs-llms` for its authors
  and DOI, and with a RePEc listing and search results for its volume,
  issue and pages; the container's network policy refused the publisher,
  doi.org and Crossref.
- Commits: `f7f9434` (step 1), `0f87b73` (step 3, with `LICENSE`, the
  changelog's entries and the network rule of `AGENTS.md`), `1630d03`
  (step 4), `e5c2166` (step 2), `e17d74d` (step 5), `3458698` (step 6),
  `cf30448` (step 8); `4d0e573` (Phase 3's note on `test_examples.py`);
  `60ae967` (docstrings that numpydoc misread); after the review of the
  phase, `90e4852` and `68b46c1`; `674039d` (this entry); after the PI's
  decisions below, `7fb53fd` (the 0.1.0 text that remained, and
  `LICENSE`), `1fdf1e7` and `b7b2cfa` (when IBS fits) and `cda9de4` (the
  floor on the SD, to `dev/TODO.md`); after their review, `18c8686` (the
  message of exit flag 0) and `0422903` (the rest).
- Verification: the documentation builds with no warning
  (`make -C docsrc github` after `make -C docsrc clean`), and every link
  into it from the README, the changelog, the skill, the notebooks and the
  package's warning resolves to a page and an anchor of the build;
  `make -C examples/scripts run` reruns the three notebooks without error in
  1 min 44 s, nearly all of it PyVBMC's, and reproduces their outputs but
  for an elapsed time and PyBADS's random tip; `make -B -C
  examples/scripts` regenerates the scripts identically; `$PY -m pytest`,
  540 tests passed and 2 deselected; the pre-commit hooks pass on the whole
  tree; the wheel ships the notebooks, `pyibs/examples/scripts/` and
  `pyibs/README.md`; `import pyibs` imports no networking module, and
  `pyibs.check_for_updates()` against PyPI reports the development install
  and the latest release, 0.1.0.
- Deviations: the phase ran in a cloud container; Opus sub-agents wrote the
  notebooks, the FAQ and the documentation site in parallel, and the
  orchestrator the rest. `dev-next` was pushed during the phase, at the
  request of the session's stop hook. Beyond the steps: the docstring of
  `IBS` moves the paragraph on its settings from Attributes, which numpydoc
  reads as attributes, to Notes (`60ae967`); "How does it work?" in the
  README and `index.rst` shows a figure of the cost and the variance of IBS,
  drawn by `dev/scripts/ibs_cost_variance.py`; `pyproject.toml` lists
  `pyibs.examples.scripts`, as PyBADS lists its own; the CI's path filters
  include `examples/`, which they left out although `pyibs.examples` is
  tested (`90e4852`). The links to PyVBMC's documentation take its 1.5
  address, `https://acerbilab.org/pyvbmc/`, which PyVBMC's README and
  `html_baseurl` give. Step 3 expected no text of 0.1.0 to remain: a
  comparison of 8-word sequences with 0.1.0's tree found the three exit
  messages of `EstimateResult` in 0.1.0's wording, now reworded after
  `ibslike.m`'s, and a sentence of the example model's docstring, now
  rewritten; the rest of what the two share is the interface that D2 keeps,
  bibliographic text, PyBADS's templates, SciPy's idiom of a dictionary
  read as attributes, and comments of `ibs_basic.m` and `psycho_gen.m`.
  `LICENSE` names the lab alone, and the README's acknowledgments thank
  Julia Maria Perathoner for her work on 0.1.0, an earlier Python port of
  IBS (PI, 2026-10-09). The acknowledgments take the grants of PyBADS's and
  PyVBMC's READMEs, and name the coding agents' maker without a model.
  Example 2 evaluates the solution with 1,000 repeats, ten times its
  target's, as `ibs_example.m` does; Example 3 checks the ELBO and the
  posterior against an exact grid integration. The Positioning's
  "unbiased, with a calibrated variance, for every dataset" is written as
  "unbiased on every dataset, with an estimate of their variance", since
  the calibration is shown in general ([1], Section 4.6) and fails where
  nearly every count is 1 (the validation).
- Review (`/doublecheck`, four read-only Opus reviewers: the code, tests,
  packaging and tooling; the FAQ; the README, the documentation's pages,
  the skill and the notebooks; the records and conventions). Fixed in
  `90e4852` and `68b46c1`: the README and `index.rst` claimed that the
  validation detected no bias under the likelihood threshold, and a
  calibrated variance on every dataset; the examples' install command named
  no versions, though Example 3 needs PyVBMC 1.5; the FAQ said that PyVBMC
  requires the SD, and that neural likelihood estimation gives posteriors at
  once; the 0.1.0 text that remained (above, reworded in `7fb53fd`); the
  CI's path filters;
  `AGENTS.md` on the text that the README shares, on the labels of PyBADS's
  and PyVBMC's FAQs that PyIBS links, and on the generation of the scripts;
  smaller corrections to the FAQ (a bound rounded down, a sizing snippet
  that gave 0 repeats, the threshold with PyVBMC, links to the lab), the
  quick start's advice on noise and its seeding example, the README's
  formula, which PyPI would show as raw LaTeX, a docstring and the figure's
  script. Not taken: positioning IBS against non-amortized neural
  simulation-based inference too, beyond the plan's Positioning; a
  `make.bat` that stops on a failed build, which PyBADS's does not either.
- Decided after the phase (PI, 2026-10-09): `LICENSE` carries no notice
  of MATLAB IBS's MIT licence on `ibslike.m`; the README keeps the grants
  of PyBADS's and PyVBMC's READMEs; the FAQ's answer on a zero SD no
  longer advises a floor on the SD in the user's target,
  `max(sd, 1 / num_reps)`, which the FAQ's writer derived: `dev/TODO.md`
  keeps it, with its derivation, to be studied; and "When should I use
  PyIBS?", its copies and the Positioning state the condition, contexts
  many and richly structured, with the board game as its one example.
- Review of the commits after the first review (`/doublecheck`, two
  read-only Opus reviewers: the code, CI and packaging; the documents and
  records). Fixed in `18c8686` and `0422903`: the message of
  exit flag 0, which claimed an unbiased estimate also under a finite
  `max_time`; the README and `index.rst`, which read as if the variance
  estimates were calibrated under the threshold too; the FAQ's claim of a
  calibrated variance on every data set, and its sentence on the cost
  under the threshold; the Positioning and this entry; `dev/TODO.md`,
  which did not stand alone; `AGENTS.md` on the slug rule, on the skill's
  citations and on what `about_us.rst` repeats; smaller corrections. Left
  for Phase 5: the README's badge of `tests.yml` names `main`, where
  nothing runs that workflow.
- After the phase, pull request #1 (branch `codex/review-1.5-fixes` into
  `dev-next`, commit `f187d60`): `IBS.__call__` refuses a `return_positive`
  that is not a boolean, and `ibs_basic` a simulated response of another
  shape, with their tests; the README, the FAQ, the documentation's pages,
  the public docstrings and the notebooks are rewritten, the notebooks
  rerun and their scripts regenerated.
- Review of pull request #1 (five read-only Opus reviewers: the code and
  tests; the FAQ; the other pages and the skill; the notebooks; the records
  and conventions). At `f187d60`, `$PY -m pytest` passed 562 tests with 2
  deselected, the pre-commit hooks passed on the whole tree, the
  documentation built with no warning, the sdist and the wheel built, both
  integration tests passed (PyBADS 1.5.1; PyVBMC at `89007a4`), and the
  notebooks reran without error, their scripts regenerating identically.
  Fixed in `3a0ed74`, `b77b681` and `919ca57`: the validation's bullet in the README and
  `index.rst`, which stated the thresholded estimates' agreement with their
  expected value, and "early stopping", which nothing defined; the
  citations of [1] for an unbiased variance estimate, which [1] calls
  calibrated (it is unbiased: its expectation is Li₂(1 − p)); the
  condition of the guarantee, which only the FAQ's copy of "When IBS fits"
  stated in full; the FAQ's answer on the exit flags, which dropped that
  flag 0 with an infinite `max_time` gives an unbiased estimate, and four
  answers ported from the IBS wiki, which lost substance or reversed its
  judgement on trial-dependent repeats; the docstrings on responses that
  mix numbers and text, which promised a `TypeError` that two of the three
  cases do not raise, and on the `rng` keyword; the catalogue's KD-6, KD-14
  and KD-18, and two entries of the changelog; the notebooks' text on the
  exit flags and the PyBADS solution, and the grid check's tolerance on
  the SDs, as large as the differences it judged (0.5% in place of 2%);
  smaller corrections. The FAQ's five labels with underscores, whose
  anchors Sphinx writes with hyphens, take hyphens before a release ships
  them.
- Decided after that review (PI, 2026-10-10): D25 (`return_positive`); D22
  covers `ibs_basic`; `AGENTS.md` states that an FAQ label does not change
  once a release has shipped it, in place of the list of the labels that
  other files link, which step 5 asked for after PyBADS's; the overflow of
  extreme trial weights, which the pull request left for later, is an item
  of `dev/TODO.md`; the notebooks' outputs are those of a rerun in the
  session's container, which reproduces those of `d7f6b72`, where the pull
  request's came from another machine, on which PyBADS and PyVBMC took
  other paths from the same seeds, as `AGENTS.md` now says they can.
- For Phase 5: the README, `installation.rst` and `development.rst` install
  `pyvbmc>=1.5`, and the links to PyVBMC's FAQ resolve, once PyVBMC 1.5.0 is
  on PyPI and its documentation published (2026-10-13); the README's badges
  of `docs.yml` and `build.yml` wait for those workflows; the README and
  `installation.rst` promise conda-forge's package for Python 3.10, which
  its recipe has to keep; the README's badge of `tests.yml` names `main`,
  where nothing runs that workflow, so it shows no status until a run
  there (a dispatch after the merge, a schedule, or another target); the
  notebooks are rerun before the release. For Phase 6: the labels of
  PyBADS's and PyVBMC's FAQs that PyIBS links join those repositories'
  lists of linked labels.

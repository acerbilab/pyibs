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
| Time limit | `MaxTime`, exit flag 2. Vectorized path: a trial's value averages its repeats with a positive count, including the partial count of the repeat it was sampling. Loop path: averages its completed repeats, NaN when it has none | None | Each trial's value averages its completed repeats; a trial with none raises; exit flag 2 and a warning (D3, deliberate differences) |
| Outputs | Value; variance, or SD with `ReturnStd`; exit flag; `funcCount`, `NsamplesPerTrial`, per-trial values and variances | Per-repeat values and variances, per-trial sums, calls, samples, seconds | 0.1.0's `additional_output` forms, as Python floats; `"full"` adds the per-trial arrays (D2, D10) |
| Sample count | The vectorized path adds the number of open trials per call, whatever the samples requested of each | Every simulated row | As the engine (deliberate difference) |
| Simulator | `fun(params, dmat, varargin{:})`, global random state | Called with the generator of the draw | `sample_from_model(params, design_rows)`, with `rng=` when its signature has a parameter named `rng` (D8, D18) |
| Matching | Every column must agree | Every column must agree; refuses kinds that NumPy never finds equal | As the engine |
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
- One heavy process at a time (`AGENTS.md`, "Setup and commands"). Long
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

**Status**: done through the review of step 11; the push awaits the PI
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
11. [~] `/doublecheck` on the phase. Then, on the PI's instruction,
    `git push -u origin dev-next`, and check that the smoke run of
    `tests.yml` passes (`gh run list --branch dev-next`).

**Verification**:
- [x] `$PY -m pytest` passes; the Worklog records the number of tests and
      the runtime.
- [x] The check of `dev/private/extraction.md` passes, and
      `git ls-files dev/private` prints nothing.
- [x] `$PY -m pre_commit run --all-files` passes.
- [x] The installation, build and YAML checks of steps 8 and 9 pass.
- [ ] After the push, the smoke run of `tests.yml` passes.

### Phase 1: the interface and parity with `ibslike.m`

**Status**: pending
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
1. Record `git -C ../ibs rev-parse HEAD` in the Worklog; if it differs
   from Phase 0's, say what changed in `ibslike.m` and `ibs_basic.m`.
2. `pyibs/_sampler.py`, per the parity table, reading `../ibs/ibslike.m`
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
     `None` is decided at the object's first call by timing one simulation
     of all trials against `vectorized_threshold`, as `ibslike.m` does
     (`False` at the threshold or above, or when `num_reps == 1`), and the
     decision is kept for the object's later calls, readable as an
     attribute. `True` with `num_reps == 1` falls back to `False` with a
     warning, as in `ibslike.m`.
   - `max_time` (D3): checked after every call; once exceeded, sampling
     stops, each trial's value averages its completed repeats, a trial
     with none raises `IBSSamplingError`, and the exit flag is 2.
   - The sampler's tests whose expectations depend on the schedule (exact
     calls, samples per call) are updated, and the Worklog lists each;
     the statistical tests stay as they are.
3. `pyibs/ibs.py` replaces 0.1.0's, with the class `IBS` (D2, D3, D8, D10,
   D16):
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
     every setting, raising `ValueError` or `TypeError` with a message
     that names the setting and what it takes.
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
     per-trial arrays `neg_logl_trials` and `neg_logl_var_trials`, all
     shown by its `__repr__`. `"none"` is accepted as None, as in 0.1.0;
     any other value raises `ValueError`. As in `ibslike.m`,
     `return_positive` changes the sign of the total only.
   - Exit flags 0, 1 and 2, with 0.1.0's messages for them; reaching
     `max_time` also issues a `UserWarning` (D3). The cap raises
     `IBSSamplingError`, a `RuntimeError` naming the trials over the cap.
   - A zero variance is returned as computed, with the `UserWarning` of
     D10; its link points to the FAQ answer that Phase 4 writes, under the
     published documentation's address.
4. `pyibs/ibs_basic.py`: read `../ibs/ibs_basic.m`; keep the function's
   signature, make a missing design work (the simulator then receives the
   trial index, as `IBS` does), compare every column, and pass the
   generator as `IBS` does. `pyibs/__init__.py` exports the public names
   and `__version__` (from `importlib.metadata`).
5. The example model: `examples/psycho_model.py`, after
   `../ibs/psycho_gen.m` and `psycho_nll.m`, with the simulator and the
   closed-form log-likelihood. In `pyproject.toml`, `packages` gains
   `pyibs.examples`, with `package-dir = {"pyibs.examples" = "examples"}`
   and `[tool.setuptools.package-data]` `"pyibs.examples" = ["*.ipynb"]`,
   as in PyBADS. Remove `pyibs/psycho_generator.py`,
   `pyibs/psycho_neg_logl.py` and the three notebooks from `pyibs/`
   (D13). Commit steps 2 to 5.
6. Tests:
   - `test_ibs.py`: each output form and its types (`type(res) is tuple`,
     Python floats); `return_positive`; scalar and per-trial weights;
     responses with several columns, text responses and a design of None;
     the exit flags; the cap's error; the warning on a zero variance
     (trials that always match); `max_time` with a simulator that
     sleeps, with margins wide enough for slow CI runners; validation
     errors, and whole-number floats accepted for the counts;
     reproducibility (two objects with one seed and a simulator that takes
     the generator give equal estimates; a simulator without a generator
     parameter works); and agreement in distribution of the `vectorized`
     settings.
   - `test_ibs_basic.py`, and `test_examples.py`: the IBS estimate of the
     example model agrees with its closed form within 4.5 standard errors
     at three seeded parameter vectors.
   - `test_ibslike_ports.py` calls the public `IBS`.
7. `pyibs/README.md`: the catalogue of deliberate differences from
   `ibslike.m` 0.96, after `../pybads/pybads/bads/README.md`, one entry per
   deliberate difference of the parity table and of steps 2 to 4, each
   with its reason. Every statement about what `ibslike.m` does is checked
   against `../ibs/ibslike.m` and cites its lines; descriptions of
   `ibslike.m` in the engine's docstrings are not copied unchecked.
8. `CHANGELOG.md`, under `Unreleased`: the entries for the changes of this
   phase, and the "Upgrading from 0.1.0" list that opens the section,
   covering the defects listed under Context and the changed behaviour:
   the cap raises; `max_iter` counts samples; `max_mem` defaults to
   `ibslike.m`'s formula instead of 1e6; acceleration is deterministic by
   default; the threshold's semantics; the example modules leave the
   package; Python 3.10 or newer.
9. `AGENTS.md`: an "Architecture" section (the modules and what each
   owns); under "What spans files", the catalogue as the list of
   deliberate differences, which a change that adds or removes one
   updates; a section "Tests and their traps" (the `max_time` test's
   timing margins); and the sentence under "The project" that says the
   tree holds the 0.1.0 code goes. Commit.

**Verification**:
- [ ] `$PY -m pytest` passes, the ports of `runtest1` to `runtest3`
      included; the Worklog records the number of tests and the runtime.
- [ ] `$PY -m pre_commit run --all-files` passes.
- [ ] Every row of the parity table is implemented as its last column
      says, and the catalogue lists every deliberate difference.

### Phase 2: review against `ibslike.m`

**Status**: pending
**Executor**: Opus (orchestrator), with two Opus sub-agents that only read
and reason (no test runs or other heavy processes).
**Needs**: `../ibs`, `../pubs-llms`.
**Goal**: every difference between PyIBS and `ibslike.m` found, and either
listed in the catalogue with its reason or fixed.

**Steps**:
1. Reviewer A compares `../ibs/ibslike.m` and `../ibs/ibs_basic.m` (the
   commit recorded in Phase 1) with `pyibs/` line by line: options and
   defaults, validation, the sampling schedule, the threshold, the cap,
   the time limit, outputs, exit flags, errors and the self-tests. It
   reports each behavioural difference with file and line on both sides,
   and checks every statement about `ibslike.m` in the docstrings and the
   catalogue against its source.
2. Reviewer B checks the package against [1] and for internal
   correctness: the estimator and its variance estimate, the weights, the
   independence of the repeats under the sampler's schedule, the
   threshold of Appendix C.1, the cost counts, and edge cases (one trial,
   `num_reps=1`, trials that always match, a zero variance, text, bytes
   and object responses, a design of None).
3. Consolidate both reports into the ledger
   `dev/results/<YYYY-MM-DD>-port-review.md`: each finding with its
   verdict (deliberate difference, defect, or no issue) and its fix,
   catalogue entry or `dev/TODO.md` item. The PI rules on the verdicts.
4. Fix the defects, each with a test that fails before the fix, and
   update the catalogue and the changelog. Index the ledger in
   `dev/README.md`. Commit.

**Verification**:
- [ ] Every finding of the ledger has a verdict and an outcome.
- [ ] The suite and the pre-commit hooks pass.

### Phase 3: statistical validation, PyBADS and PyVBMC, timing

**Status**: pending
**Executor**: Opus (orchestrator), running one heavy process at a time.
**Needs**: PyBADS and PyVBMC as step 2 installs them.
**Goal**: evidence that PyIBS 1.5's estimates are unbiased and calibrated
across models and settings, that PyBADS 1.5 and PyVBMC 1.5 run with it as
their target, and how its speed compares with 0.1.0's.

**Steps**:
1. `dev/scripts/validate.py`, over these models, each with an exact
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
   `max(Y, -T)`, and ended repeats draw fewer samples.
   Separately, a model whose trials all have p = 1 returns a value and a
   variance of exactly 0. Report every failing cell to the PI before
   going on.
2. Integration tests, `pyibs/testing/integration/test_pybads.py` and
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
3. `dev/scripts/timing.py`: the wall time of one estimate at N = 100,
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
4. The record: `dev/experiments/validation_<YYYYMMDD>/` with its
   `README.md` and provenance (`dev/README.md`), and the summary
   `dev/results/<YYYY-MM-DD>-validation.md`, both indexed in
   `dev/README.md`. Commit.

**Verification**:
- [ ] Every cell of step 1 passes, or the PI has ruled on its failure.
- [ ] Both integration files pass.
- [ ] The record holds its provenance.

### Phase 4: documentation, examples, update check and skill

**Status**: pending
**Executor**: Opus (orchestrator); the documentation site, the FAQ and the
notebooks may each go to an Opus sub-agent, one at a time for anything
that runs code.
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
- IBS remains the method of choice where amortization is hard, because
  each trial's context can be unique: the board positions of a model of
  game play, each of which may occur once ([1], Section 5.4; B. van
  Opheusden et al., 2023, "Expertise increases planning depth in human
  gameplay", Nature). And it serves where per-dataset guarantees matter:
  amortized estimates can fail on a given dataset, and need diagnostics
  and a fallback (C. Li et al., 2026, "Amortized Bayesian Workflow",
  Transactions on Machine Learning Research,
  https://openreview.net/forum?id=osV7adJlKD), while IBS's estimates are
  unbiased, with a calibrated variance, for every dataset, without
  training.
- Its costs: about 1/p_i samples for trial i, so improbable responses are
  expensive (the likelihood threshold bounds the cost at poor
  parameters); responses must be discrete, or binned.
The Nature reference is verified (authors, volume, pages, DOI) before it
is cited; the transcriptions in `../pubs-llms/publications/`
(`vanopheusden2020unbiased_*`, `li2026amortized_*`) are the sources for
the other two.

**Steps**:
1. The update check (D12): `pyibs/_update_check.py` with
   `check_for_updates()`, exported by `pyibs/__init__.py`, copied and
   adapted from `../pybads/pybads/_update_check.py`, with its tests
   (which do not reach the network). It keeps PyBADS's rule that its
   networking modules are imported inside the function.
   `../pybads/dev/plans/version-check.md` is the design; its old-release
   reminder and `RELEASE_DATE` are not carried over.
2. `examples/`: notebooks 1, basic use and calibration, after
   `../ibs/ibs_example.m`; 2, maximum-likelihood estimation with PyBADS;
   3, posterior and evidence with PyVBMC; and
   `examples/scripts/Makefile` after PyBADS's. Rerun the notebooks with
   the venv's interpreter first on `PATH`:
   `PATH="$PWD/.venv/Scripts:$PATH" make -C examples/scripts run`
   (`.venv/bin` elsewhere), and commit their outputs.
3. `README.md`, after `../pybads/README.md`'s sections: What is it?,
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
4. `docsrc/` after `../pybads/docsrc/`: `Makefile`, `make.bat`,
   `.nojekyll` (which the `github` target copies into `docs/`), and under
   `docsrc/source/`: `conf.py`, `index.rst`, `installation.rst`,
   `quickstart.rst`, `documentation.rst`, hand-written API pages under
   `api/` (`IBS`, `EstimateResult`, `IBSSamplingError`, `ibs_basic`,
   `check_for_updates`), `examples.rst`, `faq.md`, `development.rst`,
   `about_us.rst`, `_static/` and `css/`.
5. `docsrc/source/faq.md`: port the FAQ of `../ibs.wiki` (the commit of
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
6. `skills/pyibs/SKILL.md` after `../pybads/skills/pybads/SKILL.md`; what
   it tells an agent about when PyIBS fits a problem follows the
   Positioning above.
7. Build the documentation:
   `PATH="$PWD/.venv/Scripts:$PATH" make -C docsrc github`
   (`.venv/bin` elsewhere).
8. `AGENTS.md`: the documentation build, the FAQ's linked labels, the
   examples and their rerun, and the network rule of the update check, as
   in `../pybads/AGENTS.md`. Commit.

**Verification**:
- [ ] The documentation builds without warnings.
- [ ] `make -C examples/scripts run` reruns every notebook without error.
- [ ] The suite and the pre-commit hooks pass.

### Phase 5: release

**Status**: pending
**Executor**: Opus (orchestrator); each outward step on the PI's
instruction.
**Needs**: `../pybads`; the GitHub CLI `gh`, signed in to an account that
can fork repositories.
**Goal**: PyIBS 1.5.0 on GitHub, PyPI and conda-forge, by the procedure of
`../pybads/AGENTS.md`, "Setup and commands".

**Steps**:
1. Copy `../pybads/.github/workflows/build.yml`, `release.yml` (trusted
   publishing through the `pypi` environment, which admits only `v*`
   tags) and `docs.yml`, adapted to PyIBS. Add the release procedure to
   `AGENTS.md`. Commit.
2. The release gate: the full suite, the pre-commit hooks, the
   integration tests, the documentation build, and the notebooks rerun.
   Then `/doublecheck` on the whole of `dev-next` against `main`.
3. From this step on, only with both accesses of "Release access"
   (Context). `CHANGELOG.md`: `Unreleased` becomes `[1.5.0] - <date>`
   under a new, empty `Unreleased`. Commit.
4. The PI creates the branch `gh-pages` on `origin`, which `docs.yml`
   checks out, holding only an empty `.nojekyll` at its root, as PyBADS's
   and PyVBMC's do (`docs.yml` copies `docs/*`, which skips dotfiles, and
   without `.nojekyll` GitHub Pages drops `_static/` and serves the site
   unstyled): `git switch --orphan gh-pages`, `touch .nojekyll`,
   `git add .nojekyll`, `git commit -m "docs: gh-pages"`,
   `git push origin gh-pages`, `git switch dev-next`. The PI sets GitHub
   Pages to serve it, pushes `dev-next`, and opens the pull request into
   `main`. CI passes; the PI squash-merges it, and
   sets the branch protection of `main` to the checks' names.
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
  unfinished repeat, whose value depends on the schedule, and its loop
  path returns NaN for a trial with no completed repeat).
- **D4. Acceleration grows after every call by default, and the time
  rule of `ibslike.m` is opt-in** — so that a seed reproduces a run, as in
  PyBADS and PyVBMC. The values of complete repeats do not depend on the
  schedule; it changes the cost, the variance estimate of a repeat that
  the threshold ends, and, under the time limit, which repeats complete.
  Rejected: `ibslike.m`'s default, which makes the samples requested
  depend on the wall-clock time.
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
  call, and kept** (PI, 2026-10-09) — by timing one simulation of all
  trials, as `ibslike.m` does, so the decision follows MATLAB's rule
  while later calls stay on one schedule; `True` or `False` given
  explicitly makes a run fully reproducible. Rejected: `ibslike.m`'s
  decision at every call (results depend on the timing whenever the
  decision flips).

## Open Questions

None. The PI settled the plan's questions on 2026-10-09: D17 to D19, and
timings against 0.1.0 only, with no MATLAB installation in the
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
  holds no file of `dev/`; the three workflows parse. The smoke run after
  the push is pending.
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

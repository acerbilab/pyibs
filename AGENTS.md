## About this file

This file states what holds in this repository, for an agent who would not
meet it at the point of need: couplings that span files, procedures that
gate a change, traps that fail silently or at a cost, and conventions that
nothing enforces. It is not a record of changes: what a change did belongs
to its commit and pull request. What an agent meets where it matters stays
there, in a module's docstring or the user documentation. Add a line only
when an agent without it would go wrong, and write it as a fact about the
repository as it stands.

## The project

PyIBS is the Python implementation of inverse binomial sampling (IBS; van
Opheusden, Acerbi & Ma, 2020, PLOS Computational Biology,
https://doi.org/10.1371/journal.pcbi.1008483): unbiased estimates of the
log-likelihood of a model that can be simulated but not written down, for
data with discrete responses, with an estimate of their variance. Its
reference implementation is the MATLAB function `ibslike.m` of
`acerbilab/ibs`. With PyBADS and PyVBMC it makes up the lab's tools for
fitting models to data (https://acerbilab.org/model-fitting/): PyBADS
optimizes an IBS estimate as a noisy target, and PyVBMC uses it for
posterior and evidence inference.

The released version is 0.1.0, published on PyPI and conda-forge as
`pyibs`. Version 1.5.0 is prepared on the branch `dev-next` by the plan
`dev/plans/pyibs-1.5.md`, which states its scope, its phases and its
decisions.

- `dev/` holds the maintainer records: plans, findings, the evidence they
  cite and the tooling that produced it. `dev/README.md` says where each
  kind of record goes.
- `dev/private/` is gitignored and holds maintainer notes that are not
  published. A tracked file may point to a note there by its path, but
  never restates it.
- `docsrc/` is the Sphinx source of the documentation, whose links name
  its address, https://acerbilab.github.io/pyibs/; `docs/` is its
  gitignored build output. `examples/` holds the example notebooks and the
  example model, installed as `pyibs.examples`. `skills/pyibs/SKILL.md`
  is the skill that points a coding agent to the documentation.

### Sibling repositories

Work that compares with MATLAB, runs PyBADS or PyVBMC, or writes the
documentation reads sibling checkouts, cloned next to this repository when
absent. Each has its own `AGENTS.md` or README; read it
before relying on more than this file states about it.

| Path | Clone from | Used for |
| :--- | :--- | :--- |
| `../ibs` | https://github.com/acerbilab/ibs | `ibslike.m`, the reference, with `ibs_basic.m`, the tutorial `ibs_example.m` and the example model `psycho_gen.m`, `psycho_nll.m` |
| `../ibs.wiki` | https://github.com/acerbilab/ibs.wiki.git | the FAQ that the PyIBS FAQ ports |
| `../pybads` | https://github.com/acerbilab/pybads | the conventions this repository follows: `../pybads/AGENTS.md` is the model for its tooling, release procedure and records |
| `../pyvbmc` | https://github.com/acerbilab/pyvbmc | as `../pybads`, and the CI matrix |
| `../pubs-llms` | https://github.com/acerbilab/pubs-llms | the IBS paper as Markdown, `publications/vanopheusden2020unbiased_{main,appendix,backmatter}.md` |
| `../model-fitting` | https://github.com/acerbilab/model-fitting | the lab's page of its tools for fitting models to data, whose PyIBS card links this package |

A record of a comparison names the commit of the checkout it read.

`ibslike.m` runs unmodified under GNU Octave (8.4.0, the package of Ubuntu
24.04, `apt-get install octave`) with `dev/scripts/octave/compat` on
Octave's path, which supplies the `fields` that `ibslike.m` calls and
Octave lacks. `ibslike('test')` also plots, which needs a graphics
toolkit; `dev/scripts/octave/headless` supplies stand-ins that draw
nothing. From the repository root:

```console
octave-cli --norc -q --eval "warning('off', 'Octave:shadowed-function'); addpath('../ibs', 'dev/scripts/octave/compat', 'dev/scripts/octave/headless'); ibslike('test')"
```

Octave is not MATLAB, and accepts some code that MATLAB refuses: its `&&`
takes arrays, which it reduces with `all`. A record that rests on an
Octave run says so, names Octave's version, and names the semantic that
it takes MATLAB to share.

## PyBADS and PyVBMC

PyIBS's estimates reach PyBADS 1.5 and PyVBMC 1.5 through their
noisy-target interface: with `options={"specify_target_noise": True}`, the
target returns a pair `(value, sd)`.

- PyBADS minimizes the value, so it takes the negative log-likelihood. It
  accepts only a Python `tuple` of length 2 (`type(res) is tuple`): a list
  or an array raises `ValueError`.
- PyVBMC takes the log joint density, or the log-likelihood when `prior=`
  or `log_prior=` is given, to which it then adds the log prior itself. It
  unpacks any pair.
- Both raise `ValueError` on an SD that is not finite and strictly
  positive. The IBS variance estimate is exactly zero when every trial
  matches at its first sample.
- Both call the target with one parameter vector, 1-D, in the original
  parameter space.

The integration tests, `pyibs/testing/integration/`, fit the example model
with each of them, an IBS estimate as the noisy target. They carry the
marker `integration`, which `pyibs/testing/conftest.py` registers, so that
it is known outside the checkout too, and which `addopts` deselects;
`pytest --pyargs pyibs`, run outside the checkout, reads no `addopts` and
so runs them whenever the packages are installed. Each file skips when its
package is absent or older than it needs: PyBADS 1.5.1, read from its
version, and PyVBMC 1.5, read from the `seed` parameter of `VBMC`, since an
install from an untagged checkout of PyVBMC reads its version as
1.0.5.devN. They are installed into the venv with
`uv pip install "pybads>=1.5.1" "pyvbmc>=1.5.0"`. PyVBMC 1.5.0 is not on
PyPI before 2026-10-13; until then it comes from its release branch,
`uv pip install "pyvbmc @ git+https://github.com/acerbilab/pyvbmc@feat-release-1.5-preparation"`.
Each file is a heavy process (about 40 s and 2 min), run unbuffered and
logged:
`$PY -u -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v > dev/scripts/runs/it_pybads_$(date +%s).log 2>&1`,
and `test_pyvbmc.py` likewise.

## Setup and commands

The development environment is a venv at `.venv` (gitignored). No shell
activates it, and a bare `python` is another installation, so every
command names the venv's interpreter, written `$PY` here:
`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere. The
venv is created with uv:

```console
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
$PY -m pre_commit install
```

The version comes from git tags through setuptools_scm, which writes the
gitignored `pyibs/_version.py`.

`pyproject.toml` is authoritative; `setup.py` is a shim. The `packages` of
`pyproject.toml` lists every directory of the package, so a new
subpackage is added there; the build warns "Package would be ignored" for
one that is missing. The tests ship in the wheel and run from an installed
package as `pytest --pyargs pyibs`; what they need is the `test` extra,
and only the test modules (`test_*.py` under `pyibs/testing/`) import
pytest.

```console
$PY -m pytest                                           # CI adds -x -vv
$PY -m pytest pyibs/testing/test_sampler.py::test_cost
```

The tests live in `pyibs/testing/`, and default discovery is limited to
them (`testpaths` in `pyproject.toml`), with the tests marked
`integration` deselected (`addopts`).

The test jobs are defined once, in `.github/workflows/test-matrix.yml`, and
`tests.yml` and `merge-tests.yml` run both whenever they run the tests. The
matrix job installs PyIBS with its `test` extra and runs the suite on the
operating systems and Python versions it is given. The minimum-versions job
runs it on Ubuntu with Python 3.10 and the lowest NumPy, SciPy and pytest
that `pyproject.toml` allows, which uv resolves
(`--resolution lowest-direct`). That job, the default `python-version` list
of `test-matrix.yml` and the dispatch list of `tests.yml` name Python 3.10
themselves, so a change of `requires-python` changes them too. The job also
needs a lower bound on every requirement of `dependencies` and of the
`test` extra: uv resolves a requirement without one to its first release,
with only a warning. `merge-tests.yml` runs the full matrix (Ubuntu,
Windows, macOS × Python 3.10–3.14) on a pull request to `main` or to a
`dev*` branch, only when its changes against that base touch `pyibs/`,
`examples/` (installed as `pyibs.examples`), `pyproject.toml`, `setup.py`
or one of the three test workflows, so that a
Dependabot update of an action they use is tested before it merges; a pull
request that changes anything else runs no tests. `tests.yml` runs the full
matrix on dispatch, and a smoke run, the matrix reduced to Ubuntu with
Python 3.14, on each push to a `dev*` branch that touches `pyibs/`,
`examples/`, `pyproject.toml`, `setup.py`, `tests.yml` or
`test-matrix.yml`.

The documentation is built with
`PATH="$PWD/.venv/Scripts:$PATH" make -C docsrc github` on Windows
(`.venv/bin` elsewhere; the target calls `sphinx-build`), which copies the
notebooks of `examples/` into `docsrc/source/_examples/`, builds the site,
copies it with `.nojekyll` into `docs/`, and removes the copies. The build
gives no warning, and a change keeps it so. Nothing generates the API
pages: a new public class or function needs a hand-written `.rst` under
`docsrc/source/api/` and an entry in the toctree that owns it
(`api/classes/classes.rst` or `api/functions/functions.rst`, and
`documentation.rst` for a headline page). numpydoc reads every line of a
docstring's Attributes section as an attribute, so a paragraph there
renders as bogus attributes and goes in Notes instead, and it shows a
property's own docstring in place of the property's Attributes entry, so
the two say the same. The figure of "How does it work?", which the README
and `index.rst` show, is drawn by `dev/scripts/ibs_cost_variance.py`.

The notebooks in `examples/` ship in the wheel as `pyibs.examples` and are
rendered without execution by the documentation's build; no CI job runs
them, so a change that breaks one goes unnoticed. `make -C
examples/scripts run`, with the venv's interpreter first on `PATH` (the
target calls `python`) and nbconvert, ipykernel, matplotlib, PyBADS and
PyVBMC installed, reruns them in place in about two minutes, nearly all of
it PyVBMC's; nbconvert called directly would save each flush of a cell's
output as an output of its own, and the cells' execution times in their
metadata. The target sets `PYBADS_NO_UPDATE_REMINDER` and
`PYVBMC_NO_UPDATE_REMINDER`, so that the outputs show neither package's
reminder of an old release. On one machine, their seeds fix their
outputs but for the elapsed times and the tip that PyBADS draws at random.
On another platform, or with other numerical libraries, PyBADS's and
PyVBMC's runs can take another path from the same seeds, and PyVBMC's also
where its cached performance calibration differs. The notebooks'
markdown describes their outputs, so a rerun whose outputs change is read
against it. `examples/scripts/*.py` are generated from the notebooks by
`make -B -C examples/scripts` (GNU Make, with nbconvert, IPython, and
black and isort at the versions of the pre-commit hooks), after the
notebooks have run: for a notebook never run, nbconvert leaves out the
script's closing blank line, and the Makefile's `head -n -1` then cuts its
last line of code. The scripts are not edited by hand.

Formatting is enforced by the pre-commit hooks alone (black at line length
79 on every Python file and the notebooks' code cells, isort with the black
profile, pycln); no CI job checks it, and the whole tree passes them.

On the PI's workstation, run one heavy process at a time (the test suite,
a validation or timing run, a PyBADS or PyVBMC run): concurrent runs, each
multi-threaded, can bring it down. A cloud session's container has no such
limit. A long run writes its output unbuffered
(`python -u`) to a uniquely named log under `dev/scripts/runs/`, which a
fresh clone creates first (`mkdir -p dev/scripts/runs`).

## Architecture

`IBS` (`pyibs/ibs.py`) is the interface. It holds its settings in a
`_Settings` of the sampler, which checks them, with the generator `rng`
and the decision of `vectorized=None`; each call checks its arguments,
runs one draw of `num_reps` repeats, and turns it into the outputs (a
float, a tuple or an `EstimateResult`), the exit flag and the warnings.
`pyibs/_sampler.py` owns the sampling: `_Settings` and its checks; `sample`,
one draw, with the schedule, the cap, the likelihood threshold
(`_MatchCounts`) and the time limit (`_limited_estimates`); `first_round`,
the timing call of `vectorized=None`; `_simulate`, one simulator call with
its checks of the output's shape and kind; and `IBSSamplingError`.
`pyibs/_estimates.py` owns the per-count formulas, the check of the trial
weights and the reduction of counts to estimates (`repeat_estimates`).
`pyibs/ibs_basic.py` is the didactic loop over the trials, which shares the
check of the responses and the generator with `IBS`. `examples/` is
installed as `pyibs.examples`, with the example model in
`psycho_model.py`. `pyibs/testing/` holds the tests, with `_exact.py`
(exact IBS draws from geometric counts, and the exact moments) and
`_helpers.py` (the scripted and Bernoulli simulators, and `ibslike.m`'s
samples per call).

The interface and the sampler name some settings differently:
`num_samples_per_call` (0 for the default) is `initial_samples` (None),
`max_iter` is `max_samples_per_trial`, `neg_logl_threshold` (`np.inf` for
none) is `neg_loglik_threshold` (None), and `response_matrix` and
`design_matrix` are `responses` and `design`. `IBS` checks those settings
itself, and those that only it has (`vectorized`,
`vectorized_threshold`), before `_Settings` checks them again under its
names; it passes `("max_iter", "num_reps")` to `_Settings.names` for the
messages of `IBSSamplingError`. An error thus names the argument the user
gave.

## What spans files

- **The catalogue of deliberate differences.** `pyibs/README.md` lists
  every deliberate difference between PyIBS and `ibslike.m` 0.96, and
  `ibs_basic.m`, each with its reason and the lines of `ibslike.m` it cites
  at `2229c00`; a difference that it does not list is a defect until shown
  otherwise. A change that adds, removes or alters one updates its entry,
  and the docstrings that describe `ibslike.m`, in `pyibs/_sampler.py`
  above all, agree with it.
- **The FAQ.** `docsrc/source/faq.md` quotes settings and their defaults,
  messages and warnings, the record of the validation, and PyBADS's and
  PyVBMC's interfaces and messages, and nothing runs its snippets or checks
  them against the code: a change to one of those is made in the FAQ by
  hand. Its table of contents is written out by hand, and
  `skills/pyibs/SKILL.md` names its sections and questions by their
  titles, so a question added or renamed is updated in both. Each question
  carries a label `(faq-<slug>)=`, made from the question's text when the
  question is added: lowercased, its punctuation dropped but for hyphens,
  and its spaces and underscores as hyphens, so that the label is also the
  anchor that Sphinx writes. A label does not change once a release has
  shipped it, even when its question is reworded: the warning on a
  variance estimate of 0 (`_FAQ_ZERO_SD` in `pyibs/ibs.py`) links one from
  every installed copy, the README of each release is its page on PyPI,
  and the notebooks ship in the wheel. Before a label is changed, or its
  question moved or removed, `git grep` for the label finds the files that
  link it. The FAQ and the notebooks also link labels of the FAQs of PyBADS
  (`acerbilab.github.io/pybads/faq.html#…`) and PyVBMC
  (`acerbilab.org/pyvbmc/faq.html#…`), which nothing here checks: grep for
  those addresses to check them against the two repositories' `faq.md`.
- **When IBS fits.** What PyIBS is for, and when amortized
  simulation-based inference or a closed-form likelihood serves better, is
  stated, with its citations, in the README's "When should I use PyIBS?",
  its copy in `index.rst` and the FAQ's "General" section, and without them
  in the skill's "When PyIBS fits", so a change to it is made in all four.
- **Text that the README shares.** `index.rst` restates the README's "What
  is it?", "What's new in PyIBS 1.5", "How does it work?" and "When should
  I use PyIBS?", its references, citation and acknowledgments, which
  `about_us.rst` repeats in part (the thanks to 0.1.0's author, the coding
  agents and the grants); `installation.rst` and `quickstart.rst` copy its
  Installation and Quick start; and the skill quotes its section titles. The
  requirements, Python 3.10, NumPy 2.0 and SciPy 1.13, are stated in the
  README, `index.rst`, `installation.rst`, `development.rst`, the FAQ and
  the changelog besides `pyproject.toml`. A change to one copy is made in
  the others.
- **The shared IBS reduction.** `repeat_estimates` in
  `pyibs/_estimates.py` turns matching counts into per-repeat values,
  variance estimates and per-trial sums, for the sampler and for the exact
  draws of the tests (`pyibs/testing/_exact.py`). Two of its properties
  make each repeat's value and variance estimate depend, bitwise, on that
  repeat's counts alone: its sum over the trials is
  `np.sum(... * w, axis=1)`, never `@`; and each term equals `ibs_loglik`
  or `ibs_var` of its count bitwise, whether looked up in its table of the
  formulas, whose extent depends on the size of the count matrix, or
  evaluated directly. Its callers rely on them when they reduce some of
  the repeats at a time: the exact draws reduce their counts in blocks,
  and the sampler, under the likelihood threshold, only the repeats that
  the threshold did not end. `test_chunking_is_bitwise_invariant`
  (`test_exact.py`), `test_unreachable_threshold_changes_nothing`
  (`test_threshold.py`) and the bitwise tests of `test_estimates.py` check
  them.

## Tests and their traps

- Timing decides the time limit, the decision of `vectorized=None` and the
  rule of `acceleration_threshold`. Their tests run on a fake clock that
  the simulator advances, patched over `pyibs._sampler.time`, which times
  the simulator and checks the limit, and over `pyibs.ibs.time`, from which
  `IBS.__call__` counts `max_time` (the `clock` fixture of `test_ibs.py`,
  `FakeTime` in `test_sampler.py`). Two tests of `test_ibs.py` sleep,
  `test_max_time_stops_a_slow_simulator` and
  `test_max_time_raises_for_a_trial_without_a_completed_count`: each call
  sleeps 0.02 s under `max_time=0.05`, every trial that matches does so in
  the first call and never again, so the expected values hold whichever
  call the limit stops after, however slow the runner; without the limit,
  the cap would end the draw with another error after 67 calls. A new test
  that involves time takes one of these two forms, never a margin that a
  slow CI runner can miss.
- `vectorized=None` decides by timing, so a test that compares the
  estimates of two `IBS` objects gives `vectorized` or runs on the fake
  clock.
- The ports of `ibslike.m`'s self-tests (`test_ibslike_ports.py`) check
  `runtest1` and `runtest3` at three seeds and pass at two, with a warning
  for a seed that fails, and `runtest2` at one seed with `ibslike.m`'s
  fixed tolerances. They run through `IBS` with `vectorized=True`, and
  their draws change whenever the sampling consumes the generator
  differently.
- A call that returns a variance estimate of 0, as when every count is 1,
  issues a `UserWarning` with `additional_output` `"var"`, `"std"` or
  `"full"` (not with the value alone), which a test expects or filters.

## Conventions

- **Commits** follow conventional commits. A `Co-Authored-By:` line is
  fine; a `Claude-Session:` trailer is not, even where the session's own
  attribution instructions ask for one. Work collects on the long-lived
  branch `dev-next`. Changes reach `main` through pull requests, which are
  squash-merged and titled `<type>: <summary> (#NN)`; after such a merge,
  `dev-next` is reset onto `main`, keeping only the commits made after the
  merged head, and force-pushed, or the next pull request lists the merged
  commits again.
- **Changelog.** A change to results, to the interface, or to what a
  script sees or has to handle gets an entry in `CHANGELOG.md` under
  `Unreleased`, in the commit that makes it, and so does a speed-up that a
  user notices, with its measured size and a link to its record under
  `dev/`; a message's wording, a minor speed-up or a change to the tests
  gets none. An entry is one or two sentences, written for users, on what
  a user notices, relative to the last release: a fix to a change that no
  release has shipped edits that change's entry, and the reasons and the
  comparison with `ibslike.m` belong in the catalogue (`pyibs/README.md`)
  and the records under `dev/`, to which an entry can point. The fixes are
  grouped under a few themes (`####` headings under Fixed), and changes of
  one kind, such as new checks of the settings' values, share one entry.
  A change that can stop a script written for the last release, or change
  its results, also has a line in the "Upgrading from" list that opens the
  section: the change's only mention when that line says all a user
  needs, and otherwise a pointer to its entry, kept in step with it;
  changes of one kind share a line.
- **Code** follows PyBADS and PyVBMC: plain NumPy/SciPy, numpydoc
  docstrings, black at line length 79, isort with the black profile and
  pycln, which the pre-commit hooks enforce ("Setup and commands").
- **Random numbers.** Every random draw of the package goes through an
  explicit `numpy.random.Generator`, so that a seed reproduces a run
  wherever no timing decides the sampling, and every test that draws
  seeds its generator.
- **Tests.** Statistical tolerances are stated in standard errors (4.5 by
  default), and a failing statistical test is investigated, never
  reseeded.
- **Network access and output.** The package opens a network connection
  only in `pyibs.check_for_updates()` (`pyibs/_update_check.py`), which
  the user calls, and which imports its networking modules inside the
  function; its tests replace `urllib.request.urlopen` and reach no
  network. Nothing else in the package prints: a call reports through its
  return value, its exit flag and Python's `warnings`.
- **Links to the lab.** In what ships or is published, a link that names
  Luigi Acerbi goes to his personal page, https://lacerbi.github.io/. A
  link to the group goes preferably to its main page,
  https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence.
  A paragraph that sends the reader to another of the lab's methods
  (PyBADS, PyVBMC, MATLAB IBS) links the lab's page of them,
  https://acerbilab.org/model-fitting/, with the text "tools for fitting
  models to data".

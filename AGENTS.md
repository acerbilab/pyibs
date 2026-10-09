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
decisions. The package holds the engine (`pyibs/_estimates.py`,
`pyibs/_sampler.py`) and its tests next to the 0.1.0 interface, which
Phase 1 of the plan replaces.

- `dev/` holds the maintainer records: plans, findings, the evidence they
  cite and the tooling that produced it. `dev/README.md` says where each
  kind of record goes.
- `dev/private/` is gitignored and holds maintainer notes that are not
  published. A tracked file may point to a note there by its path, but
  never restates it.

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
and no module of the package imports pytest.

```console
$PY -m pytest                                           # CI adds -x -vv
$PY -m pytest pyibs/testing/test_sampler.py::test_cost
```

The tests live in `pyibs/testing/`, and default discovery is limited to
them (`testpaths` in `pyproject.toml`), with the tests marked
`integration` deselected (`addopts`).

The test jobs are defined once, in `.github/workflows/test-matrix.yml`, and
every run of `tests.yml` or `merge-tests.yml` includes both. The matrix job
installs PyIBS with its `test` extra and runs the suite on the operating
systems and Python versions it is given. The minimum-versions job runs it
on Ubuntu with Python 3.10 and the lowest NumPy, SciPy and pytest that
`pyproject.toml` allows, which uv resolves (`--resolution lowest-direct`).
That job and the matrix lists name Python 3.10 themselves, so a change of
`requires-python` changes them too. `merge-tests.yml` runs the full matrix
(Ubuntu, Windows, macOS × Python 3.10–3.14) on a pull request to `main` or
to a `dev*` branch, only when its changes against that base touch `pyibs/`,
`pyproject.toml`, `setup.py` or one of the three test workflows, so that a
Dependabot update of an action they use is tested before it merges; a pull
request that changes anything else runs no tests. `tests.yml` runs the full
matrix on dispatch, and a smoke run, the matrix reduced to Ubuntu with
Python 3.14, on each push to a `dev*` branch that touches the package,
`pyproject.toml`, `setup.py`, `tests.yml` or `test-matrix.yml`.

Formatting is enforced by the pre-commit hooks alone (black at line length
79 on every Python file and the notebooks' code cells, isort with the black
profile, pycln); no CI job checks it, and the whole tree passes them.

Run one heavy process at a time (the test suite, a validation or timing
run, a PyBADS or PyVBMC run): concurrent runs, each multi-threaded, can
bring a workstation down. A long run writes its output unbuffered
(`python -u`) to a uniquely named log under `dev/scripts/runs/`, which a
fresh clone creates first (`mkdir -p dev/scripts/runs`).

## What spans files

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
  comparison with `ibslike.m` belong in the records under `dev/`, to which
  an entry can point. The fixes are grouped under a few themes (`####`
  headings under Fixed), and changes of one kind, such as new checks of
  the settings' values, share one entry. A change that can stop a script
  written for the last release, or change its results, also has a line in
  the "Upgrading from" list that opens the section: the change's only
  mention when that line says all a user needs, and otherwise a pointer
  to its entry, kept in step with it; changes of one kind share a line.
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
- **Links to the lab.** In what ships or is published, a link that names
  Luigi Acerbi goes to his personal page, https://lacerbi.github.io/. A
  link to the group goes preferably to its main page,
  https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence.
  A paragraph that sends the reader to another of the lab's methods
  (PyBADS, PyVBMC, MATLAB IBS) links the lab's page of them,
  https://acerbilab.org/model-fitting/, with the text "tools for fitting
  models to data".

********************************************
Instructions for developers and contributors
********************************************

PyIBS implements inverse binomial sampling in Python. It requires Python
3.10 or later, NumPy 2.0 or later, and SciPy 1.13 or later. The reference
implementation is ``ibslike.m`` in
:labrepos:`MATLAB IBS <ibs>`; the
:mainbranch:`catalogue of deliberate differences <pyibs/README.md>` lists
the intentional differences and explains their reasons.

The published documentation is at https://acerbilab.github.io/pyibs/.

Installation instructions for developers
########################################

For a source checkout with development dependencies:

1. Clone the PyIBS GitHub repository::

     git clone https://github.com/acerbilab/pyibs
     cd pyibs

2. Create a virtual environment with `uv <https://docs.astral.sh/uv/>`__ and
   install PyIBS in editable mode, so changes to the source take effect
   without reinstalling::

     uv venv --python 3.12 .venv
     uv pip install -e ".[dev]"

3. Install the pre-commit hooks, which format and check staged files before
   a commit::

     .venv/bin/python -m pre_commit install

Commands on this page use the environment's interpreter,
``.venv/bin/python``. On Windows, replace this path with
``.venv\Scripts\python.exe``, including in the hook-installation command
above. No environment activation is needed. ``setuptools_scm`` derives
PyIBS's version from the Git tags.

Declare dependencies in ``pyproject.toml``: runtime requirements under
``dependencies``, test requirements under the ``test`` extra, and development
requirements under ``dev``. The development extra also includes the test
requirements. Every runtime and test dependency needs a lower bound; CI
tests the lowest permitted versions.

Coding conventions
##################

Useful background on Python style:

- `PEP 8 -- Style Guide for Python Code <https://peps.python.org/pep-0008/>`__
- `Code style in The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__

The repository conventions below keep contributions consistent with the
existing code.

Code formatting
---------------

The pre-commit hooks run `Black <https://pypi.org/project/black/>`__ at a
line length of 79, `isort <https://pycqa.github.io/isort/>`__ with Black's
profile, and `pycln <https://hadialqattan.github.io/pycln/>`__. To check every
file::

    .venv/bin/python -m pre_commit run -a

Once installed, the hooks check staged files at each commit and may reformat
them. Restage any modified files before committing again. The hooks are
defined in ``.pre-commit-config.yaml`` and their formatting settings in
``pyproject.toml``. CI does not check formatting, so run the hooks locally.

Docstrings
----------

The docstrings follow the
`NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.
Use the `numpydoc example <https://numpydoc.readthedocs.io/en/latest/example.html>`__
and the docstrings in :mainbranch:`pyibs/ibs.py <pyibs/ibs.py>` as guides.

Random numbers
--------------

Pass an explicit ``numpy.random.Generator`` to code that draws random
numbers. A fixed seed must reproduce a run when the sampling schedule does
not depend on timing.

Exceptions
----------

Use `standard Python exceptions <https://docs.python.org/3/library/exceptions.html>`__
where appropriate.

Testing
-------

The ``pytest`` suite lives in ``pyibs/testing/``. It ships with the package
and can be run after installing the ``test`` extra with
``pytest --pyargs pyibs``. From a development checkout, run the full default
suite or a selected test with::

    .venv/bin/python -m pytest
    .venv/bin/python -m pytest pyibs/testing/test_sampler.py::test_cost

Test requirements:

- Run the relevant tests before submitting a change. CI runs the full
  matrix on pull requests to ``main`` or ``dev*`` branches that change the
  package, examples, installation metadata, or test workflows. It covers
  Windows, Linux, and macOS with Python 3.10–3.14, plus Python 3.10 with the
  lowest NumPy, SciPy, and pytest versions allowed by ``pyproject.toml``.
- Seed every test that draws random numbers so its failure can be
  reproduced.
- State statistical tolerances in standard errors, 4.5 by default.
  Investigate a failing statistical test; do not reseed it to make it pass.
- Use a fake clock advanced by the simulator for timing tests, or make the
  expected values independent of the machine's speed. Avoid timing margins
  that a slow runner can miss.
- Some tests port the self-tests of ``ibslike.m``, the reference
  implementation.

The integration tests in ``pyibs/testing/integration/`` fit the example
model with PyBADS or PyVBMC, using IBS estimates as the noisy target. The
default checkout run excludes the ``integration`` marker. Each integration
test skips when its package is absent or too old: it requires PyBADS 1.5.1
or later or PyVBMC 1.5 or later, respectively. These tests take a few
minutes. To run them::

    uv pip install "pybads>=1.5.1" "pyvbmc>=1.5.0"
    .venv/bin/python -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v
    .venv/bin/python -m pytest -m integration pyibs/testing/integration/test_pyvbmc.py -s -v

Code documentation
------------------

Build the documentation with
`Sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_. The
source is in :mainbranch:`docsrc <docsrc>`.
From the repository root, with the environment's ``bin`` directory
(``Scripts`` on Windows) first on the ``PATH``::

    PATH="$PWD/.venv/bin:$PATH" make -C docsrc github

The target copies the example notebooks into the documentation source,
builds the site, copies it to the gitignored ``docs/`` directory, and removes
the temporary notebook copies. Open ``docs/index.html`` to view the result.
On Windows, use ``.venv/Scripts`` in the ``PATH`` command above, or run
``.\make.bat github`` from ``docsrc`` in ``cmd`` with that directory on
``PATH``. The build must produce no warnings.

If pages or navigation do not update correctly, clear the generated site
in ``docs/`` and the cached ``docsrc/_build`` directory, then rebuild::

    make -C docsrc clean

On Windows, run ``.\make.bat clean`` from ``docsrc`` in ``cmd`` instead.

General structure
.................

Each public class or function needs a hand-written ``.rst`` page in
``docsrc/source/api/classes/`` or ``docsrc/source/api/functions/``. Give it a
short introduction and an ``autoclass`` or ``autofunction`` directive to
render the docstring, for example::

    .. autoclass:: pyibs.IBS
       :members:

Add the page to its table of contents: ``documentation.rst`` for a headline
page, and ``api/classes/classes.rst`` or ``api/functions/functions.rst``.
Keep the documentation in step with changes to the public API.

Examples
........

The example notebooks live in ``examples/`` and are installed as
``pyibs.examples``. Their model is in ``psycho_model.py``. The documentation
renders saved outputs without executing the notebooks.
To rerun them in place, with PyBADS, PyVBMC, matplotlib and ipykernel
installed in the environment::

    uv pip install nbconvert ipykernel matplotlib "pybads>=1.5.1" "pyvbmc>=1.5.0"
    PATH="$PWD/.venv/bin:$PATH" make -C examples/scripts run

After rerunning the notebooks, regenerate their code in
``examples/scripts/`` with ``make -B -C examples/scripts``. Keep the
environment first on ``PATH`` and install nbconvert, IPython, and the Black
and isort versions specified by the pre-commit hooks. Edit the notebooks;
the scripts are generated files.

``git`` commits
---------------

Commits follow the
`conventional commits <https://www.conventionalcommits.org/en/v1.0.0/>`__
style. See the
`cheat sheet <https://cheatography.com/albelop/cheat-sheets/conventional-commits/>`__
for examples.

Use a feature branch for a contribution and submit completed changes with
passing tests through a pull request. For example, replacing
``<new-feature>`` with your branch name::

    git checkout -b <new-feature>
    [... do stuff and commit ...]
    git push -u origin <new-feature>
    [... open a pull request on GitHub when ready ...]

See `Feature Branch <https://martinfowler.com/bliki/FeatureBranch.html>`__
for background on this workflow.

Changelog
---------

In the same commit as a change to results, the interface, or what a
script sees or has to handle, add one or two sentences for users under
``Unreleased`` in :mainbranch:`CHANGELOG.md <CHANGELOG.md>`. A change that
can break a script written for the last release or alter its results also
needs an entry in the opening "Upgrading from" list. When a change adds,
removes, or alters an intentional difference from ``ibslike.m``, update the
corresponding entry in the :mainbranch:`catalogue <pyibs/README.md>`.

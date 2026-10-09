********************************************
Instructions for developers and contributors
********************************************

PyIBS is the Python implementation of inverse binomial sampling (it requires
Python 3.10 or later). Its reference implementation is ``ibslike.m`` of
:labrepos:`MATLAB IBS <ibs>`; the
:mainbranch:`catalogue of deliberate differences <pyibs/README.md>` lists
every place where PyIBS differs from it on purpose, with the reason.

The documentation is available at: https://acerbilab.github.io/pyibs/

Installation instructions for developers
########################################

Release versions of PyIBS are available via ``pip`` and ``conda-forge``, but
developers will need to work with the latest source code. They should follow
these steps to install:

1. Clone the PyIBS GitHub repository::

     git clone https://github.com/acerbilab/pyibs
     cd pyibs

2. Create a virtual environment in the checkout with
   `uv <https://docs.astral.sh/uv/>`__, and install PyIBS in it, editable,
   with its development dependencies::

     uv venv --python 3.12 .venv
     uv pip install -e ".[dev]"

3. Install the pre-commit hooks::

     .venv/bin/python -m pre_commit install

The commands below call the environment's interpreter, ``.venv/bin/python``
(``.venv\Scripts\python.exe`` on Windows), by its path, so they need no
activated environment. The version of PyIBS comes from the git tags, through
``setuptools_scm``.

We are using the dependencies listed in ``pyproject.toml``. Please list all
used dependencies there. Dependencies are separated into basic dependencies,
the dependencies of the test suite under ``test``, and the development
dependencies under ``dev``, which include those of ``test``. Every
dependency of PyIBS and of its ``test`` extra has a lower bound, which a CI
job tests.

Coding conventions
##################

We try to follow common conventions whenever possible. Some useful reading:

- `PEP 8 -- Style Guide for Python Code <https://peps.python.org/pep-0008/>`__
- `Code style in The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__

These basic rules should be followed to ensure coherence and to make it easy
for third parties to contribute. In the following, we list more detailed
conventions. Please read carefully if you are contributing to PyIBS.

Code formatting
---------------

The code is formatted with `Black <https://pypi.org/project/black/>`__ at a
line length of 79, `isort <https://pycqa.github.io/isort/>`__ with Black's
profile, and `pycln <https://hadialqattan.github.io/pycln/>`__, with the help
of the pre-commit hooks installed above. To run them on every file::

    .venv/bin/python -m pre_commit run -a

After installation, when you try to commit the staged files, git will
automatically check the files and modify them for meeting the requirements
of the hooks in ``.pre-commit-config.yaml``. The settings of the hooks are
specified in ``pyproject.toml``. You need to restage the file if it gets
modified by the hooks. No CI job checks the formatting: the hooks are its
only check.

Docstrings
----------

The docstrings follow the
`NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.
See an example of a correct docstring from NumPy
`here <https://numpydoc.readthedocs.io/en/latest/example.html>`__, and the
docstrings of :mainbranch:`pyibs/ibs.py <pyibs/ibs.py>` for the style of
PyIBS.

Random numbers
--------------

Every random draw of the package goes through an explicit
``numpy.random.Generator``, so that a seed reproduces a run wherever no
timing decides the sampling.

Exceptions
----------

Please use standard Python exceptions whenever it is sensible. Here is a
list of those `exceptions <https://docs.python.org/3/library/exceptions.html>`__.

Testing
-------

The tests live in ``pyibs/testing/`` and use ``pytest``. They ship with the
package, so they also run from an installed PyIBS, as
``pytest --pyargs pyibs``. From the checkout::

    .venv/bin/python -m pytest
    .venv/bin/python -m pytest pyibs/testing/test_sampler.py::test_cost

A few comments about testing:

- Testing is mandatory! The full suite of tests runs on every pull request
  to ``main`` that changes the package or its tests, on Windows, Linux and
  macOS, with every supported Python version, and on Python 3.10 with the
  lowest versions of NumPy, SciPy and pytest that ``pyproject.toml`` allows.
- Every test that draws random numbers seeds its generator, so that a
  failure repeats when the test is rerun.
- Statistical tolerances are stated in standard errors, 4.5 by default, and
  a failing statistical test is investigated, never reseeded.
- A test that involves time runs on a fake clock that the simulator
  advances, or is written so that its expected values hold however slow the
  machine; never on a margin that a slow machine can miss.
- Some tests port the self-tests of ``ibslike.m``, the reference
  implementation.

The integration tests, in ``pyibs/testing/integration/``, fit the example
model with PyBADS and with PyVBMC, an IBS estimate as their noisy target.
They carry the marker ``integration``, which the default run leaves out, and
skip when PyBADS 1.5.1 or later, or PyVBMC 1.5 or later, is not installed.
They take a few minutes; to run them::

    uv pip install "pybads>=1.5.1" "pyvbmc>=1.5.0"
    .venv/bin/python -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v
    .venv/bin/python -m pytest -m integration pyibs/testing/integration/test_pyvbmc.py -s -v

Code documentation
------------------

We build the PyIBS documentation with
`Sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_. The
source of the documentation is in the :mainbranch:`docsrc folder <docsrc>`.
From the repository root, with the environment's ``bin`` directory
(``Scripts`` on Windows) first on the ``PATH``::

    PATH="$PWD/.venv/bin:$PATH" make -C docsrc github

This copies the example notebooks into the documentation's source, builds
the documentation, and copies it to ``docs/`` in the repository root, which
git ignores; open ``docs/index.html`` to view it. On Windows, run
``.\make.bat github`` from ``docsrc`` with ``cmd`` instead. The build should
issue no warnings.

If it seems that the documentation does not update correctly (e.g., items
not appearing in the sidebar or table of content), delete the ``docs/``
folder and the cached folder ``docsrc/_build`` before building the
documentation again::

    make -C docsrc clean

(If you are using Windows, run ``.\make.bat clean`` with ``cmd`` instead.)

General structure
.................

Nothing generates the pages of the API: each public class or function has
a hand-written ``.rst`` file under ``docsrc/source/api/``, ``classes`` or
``functions``, with a short introduction and the ``autoclass`` or
``autofunction`` directive that renders its docstring, for example::

    .. autoclass:: pyibs.IBS
       :members:

A new public class or function needs such a file, and an entry in the
table of contents that owns it: ``documentation.rst`` for a headline page,
and ``api/classes/classes.rst`` or ``api/functions/functions.rst``.
Please keep the documentation up to date.

Examples
........

The example notebooks are in ``examples/``, which is installed as
``pyibs.examples``, with the example model in ``psycho_model.py``. The
documentation renders them with their saved outputs, without running them.
To rerun them in place, with PyBADS, PyVBMC, matplotlib and ipykernel
installed in the environment::

    uv pip install nbconvert ipykernel matplotlib "pybads>=1.5.1" "pyvbmc>=1.5.0"
    PATH="$PWD/.venv/bin:$PATH" make -C examples/scripts run

The scripts in ``examples/scripts/`` hold the code of the notebooks; they
are generated from them, not edited, with ``make -B -C examples/scripts``,
which needs nbconvert, IPython, and Black and isort at the versions of the
pre-commit hooks, with the environment first on the ``PATH``.

``git`` commits
---------------

Commits follow the
`conventional commits <https://www.conventionalcommits.org/en/v1.0.0/>`__
style. This makes it easier to collaborate on the project. A cheat sheet can
be found `here <https://cheatography.com/albelop/cheat-sheets/conventional-commits/>`__.

Please do not submit pull requests with unfinished code or code which does
not pass all tests. Work on feature branches whenever possible and sensible.
Changes reach the main branch through pull requests.
`Read this <https://martinfowler.com/bliki/FeatureBranch.html>`__ ::

    git checkout -b <new-feature>
    [... do stuff and commit ...]
    git push -u origin <new-feature>
    [... when finished created pull request on github ...]

If you switch to an existing branch using ``git checkout``, remember to
``pull`` before making any change as it is not done automatically.

Changelog
---------

Each change to results, to the interface, or to what a script sees is listed
in :mainbranch:`CHANGELOG.md <CHANGELOG.md>` under ``Unreleased``, in the
commit that makes it, in one or two sentences written for users. A change
that can stop a script written for the last release, or change its results,
also has a line in the list "Upgrading from" that opens the section. A
change that adds, removes or alters a deliberate difference from
``ibslike.m`` also updates its entry in the
:mainbranch:`catalogue <pyibs/README.md>`.

************
Installation
************

Install PyIBS from PyPI or conda-forge. It requires Python 3.10 or later,
NumPy 2.0 or later, and SciPy 1.13 or later. NumPy and SciPy are its only
runtime dependencies.

1. Install or upgrade with pip::

     python -m pip install --upgrade pyibs

   Or use Conda::

     conda install --channel=conda-forge "pyibs>=1.5"

   The minimum version in the Conda command prevents it from silently
   selecting PyIBS 0.1.0 in an environment with NumPy 1.x.

   Run ``pyibs.check_for_updates()`` to check for a newer release. See the
   :ref:`FAQ <faq-how-do-i-know-whether-a-newer-version-of-pyibs-exists>`.

2. (Optional) To fit models with PyIBS's estimates, as the example
   notebooks do, install `PyBADS <https://acerbilab.github.io/pybads/>`__
   and `PyVBMC <https://acerbilab.org/pyvbmc/>`__, among the lab's
   `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__,
   and `Jupyter Notebook <https://jupyter.org/install>`__ to run the
   notebooks::

     python -m pip install --upgrade "pybads>=1.5.1" "pyvbmc>=1.5" notebook

   The notebooks are included with PyIBS. Find their installation folder
   with::

     python -c "import os, pyibs.examples; print(os.path.dirname(pyibs.examples.__file__))"

To check your installation, install the test dependencies and run the suite::

  python -m pip install --upgrade "pyibs[test]"
  python -m pytest --pyargs pyibs

If PyBADS 1.5.1 or later or PyVBMC 1.5 or later is installed, the suite also
fits the example model with that package. These integration tests take a
few minutes. To skip them, run
``python -m pytest --pyargs pyibs -m "not integration"``.

To install from source, follow the
:doc:`instructions for developers and contributors <development>`.
